"""E4 Colab runtime I/O robustness (Audit 0040).

A real A100 full-training attempt passed the complete alignment preflight and then
died when the notebook opened the governed train split a *second* time:

    OSError: [Errno 107] Transport endpoint is not connected

The split paths were still on the long-lived Google Drive FUSE mount. Recovering
by hand also exposed a second problem: building all 33,826 train W2NER contracts —
each carrying an O(n²) relation grid — into one Python list drove system RAM to
roughly 30 GB *before* the encoder was even loaded, and a failed attempt left
partially built objects reachable through the notebook traceback, so a retry
allocated on top of that.

This module fixes both classes of problem, and every piece is injectable so it can
be tested without ``google.colab``, without Drive, and without a GPU:

* :class:`DriveAdapter` / :func:`ensure_drive_healthy` — a real read/stat probe and
  **at most one** bounded remount;
* :func:`materialize_governed_splits` — copy only train and validation to local
  ``/content`` storage, atomically and hash-verified, before any heavy work;
* :class:`GovernedW2NERContractSource` — a repeatable, memory-bounded iterator so
  no full-corpus contract list ever exists;
* :class:`BoundedAlignmentReport` — a fixed-size sample plus scalar aggregates;
* :func:`persist_artifact` — local-first, reload-checked, hash-verified writes to
  Drive that stop cleanly instead of silently continuing.

Nothing here trains, and internal_test is refused by name.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DRIVE_HEALTH_PROBE_VERSION = "e4-drive-health-probe-v1"
LOCAL_MATERIALIZATION_VERSION = "e4-local-split-materialization-v1"
CONTRACT_STREAM_VERSION = "e4-governed-contract-stream-v1"
ARTIFACT_PERSISTENCE_VERSION = "e4-local-first-persistence-v1"

# Default local (non-Drive) runtime root used by the Colab notebook.
DEFAULT_RUNTIME_ROOT = "/content/mednorm_vi_runtime"

# internal_test is frozen. It is refused by name so a generic directory-wide copy
# can never smuggle it into the runtime.
FORBIDDEN_SPLIT_NAMES = frozenset({"internal_test"})

# Copy in bounded chunks; a governed JSONL is never loaded whole into memory.
COPY_CHUNK_BYTES = 1 << 20

# Markers of a broken FUSE mount, matched case-insensitively against the message.
TRANSPORT_FAILURE_MARKERS: tuple[str, ...] = (
    "transport endpoint is not connected",
    "socket not connected",
    "input/output error",
)
TRANSPORT_ERRNOS: frozenset[int] = frozenset({errno.ENOTCONN, errno.EIO, errno.ESHUTDOWN})


class DriveHealthError(RuntimeError):
    """Raised when Drive is unusable and the bounded recovery did not fix it."""


class ArtifactPersistenceError(RuntimeError):
    """Raised when an artifact could not be persisted with a verified digest."""


class GovernedStreamError(RuntimeError):
    """Raised when a governed streaming source violates its contract."""


# ---------------------------------------------------------------------------
# A. Drive health and bounded remount
# ---------------------------------------------------------------------------


def is_transport_failure(error: BaseException) -> bool:
    """True for a broken-FUSE failure, false for an ordinary missing/bad file.

    The distinction matters: a transport failure is worth one remount, while a
    missing or corrupted source file must surface immediately and never be masked
    by remounting.
    """
    if isinstance(error, OSError) and error.errno in TRANSPORT_ERRNOS:
        return True
    message = str(error).lower()
    return any(marker in message for marker in TRANSPORT_FAILURE_MARKERS)


@dataclass(frozen=True, slots=True)
class DriveAdapter:
    """Injected mount operations, so tests never import ``google.colab``."""

    mount: Callable[[], None]
    flush_and_unmount: Callable[[], None] | None = None
    force_unmount: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class DriveHealthReport:
    drive_health_checked: bool
    drive_healthy: bool
    drive_remount_attempted: bool
    drive_remount_succeeded: bool
    drive_health_probe_version: str = DRIVE_HEALTH_PROBE_VERSION
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "drive_health_checked": self.drive_health_checked,
            "drive_healthy": self.drive_healthy,
            "drive_remount_attempted": self.drive_remount_attempted,
            "drive_remount_succeeded": self.drive_remount_succeeded,
            "drive_health_probe_version": self.drive_health_probe_version,
            "detail": self.detail,
        }


def default_drive_probe(
    mount_point: str | Path, drive_root: str | Path,
    *, probe_files: Sequence[str | Path] = (),
) -> None:
    """Actually touch the filesystem. ``Path.exists()`` alone is not a probe.

    A broken FUSE endpoint frequently still answers ``exists()`` while every real
    operation raises ENOTCONN, which is precisely how the real run failed.
    """
    mount = Path(mount_point)
    root = Path(drive_root)
    os.stat(mount)  # raises OSError(ENOTCONN) on a broken endpoint
    if not mount.is_dir():
        raise DriveHealthError(f"Drive mount point is not a directory: {mount}")
    os.stat(root)
    if not root.is_dir():
        raise DriveHealthError(f"Drive root is unavailable: {root}")
    # A directory listing and a real byte read exercise the transport itself.
    next(iter(os.scandir(root)), None)
    for probe_file in probe_files:
        target = Path(probe_file)
        os.stat(target)
        with target.open("rb") as handle:
            handle.read(1)


def ensure_drive_healthy(
    mount_point: str | Path,
    drive_root: str | Path,
    *,
    adapter: DriveAdapter,
    probe: Callable[[], None] | None = None,
    probe_files: Sequence[str | Path] = (),
    max_remount_attempts: int = 1,
) -> DriveHealthReport:
    """Probe Drive, and recover with **at most one** remount.

    A healthy Drive is never remounted. A non-transport failure (a missing or
    corrupted source file) is re-raised untouched rather than hidden behind a
    remount. There is no retry loop: the budget is exhausted after
    ``max_remount_attempts`` and the failure is reported.
    """
    if max_remount_attempts < 0:
        raise ValueError("max_remount_attempts must be >= 0")
    run_probe = probe or (
        lambda: default_drive_probe(mount_point, drive_root, probe_files=probe_files))

    try:
        run_probe()
    except Exception as first_error:  # noqa: BLE001 - classified immediately below
        if not is_transport_failure(first_error):
            raise
        if max_remount_attempts == 0:
            raise DriveHealthError(
                f"Drive transport failure and no remount budget: {first_error}"
            ) from first_error

        # Bounded recovery: flush politely if we can, force-unmount only if we
        # must, then a single explicit force_remount.
        if adapter.flush_and_unmount is not None:
            try:
                adapter.flush_and_unmount()
            except Exception:  # noqa: BLE001 - a broken mount often cannot flush
                if adapter.force_unmount is not None:
                    try:
                        adapter.force_unmount()
                    except Exception:  # noqa: BLE001 - best effort before remount
                        pass
        elif adapter.force_unmount is not None:
            try:
                adapter.force_unmount()
            except Exception:  # noqa: BLE001 - best effort before remount
                pass

        adapter.mount()
        try:
            run_probe()
        except Exception as second_error:  # noqa: BLE001 - bounded: no further retry
            raise DriveHealthError(
                "Drive is still unusable after one remount attempt: "
                f"{second_error}"
            ) from second_error
        return DriveHealthReport(
            drive_health_checked=True, drive_healthy=True,
            drive_remount_attempted=True, drive_remount_succeeded=True,
            detail=f"recovered after transport failure: {first_error}")

    return DriveHealthReport(
        drive_health_checked=True, drive_healthy=True,
        drive_remount_attempted=False, drive_remount_succeeded=False,
        detail="healthy on first probe")


# ---------------------------------------------------------------------------
# B. Local governed-split materialization
# ---------------------------------------------------------------------------


def sha256_and_size(path: str | Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> tuple[str, int]:
    """Streaming digest + size. Never reads the whole file into memory."""
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


@dataclass(frozen=True, slots=True)
class MaterializedSplit:
    """One governed split copied to local runtime storage."""

    split: str
    source_path: str
    runtime_path: str
    sha256: str
    size_bytes: int
    reused: bool
    materialization_version: str = LOCAL_MATERIALIZATION_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_path": self.source_path,
            "runtime_path": self.runtime_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "reused": self.reused,
            "local_materialization_version": self.materialization_version,
        }


def materialize_split(
    split: str,
    source_path: str | Path,
    runtime_dir: str | Path,
    expected_sha256: str,
    *,
    chunk_bytes: int = COPY_CHUNK_BYTES,
) -> MaterializedSplit:
    """Copy one governed split to local storage, atomically and hash-verified.

    The governed SHA-256 stays authoritative: the copy is accepted only when its
    digest matches exactly, so materialization cannot alter examples, ordering,
    text, offsets, entities or split identity.
    """
    if split in FORBIDDEN_SPLIT_NAMES:
        raise GovernedStreamError(
            f"refusing to materialize the frozen split {split!r}")
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64:
        raise GovernedStreamError(
            f"expected SHA-256 for {split!r} must be 64 hex characters, got "
            f"{len(expected)}")

    destination = Path(runtime_dir) / f"{split}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        digest, size = sha256_and_size(destination, chunk_bytes=chunk_bytes)
        if digest == expected:
            return MaterializedSplit(
                split=split, source_path=str(source_path),
                runtime_path=str(destination), sha256=digest,
                size_bytes=size, reused=True)

    source = Path(source_path)
    source_size = source.stat().st_size
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    hasher = hashlib.sha256()
    written = 0
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            while True:
                chunk = reader.read(chunk_bytes)
                if not chunk:
                    break
                writer.write(chunk)
                hasher.update(chunk)
                written += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if written != source_size:
            raise GovernedStreamError(
                f"{split}: copied {written} bytes but the source is {source_size}")
        if hasher.hexdigest() != expected:
            raise GovernedStreamError(
                f"{split}: local copy digest {hasher.hexdigest()} does not match the "
                f"governed digest {expected}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    # Re-verify after the replace, so a truncated or swapped file cannot survive.
    final_digest, final_size = sha256_and_size(destination, chunk_bytes=chunk_bytes)
    if final_digest != expected or final_size != source_size:
        destination.unlink(missing_ok=True)
        raise GovernedStreamError(
            f"{split}: materialized file failed post-replace verification")
    return MaterializedSplit(
        split=split, source_path=str(source), runtime_path=str(destination),
        sha256=final_digest, size_bytes=final_size, reused=False)


def materialize_governed_splits(
    sources: Mapping[str, tuple[str | Path, str]],
    runtime_dir: str | Path,
    *,
    chunk_bytes: int = COPY_CHUNK_BYTES,
) -> dict[str, MaterializedSplit]:
    """Materialize each ``split -> (source path, governed sha256)`` locally."""
    forbidden = sorted(set(sources) & FORBIDDEN_SPLIT_NAMES)
    if forbidden:
        raise GovernedStreamError(
            f"refusing to materialize frozen split(s): {', '.join(forbidden)}")
    return {
        split: materialize_split(
            split, source_path, runtime_dir, expected, chunk_bytes=chunk_bytes)
        for split, (source_path, expected) in sorted(sources.items())
    }


def materialization_summary(
    materialized: Mapping[str, MaterializedSplit],
    sources: Mapping[str, tuple[str | Path, str]],
) -> dict[str, Any]:
    """Config/manifest fields describing where the corpus was read from."""
    train = materialized["train"]
    validation = materialized["validation"]
    return {
        "drive_source_train_path": str(sources["train"][0]),
        "drive_source_validation_path": str(sources["validation"][0]),
        "runtime_train_path": train.runtime_path,
        "runtime_validation_path": validation.runtime_path,
        "train_sha256": train.sha256,
        "validation_sha256": validation.sha256,
        "train_size_bytes": train.size_bytes,
        "validation_size_bytes": validation.size_bytes,
        "local_materialization_version": LOCAL_MATERIALIZATION_VERSION,
        "local_materialization_reused": {
            "train": train.reused, "validation": validation.reused},
        "internal_test_materialized": False,
    }


# ---------------------------------------------------------------------------
# C/D. Memory-bounded contract streaming and bounded reporting
# ---------------------------------------------------------------------------


@dataclass
class BoundedAlignmentReport:
    """A fixed-size sample plus scalar aggregates — never one entry per example.

    The previous notebook kept one report dictionary per example, i.e. 34,871 of
    them, on top of the contracts themselves.
    """

    sample_limit: int = 3
    samples: list[dict[str, Any]] = field(default_factory=list)
    examples: int = 0
    max_atomic_words: int = 0
    max_model_words: int = 0
    max_encoded_tokens: int = 0
    label_count: int = 0

    def observe(self, report: Mapping[str, Any]) -> None:
        self.examples += 1
        self.max_atomic_words = max(
            self.max_atomic_words, int(report.get("atomic_word_count", 0)))
        self.max_model_words = max(
            self.max_model_words, int(report.get("model_word_count", 0)))
        self.max_encoded_tokens = max(
            self.max_encoded_tokens, int(report.get("encoded_tokens", 0)))
        self.label_count = max(self.label_count, int(report.get("label_count", 0)))
        if len(self.samples) < self.sample_limit:
            self.samples.append(dict(report))

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "max_atomic_words": self.max_atomic_words,
            "max_model_words": self.max_model_words,
            "max_encoded_tokens": self.max_encoded_tokens,
            "label_count": self.label_count,
            "sample_limit": self.sample_limit,
            "samples": [dict(sample) for sample in self.samples],
            "retained_report_count": len(self.samples),
        }


@dataclass(frozen=True, slots=True)
class GovernedW2NERContractSource:
    """A repeatable, memory-bounded source of governed W2NER contracts.

    Iterating opens the local JSONL and yields **one** contract at a time. Each
    ``iter()`` is an independent pass, so an epoch and a validation sweep each get
    a fresh iterator without anything being retained between them. No relation
    grid, projection or alignment report survives past its own iteration step.
    """

    split: str
    path: str
    expected_sha256: str
    expected_example_count: int
    tokenizer: Any
    build_contract: Callable[[str, Mapping[str, Any]], tuple[Any, dict[str, Any]]]
    max_words: int
    max_model_tokens: int
    input_contract_version: str
    max_rows: int | None = None
    stream_version: str = CONTRACT_STREAM_VERSION

    def __post_init__(self) -> None:
        if self.split in FORBIDDEN_SPLIT_NAMES:
            raise GovernedStreamError(
                f"refusing to stream the frozen split {self.split!r}")
        if self.max_rows is not None and self.max_rows <= 0:
            raise GovernedStreamError("max_rows must be positive when supplied")

    @property
    def example_count(self) -> int:
        """Examples this source yields, honouring a bounded smoke row limit."""
        if self.max_rows is None:
            return self.expected_example_count
        return min(self.max_rows, self.expected_example_count)

    def verify_identity(self) -> str:
        """Confirm the local file is still the governed split before streaming."""
        digest, _size = sha256_and_size(self.path)
        if digest != self.expected_sha256:
            raise GovernedStreamError(
                f"{self.split}: local split digest {digest} does not match the "
                f"governed digest {self.expected_sha256}")
        return digest

    def iter_contracts(
        self, *, reporter: BoundedAlignmentReport | None = None,
    ) -> Iterator[Any]:
        """Yield contracts one at a time. Deterministic file order is preserved."""
        import json

        yielded = 0
        with Path(self.path).open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if self.max_rows is not None and index >= self.max_rows:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                contract, report = self.build_contract(self.split, row)
                if reporter is not None:
                    reporter.observe({"row_index": index, **report})
                yielded += 1
                yield contract
                # The caller's reference is the only one; nothing is accumulated
                # here, so the contract and its O(n^2) grid become collectable as
                # soon as the consumer moves on.
                del contract, report, row
        if yielded == 0:
            raise GovernedStreamError(f"{self.split}: no contracts were produced")
        if self.max_rows is None and yielded != self.expected_example_count:
            raise GovernedStreamError(
                f"{self.split}: streamed {yielded} examples but the governed split "
                f"declares {self.expected_example_count}")

    def __iter__(self) -> Iterator[Any]:
        return self.iter_contracts()

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "runtime_path": self.path,
            "expected_sha256": self.expected_sha256,
            "expected_example_count": self.expected_example_count,
            "example_count": self.example_count,
            "max_words": self.max_words,
            "max_model_tokens": self.max_model_tokens,
            "input_contract_version": self.input_contract_version,
            "max_rows": self.max_rows,
            "stream_version": self.stream_version,
        }


def assert_not_materialized(candidate: Any, *, label: str) -> None:
    """Structural guard: full mode must never hold a contract list/tuple.

    This is a *structural* assertion about the object handed to training, not a
    claim about a universal RAM ceiling.
    """
    if isinstance(candidate, (list, tuple, set, dict)):
        raise GovernedStreamError(
            f"{label} must be a repeatable streaming source, not a materialized "
            f"{type(candidate).__name__} of contracts")


# ---------------------------------------------------------------------------
# F. Memory reporting
# ---------------------------------------------------------------------------


def memory_snapshot(label: str) -> dict[str, Any]:
    """Process RSS and available system RAM, best effort and dependency-free.

    Reported as observation only: no universal RAM ceiling is claimed from it.
    """
    snapshot: dict[str, Any] = {"label": label, "rss_bytes": -1,
                                "available_bytes": -1, "source": "unavailable"}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    snapshot["rss_bytes"] = int(line.split()[1]) * 1024
                    snapshot["source"] = "proc"
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    snapshot["available_bytes"] = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    for key in ("rss_bytes", "available_bytes"):
        value = snapshot[key]
        snapshot[key.replace("_bytes", "_gib")] = (
            round(value / (1 << 30), 3) if value >= 0 else -1.0)
    return snapshot


# ---------------------------------------------------------------------------
# G. Local-first verified persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersistedArtifact:
    name: str
    staging_path: str
    persistent_path: str
    sha256: str
    verified: bool
    remount_attempted: bool
    atomic_replace_used: bool
    persistence_version: str = ARTIFACT_PERSISTENCE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "staging_path": self.staging_path,
            "persistent_path": self.persistent_path,
            "sha256": self.sha256,
            "verified": self.verified,
            "remount_attempted": self.remount_attempted,
            "atomic_replace_used": self.atomic_replace_used,
            "persistence_version": self.persistence_version,
        }


def _atomic_replace(temporary: Path, target: Path) -> bool:
    """``os.replace`` when the filesystem supports it, else a bounded fallback."""
    try:
        os.replace(temporary, target)
        return True
    except OSError:
        shutil.copyfile(temporary, target)
        temporary.unlink(missing_ok=True)
        return False


def persist_artifact(
    name: str,
    staging_path: str | Path,
    persistent_path: str | Path,
    *,
    drive_health: Callable[[], DriveHealthReport],
    reload_check: Callable[[Path], None] | None = None,
    chunk_bytes: int = COPY_CHUNK_BYTES,
    max_remount_attempts: int = 1,
) -> PersistedArtifact:
    """Write local-first, verify, then sync to Drive with a verified digest.

    Order: hash the staged file, prove it reloads, check Drive, copy to a
    temporary persistent path, atomically replace, re-hash the persistent copy and
    require equality. A transport failure buys **one** bounded remount and one
    retry; after that the staged local file is preserved and the failure is
    reported rather than silently continuing.
    """
    staging = Path(staging_path)
    if not staging.is_file():
        raise ArtifactPersistenceError(f"{name}: staged artifact is missing: {staging}")
    local_digest, _size = sha256_and_size(staging, chunk_bytes=chunk_bytes)
    if reload_check is not None:
        reload_check(staging)

    target = Path(persistent_path)
    remount_attempted = False
    last_error: BaseException | None = None

    for attempt in range(max_remount_attempts + 1):
        try:
            report = drive_health()
            remount_attempted = remount_attempted or report.drive_remount_attempted
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
            try:
                shutil.copyfile(staging, temporary)
                atomic = _atomic_replace(temporary, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            persistent_digest, _persistent_size = sha256_and_size(
                target, chunk_bytes=chunk_bytes)
            if persistent_digest != local_digest:
                raise ArtifactPersistenceError(
                    f"{name}: persistent digest {persistent_digest} does not match "
                    f"the staged digest {local_digest}")
            return PersistedArtifact(
                name=name, staging_path=str(staging), persistent_path=str(target),
                sha256=local_digest, verified=True,
                remount_attempted=remount_attempted, atomic_replace_used=atomic)
        except Exception as error:  # noqa: BLE001 - classified for bounded retry
            last_error = error
            if attempt >= max_remount_attempts or not is_transport_failure(error):
                break
            remount_attempted = True

    raise ArtifactPersistenceError(
        f"{name}: persistent checkpoint custody FAILED; the verified local copy is "
        f"preserved at {staging} (sha256 {local_digest}). This epoch must NOT be "
        f"treated as resumable from Drive. Cause: {last_error}")


def persist_artifacts(
    items: Sequence[tuple[str, str | Path, str | Path]],
    *,
    drive_health: Callable[[], DriveHealthReport],
    reload_checks: Mapping[str, Callable[[Path], None]] | None = None,
) -> dict[str, PersistedArtifact]:
    """Persist several artifacts under the same local-first policy."""
    checks = dict(reload_checks or {})
    persisted: dict[str, PersistedArtifact] = {}
    for name, staging, target in items:
        persisted[name] = persist_artifact(
            name, staging, target, drive_health=drive_health,
            reload_check=checks.get(name))
    return persisted


__all__ = [
    "ARTIFACT_PERSISTENCE_VERSION",
    "CONTRACT_STREAM_VERSION",
    "COPY_CHUNK_BYTES",
    "DEFAULT_RUNTIME_ROOT",
    "DRIVE_HEALTH_PROBE_VERSION",
    "FORBIDDEN_SPLIT_NAMES",
    "LOCAL_MATERIALIZATION_VERSION",
    "TRANSPORT_ERRNOS",
    "TRANSPORT_FAILURE_MARKERS",
    "ArtifactPersistenceError",
    "BoundedAlignmentReport",
    "DriveAdapter",
    "DriveHealthError",
    "DriveHealthReport",
    "GovernedStreamError",
    "GovernedW2NERContractSource",
    "MaterializedSplit",
    "PersistedArtifact",
    "assert_not_materialized",
    "default_drive_probe",
    "ensure_drive_healthy",
    "is_transport_failure",
    "materialization_summary",
    "materialize_governed_splits",
    "materialize_split",
    "memory_snapshot",
    "persist_artifact",
    "persist_artifacts",
    "sha256_and_size",
]
