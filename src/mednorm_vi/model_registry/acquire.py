"""Governed model acquisition: resolve, pin, download, verify (Audit 0061).

Every previous milestone refused to download a model because nothing here could prove
*what* had been downloaded. This module is that proof. It resolves the current immutable
commit SHA from the Hugging Face API, downloads **only** that revision into a
revision-addressed directory, hashes every file, reads the license from the repository
metadata, and counts parameters from the loaded model rather than from a planning table.

**It fails closed at every step.** A revision that cannot be resolved, a license that
cannot be read, a file whose hash cannot be computed, or a parameter count that cannot
be obtained programmatically all abort the acquisition rather than produce a
half-verified artifact. A model the repository cannot describe exactly is a model it
will not run.

The revision is never hard-coded. Audit 0061's instruction was explicit — "do not invent
or hard-code a revision from memory" — and a remembered SHA is indistinguishable from a
fabricated one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ACQUISITION_VERSION = "governed-model-acquisition-v1"

#: A registerable checkpoint must expose a config and at least one weights file.
#: Tokenizer identity is resolved from the config's declared backbone rather than by
#: filename: GLiNER ships no tokenizer files of its own, it names the backbone whose
#: tokenizer it loads, and requiring a `tokenizer*` file would reject a perfectly
#: well-specified model for a packaging convention it does not use.
REQUIRED_CONFIG_PATTERNS: tuple[str, ...] = ("config",)
REQUIRED_WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin", ".pt")

#: What an acquisition is expected to be. A companion tokenizer legitimately carries no
#: weights, and a model legitimately carries no tokenizer when it names a backbone; the
#: caller declares which, and verification is against that declaration rather than
#: against one shape that would be wrong for the other.
KIND_MODEL = "model"
KIND_TOKENIZER = "tokenizer"
ARTIFACT_KINDS: tuple[str, ...] = (KIND_MODEL, KIND_TOKENIZER)


class AcquisitionFailed(RuntimeError):
    """A governed acquisition step could not be verified. Nothing is registered."""


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One downloaded file and its digest."""

    name: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class AcquiredModel:
    """A fully verified local model artifact."""

    model_id: str
    pinned_revision: str
    license_id: str
    local_path: str
    files: tuple[FileRecord, ...] = field(default_factory=tuple)
    total_bytes: int = 0
    parameter_count: int | None = None
    parameter_count_status: str = "NOT_VERIFIED"
    source_url: str = ""
    tokenizer_identity: str = ""
    config_identity: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "acquisition_version": ACQUISITION_VERSION,
            "model_id": self.model_id,
            "pinned_revision": self.pinned_revision,
            "license": self.license_id,
            "source_url": self.source_url,
            "local_path": self.local_path,
            "files": [f.as_dict() for f in self.files],
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "parameter_count": self.parameter_count,
            "parameter_count_status": self.parameter_count_status,
            "tokenizer_identity": self.tokenizer_identity,
            "config_identity": self.config_identity,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_revision(model_id: str) -> tuple[str, str]:
    """Resolve ``model_id`` to its current immutable commit SHA and license.

    The SHA comes from the Hub API, never from this file. Both values are required:
    a model whose license cannot be read is not acquirable under the project's
    governance rules, however permissive it might turn out to be.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise AcquisitionFailed(f"huggingface_hub unavailable: {exc}") from exc

    info = HfApi().model_info(repo_id=model_id, files_metadata=False)
    revision = str(getattr(info, "sha", "") or "")
    if len(revision) != 40 or not all(c in "0123456789abcdef" for c in revision):
        raise AcquisitionFailed(
            f"{model_id}: could not resolve a 40-hex immutable commit SHA (got {revision!r})"
        )
    card = getattr(info, "card_data", None) or {}
    license_id = ""
    if hasattr(card, "get"):
        license_id = str(card.get("license", "") or "")
    if not license_id:
        license_id = str((getattr(info, "tags", None) or []) and "" or "")
        for tag in getattr(info, "tags", []) or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_id = tag.split(":", 1)[1]
                break
    if not license_id:
        raise AcquisitionFailed(f"{model_id}: license could not be read from repository metadata")
    return revision, license_id


def acquire_model(
    *,
    model_id: str,
    root: str | Path,
    allow_patterns: Sequence[str] | None = None,
    expected_license: str | None = None,
    kind: str = KIND_MODEL,
) -> AcquiredModel:
    """Download exactly one resolved revision into ``root/<model>/<revision>/``."""
    if kind not in ARTIFACT_KINDS:
        raise AcquisitionFailed(f"unknown artifact kind {kind!r}; expected one of {ARTIFACT_KINDS}")
    revision, license_id = resolve_revision(model_id)
    if expected_license is not None and license_id.lower() != expected_license.lower():
        raise AcquisitionFailed(
            f"{model_id}: license {license_id!r} does not match expected {expected_license!r}"
        )

    target = Path(root) / revision
    if target.exists() and any(target.iterdir()):
        raise AcquisitionFailed(
            f"refusing to overwrite an existing model directory: {target}. "
            "A revision-addressed path is immutable by construction."
        )
    target.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise AcquisitionFailed(f"huggingface_hub unavailable: {exc}") from exc

    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(target),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    except Exception as exc:  # noqa: BLE001 - any failure must abort, not degrade
        shutil.rmtree(target, ignore_errors=True)
        raise AcquisitionFailed(f"{model_id}@{revision}: download failed: {exc}") from exc

    files = tuple(
        FileRecord(
            name=path.relative_to(target).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(target.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    )
    if not files:
        raise AcquisitionFailed(f"{model_id}@{revision}: no files were downloaded")
    names = " ".join(f.name for f in files)
    for pattern in REQUIRED_CONFIG_PATTERNS:
        if pattern not in names:
            raise AcquisitionFailed(
                f"{model_id}@{revision}: no {pattern} file present; refusing to register"
            )
    if kind == KIND_MODEL and not any(f.name.endswith(REQUIRED_WEIGHT_SUFFIXES) for f in files):
        raise AcquisitionFailed(
            f"{model_id}@{revision}: no weights file present; refusing to register a model"
        )
    if kind == KIND_TOKENIZER and not any("tokenizer" in f.name or "spm" in f.name for f in files):
        raise AcquisitionFailed(
            f"{model_id}@{revision}: no tokenizer file present; refusing to register a tokenizer"
        )

    config_identity, tokenizer_identity = _identities(target, files)

    return AcquiredModel(
        model_id=model_id,
        pinned_revision=revision,
        license_id=license_id,
        local_path=target.as_posix(),
        files=files,
        total_bytes=sum(f.size_bytes for f in files),
        source_url=f"https://huggingface.co/{model_id}/tree/{revision}",
        config_identity=config_identity,
        tokenizer_identity=tokenizer_identity,
    )


def _identities(target: Path, files: Sequence[FileRecord]) -> tuple[str, str]:
    """Config digest and the backbone whose tokenizer this checkpoint expects.

    Recorded rather than assumed: a checkpoint that silently changed backbone would
    change tokenization and therefore every offset it produces.
    """
    config_files = [f for f in files if f.name.endswith(".json") and "config" in f.name]
    if not config_files:
        raise AcquisitionFailed("no config json found; cannot record config identity")
    primary = config_files[0]
    payload = json.loads((target / primary.name).read_text(encoding="utf-8"))
    backbone = str(
        payload.get("model_name") or payload.get("_name_or_path") or payload.get("model_type") or ""
    )
    if not backbone:
        raise AcquisitionFailed(
            f"{primary.name}: declares no backbone model name; tokenizer identity unverifiable"
        )
    return f"{primary.name}:{primary.sha256}", backbone


def count_parameters(model: Any) -> int:
    """Programmatic parameter count from a loaded torch module.

    Deliberately not read from a config or a planning table: spec §17's budget is a
    statement about what the process actually loads, and Audit 0051 recorded a case
    where a declared estimate and the measured count differed.
    """
    inner = getattr(model, "model", model)
    params = getattr(inner, "parameters", None)
    if params is None:
        raise AcquisitionFailed("loaded object exposes no .parameters(); cannot count")
    total = sum(int(p.numel()) for p in params())
    if total <= 0:
        raise AcquisitionFailed("parameter count resolved to zero; refusing to register")
    return total


__all__ = [
    "ACQUISITION_VERSION",
    "ARTIFACT_KINDS",
    "KIND_MODEL",
    "KIND_TOKENIZER",
    "REQUIRED_CONFIG_PATTERNS",
    "REQUIRED_WEIGHT_SUFFIXES",
    "AcquiredModel",
    "AcquisitionFailed",
    "FileRecord",
    "acquire_model",
    "count_parameters",
    "resolve_revision",
    "sha256_file",
]
