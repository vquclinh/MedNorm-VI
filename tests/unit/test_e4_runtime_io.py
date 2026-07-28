"""E4 Colab I/O robustness, local materialization and streaming (Audit 0040).

The anchor failure is the real A100 run: the alignment preflight passed, then the
notebook reopened the governed train split on the Drive FUSE mount and hit
``OSError: [Errno 107] Transport endpoint is not connected``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from mednorm_vi.training.phase2.e4.runtime_io import (
    ARTIFACT_PERSISTENCE_VERSION,
    CONTRACT_STREAM_VERSION,
    DRIVE_HEALTH_PROBE_VERSION,
    FORBIDDEN_SPLIT_NAMES,
    LOCAL_MATERIALIZATION_VERSION,
    ArtifactPersistenceError,
    BoundedAlignmentReport,
    DriveAdapter,
    DriveHealthError,
    GovernedStreamError,
    GovernedW2NERContractSource,
    assert_not_materialized,
    default_drive_probe,
    ensure_drive_healthy,
    is_transport_failure,
    materialization_summary,
    materialize_governed_splits,
    materialize_split,
    memory_snapshot,
    persist_artifact,
    sha256_and_size,
)

REPO = Path(__file__).resolve().parents[2]

# The real governed digests. The Audit-0040 report pasted the train value with 63
# characters; the authoritative 64-character digest is used everywhere.
GOVERNED_TRAIN_SHA256 = "892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a"
GOVERNED_VALIDATION_SHA256 = "ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103"

TRANSPORT_MESSAGE = "Transport endpoint is not connected"


def _write(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _Recorder:
    """Counts mount operations so 'exactly one remount' is provable."""

    def __init__(self) -> None:
        self.mounts = 0
        self.flushes = 0
        self.force_unmounts = 0

    def adapter(self, *, flush: bool = True, force: bool = False) -> DriveAdapter:
        def mount() -> None:
            self.mounts += 1

        def flush_and_unmount() -> None:
            self.flushes += 1

        def force_unmount() -> None:
            self.force_unmounts += 1

        return DriveAdapter(
            mount=mount,
            flush_and_unmount=flush_and_unmount if flush else None,
            force_unmount=force_unmount if force else None,
        )


# ---------------------------------------------------------------------------
# A. Drive health and bounded remount
# ---------------------------------------------------------------------------


def test_transport_failures_are_recognised_by_errno_and_message() -> None:
    assert is_transport_failure(OSError(errno.ENOTCONN, TRANSPORT_MESSAGE))
    assert is_transport_failure(OSError(TRANSPORT_MESSAGE))
    assert is_transport_failure(RuntimeError("Transport endpoint is not connected"))
    assert is_transport_failure(OSError(errno.EIO, "Input/output error"))


def test_ordinary_missing_file_is_not_a_transport_failure() -> None:
    assert not is_transport_failure(FileNotFoundError("no such file"))
    assert not is_transport_failure(ValueError("corrupt json"))


def test_healthy_drive_is_never_remounted() -> None:
    recorder = _Recorder()
    report = ensure_drive_healthy(
        "/mount", "/mount/root", adapter=recorder.adapter(), probe=lambda: None)
    assert report.drive_healthy is True
    assert report.drive_health_checked is True
    assert report.drive_remount_attempted is False
    assert report.drive_remount_succeeded is False
    assert report.drive_health_probe_version == DRIVE_HEALTH_PROBE_VERSION
    assert recorder.mounts == 0
    assert recorder.flushes == 0


def test_errno_107_triggers_exactly_one_remount_and_recovers() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    def probe() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ENOTCONN, TRANSPORT_MESSAGE)

    report = ensure_drive_healthy(
        "/mount", "/mount/root", adapter=recorder.adapter(), probe=probe)
    assert report.drive_healthy is True
    assert report.drive_remount_attempted is True
    assert report.drive_remount_succeeded is True
    assert recorder.mounts == 1
    assert recorder.flushes == 1
    assert calls["n"] == 2


def test_transport_message_without_errno_also_triggers_one_remount() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    def probe() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(TRANSPORT_MESSAGE)

    report = ensure_drive_healthy(
        "/mount", "/mount/root", adapter=recorder.adapter(), probe=probe)
    assert report.drive_remount_succeeded is True
    assert recorder.mounts == 1


def test_failed_post_remount_probe_raises_and_does_not_retry() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    def probe() -> None:
        calls["n"] += 1
        raise OSError(errno.ENOTCONN, TRANSPORT_MESSAGE)

    with pytest.raises(DriveHealthError, match="after one remount"):
        ensure_drive_healthy(
            "/mount", "/mount/root", adapter=recorder.adapter(), probe=probe)
    # Exactly one remount, exactly two probes: no infinite loop.
    assert recorder.mounts == 1
    assert calls["n"] == 2


def test_zero_remount_budget_reports_the_transport_failure() -> None:
    recorder = _Recorder()
    with pytest.raises(DriveHealthError, match="no remount budget"):
        ensure_drive_healthy(
            "/mount", "/mount/root", adapter=recorder.adapter(),
            probe=lambda: (_ for _ in ()).throw(OSError(errno.ENOTCONN, TRANSPORT_MESSAGE)),
            max_remount_attempts=0)
    assert recorder.mounts == 0


def test_a_corrupted_or_missing_source_file_is_never_hidden_by_a_remount() -> None:
    recorder = _Recorder()

    def probe() -> None:
        raise FileNotFoundError("governed split is missing")

    with pytest.raises(FileNotFoundError):
        ensure_drive_healthy(
            "/mount", "/mount/root", adapter=recorder.adapter(), probe=probe)
    assert recorder.mounts == 0


def test_force_unmount_is_used_only_when_flush_fails() -> None:
    recorder = _Recorder()
    calls = {"n": 0}

    def probe() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ENOTCONN, TRANSPORT_MESSAGE)

    adapter = recorder.adapter(force=True)
    broken = DriveAdapter(
        mount=adapter.mount,
        flush_and_unmount=lambda: (_ for _ in ()).throw(OSError("cannot flush")),
        force_unmount=adapter.force_unmount,
    )
    report = ensure_drive_healthy("/mount", "/mount/root", adapter=broken, probe=probe)
    assert report.drive_remount_succeeded is True
    assert recorder.force_unmounts == 1
    assert recorder.mounts == 1


def test_default_probe_performs_a_real_read(tmp_path: Path) -> None:
    root = tmp_path / "MyDrive"
    target = root / "data" / "train.jsonl"
    _write(target, '{"a": 1}\n')
    default_drive_probe(tmp_path, root, probe_files=(target,))
    with pytest.raises(OSError):
        default_drive_probe(tmp_path, root, probe_files=(root / "absent.jsonl",))


def test_default_probe_rejects_an_unavailable_drive_root(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        default_drive_probe(tmp_path, tmp_path / "missing-root")


# ---------------------------------------------------------------------------
# B. Local governed-split materialization
# ---------------------------------------------------------------------------


def test_copy_is_atomic_hash_verified_and_leaves_no_temporary(tmp_path: Path) -> None:
    source = tmp_path / "drive" / "train.jsonl"
    digest = _write(source, '{"example_id": "a"}\n{"example_id": "b"}\n')
    runtime = tmp_path / "runtime" / "splits"

    result = materialize_split("train", source, runtime, digest)
    assert result.reused is False
    assert result.sha256 == digest
    assert Path(result.runtime_path).read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8")
    assert result.size_bytes == source.stat().st_size
    assert list(runtime.glob("*.tmp-*")) == []
    assert result.materialization_version == LOCAL_MATERIALIZATION_VERSION


def test_a_valid_local_copy_is_reused(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    digest = _write(source, '{"x": 1}\n')
    runtime = tmp_path / "runtime"
    first = materialize_split("train", source, runtime, digest)
    second = materialize_split("train", source, runtime, digest)
    assert first.reused is False
    assert second.reused is True
    assert second.sha256 == digest


def test_a_stale_local_copy_is_replaced(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    digest = _write(source, '{"x": 2}\n')
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "train.jsonl").write_text("stale\n", encoding="utf-8")
    result = materialize_split("train", source, runtime, digest)
    assert result.reused is False
    assert result.sha256 == digest


def test_hash_mismatch_rejects_the_local_copy(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write(source, '{"x": 1}\n')
    runtime = tmp_path / "runtime"
    with pytest.raises(GovernedStreamError, match="does not match the governed digest"):
        materialize_split("train", source, runtime, "0" * 64)
    assert not (runtime / "train.jsonl").exists()
    assert list(runtime.glob("*.tmp-*")) == []


def test_a_truncated_expected_digest_is_rejected(tmp_path: Path) -> None:
    """The Audit-0040 report pasted a 63-character train digest."""
    source = tmp_path / "train.jsonl"
    _write(source, '{"x": 1}\n')
    with pytest.raises(GovernedStreamError, match="64 hex characters"):
        materialize_split("train", source, tmp_path / "runtime",
                          GOVERNED_TRAIN_SHA256[:-1])


def test_internal_test_is_never_materialized(tmp_path: Path) -> None:
    source = tmp_path / "internal_test.jsonl"
    digest = _write(source, '{"x": 1}\n')
    with pytest.raises(GovernedStreamError, match="frozen split"):
        materialize_split("internal_test", source, tmp_path / "runtime", digest)
    with pytest.raises(GovernedStreamError, match="frozen split"):
        materialize_governed_splits(
            {"internal_test": (source, digest)}, tmp_path / "runtime")
    assert "internal_test" in FORBIDDEN_SPLIT_NAMES


def test_summary_reports_drive_sources_and_runtime_paths(tmp_path: Path) -> None:
    train = tmp_path / "drive" / "train.jsonl"
    validation = tmp_path / "drive" / "validation.jsonl"
    train_digest = _write(train, '{"t": 1}\n')
    validation_digest = _write(validation, '{"v": 1}\n')
    sources = {"train": (train, train_digest), "validation": (validation, validation_digest)}
    materialized = materialize_governed_splits(sources, tmp_path / "runtime")
    summary = materialization_summary(materialized, sources)
    assert summary["drive_source_train_path"] == str(train)
    assert summary["drive_source_validation_path"] == str(validation)
    assert summary["runtime_train_path"].endswith("train.jsonl")
    assert "/drive/" not in summary["runtime_train_path"]
    assert summary["train_sha256"] == train_digest
    assert summary["validation_sha256"] == validation_digest
    assert summary["train_size_bytes"] == train.stat().st_size
    assert summary["local_materialization_version"] == LOCAL_MATERIALIZATION_VERSION
    assert summary["local_materialization_reused"] == {"train": False, "validation": False}
    assert summary["internal_test_materialized"] is False


def test_real_governed_splits_materialize_with_their_authoritative_digests(
    tmp_path: Path,
) -> None:
    splits = REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1" / "splits"
    if not (splits / "train.jsonl").is_file():
        pytest.skip("governed corpus is not present locally")
    materialized = materialize_governed_splits({
        "train": (splits / "train.jsonl", GOVERNED_TRAIN_SHA256),
        "validation": (splits / "validation.jsonl", GOVERNED_VALIDATION_SHA256),
    }, tmp_path / "runtime")
    assert materialized["train"].sha256 == GOVERNED_TRAIN_SHA256
    assert materialized["validation"].sha256 == GOVERNED_VALIDATION_SHA256


def test_sha256_and_size_streams_without_loading_the_file(tmp_path: Path) -> None:
    target = tmp_path / "big.jsonl"
    payload = "x" * 10_000
    digest = _write(target, payload)
    assert sha256_and_size(target, chunk_bytes=64) == (digest, len(payload))


# ---------------------------------------------------------------------------
# C. Streaming contract source
# ---------------------------------------------------------------------------


class _FakeContract:
    """Stand-in carrying a grid-sized payload, so retention would be visible."""

    def __init__(self, example_id: str, words: int) -> None:
        self.example_id = example_id
        self.word_count = words
        self.label_count = 7
        self.grid = [[0] * words for _ in range(words)]


def _source(tmp_path: Path, rows: int = 5, **overrides) -> GovernedW2NERContractSource:
    path = tmp_path / "train.jsonl"
    payload = "".join(
        json.dumps({"example_id": f"e{index}", "text": "a b"}) + "\n"
        for index in range(rows))
    digest = _write(path, payload)
    built: list[str] = []

    def build(split: str, row: dict) -> tuple[_FakeContract, dict]:
        built.append(row["example_id"])
        contract = _FakeContract(row["example_id"], words=4)
        return contract, {"split": split, "atomic_word_count": 4,
                          "model_word_count": 3, "encoded_tokens": 6, "label_count": 7}

    kwargs = dict(
        split="train", path=str(path), expected_sha256=digest,
        expected_example_count=rows, tokenizer=object(), build_contract=build,
        max_words=256, max_model_tokens=512, input_contract_version="v1")
    kwargs.update(overrides)
    source = GovernedW2NERContractSource(**kwargs)  # type: ignore[arg-type]
    source_built = built  # noqa: F841 - retained for readability in failures
    return source


def test_streaming_yields_one_contract_at_a_time(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=5)
    iterator = source.iter_contracts()
    first = next(iterator)
    assert isinstance(first, _FakeContract)
    assert first.example_id == "e0"
    remaining = list(iterator)
    assert [item.example_id for item in remaining] == ["e1", "e2", "e3", "e4"]


def test_each_epoch_gets_a_new_independent_iterator(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=3)
    first_epoch = [item.example_id for item in source.iter_contracts()]
    second_epoch = [item.example_id for item in source.iter_contracts()]
    third_epoch = [item.example_id for item in source]
    assert first_epoch == second_epoch == third_epoch == ["e0", "e1", "e2"]


def test_each_validation_pass_gets_a_new_iterator(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=2)
    one = source.iter_contracts()
    two = source.iter_contracts()
    assert one is not two
    assert [item.example_id for item in one] == [item.example_id for item in two]


def test_deterministic_file_order_is_preserved(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=6)
    assert [item.example_id for item in source] == [f"e{i}" for i in range(6)]


def test_the_source_is_not_a_materialized_collection(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=3)
    assert_not_materialized(source, label="train_source")
    for materialized in ([1, 2], (1, 2), {1, 2}, {"a": 1}):
        with pytest.raises(GovernedStreamError, match="streaming source"):
            assert_not_materialized(materialized, label="train_source")


def test_a_short_stream_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=3, expected_example_count=5)
    with pytest.raises(GovernedStreamError, match="streamed 3 examples"):
        list(source.iter_contracts())


def test_max_rows_bounds_a_smoke_stream(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=10, max_rows=8)
    assert source.example_count == 8
    assert len(list(source.iter_contracts())) == 8


def test_identity_is_verified_against_the_governed_digest(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=2)
    assert source.verify_identity() == source.expected_sha256
    Path(source.path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(GovernedStreamError, match="does not match the governed digest"):
        source.verify_identity()


def test_streaming_the_frozen_split_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GovernedStreamError, match="frozen split"):
        _source(tmp_path, split="internal_test")


def test_stream_version_is_recorded(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=1)
    assert source.stream_version == CONTRACT_STREAM_VERSION
    assert source.as_dict()["stream_version"] == CONTRACT_STREAM_VERSION


# ---------------------------------------------------------------------------
# D. Bounded reporting
# ---------------------------------------------------------------------------


def test_reporting_keeps_a_bounded_sample_and_scalar_aggregates(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=50)
    report = BoundedAlignmentReport(sample_limit=3)
    for contract in source.iter_contracts(reporter=report):
        del contract
    assert report.examples == 50
    assert len(report.samples) == 3
    assert report.as_dict()["retained_report_count"] == 3
    assert report.max_atomic_words == 4
    assert report.max_model_words == 3
    assert report.max_encoded_tokens == 6
    assert report.label_count == 7


def test_reporting_tracks_maxima_across_examples() -> None:
    report = BoundedAlignmentReport(sample_limit=1)
    report.observe({"atomic_word_count": 10, "model_word_count": 8,
                    "encoded_tokens": 20, "label_count": 7})
    report.observe({"atomic_word_count": 162, "model_word_count": 162,
                    "encoded_tokens": 213, "label_count": 7})
    assert report.max_atomic_words == 162
    assert report.max_model_words == 162
    assert report.max_encoded_tokens == 213
    assert report.examples == 2
    assert len(report.samples) == 1


# ---------------------------------------------------------------------------
# F. Memory reporting
# ---------------------------------------------------------------------------


def test_memory_snapshot_reports_rss_and_available_ram() -> None:
    snapshot = memory_snapshot("probe")
    assert snapshot["label"] == "probe"
    assert set(snapshot) >= {"rss_bytes", "available_bytes", "rss_gib", "available_gib"}
    # No universal RAM ceiling is asserted here; the values are observations.
    assert snapshot["rss_bytes"] == -1 or snapshot["rss_bytes"] > 0


# ---------------------------------------------------------------------------
# G. Local-first verified persistence
# ---------------------------------------------------------------------------


def _healthy():
    from mednorm_vi.training.phase2.e4.runtime_io import DriveHealthReport

    return DriveHealthReport(True, True, False, False)


def test_local_first_save_verifies_the_persistent_digest(tmp_path: Path) -> None:
    staged = tmp_path / "staging" / "best.pt"
    digest = _write(staged, "checkpoint-bytes")
    target = tmp_path / "drive" / "checkpoints" / "best.pt"
    reloaded: list[Path] = []

    persisted = persist_artifact(
        "best.pt", staged, target, drive_health=_healthy,
        reload_check=reloaded.append)
    assert persisted.verified is True
    assert persisted.sha256 == digest
    assert Path(persisted.persistent_path).read_text(encoding="utf-8") == "checkpoint-bytes"
    assert reloaded == [staged]  # reload check ran BEFORE the Drive sync
    assert persisted.persistence_version == ARTIFACT_PERSISTENCE_VERSION
    assert list(target.parent.glob("*.tmp-*")) == []


def test_a_failing_reload_check_stops_before_touching_drive(tmp_path: Path) -> None:
    staged = tmp_path / "best.pt"
    _write(staged, "bytes")
    target = tmp_path / "drive" / "best.pt"

    def bad_reload(_path: Path) -> None:
        raise AssertionError("checkpoint payload missing keys")

    with pytest.raises(AssertionError):
        persist_artifact("best.pt", staged, target, drive_health=_healthy,
                         reload_check=bad_reload)
    assert not target.exists()


def test_one_bounded_remount_retry_during_persistence(tmp_path: Path) -> None:
    staged = tmp_path / "best.pt"
    digest = _write(staged, "bytes")
    target = tmp_path / "drive" / "best.pt"
    calls = {"n": 0}

    def flaky_health():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ENOTCONN, TRANSPORT_MESSAGE)
        return _healthy()

    persisted = persist_artifact(
        "best.pt", staged, target, drive_health=flaky_health)
    assert persisted.verified is True
    assert persisted.sha256 == digest
    assert persisted.remount_attempted is True
    assert calls["n"] == 2


def test_persistence_failure_stops_and_preserves_the_local_copy(tmp_path: Path) -> None:
    staged = tmp_path / "best.pt"
    digest = _write(staged, "bytes")
    target = tmp_path / "drive" / "best.pt"

    def always_broken():
        raise OSError(errno.ENOTCONN, TRANSPORT_MESSAGE)

    with pytest.raises(ArtifactPersistenceError) as excinfo:
        persist_artifact("best.pt", staged, target, drive_health=always_broken)
    message = str(excinfo.value)
    assert "custody FAILED" in message
    assert str(staged) in message
    assert digest in message
    assert "must NOT be treated as resumable from Drive" in message
    # The verified local copy survives so the epoch's work is not lost.
    assert staged.is_file()


def test_a_missing_staged_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactPersistenceError, match="staged artifact is missing"):
        persist_artifact("best.pt", tmp_path / "absent.pt", tmp_path / "drive" / "best.pt",
                         drive_health=_healthy)


# ---------------------------------------------------------------------------
# H/I. Notebook wiring
# ---------------------------------------------------------------------------


def test_no_model_checkpoint_cache_or_archive_is_tracked_in_git() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip"))
        assert not line.startswith(
            ("artifacts/", "weights/", "caches/", "checkpoint/", ".claude/"))
        assert Path(line).name not in {"CLAUDE.md", "AGENTS.md"}
