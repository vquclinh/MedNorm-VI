"""E4 read-only checkpoint probe (Audit 0044).

Audit 0043 shipped a probe that could never run: `run_collapse_diagnosis` bound
`probes = ()` and called `require_checkpoint()` only for its raise-on-absence
side effect, so arriving weights left section E blocked with an empty reason.

The first two tests here are the regression tests for exactly that bug — present
checkpoints must reach the probe, and a probe that cannot run must say *why*.
The rest lock down that the probe stays forward-only, bounded, one-checkpoint-at-
a-time, and silent about clinical text.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mednorm_vi.mention_factory.w2ner import W2NERLabelVocab
from mednorm_vi.training.phase2.e4_checkpoint_probe import (
    CHECKPOINT_ROLES,
    DEFAULT_SAMPLE_CAPACITY,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_W2NER_HEAD_KEYS,
    BoundedSample,
    CheckpointRestoreError,
    GridLogitReport,
    ProbeDependencyError,
    RestorationReport,
    checkpoint_payload,
    classify_probe_outcome,
    inspect_checkpoint,
)
from mednorm_vi.training.phase2.e4_collapse_diagnosis import (
    ALL_BACKGROUND_LOSS_COLLAPSE,
    CHECKPOINT_RESTORE_FAILURE,
    ROOT_CAUSE_NOT_YET_PROVEN,
    CheckpointEvidenceUnavailable,
    CheckpointProbeReport,
    E4DiagnosisError,
    run_collapse_diagnosis,
    verify_artifact_integrity,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO / "local-artifacts" / "e4_phobert_w2ner_full_v1"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
SPLIT_ROOT = REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1" / "splits"
TRAIN_SPLIT = SPLIT_ROOT / "train.jsonl"
VALIDATION_SPLIT = SPLIT_ROOT / "validation.jsonl"

# The artifact's non-checkpoint files. Pinned in Audit 0043 and re-asserted here:
# a diagnostic that mutated its own evidence would invalidate both audits.
E4_ARTIFACT_SHA256: dict[str, str] = {
    "training_manifest.json":
        "52b8673a2bd5f4b99bda1e6b10eb2cf8debdb08b5ee7eb6a51f5cbe71d948d73",
    "resolved_config.json":
        "c7d47737b46facc1df024f7cc8c2910cc6a9e3de3231f3d904f34b3f4c476a96",
    "validation_metrics.json":
        "c563ea63fde15c174ca43b1c7c7e341e20e0c3f3a6e941f271f56311a5b1cd50",
    "grid_target_statistics.json":
        "0d959c8f4eea428babece37106175d3f0db7a10cbfb3a5ea2f837c4ae0bbf05a",
    "e4_alignment_diagnostic.json":
        "86ad2fc50bb9084d3a6ba2328c9b571a77aa0067c82fcc0481bb70b5d2c34bc7",
    "logs/training_history.jsonl":
        "2a2127de035cfd8f4875b8026c091a34636e7b20fbef984886670506382363b6",
    "logs/training_progress.jsonl":
        "30e9e359a04b19f39a0b52d295f34b480e45f9542df182ac31023dceb9177ff8",
}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


checkpoints_required = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "best.pt").is_file() or not _torch_available(),
    reason="E4 checkpoints or torch are not available in this environment")
artifact_required = pytest.mark.skipif(
    not ARTIFACT_DIR.is_dir(), reason="local E4 artifact is not present")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _probe_stub(role: str, **overrides: Any) -> CheckpointProbeReport:
    fields: dict[str, Any] = {
        "role": role,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[role],
        "epoch": 2 if role == "best" else 12,
        "predicted_mention_total": 0,
        "gold_mention_total": 1991,
        "true_positives": 0,
        "exact_precision": 0.0,
        "exact_recall": 0.0,
        "exact_f1": 0.0,
        "predictions_by_entity_type": {},
        "predicted_labels_by_class": {"NONE": 1_491_764},
        "background_label_count": 1_491_764,
        "non_background_label_count": 0,
        "background_logit_quantiles": {"p50": 13.6},
        "strongest_non_background_logit_quantiles": {"p50": -3.0},
        "gold_positive_cell_predicted_labels": {"NONE": 6638},
        "gold_positive_cell_background_rate": 1.0,
        "decoder_input_positive_relations": 0,
        "decoder_output_mention_count": 0,
        "w2ner_head_keys_restored": EXPECTED_W2NER_HEAD_KEYS,
    }
    fields.update(overrides)
    return CheckpointProbeReport(**fields)


def _diagnosis(probe_runner: Any) -> Any:
    return run_collapse_diagnosis(
        artifact_dir=ARTIFACT_DIR,
        split_paths={"train": TRAIN_SPLIT, "validation": VALIDATION_SPLIT},
        max_words=256, limit=11_000, probe_runner=probe_runner)


# ---------------------------------------------------------------------------
# The Audit-0043 regression: present checkpoints must reach the probe
# ---------------------------------------------------------------------------


@artifact_required
def test_present_checkpoints_invoke_the_probe_runner() -> None:
    """The exact bug: `probes` was bound to () and the runner never called."""
    calls: list[tuple[Any, Any]] = []

    def runner(artifact_dir: Any, split_paths: Any) -> Any:
        calls.append((artifact_dir, split_paths))
        return (_probe_stub("best"), _probe_stub("latest")), (
            {"role": "best", "w2ner_head_restored": True},
            {"role": "latest", "w2ner_head_restored": True})

    integrity = verify_artifact_integrity(ARTIFACT_DIR)
    if not integrity.checkpoints_present:
        pytest.skip("checkpoints are not present in the local artifact")

    diagnosis = _diagnosis(runner)
    assert len(calls) == 1, "the probe runner was never invoked"
    assert len(diagnosis.probes) == 2
    assert diagnosis.probe_blocked_reason == ""
    assert {probe.role for probe in diagnosis.probes} == set(CHECKPOINT_ROLES)


@artifact_required
def test_a_present_checkpoint_is_never_reported_as_absent() -> None:
    integrity = verify_artifact_integrity(ARTIFACT_DIR)
    if not integrity.checkpoints_present:
        pytest.skip("checkpoints are not present in the local artifact")
    assert "checkpoints/best.pt" not in integrity.missing_files
    assert "checkpoints/latest.pt" not in integrity.missing_files
    assert integrity.checkpoint_hash_matches == {"best": True, "latest": True}
    assert integrity.ok


@artifact_required
def test_probe_failure_is_never_flattened_into_a_bare_blocked() -> None:
    """A dependency failure must carry its type, location and next action."""
    def runner(_artifact_dir: Any, _split_paths: Any) -> Any:
        raise ProbeDependencyError(
            "vinai/phobert-large config.json", "/nowhere/cache",
            "allow network access to huggingface.co")

    diagnosis = _diagnosis(runner)
    assert diagnosis.probes == ()
    assert "vinai/phobert-large config.json" in diagnosis.probe_blocked_reason
    assert "allow network access" in diagnosis.probe_blocked_reason
    detail = diagnosis.probe_blocked_detail
    assert detail["exception_type"] == "ProbeDependencyError"
    assert detail["dependency"] == "vinai/phobert-large config.json"
    assert detail["location"] == "/nowhere/cache"
    assert detail["next_action"]


@artifact_required
def test_a_restore_failure_reaches_the_verdict_as_a_restore_failure() -> None:
    def runner(_artifact_dir: Any, _split_paths: Any) -> Any:
        return (_probe_stub("best"),), ({"role": "best", "w2ner_head_restored": False},)

    diagnosis = _diagnosis(runner)
    assert diagnosis.verdict.verdict == CHECKPOINT_RESTORE_FAILURE


@artifact_required
def test_probe_evidence_promotes_the_collapse_verdict() -> None:
    def runner(_artifact_dir: Any, _split_paths: Any) -> Any:
        return (_probe_stub("best"), _probe_stub("latest")), (
            {"role": "best", "w2ner_head_restored": True},
            {"role": "latest", "w2ner_head_restored": True})

    diagnosis = _diagnosis(runner)
    assert diagnosis.verdict.verdict == ALL_BACKGROUND_LOSS_COLLAPSE
    assert CHECKPOINT_RESTORE_FAILURE in diagnosis.verdict.ruled_out
    assert diagnosis.verdict.missing_evidence == ()


def test_missing_checkpoint_still_produces_a_precise_blocked_reason(
    tmp_path: Path,
) -> None:
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "training_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "resolved_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "grid_target_statistics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "e4_alignment_diagnostic.json").write_text("{}", encoding="utf-8")
    (tmp_path / "logs" / "training_history.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "training_progress.jsonl").write_text("", encoding="utf-8")

    called: list[int] = []

    def runner(_artifact_dir: Any, _split_paths: Any) -> Any:
        called.append(1)
        return (), ()

    diagnosis = run_collapse_diagnosis(
        artifact_dir=tmp_path, split_paths={"validation": VALIDATION_SPLIT},
        limit=50, probe_runner=runner)
    assert called == [], "the probe must not run when the weights are absent"
    assert "best.pt is not present" in diagnosis.probe_blocked_reason
    assert EXPECTED_CHECKPOINT_SHA256["best"] in diagnosis.probe_blocked_reason
    assert diagnosis.probe_blocked_detail["exception_type"] == (
        "CheckpointEvidenceUnavailable")
    assert diagnosis.probe_blocked_detail["next_action"]
    assert diagnosis.verdict.verdict == ROOT_CAUSE_NOT_YET_PROVEN


def test_probe_dependency_error_is_a_diagnosis_error_with_a_remedy() -> None:
    error = ProbeDependencyError("torch", "python environment", "pip install torch")
    assert isinstance(error, E4DiagnosisError)
    payload = error.as_dict()
    assert payload["exception_type"] == "ProbeDependencyError"
    assert payload["next_action"] == "pip install torch"
    assert "torch" in str(error) and "pip install torch" in str(error)


def test_checkpoint_evidence_unavailable_is_distinct_from_a_dependency_error() -> None:
    """"we could not look" and "a dependency is missing" must stay separable."""
    assert issubclass(CheckpointEvidenceUnavailable, E4DiagnosisError)
    assert not issubclass(CheckpointEvidenceUnavailable, ProbeDependencyError)
    assert not issubclass(ProbeDependencyError, CheckpointEvidenceUnavailable)


# ---------------------------------------------------------------------------
# Payload inspection, read-only and one at a time
# ---------------------------------------------------------------------------


@checkpoints_required
@pytest.mark.parametrize("role", CHECKPOINT_ROLES)
def test_checkpoint_payload_inspection(role: str) -> None:
    inspection = inspect_checkpoint(CHECKPOINT_DIR / f"{role}.pt", role=role)
    assert inspection.sha256 == EXPECTED_CHECKPOINT_SHA256[role]
    assert inspection.sha256_matches_expected
    assert inspection.payload_type == "dict"
    assert inspection.missing_required_keys == ()
    assert inspection.schema_ok
    assert inspection.model_state_subkeys == ("base_model", "w2ner_head")
    assert inspection.w2ner_head_keys == EXPECTED_W2NER_HEAD_KEYS
    assert inspection.base_model_tensor_count == 391
    assert inspection.parameter_accounting_reconciles
    assert inspection.declared_parameter_count == 371_289_161
    assert inspection.e4_input_contract_version == "e4-atomic-grid-word-v1"
    assert inspection.e4_checkpoint_schema_version == "phase2-e4-checkpoint-v2"
    assert inspection.mode == "full"
    assert inspection.expert_id == "E4_phobert_w2ner"
    assert inspection.internal_test_accessed is False
    assert inspection.optimizer_state_present
    assert inspection.scheduler_state_present is False
    assert inspection.precision_mode == "bf16"
    assert list(inspection.label_space) == list(W2NERLabelVocab().type_order)
    assert inspection.as_dict()["tensor_values_read"] is False


@checkpoints_required
def test_best_is_epoch_two_and_latest_is_epoch_twelve() -> None:
    best = inspect_checkpoint(CHECKPOINT_DIR / "best.pt", role="best")
    latest = inspect_checkpoint(CHECKPOINT_DIR / "latest.pt", role="latest")
    assert best.epoch == 2
    assert latest.epoch == 12
    assert best.optimizer_steps == 8458
    assert latest.optimizer_steps == 50748
    assert best.w2ner_head_shapes == latest.w2ner_head_shapes


@checkpoints_required
def test_w2ner_head_has_the_expected_atomic_projection_geometry() -> None:
    """1027 = hidden 1024 + ATOMIC_FEATURE_DIM 3; classifier is 7 x 2054."""
    inspection = inspect_checkpoint(CHECKPOINT_DIR / "best.pt", role="best")
    shapes = inspection.w2ner_head_shapes
    assert shapes["left.weight"] == (1027, 1027)
    assert shapes["right.weight"] == (1027, 1027)
    assert shapes["classifier.weight"] == (7, 2054)
    assert shapes["classifier.bias"] == (7,)
    assert inspection.w2ner_head_parameter_count == 2_125_897


@checkpoints_required
def test_checkpoint_payload_is_released_before_the_next_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ~4.4 GB payloads must never be live at once."""
    import torch

    live: list[str] = []
    peak = 0
    real_load = torch.load

    def counting_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal peak
        live.append(str(args[0] if args else kwargs.get("f")))
        peak = max(peak, len(live))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    for role in CHECKPOINT_ROLES:
        with checkpoint_payload(CHECKPOINT_DIR / f"{role}.pt") as payload:
            assert "model_state" in payload
        live.pop()
    assert peak == 1, "a second checkpoint was opened before the first was released"


@checkpoints_required
def test_checkpoint_payload_uses_memory_mapped_read_only_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    seen: dict[str, Any] = {}
    real_load = torch.load

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    with checkpoint_payload(CHECKPOINT_DIR / "best.pt"):
        pass
    assert seen["map_location"] == "cpu"
    assert seen["mmap"] is True


def test_checkpoint_payload_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProbeDependencyError) as excinfo:
        with checkpoint_payload(tmp_path / "best.pt"):
            pass
    assert str(tmp_path / "best.pt") in str(excinfo.value)
    assert excinfo.value.as_dict()["next_action"]


# ---------------------------------------------------------------------------
# The artifact is never written to, and never tracked
# ---------------------------------------------------------------------------


@artifact_required
def test_artifact_non_checkpoint_files_are_byte_identical() -> None:
    for name, expected in E4_ARTIFACT_SHA256.items():
        path = ARTIFACT_DIR / name
        assert path.is_file(), name
        assert _sha256(path) == expected, f"{name} was modified by a diagnostic"


@checkpoints_required
@pytest.mark.parametrize("role", CHECKPOINT_ROLES)
def test_checkpoint_files_are_byte_identical(role: str) -> None:
    assert _sha256(CHECKPOINT_DIR / f"{role}.pt") == EXPECTED_CHECKPOINT_SHA256[role]


def test_no_checkpoint_is_tracked_by_git() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    for path in tracked:
        assert not path.endswith((".pt", ".bin", ".safetensors", ".ckpt")), path
        assert not path.startswith("local-artifacts/"), path


@artifact_required
def test_the_checkpoints_are_git_ignored() -> None:
    for role in CHECKPOINT_ROLES:
        result = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-q",
             f"local-artifacts/e4_phobert_w2ner_full_v1/checkpoints/{role}.pt"],
            capture_output=True)
        assert result.returncode == 0, f"{role}.pt must be git-ignored"


# ---------------------------------------------------------------------------
# Forward-only: no optimizer, no backward, no training
# ---------------------------------------------------------------------------


def _executable_tokens(path: Path) -> str:
    """Space-joined code tokens, with comments and string literals removed.

    Two traps this avoids. A prose guarantee ("no ``.backward()`` is ever
    called") lives in a docstring and would satisfy a naive substring search —
    docstrings are STRING tokens, not comments, so both are dropped. And a
    substring search over raw text matches ``backward`` inside the legitimate
    field ``backward_passes``; joining *tokens* with spaces makes the search
    exact, so ``" backward ( "`` matches the call and not the field.
    """
    import io
    import tokenize

    kept: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if token.string.strip():
                kept.append(token.string)
    return " " + " ".join(kept) + " "


PROBE_SOURCE = (REPO / "src" / "mednorm_vi" / "training" / "phase2"
                / "e4_checkpoint_probe.py")


def test_probe_source_constructs_no_optimizer_and_never_calls_backward() -> None:
    tokens = _executable_tokens(PROBE_SOURCE)
    for forbidden in (" backward ( ", " AdamW ", " optim ", " save ( ",
                      " cross_entropy ( ", " requires_grad_ ( True ) ",
                      " zero_grad ( ", " step ( "):
        assert forbidden not in tokens, forbidden
    # And the guarantees it must positively make.
    assert " torch . no_grad ( ) " in tokens
    assert " . eval ( ) " in tokens
    assert " requires_grad_ ( False ) " in tokens


def test_the_source_scanner_would_actually_catch_a_backward_call(
    tmp_path: Path,
) -> None:
    """Guard the guard: prose must not pass, a real call must not slip through,
    and a field merely named ``backward_passes`` must not be a false positive."""
    prose_only = tmp_path / "prose.py"
    prose_only.write_text(
        '"""This module never calls .backward()."""\n', encoding="utf-8")
    assert " backward ( " not in _executable_tokens(prose_only)

    field_named_backward = tmp_path / "field.py"
    field_named_backward.write_text("backward_passes = 405912\n", encoding="utf-8")
    assert " backward ( " not in _executable_tokens(field_named_backward)

    real_call = tmp_path / "real.py"
    real_call.write_text("def train(loss):\n    loss.backward()\n", encoding="utf-8")
    assert " backward ( " in _executable_tokens(real_call)


def test_probe_source_never_opens_the_frozen_split() -> None:
    source = PROBE_SOURCE.read_text(encoding="utf-8")
    assert "internal_test.jsonl" not in source
    assert 'split="internal_test"' not in source
    assert " assert_split_allowed " in _executable_tokens(PROBE_SOURCE)


def test_probe_source_does_not_download_pretrained_weights() -> None:
    """The checkpoint supplies the encoder. Loading pretrained weights first
    would let a checkpoint that lacks them appear to restore cleanly."""
    tokens = _executable_tokens(PROBE_SOURCE)
    assert " AutoModel . from_config ( " in tokens
    assert " AutoModel . from_pretrained " not in tokens


# ---------------------------------------------------------------------------
# Bounded aggregation and reporting
# ---------------------------------------------------------------------------


def test_bounded_sample_is_capped_deterministic_and_exact_where_it_claims() -> None:
    sample = BoundedSample(capacity=64)
    for value in range(10_000):
        sample.observe(float(value))
    assert sample.count == 10_000
    assert sample.minimum == 0.0
    assert sample.maximum == 9999.0
    assert sample.mean == pytest.approx(4999.5)
    assert len(sample.as_dict()["sampled_for_quantiles"] * [0]) <= 64
    quantiles = sample.quantiles()
    assert quantiles["p50"] == pytest.approx(4999.5, abs=200)
    assert quantiles["p01"] < quantiles["p50"] < quantiles["p99"]
    # Determinism: no seed anywhere.
    twin = BoundedSample(capacity=64)
    for value in range(10_000):
        twin.observe(float(value))
    assert twin.as_dict() == sample.as_dict()


def test_bounded_sample_default_capacity_bounds_a_full_validation_sweep() -> None:
    assert DEFAULT_SAMPLE_CAPACITY < 1_491_764
    sample = BoundedSample()
    for value in range(1_491_764):
        sample.observe(float(value % 97))
    assert len(sample.quantiles()) == 7
    assert sample.count == 1_491_764


def test_empty_bounded_sample_reports_nothing_rather_than_zero() -> None:
    assert BoundedSample().as_dict() == {"count": 0}
    assert BoundedSample().quantiles() == {}


def _logit_report(**overrides: Any) -> GridLogitReport:
    fields: dict[str, Any] = {
        "role": "latest",
        "grid_cells": 1_491_764,
        "predicted_labels_by_class": {"NONE": 1_491_764},
        "none_predictions": 1_491_764,
        "non_none_predictions": 0,
        "gold_positive_cells": 6638,
        "gold_positive_predicted_as_none": 6638,
        "gold_positive_predicted_correct_class": 0,
        "gold_positive_predicted_labels": {"NONE": 6638},
        "none_logits": {"mean": 13.5},
        "strongest_non_none_logits": {"mean": -3.0},
        "non_none_margin_over_none": {"mean": -16.5},
        "gold_positive_margin_over_none": {"mean": -15.0},
        "decoder_input_thw_relations": 0,
        "decoder_input_nnw_relations": 0,
        "decoder_output_mentions": 0,
    }
    fields.update(overrides)
    return GridLogitReport(**fields)


def _restoration(ok: bool = True) -> RestorationReport:
    return RestorationReport(
        role="latest", strictness="strict=False, then every key reported",
        base_missing_keys=(), base_unexpected_keys=(),
        head_missing_keys=() if ok else ("classifier.weight",),
        head_unexpected_keys=(),
        w2ner_head_restored=ok, base_model_restored=True,
        instantiated_parameter_count=371_289_161,
        checkpoint_parameter_count=371_289_161,
        checkpoint_epoch=12, base_weights_downloaded=False,
        initialization="architecture from config.json, all weights from the checkpoint")


def test_grid_logit_rates_are_derived_not_stored() -> None:
    report = _logit_report()
    assert report.gold_positive_background_rate == 1.0
    assert report.gold_positive_correct_class_rate == 0.0
    assert report.non_none_prediction_rate == 0.0
    assert report.as_dict()["internal_test_accessed"] is False


def test_outcome_classifier_keys_on_thw_not_on_non_background() -> None:
    """A stray NNW cell with no THW yields zero mentions, exactly like pure
    NONE. Treating "some non-background cell exists" as evidence against
    background collapse would misread precisely that case."""
    stray_nnw = _logit_report(
        non_none_predictions=2, predicted_labels_by_class={"NONE": 1_491_762, "NNW": 2},
        decoder_input_nnw_relations=2, decoder_input_thw_relations=0)
    assert classify_probe_outcome(
        _probe_stub("latest"), stray_nnw, _restoration()) == (
            "trained_head_emits_no_entity_relation_only_background")


def test_outcome_classifier_separates_the_four_mechanisms() -> None:
    assert classify_probe_outcome(
        _probe_stub("latest"), _logit_report(), _restoration(ok=False)) == (
            "checkpoint_does_not_restore_the_trained_head")
    assert classify_probe_outcome(
        _probe_stub("latest"), _logit_report(), _restoration()) == (
            "trained_head_emits_no_entity_relation_only_background")
    assert classify_probe_outcome(
        _probe_stub("latest"),
        _logit_report(decoder_input_thw_relations=40, decoder_output_mentions=0),
        _restoration()) == "head_emits_positive_relations_but_the_decoder_loses_them"
    assert classify_probe_outcome(
        _probe_stub("latest", predicted_mention_total=12),
        _logit_report(decoder_input_thw_relations=40, decoder_output_mentions=12),
        _restoration()) == "head_and_decoder_both_produce_mentions"


def test_restoration_report_requires_every_condition_to_be_ok() -> None:
    assert _restoration().ok
    assert not _restoration(ok=False).ok
    mismatch = RestorationReport(
        role="best", strictness="strict", base_missing_keys=("embeddings.weight",),
        base_unexpected_keys=(), head_missing_keys=(), head_unexpected_keys=(),
        w2ner_head_restored=True, base_model_restored=False,
        instantiated_parameter_count=371_289_161,
        checkpoint_parameter_count=371_289_161, checkpoint_epoch=2,
        base_weights_downloaded=False, initialization="config only")
    assert not mismatch.ok
    assert mismatch.as_dict()["base_missing_keys"] == ["embeddings.weight"]


def test_restore_error_is_a_diagnosis_error() -> None:
    assert issubclass(CheckpointRestoreError, E4DiagnosisError)


# ---------------------------------------------------------------------------
# No clinical text anywhere in a probe report
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not VALIDATION_SPLIT.is_file(), reason="governed corpus absent")
def test_probe_reports_contain_no_governed_document_text() -> None:
    texts = [
        json.loads(line)["text"]
        for line in VALIDATION_SPLIT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payloads = [
        _probe_stub("best").as_dict(),
        _logit_report().as_dict(),
        _restoration().as_dict(),
    ]
    if (CHECKPOINT_DIR / "best.pt").is_file() and _torch_available():
        payloads.append(inspect_checkpoint(CHECKPOINT_DIR / "best.pt", role="best")
                        .as_dict())
    serialized = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    for text in texts:
        stripped = text.strip()
        if len(stripped) >= 24:
            assert stripped not in serialized


# ---------------------------------------------------------------------------
# The CLI never emits a bare BLOCKED
# ---------------------------------------------------------------------------


def test_cli_never_prints_a_bare_blocked_line() -> None:
    source = (REPO / "scripts" / "diagnose_e4_collapse.py").read_text(encoding="utf-8")
    assert '"  BLOCKED"' not in source
    assert '"  NOT EXECUTED"' in source
    assert " probe_blocked_detail " in _executable_tokens(
        REPO / "scripts" / "diagnose_e4_collapse.py")
