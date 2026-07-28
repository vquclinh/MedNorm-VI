"""E4 post-training collapse diagnosis (Audit 0043).

The completed full run is engineering-complete and quality-failed. These tests
lock down the *diagnosis*, not the model: that the gold-grid round-trip is exact,
that one label ordering is shared end to end, that the class distribution is
counted rather than assumed, and — most importantly — that a missing checkpoint
can never be mistaken for evidence of collapse.

No test trains, downloads a model, opens internal_test, or writes into the
immutable local artifact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from mednorm_vi.mention_factory.w2ner import (
    W2NER_NONE,
    EntitySpan,
    W2NERLabelVocab,
    build_w2ner_grid,
    decode_w2ner_grid,
    tokenize_atomic_words,
)
from mednorm_vi.training.phase2.e4_collapse_diagnosis import (
    ALL_BACKGROUND_LOSS_COLLAPSE,
    CHECKPOINT_RESTORE_FAILURE,
    DECODER_THRESHOLD_FAILURE,
    E4_ARTIFACT_CHECKPOINT_FILES,
    E4_FULL_BEST_CHECKPOINT_SHA256,
    E4_FULL_EXPECTED,
    E4_FULL_LATEST_CHECKPOINT_SHA256,
    LABEL_MAPPING_MISMATCH,
    ROOT_CAUSE_NOT_YET_PROVEN,
    TARGET_DECODER_MISMATCH,
    UNAVAILABLE,
    CheckpointEvidenceUnavailable,
    CheckpointProbeReport,
    E4DiagnosisError,
    GovernedExample,
    RoundTripReport,
    assert_no_clinical_text,
    assert_split_allowed,
    audit_loss_contract,
    constant_predictor_gap,
    gold_grid_round_trip,
    grid_class_distribution,
    inspect_checkpoint_payload,
    load_governed_examples,
    measure_corpus_composition,
    reconstruct_epoch_history,
    require_checkpoint,
    resolve_verdict,
    run_collapse_diagnosis,
    trace_label_contract,
    verify_artifact_integrity,
)
from mednorm_vi.training.phase2.e4_tiny_overfit import (
    TINY_OVERFIT_AUTHORIZATION,
    TINY_OVERFIT_MAX_EPOCHS,
    TINY_OVERFIT_MIN_EXAMPLES,
    TINY_OVERFIT_REQUIRED_TYPES,
    TINY_OVERFIT_TARGET_EXACT_F1,
    TinyOverfitError,
    assert_artifact_dir_is_not_protected,
    assert_tiny_overfit_authorized,
    build_tiny_overfit_resolved_config,
    build_tiny_overfit_targets,
    score_predicted_grid,
    select_tiny_overfit_examples,
    should_stop_tiny_overfit,
    summarize_tiny_overfit,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO / "local-artifacts" / "e4_phobert_w2ner_full_v1"
SPLIT_ROOT = REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1" / "splits"
TRAIN_SPLIT = SPLIT_ROOT / "train.jsonl"
VALIDATION_SPLIT = SPLIT_ROOT / "validation.jsonl"
DIAGNOSTIC_NOTEBOOK = REPO / "notebooks" / "MedNorm_E4_TinyOverfit_Diagnostic.ipynb"
DIAGNOSTIC_CONFIG = REPO / "configs" / "training" / "phase2_e4_tiny_overfit_diagnostic.yaml"

# The completed run's own recorded digests. Pinned so a swapped or regenerated
# artifact is a loud failure rather than a quiet change of evidence.
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

# Protected E4 implementation paths. This milestone diagnoses the run; it must
# not change what produced it. Digests are recomputed, not trusted from git.
E4_PROTECTED_SHA256: dict[str, str] = {
    "notebooks/MedNorm_E4_PhoBERT_W2NER_Training.ipynb":
        "85040fbb824521582d5d04d7c45af293db7927b8a746a4d5b52558b32ce92813",
    "configs/training/phase2_e4_phobert_w2ner_colab.yaml":
        "054d376570eafe611cd1a2a35c3e24aaa30390a8a48f775a838d47b6500942ac",
    "src/mednorm_vi/training/phase2/e4_runtime_io.py":
        "ad74031f9a909eb6d40eddeed1bb3f0eeb9183f015e3024d7d8c425df3b43e77",
    "src/mednorm_vi/training/phase2/e4_progress.py":
        "2d770fa9c2c5551d4183d9d671c06d7965cda1ef104410de72d57adc29e95474",
    "src/mednorm_vi/training/phase2/e4_w2ner_training.py":
        "fc44befd49bd9a56a4efad49fff65ef17686bf9272689828f850e1d87cff48dd",
    "src/mednorm_vi/training/phase2/e4_alignment_diagnostic.py":
        "ec610e190bf65e9033ccad7dd21a17b3f4403ff0a3f98f4192eea77d1c3e7b1c",
    "src/mednorm_vi/mention_factory/w2ner.py":
        "2ca5d434e4a4a252ee8b3cd942b3c4a4566fd9d9591a7a024426e21d53f1fd4f",
    "src/mednorm_vi/evaluation/exact_mention.py":
        "7b2ba8fd72afdde715f90ac321c85cf3ea1d6e88e3eb98701903b463f14e07f0",
    "docs/MedNorm-VI_Architecture.pdf":
        "0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b",
}

artifact_required = pytest.mark.skipif(
    not ARTIFACT_DIR.is_dir(), reason="local E4 artifact is not present")
corpus_required = pytest.mark.skipif(
    not TRAIN_SPLIT.is_file(), reason="governed corpus is not present")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _example(text: str, entities: tuple[EntitySpan, ...] = (), row: int = 0,
             source: str = "unit") -> GovernedExample:
    return GovernedExample(
        row_index=row, document_id=f"unit-{row:04d}", source_dataset=source,
        text=text, entities=entities)


# ---------------------------------------------------------------------------
# Immutable artifact hash verification
# ---------------------------------------------------------------------------


@artifact_required
def test_local_artifact_files_match_their_recorded_digests() -> None:
    for name, expected in E4_ARTIFACT_SHA256.items():
        path = ARTIFACT_DIR / name
        assert path.is_file(), f"missing artifact file {name}"
        assert _sha256(path) == expected, f"{name} changed since Audit 0043"


@artifact_required
def test_artifact_declares_the_recorded_checkpoint_hashes() -> None:
    report = verify_artifact_integrity(ARTIFACT_DIR)
    assert report.declared_checkpoint_hashes["best"] == E4_FULL_BEST_CHECKPOINT_SHA256
    assert report.declared_checkpoint_hashes["latest"] == E4_FULL_LATEST_CHECKPOINT_SHA256


@artifact_required
def test_artifact_integrity_reports_no_inconsistency() -> None:
    report = verify_artifact_integrity(ARTIFACT_DIR)
    assert report.inconsistencies == (), report.inconsistencies
    assert report.ok


@artifact_required
def test_artifact_records_the_completed_full_run_accounting() -> None:
    report = verify_artifact_integrity(ARTIFACT_DIR)
    manifest, metrics = report.manifest, report.validation_metrics
    assert manifest["completed_epochs"] == E4_FULL_EXPECTED["completed_epochs"]
    assert manifest["optimizer_steps"] == E4_FULL_EXPECTED["optimizer_steps"]
    assert metrics["backward_passes"] == E4_FULL_EXPECTED["backward_passes"]
    assert metrics["best_epoch"] == E4_FULL_EXPECTED["best_epoch"]
    assert manifest["internal_test_accessed"] is False
    assert metrics["internal_test_accessed"] is False
    # The quality gate failure itself, recorded rather than softened.
    assert metrics["validation_predicted_total"] == 0
    assert metrics["validation_gold_total"] == 1991
    assert metrics["validation_exact_f1"] == 0.0


@artifact_required
def test_missing_checkpoints_are_reported_as_absent_not_as_inconsistent() -> None:
    """Absent weights are an evidence gap, never a contradiction in the artifact."""
    report = verify_artifact_integrity(ARTIFACT_DIR)
    if report.checkpoints_present:
        pytest.skip("checkpoints have been downloaded into the artifact")
    assert set(report.missing_files) == set(E4_ARTIFACT_CHECKPOINT_FILES)
    assert report.ok


def test_integrity_flags_a_declared_hash_that_does_not_match_the_run(
    tmp_path: Path,
) -> None:
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "training_manifest.json").write_text(json.dumps({
        "expert_id": "E4_phobert_w2ner", "stage_id": "phase2-e4-phobert-w2ner-v2",
        "mode": "full", "model_id": "vinai/phobert-large",
        "model_revision": E4_FULL_EXPECTED["model_revision"],
        "completed_epochs": 12, "optimizer_steps": 50748,
        "parameter_count": 371289161, "internal_test_accessed": False,
        "checkpoint_hashes": {"best": "0" * 64, "latest": "1" * 64},
    }), encoding="utf-8")
    (tmp_path / "resolved_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_metrics.json").write_text(json.dumps({
        "best_epoch": 2, "backward_passes": 405912, "internal_test_accessed": False,
    }), encoding="utf-8")
    (tmp_path / "grid_target_statistics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "e4_alignment_diagnostic.json").write_text("{}", encoding="utf-8")
    (tmp_path / "logs" / "training_history.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "training_progress.jsonl").write_text("", encoding="utf-8")
    report = verify_artifact_integrity(tmp_path)
    assert not report.ok
    assert any("does not match the recorded run hash" in item
               for item in report.inconsistencies)


def test_integrity_refuses_a_missing_artifact_directory(tmp_path: Path) -> None:
    with pytest.raises(E4DiagnosisError, match="does not exist"):
        verify_artifact_integrity(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Training-history epoch reconstruction
# ---------------------------------------------------------------------------


@artifact_required
def test_epoch_history_reconstructs_all_twelve_epochs() -> None:
    report = reconstruct_epoch_history(ARTIFACT_DIR)
    assert [row.epoch for row in report.rows] == list(range(1, 13))
    assert all(isinstance(row.mean_training_loss, float) for row in report.rows)


@artifact_required
def test_epoch_history_locates_when_predictions_vanished() -> None:
    report = reconstruct_epoch_history(ARTIFACT_DIR)
    # Epoch 2 was the only epoch that produced a meaningful number of mentions;
    # from epoch 4 onward the model predicted nothing at all, for eight epochs.
    assert report.peak_prediction_epoch == 2
    assert report.peak_predicted_mentions == 13
    assert report.first_epoch_with_zero_predictions == 4
    assert report.last_epoch_with_any_prediction == 3
    assert report.final_predicted_mentions == 0
    by_epoch = {row.epoch: row for row in report.rows}
    assert by_epoch[2].true_positives == 2
    assert by_epoch[3].predicted_mentions == 4
    for epoch in range(4, 13):
        assert by_epoch[epoch].predicted_mentions == 0
        assert by_epoch[epoch].gold_mentions == 1991


@artifact_required
def test_epoch_history_converged_loss_is_flat_from_epoch_five() -> None:
    """The loss stops moving entirely: the run was stationary for 8 epochs."""
    report = reconstruct_epoch_history(ARTIFACT_DIR)
    tail = [float(row.mean_training_loss) for row in report.rows if row.epoch >= 5]
    assert max(tail) - min(tail) < 2e-4, tail


def test_epoch_history_marks_unread_fields_unavailable(tmp_path: Path) -> None:
    """A missing progress log must never print as "predicted 0 mentions"."""
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "training_history.jsonl").write_text(
        json.dumps({"epoch": 1, "train_loss": 0.5, "validation_exact_f1": 0.0}) + "\n",
        encoding="utf-8")
    report = reconstruct_epoch_history(tmp_path)
    assert report.rows[0].predicted_mentions == UNAVAILABLE
    assert report.rows[0].gold_mentions == UNAVAILABLE
    assert report.final_predicted_mentions == UNAVAILABLE
    assert "predicted_mentions" in report.unavailable_fields


def test_epoch_history_requires_a_history_file(tmp_path: Path) -> None:
    with pytest.raises(E4DiagnosisError, match="missing epoch history"):
        reconstruct_epoch_history(tmp_path)


# ---------------------------------------------------------------------------
# Gold grid round-trip
# ---------------------------------------------------------------------------


def test_round_trip_is_exact_on_a_constructed_multi_word_entity() -> None:
    text = "bệnh nhân bị sốt cao và ho khan ."
    entities = (
        EntitySpan(text.index("sốt cao"), text.index("sốt cao") + 7, "SYMPTOM", "sốt cao"),
        EntitySpan(text.index("ho khan"), text.index("ho khan") + 7, "SYMPTOM", "ho khan"),
    )
    report = gold_grid_round_trip([_example(text, entities)], split="train")
    assert report.passes
    assert report.gold_mentions == 2
    assert report.true_positives == 2
    assert report.failures_by_entity_type == {}


def test_round_trip_reports_a_failure_with_offsets_and_no_text() -> None:
    """A misaligned entity is reported, never repaired."""
    text = "sốt cao và ho"
    # End 4 lands inside "cao" (offsets 4:7), so the right edge has no word to
    # land on and build_w2ner_grid refuses it.
    bad = EntitySpan(0, 5, "SYMPTOM", text[0:5])
    with pytest.raises(Exception, match="not word-aligned"):
        gold_grid_round_trip([_example(text, (bad,))], split="train")


@corpus_required
def test_round_trip_is_exact_on_the_whole_governed_train_split() -> None:
    report = gold_grid_round_trip(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    assert report.examples_checked == 33826
    assert report.gold_mentions == 11720
    assert report.reconstructed_mentions == 11720
    assert report.true_positives == 11720
    assert report.exact_precision == 1.0
    assert report.exact_recall == 1.0
    assert report.exact_f1 == 1.0
    assert report.failures_by_entity_type == {}
    assert report.passes


@corpus_required
def test_round_trip_is_exact_on_the_whole_governed_validation_split() -> None:
    report = gold_grid_round_trip(
        load_governed_examples(VALIDATION_SPLIT, split="validation"),
        split="validation")
    assert report.examples_checked == 1045
    # The full run's reported gold total is exactly what the round-trip produces,
    # so the 1,991 it scored against is the real gold count and not an artefact.
    assert report.gold_mentions == 1991
    assert report.reconstructed_mentions == 1991
    assert report.exact_f1 == 1.0
    assert report.passes


def test_round_trip_report_never_carries_document_text() -> None:
    text = "bệnh nhân bị sốt cao kéo dài nhiều ngày liên tiếp không giảm ."
    entity = EntitySpan(text.index("sốt cao"), text.index("sốt cao") + 7,
                        "SYMPTOM", "sốt cao")
    report = gold_grid_round_trip([_example(text, (entity,))], split="train")
    assert_no_clinical_text(report.as_dict(), corpus_texts=[text])
    assert report.as_dict()["model_predictions_used"] is False


# ---------------------------------------------------------------------------
# Class distribution accounting
# ---------------------------------------------------------------------------


def test_class_distribution_counts_cells_exactly_on_a_known_grid() -> None:
    text = "sốt cao"  # two atomic words -> 2x2 grid
    entity = EntitySpan(0, len(text), "SYMPTOM", text)
    report = grid_class_distribution([_example(text, (entity,))], split="train")
    assert report.valid_grid_cells == 4
    # One NNW edge (word 0 -> 1) and one THW cell (tail 1, head 0).
    assert report.positive_cells == 2
    assert report.background_cells == 2
    assert report.cells_by_label["NNW"] == 1
    assert report.cells_by_label["THW:SYMPTOM"] == 1
    assert report.mentions_by_relation_pattern == {"two_word_nnw1_thw1": 1}
    assert report.entities_represented == 1


def test_class_distribution_counts_an_example_with_no_entities() -> None:
    report = grid_class_distribution([_example("không có gì .")], split="train")
    assert report.positive_cells == 0
    assert report.examples_with_zero_positive_cells == 1
    assert report.positive_to_background_ratio == 0.0


def test_class_distribution_counts_padded_cells_separately(tmp_path: Path) -> None:
    del tmp_path
    text = "sốt cao"
    report = grid_class_distribution(
        [_example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))],
        split="train", max_words=8)
    assert report.valid_grid_cells == 4
    assert report.ignored_or_padded_cells == 8 * 8 - 4


@corpus_required
def test_governed_train_distribution_is_overwhelmingly_background() -> None:
    report = grid_class_distribution(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    assert report.valid_grid_cells == 25_635_699
    assert report.positive_cells == 44_340
    assert report.background_cells == 25_591_359
    assert report.positive_cells + report.background_cells == report.valid_grid_cells
    # Every one of the 11,720 governed train entities produces target structure.
    assert report.entities_represented == 11_720
    assert report.positive_to_background_ratio < 0.002
    # 81% of training examples ask the model for nothing but background.
    assert report.examples_with_zero_positive_cells == 27_513
    # Two of the five declared entity types never occur anywhere in the corpus.
    assert report.labels_with_no_training_signal == (
        "THW:TEST_NAME", "THW:TEST_RESULT")


@corpus_required
def test_governed_validation_distribution_matches_the_scored_gold_total() -> None:
    report = grid_class_distribution(
        load_governed_examples(VALIDATION_SPLIT, split="validation"),
        split="validation")
    assert report.entities_represented == 1991
    assert report.examples_with_zero_positive_cells == 0


@corpus_required
def test_train_and_validation_entity_counts_sum_to_the_audited_total() -> None:
    """13,711 aligned entities (Audit 0038) = 11,720 train + 1,991 validation."""
    train = grid_class_distribution(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    validation = grid_class_distribution(
        load_governed_examples(VALIDATION_SPLIT, split="validation"),
        split="validation")
    assert train.entities_represented + validation.entities_represented == 13_711


@corpus_required
def test_recorded_grid_statistics_agree_with_a_fresh_measurement() -> None:
    if not ARTIFACT_DIR.is_dir():
        pytest.skip("local E4 artifact is not present")
    recorded = json.loads(
        (ARTIFACT_DIR / "grid_target_statistics.json").read_text(encoding="utf-8"))
    train = grid_class_distribution(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    validation = grid_class_distribution(
        load_governed_examples(VALIDATION_SPLIT, split="validation"),
        split="validation")
    assert recorded["train_contracts"] == train.examples
    assert recorded["validation_contracts"] == validation.examples
    assert recorded["label_count"] == len(W2NERLabelVocab().labels)
    assert recorded["max_atomic_words"] == max(
        train.max_words_observed, validation.max_words_observed)


@corpus_required
def test_converged_loss_is_barely_better_than_ignoring_the_input() -> None:
    """The decisive quantity: the run's converged loss against the best possible
    input-independent predictor. A ratio near 1.0 means the network learned
    essentially nothing beyond the class prior, however small the loss looks."""
    train = grid_class_distribution(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    comparison = constant_predictor_gap(0.013603323943778346, train)
    assert comparison["constant_predictor_loss"] == pytest.approx(0.0147641, abs=1e-6)
    assert comparison["observed_over_constant_ratio"] == pytest.approx(0.9214, abs=1e-3)
    assert comparison["improvement_over_constant_predictor"] < 0.08


def test_constant_predictor_gap_refuses_a_degenerate_baseline() -> None:
    report = grid_class_distribution([_example("abc")], split="train")
    with pytest.raises(E4DiagnosisError, match="not positive"):
        constant_predictor_gap(0.1, report)


@corpus_required
def test_training_corpus_is_grouped_by_source_and_starts_with_zero_entity_rows() -> None:
    """The E4 loop streams deterministic file order and never shuffles, so the
    on-disk grouping is part of what the optimizer actually experienced."""
    report = measure_corpus_composition(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    assert report.sources_in_file_order == (
        "phoner_covid19", "vimedner", "vimq", "vietmed_ner")
    assert report.entities_by_source["phoner_covid19"] == 0
    assert report.first_row_index_by_source["phoner_covid19"] == 0
    # Every epoch opens with 10,027 consecutive examples containing no entity.
    assert report.longest_zero_entity_run == 10_027
    assert report.longest_zero_entity_run_start == 0


@corpus_required
def test_validation_covers_only_two_of_the_four_training_sources() -> None:
    train = measure_corpus_composition(
        load_governed_examples(TRAIN_SPLIT, split="train"), split="train")
    validation = measure_corpus_composition(
        load_governed_examples(VALIDATION_SPLIT, split="validation"),
        split="validation")
    assert set(validation.sources_in_file_order) == {"vimedner", "vimq"}
    absent = set(train.sources_in_file_order) - set(validation.sources_in_file_order)
    assert absent == {"phoner_covid19", "vietmed_ner"}


# ---------------------------------------------------------------------------
# Label-ID consistency
# ---------------------------------------------------------------------------


def test_label_ordering_is_shared_by_builder_loss_and_decoder() -> None:
    trace = trace_label_contract()
    assert trace.consistent, trace.differences
    assert trace.label_order == (
        "NONE", "NNW", "THW:DIAGNOSIS", "THW:MEDICATION", "THW:SYMPTOM",
        "THW:TEST_NAME", "THW:TEST_RESULT")
    assert trace.background_label_id == 0
    assert trace.nnw_label_id == 1
    assert trace.classifier_output_size == 7
    assert trace.thw_label_ids == {
        "DIAGNOSIS": 2, "MEDICATION": 3, "SYMPTOM": 4,
        "TEST_NAME": 5, "TEST_RESULT": 6}


def test_every_traced_cell_keeps_one_index_from_target_to_decoder() -> None:
    trace = trace_label_contract()
    assert trace.traced_cells
    for cell in trace.traced_cells:
        assert cell["target_label_id"] == cell["classifier_output_index"]
        assert cell["classifier_output_index"] == cell["cross_entropy_target_index"]
        assert cell["argmax_recovers_label"] == cell["target_label"]


def test_head_relation_count_matches_the_recorded_label_count() -> None:
    """The head is sized from grid_target_statistics.label_count; the loss uses
    len(grid.vocab.labels). Both must be the same number."""
    if not ARTIFACT_DIR.is_dir():
        pytest.skip("local E4 artifact is not present")
    recorded = json.loads(
        (ARTIFACT_DIR / "grid_target_statistics.json").read_text(encoding="utf-8"))
    assert int(recorded["label_count"]) == len(W2NERLabelVocab().labels)


def test_loss_contract_records_what_the_executed_run_optimized() -> None:
    report = audit_loss_contract()
    assert report.ignore_index_used is False
    assert report.class_weights_used is False
    assert report.padded_cells_in_loss is False
    assert report.triangular_masking_applied is False
    assert report.background_class_id == 0
    assert report.positive_relation_ids == (1, 2, 3, 4, 5, 6)
    assert report.all_background_solution_is_reachable is True
    assert any("threshold" in note for note in report.notes)


def test_decoder_applies_no_score_threshold() -> None:
    """Statically provable: score defaults to 1.0 and threshold to 0.0, so no
    decoding policy can be suppressing predictions."""
    text = "sốt cao"
    grid = build_w2ner_grid(
        "unit", text, (EntitySpan(0, len(text), "SYMPTOM", text),),
        words=tokenize_atomic_words(text))
    assert len(decode_w2ner_grid(grid)) == 1
    assert len(decode_w2ner_grid(grid, threshold=0.0)) == 1


# ---------------------------------------------------------------------------
# Checkpoint-head restoration
# ---------------------------------------------------------------------------


def test_checkpoint_inspection_accepts_a_complete_payload() -> None:
    payload = {key: "x" for key in (
        "checkpoint_schema_version", "e4_checkpoint_schema_version",
        "e4_input_contract_version", "atomic_projection_version", "expert_id",
        "mode", "config_sha256", "model_revision", "tokenizer_revision")}
    payload.update({
        "epoch": 12, "optimizer_steps": 50748, "best_metric": 0.002,
        "optimizer_state": {"state": {}},
        "model_state": {
            "base_model": {"encoder.weight": None},
            "w2ner_head": {
                "left.weight": None, "right.weight": None, "classifier.weight": None},
        },
    })
    report = inspect_checkpoint_payload(payload, role="latest")
    assert report["schema_ok"] is True
    assert report["w2ner_head_restored"] is True
    assert report["w2ner_head_keys"] == [
        "classifier.weight", "left.weight", "right.weight"]


def test_checkpoint_inspection_detects_a_missing_w2ner_head() -> None:
    report = inspect_checkpoint_payload(
        {"model_state": {"base_model": {}}}, role="best")
    assert report["schema_ok"] is False
    assert report["w2ner_head_restored"] is False
    assert "w2ner_head" in report["model_state_missing_keys"]


def test_requiring_an_absent_checkpoint_names_the_blocked_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(CheckpointEvidenceUnavailable) as excinfo:
        require_checkpoint(tmp_path, "best")
    message = str(excinfo.value)
    assert ALL_BACKGROUND_LOSS_COLLAPSE in message
    assert E4_FULL_BEST_CHECKPOINT_SHA256 in message


def test_requiring_an_unknown_checkpoint_role_fails() -> None:
    with pytest.raises(E4DiagnosisError, match="unknown checkpoint role"):
        require_checkpoint(".", "penultimate")


# ---------------------------------------------------------------------------
# Verdict gating
# ---------------------------------------------------------------------------


def _passing_round_trip():
    text = "sốt cao"
    return gold_grid_round_trip(
        [_example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))], split="train")


def _distribution():
    text = "sốt cao"
    return grid_class_distribution(
        [_example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))], split="train")


def _history():
    return reconstruct_epoch_history_stub()


def reconstruct_epoch_history_stub():
    from mednorm_vi.training.phase2.e4_collapse_diagnosis import (
        EpochHistoryReport,
        EpochRow,
    )
    row = EpochRow(
        epoch=1, mean_training_loss=0.0136, predicted_mentions=0, gold_mentions=1991,
        true_positives=0, false_positives=0, false_negatives=1991,
        exact_precision=0.0, exact_recall=0.0, exact_f1=0.0, is_new_best=True,
        checkpoint_hash="x", optimizer_steps=4229, backward_passes=33826)
    return EpochHistoryReport(
        rows=(row,), first_epoch_with_zero_predictions=1,
        last_epoch_with_any_prediction=UNAVAILABLE, peak_predicted_mentions=0,
        peak_prediction_epoch=1, final_predicted_mentions=0, unavailable_fields=())


def test_zero_predictions_alone_never_yields_the_collapse_verdict() -> None:
    """The whole point of the gate: predicted_total == 0 is not a root cause."""
    verdict = resolve_verdict(
        round_trips=[_passing_round_trip()],
        label_trace=trace_label_contract(),
        distributions=[_distribution()],
        history=_history(),
        loss_contract=audit_loss_contract(),
        probes=(),
        checkpoint_inspections=(),
    )
    assert verdict.verdict == ROOT_CAUSE_NOT_YET_PROVEN
    assert verdict.supported_hypothesis == ALL_BACKGROUND_LOSS_COLLAPSE
    assert TARGET_DECODER_MISMATCH in verdict.ruled_out
    assert LABEL_MAPPING_MISMATCH in verdict.ruled_out
    assert DECODER_THRESHOLD_FAILURE in verdict.ruled_out
    assert any("grid logit" in item for item in verdict.missing_evidence)
    assert any("gold-positive cells" in item for item in verdict.missing_evidence)


def test_collapse_verdict_is_reached_only_with_probe_evidence() -> None:
    probe = CheckpointProbeReport(
        role="latest", checkpoint_sha256=E4_FULL_LATEST_CHECKPOINT_SHA256, epoch=12,
        predicted_mention_total=0, gold_mention_total=1991, true_positives=0,
        exact_precision=0.0, exact_recall=0.0, exact_f1=0.0,
        predictions_by_entity_type={}, predicted_labels_by_class={"NONE": 1_491_764},
        background_label_count=1_491_764, non_background_label_count=0,
        background_logit_quantiles={"p50": 12.0},
        strongest_non_background_logit_quantiles={"p50": -3.0},
        gold_positive_cell_predicted_labels={"NONE": 6638},
        gold_positive_cell_background_rate=1.0,
        decoder_input_positive_relations=0, decoder_output_mention_count=0,
        w2ner_head_keys_restored=("classifier.weight",))
    verdict = resolve_verdict(
        round_trips=[_passing_round_trip()],
        label_trace=trace_label_contract(),
        distributions=[_distribution()],
        history=_history(),
        loss_contract=audit_loss_contract(),
        probes=(probe,),
        checkpoint_inspections=({"w2ner_head_restored": True},),
    )
    assert verdict.verdict == ALL_BACKGROUND_LOSS_COLLAPSE
    assert CHECKPOINT_RESTORE_FAILURE in verdict.ruled_out


def test_a_failing_round_trip_is_terminal() -> None:
    """A real mismatch means the supervision itself is lossy; nothing downstream
    can be concluded, so it becomes the verdict."""
    lossy = RoundTripReport(
        split="train", examples_checked=10, gold_mentions=20,
        reconstructed_mentions=18, true_positives=18, false_positives=0,
        false_negatives=2, exact_precision=1.0, exact_recall=0.9,
        exact_f1=0.9473684210526315,
        failures_by_entity_type={"SYMPTOM": 2}, representative_failures=())
    assert not lossy.passes and not lossy.vacuous
    verdict = resolve_verdict(
        round_trips=[lossy], label_trace=trace_label_contract(),
        distributions=[_distribution()], history=_history(),
        loss_contract=audit_loss_contract())
    assert verdict.verdict == TARGET_DECODER_MISMATCH


def test_a_round_trip_with_no_gold_mention_is_inconclusive_not_a_failure() -> None:
    """A bounded scan can land entirely in the corpus's zero-entity prefix.
    Absence of evidence must not invert into a confirmed target/decoder defect."""
    vacuous = gold_grid_round_trip([_example("không có gì .")], split="train")
    assert vacuous.vacuous and not vacuous.passes
    verdict = resolve_verdict(
        round_trips=[vacuous], label_trace=trace_label_contract(),
        distributions=[_distribution()], history=_history(),
        loss_contract=audit_loss_contract())
    assert verdict.verdict == ROOT_CAUSE_NOT_YET_PROVEN
    assert TARGET_DECODER_MISMATCH not in verdict.ruled_out
    assert any("inconclusive" in item for item in verdict.missing_evidence)


def test_a_broken_checkpoint_head_is_reported_as_a_restore_failure() -> None:
    verdict = resolve_verdict(
        round_trips=[_passing_round_trip()], label_trace=trace_label_contract(),
        distributions=[_distribution()], history=_history(),
        loss_contract=audit_loss_contract(),
        checkpoint_inspections=({"w2ner_head_restored": False},))
    assert verdict.verdict == CHECKPOINT_RESTORE_FAILURE


@artifact_required
@corpus_required
def test_end_to_end_diagnosis_is_root_cause_not_yet_proven_without_weights() -> None:
    diagnosis = run_collapse_diagnosis(
        artifact_dir=ARTIFACT_DIR,
        split_paths={"train": TRAIN_SPLIT, "validation": VALIDATION_SPLIT},
        max_words=256, limit=11_000)
    if diagnosis.integrity.checkpoints_present:
        pytest.skip("checkpoints have been downloaded into the artifact")
    assert diagnosis.verdict.verdict == ROOT_CAUSE_NOT_YET_PROVEN
    assert diagnosis.probe_blocked_reason
    payload = diagnosis.as_dict()
    assert payload["local_training_performed"] is False
    assert payload["internal_test_accessed"] is False
    assert payload["organizer_inference_performed"] is False
    assert payload["output_zip_created"] is False


# ---------------------------------------------------------------------------
# internal_test is never opened
# ---------------------------------------------------------------------------


def test_internal_test_is_refused_by_name() -> None:
    with pytest.raises(E4DiagnosisError, match="frozen split"):
        assert_split_allowed("internal_test")
    with pytest.raises(E4DiagnosisError, match="frozen split"):
        list(load_governed_examples(TRAIN_SPLIT, split="internal_test"))
    with pytest.raises(E4DiagnosisError, match="frozen split"):
        gold_grid_round_trip([], split="internal_test")
    with pytest.raises(E4DiagnosisError, match="frozen split"):
        grid_class_distribution([], split="internal_test")
    with pytest.raises(E4DiagnosisError, match="frozen split"):
        measure_corpus_composition([], split="internal_test")
    with pytest.raises(E4DiagnosisError, match="frozen split"):
        select_tiny_overfit_examples([], split="internal_test")


def test_diagnosis_sources_never_mention_the_frozen_split_as_a_data_path() -> None:
    for name in ("e4_collapse_diagnosis.py", "e4_tiny_overfit.py"):
        source = (REPO / "src" / "mednorm_vi" / "training" / "phase2" / name).read_text(
            encoding="utf-8")
        assert "internal_test.jsonl" not in source


# ---------------------------------------------------------------------------
# Diagnostic output carries no clinical text
# ---------------------------------------------------------------------------


@corpus_required
def test_full_diagnosis_payload_contains_no_governed_document_text() -> None:
    texts = [
        example.text
        for example in load_governed_examples(TRAIN_SPLIT, split="train", limit=400)
    ]
    round_trip = gold_grid_round_trip(
        load_governed_examples(TRAIN_SPLIT, split="train", limit=400), split="train")
    distribution = grid_class_distribution(
        load_governed_examples(TRAIN_SPLIT, split="train", limit=400), split="train")
    composition = measure_corpus_composition(
        load_governed_examples(TRAIN_SPLIT, split="train", limit=400), split="train")
    for payload in (round_trip.as_dict(), distribution.as_dict(), composition.as_dict()):
        assert_no_clinical_text(payload, corpus_texts=texts)


def test_the_text_guard_actually_detects_a_leak() -> None:
    text = "bệnh nhân bị sốt cao kéo dài nhiều ngày liên tiếp không giảm ."
    with pytest.raises(E4DiagnosisError, match="verbatim governed document text"):
        assert_no_clinical_text({"leak": text}, corpus_texts=[text])


# ---------------------------------------------------------------------------
# Tiny-overfit diagnostic
# ---------------------------------------------------------------------------


@corpus_required
def test_tiny_overfit_selection_is_deterministic_and_covers_every_typed_class() -> None:
    first = select_tiny_overfit_examples(
        load_governed_examples(TRAIN_SPLIT, split="train"))
    second = select_tiny_overfit_examples(
        load_governed_examples(TRAIN_SPLIT, split="train"))
    assert first.as_dict() == second.as_dict()
    assert TINY_OVERFIT_MIN_EXAMPLES <= first.example_count <= 16
    assert set(first.covered_required_types) == set(TINY_OVERFIT_REQUIRED_TYPES)
    assert first.missing_required_types == ()
    assert first.entity_count > 0
    assert list(first.row_indices) == sorted(first.row_indices)


@corpus_required
def test_tiny_overfit_selection_carries_no_clinical_text() -> None:
    texts = [
        example.text
        for example in load_governed_examples(TRAIN_SPLIT, split="train", limit=20000)
    ]
    selection = select_tiny_overfit_examples(
        load_governed_examples(TRAIN_SPLIT, split="train"))
    assert_no_clinical_text(selection.as_dict(), corpus_texts=texts)


def test_tiny_overfit_selection_refuses_an_out_of_range_size() -> None:
    with pytest.raises(TinyOverfitError, match="target_size must be between"):
        select_tiny_overfit_examples([], target_size=4)


def test_tiny_overfit_selection_refuses_too_few_eligible_examples() -> None:
    text = "sốt cao"
    examples = [
        _example(text, (EntitySpan(0, len(text), "SYMPTOM", text),), row=index)
        for index in range(3)
    ]
    with pytest.raises(TinyOverfitError, match="at least"):
        select_tiny_overfit_examples(examples)


def test_tiny_overfit_refuses_to_write_into_the_immutable_full_artifact() -> None:
    with pytest.raises(TinyOverfitError, match="immutable artifact directory"):
        assert_artifact_dir_is_not_protected(
            "/content/drive/MyDrive/MedNorm-VI/artifacts/e4_phobert_w2ner_full_v1")
    with pytest.raises(TinyOverfitError, match="immutable artifact directory"):
        assert_artifact_dir_is_not_protected("artifacts/e4_phobert_w2ner_smoke_v2/logs")
    assert assert_artifact_dir_is_not_protected(
        "artifacts/e4_tiny_overfit_diagnostic_v1").name == (
            "e4_tiny_overfit_diagnostic_v1")


def test_tiny_overfit_is_committed_disabled_and_unauthorized() -> None:
    with pytest.raises(TinyOverfitError, match="disabled"):
        assert_tiny_overfit_authorized(TINY_OVERFIT_AUTHORIZATION, enabled=False)
    with pytest.raises(TinyOverfitError, match="authorization string"):
        assert_tiny_overfit_authorized("yes please", enabled=True)
    assert_tiny_overfit_authorized(TINY_OVERFIT_AUTHORIZATION, enabled=True)


def test_tiny_overfit_scoring_separates_positive_cells_from_grid_accuracy() -> None:
    """A background-everywhere prediction must not look like a good result."""
    text = "sốt cao"
    example = _example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))
    targets, gold = build_tiny_overfit_targets(example)
    all_background = tuple(tuple(0 for _ in row) for row in targets)
    score = score_predicted_grid(
        epoch=1, mean_training_loss=0.001,
        target_grids=[targets], predicted_grids=[all_background],
        gold_mention_sets=[gold], predicted_mention_sets=[set()])
    assert score.grid_cell_accuracy == 0.5      # 2 of 4 cells happen to be NONE
    assert score.positive_cell_accuracy == 0.0  # the number that matters
    assert score.background_cell_accuracy == 1.0
    assert score.exact_f1 == 0.0


def test_tiny_overfit_scoring_reports_perfect_memorization() -> None:
    text = "sốt cao"
    example = _example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))
    targets, gold = build_tiny_overfit_targets(example)
    score = score_predicted_grid(
        epoch=40, mean_training_loss=1e-6,
        target_grids=[targets], predicted_grids=[targets],
        gold_mention_sets=[gold], predicted_mention_sets=[gold])
    assert score.grid_cell_accuracy == 1.0
    assert score.positive_cell_accuracy == 1.0
    assert score.exact_f1 == 1.0
    outcome = summarize_tiny_overfit([score], stopped_reason="reached_target_exact_f1")
    assert outcome.pipeline_can_memorize
    assert "coherent" in outcome.interpretation


def test_tiny_overfit_outcome_names_the_zero_loss_zero_f1_signature() -> None:
    text = "sốt cao"
    example = _example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))
    targets, gold = build_tiny_overfit_targets(example)
    score = score_predicted_grid(
        epoch=TINY_OVERFIT_MAX_EPOCHS, mean_training_loss=1e-6,
        target_grids=[targets],
        predicted_grids=[tuple(tuple(0 for _ in row) for row in targets)],
        gold_mention_sets=[gold], predicted_mention_sets=[set()])
    outcome = summarize_tiny_overfit(
        [score], stopped_reason="reached_max_epochs_without_target_exact_f1")
    assert not outcome.pipeline_can_memorize
    assert "inconsistency" in outcome.interpretation


def test_tiny_overfit_stop_rule_is_bounded_on_both_ends() -> None:
    stop, reason = should_stop_tiny_overfit(
        epoch=5, exact_f1=TINY_OVERFIT_TARGET_EXACT_F1)
    assert stop and reason == "reached_target_exact_f1"
    stop, reason = should_stop_tiny_overfit(
        epoch=TINY_OVERFIT_MAX_EPOCHS, exact_f1=0.0)
    assert stop and reason == "reached_max_epochs_without_target_exact_f1"
    stop, reason = should_stop_tiny_overfit(epoch=5, exact_f1=0.0)
    assert not stop and reason == "continue"
    with pytest.raises(TinyOverfitError, match="1-based"):
        should_stop_tiny_overfit(epoch=0, exact_f1=0.0)


def test_tiny_overfit_resolved_config_records_what_it_is_not() -> None:
    text = "sốt cao"
    examples = [
        _example(text, (EntitySpan(0, len(text), "SYMPTOM", text),), row=index)
        for index in range(12)
    ]
    selection = select_tiny_overfit_examples(examples, required_types=("SYMPTOM",))
    config = build_tiny_overfit_resolved_config(
        selection=selection, model_revision="a" * 40, tokenizer_revision="a" * 40,
        seed=1, learning_rate=5e-5)
    assert config["mode"] == "diagnostic"
    assert config["is_a_quality_result"] is False
    assert config["produces_a_deployable_checkpoint"] is False
    assert config["may_initialize_a_full_run"] is False
    assert config["evaluates_on_the_training_subset_by_design"] is True
    assert config["internal_test_accessed"] is False


def test_tiny_overfit_targets_match_the_e4_grid_builder() -> None:
    text = "sốt cao và ho"
    entity = EntitySpan(0, 7, "SYMPTOM", text[0:7])
    example = _example(text, (entity,))
    targets, gold = build_tiny_overfit_targets(example)
    reference = build_w2ner_grid(
        example.document_id, text, (entity,), words=tokenize_atomic_words(text))
    assert targets == reference.labels
    assert gold == {(0, 7, "SYMPTOM")}


# ---------------------------------------------------------------------------
# Diagnostic notebook parsing
# ---------------------------------------------------------------------------


def _notebook_code_source(path: Path) -> str:
    """Executable notebook source only.

    Prose in a markdown cell explaining that the notebook never writes
    ``output.zip`` would otherwise satisfy a naive substring search for
    ``output.zip``. These assertions are about what the notebook *does*, so they
    read code cells only.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"] if cell.get("cell_type") == "code")


def test_diagnostic_notebook_is_valid_and_committed_disabled() -> None:
    payload = json.loads(DIAGNOSTIC_NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4 and payload["cells"]
    source = _notebook_code_source(DIAGNOSTIC_NOTEBOOK)
    assert "RUN_TINY_OVERFIT_DIAGNOSTIC = False" in source
    assert 'CONFIRM_TINY_OVERFIT = ""' in source
    # The authorization string itself lives in one place (the module constant);
    # the notebook references the symbol rather than re-spelling the literal.
    assert "TINY_OVERFIT_AUTHORIZATION" in source
    assert TINY_OVERFIT_AUTHORIZATION not in source
    assert "assert_tiny_overfit_authorized" in source
    assert "assert_artifact_dir_is_not_protected" in source


def test_diagnostic_notebook_never_trains_full_or_touches_internal_test() -> None:
    source = _notebook_code_source(DIAGNOSTIC_NOTEBOOK)
    assert "RUN_FULL_TRAINING" not in source
    # internal_test is never a data source. It appears only as the declaration
    # ``"internal_test_accessed": False`` in the printed provenance record.
    assert "internal_test.jsonl" not in source
    assert 'split="internal_test"' not in source
    assert '"internal_test_accessed": False' in source
    assert "output.zip" not in source
    # The immutable full artifact is never named as a path in executable code.
    assert "e4_phobert_w2ner_full_v1" not in source


def test_diagnostic_notebook_reports_three_separate_metrics() -> None:
    """The notebook logs whatever TinyOverfitScore.as_dict() emits, so the
    guarantee is checked at its source rather than by matching metric names in
    notebook text that would drift apart from the code."""
    source = _notebook_code_source(DIAGNOSTIC_NOTEBOOK)
    assert "score_predicted_grid" in source
    assert "should_stop_tiny_overfit" in source
    assert "summarize_tiny_overfit" in source
    text = "sốt cao"
    example = _example(text, (EntitySpan(0, len(text), "SYMPTOM", text),))
    targets, gold = build_tiny_overfit_targets(example)
    emitted = score_predicted_grid(
        epoch=1, mean_training_loss=0.0, target_grids=[targets],
        predicted_grids=[targets], gold_mention_sets=[gold],
        predicted_mention_sets=[gold]).as_dict()
    for metric in ("grid_cell_accuracy", "positive_cell_accuracy",
                   "background_cell_accuracy", "exact_precision", "exact_recall",
                   "exact_f1"):
        assert metric in emitted, metric


def test_diagnostic_config_is_committed_not_authorized() -> None:
    text = DIAGNOSTIC_CONFIG.read_text(encoding="utf-8")
    assert "status: IMPLEMENTED_NOT_AUTHORIZED" in text
    assert "enabled_by_default: false" in text
    assert 'committed_confirmation_value: ""' in text
    assert "committed_run_flag: false" in text
    assert "unchanged_from_full_run: true" in text
    assert "internal_test_allowed: false" in text


# ---------------------------------------------------------------------------
# Nothing artifact-shaped is tracked in Git
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True, capture_output=True, text=True).stdout


def test_no_checkpoint_or_artifact_file_is_tracked_in_git() -> None:
    tracked = _git("ls-files").splitlines()
    for path in tracked:
        assert not path.endswith((".pt", ".bin", ".safetensors", ".ckpt")), path
        assert not path.startswith("local-artifacts/"), path
        assert not path.startswith("checkpoint/"), path
        # reports/ and data/derived/ hold generated output. Only their
        # explanatory documentation is tracked; no artifact, split or checkpoint.
        if path.startswith(("reports/", "data/derived/")):
            assert path.endswith((".md", ".gitkeep")), path


@artifact_required
def test_the_local_artifact_directory_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q",
         "local-artifacts/e4_phobert_w2ner_full_v1/validation_metrics.json"],
        capture_output=True)
    assert result.returncode == 0, "the local E4 artifact must be git-ignored"


# ---------------------------------------------------------------------------
# The E4 implementation this milestone diagnoses is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative_path", sorted(E4_PROTECTED_SHA256))
def test_protected_e4_path_is_byte_identical(relative_path: str) -> None:
    path = REPO / relative_path
    assert path.is_file(), relative_path
    assert _sha256(path) == E4_PROTECTED_SHA256[relative_path], (
        f"{relative_path} changed; this milestone diagnoses the E4 run and must "
        "not modify what produced it")


def test_w2ner_none_label_is_still_the_background_class() -> None:
    assert W2NER_NONE == "NONE"
    assert W2NERLabelVocab().labels[0] == W2NER_NONE
