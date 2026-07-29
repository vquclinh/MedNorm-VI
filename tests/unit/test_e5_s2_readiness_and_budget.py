"""E5 / S2 readiness and parameter budgeting (Audit 0042; retitled in Audit 0051).

Audit 0051 removed section D. It exercised ``governance.post_e4_gates``, which
blocked nine downstream tasks — including learned-L4-v2 training and leaderboard
submission — on a validated E4 full artifact. E4 is
RETIRED_FROM_ACTIVE_ARCHITECTURE and no such artifact can ever exist, so the gate
was permanently shut and was blocking live work behind a dead expert. The module and
its tests are gone; ``tests/unit/test_e4_retirement.py`` asserts E4 stays
unreachable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from mednorm_vi.governance.parameter_budget import (
    MAX_DEPLOYMENT_PARAMETERS,
    METHOD_COUNTED_FROM_CONFIG,
    METHOD_PUBLISHED_ESTIMATE,
    STATUS_IMPLEMENTED,
    STATUS_PLANNED,
    STATUS_TRAINED,
    CandidateModel,
    CandidateRegistry,
    DeploymentBudgetExceeded,
    ParameterBudgetError,
    UnverifiedDeploymentComponent,
    compute_deployment_budget,
    count_parameters,
    load_candidate_registry,
    load_deployment_selection,
)
from mednorm_vi.training.phase2.e5_mrc_training import E5TrainingContractError
from mednorm_vi.training.phase2.e5_readiness import (
    E5_RESUME_COMPATIBILITY_FIELDS,
    assert_compatible_e5_resume,
    assert_e5_checkpoint_custody,
    build_e5_full_resolved_config,
    e5_readiness_report,
    plan_e5_accumulation,
    resolve_e5_precision,
    resolve_e5_weight_format,
    scan_supervised_types,
)
from mednorm_vi.training.phase2.s2_assertion_training import (
    ASSERTION_ELIGIBLE_TYPES,
    ASSERTION_INELIGIBLE_TYPES,
    ASSERTION_LABEL_ORDER,
    PROVENANCE_TRUSTED,
    PROVENANCE_WEAK,
    SUPERVISED_FALSE,
    SUPERVISED_TRUE,
    UNKNOWN,
    AssertionExample,
    AssertionLabels,
    S2AssertionContractError,
    assert_no_document_leakage,
    assert_trainable,
    build_assertion_examples,
    build_s2_assertion_head,
    build_s2_checkpoint_payload,
    build_s2_resolved_config,
    extract_assertion_labels,
    multi_label_metrics,
    s2_config_sha256,
    s2_head_parameter_count,
    scan_assertion_supervision,
    validate_s2_artifact,
    validate_s2_checkpoint_payload,
)

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1"
REGISTRY_PATH = REPO / "configs" / "models" / "candidate_model_registry.yaml"
DEPLOYMENT_PATH = REPO / "configs" / "models" / "deployment_budget_template.yaml"

# Verified programmatically from the locally cached config (Audit 0042).
VIHEALTHBERT_PARAMETERS = 134_998_272
S2_HEAD_PARAMETERS = 9_228


def _corpus_available() -> bool:
    return (CORPUS / "splits" / "train.jsonl").is_file()


# ---------------------------------------------------------------------------
# C. Candidate vs deployment inventories
# ---------------------------------------------------------------------------


def _registry(*components: CandidateModel) -> CandidateRegistry:
    return CandidateRegistry(components=components)


def _verified(component_id: str, total: int, **overrides) -> CandidateModel:
    payload = dict(
        component_id=component_id, architecture_layer="L3", training_stage="S1",
        status=STATUS_TRAINED, model_id=f"org/{component_id}",
        total_parameters=total, trainable_parameters=total,
        parameter_count_method=METHOD_COUNTED_FROM_CONFIG,
        parameter_count_verified=True)
    payload.update(overrides)
    return CandidateModel(**payload)  # type: ignore[arg-type]


def test_candidate_total_may_exceed_9b_without_failure() -> None:
    registry = _registry(
        _verified("big_a", 5_000_000_000), _verified("big_b", 5_000_000_000))
    summary = registry.summary()
    assert summary["candidate_total_parameters"] == 10_000_000_000
    assert summary["candidate_total_exceeds_9b"] is True
    assert summary["gate_applies_to_candidates"] is False
    # No exception: the gate is not applied to the research inventory.


def test_deployment_within_budget_passes() -> None:
    registry = _registry(
        _verified("a", 4_000_000_000), _verified("b", 4_000_000_000))
    report = compute_deployment_budget(registry, ["a", "b"])
    assert report.within_budget is True
    assert report.total_loaded_parameters == 8_000_000_000
    assert report.remaining_margin == MAX_DEPLOYMENT_PARAMETERS - 8_000_000_000


def test_deployment_above_budget_fails() -> None:
    registry = _registry(
        _verified("a", 5_000_000_000), _verified("b", 5_000_000_000))
    with pytest.raises(DeploymentBudgetExceeded, match="exceeds"):
        compute_deployment_budget(registry, ["a", "b"])
    # The same selection can still be *reported* without enforcement.
    report = compute_deployment_budget(registry, ["a", "b"], enforce=False)
    assert report.within_budget is False
    assert report.remaining_margin < 0


def test_unknown_selected_count_fails_closed() -> None:
    registry = _registry(
        _verified("a", 1_000),
        CandidateModel(component_id="planned", architecture_layer="L5",
                       training_stage="S3", status=STATUS_PLANNED))
    with pytest.raises(UnverifiedDeploymentComponent, match="fails closed"):
        compute_deployment_budget(registry, ["a", "planned"])


def test_a_published_estimate_is_not_good_enough_for_deployment() -> None:
    registry = _registry(CandidateModel(
        component_id="estimated", architecture_layer="L3", training_stage="S1",
        status=STATUS_IMPLEMENTED, model_id="org/x", total_parameters=370_000_000,
        parameter_count_method=METHOD_PUBLISHED_ESTIMATE,
        parameter_count_verified=False))
    with pytest.raises(UnverifiedDeploymentComponent):
        compute_deployment_budget(registry, ["estimated"])


def test_shared_base_loaded_once_is_counted_once() -> None:
    registry = _registry(
        _verified("mention_head", VIHEALTHBERT_PARAMETERS,
                  shares_weights_with="vihealthbert_base"),
        _verified("assertion_head", VIHEALTHBERT_PARAMETERS,
                  shares_weights_with="vihealthbert_base",
                  adapter_parameters=S2_HEAD_PARAMETERS))
    report = compute_deployment_budget(registry, ["mention_head", "assertion_head"])
    # The backbone is counted once; the second head contributes only its adapter.
    assert report.total_loaded_parameters == VIHEALTHBERT_PARAMETERS + S2_HEAD_PARAMETERS
    assert report.components[1].counted_once is False
    assert report.components[1].counted_parameters == S2_HEAD_PARAMETERS


def test_independent_checkpoints_are_both_counted() -> None:
    registry = _registry(_verified("a", 1_000_000), _verified("b", 2_000_000))
    report = compute_deployment_budget(registry, ["a", "b"])
    assert report.total_loaded_parameters == 3_000_000
    assert all(component.counted_once for component in report.components)


def test_lora_counts_base_plus_adapter() -> None:
    registry = _registry(_verified(
        "qwen_lora", 4_000_000_000, adapter_parameters=50_000_000))
    report = compute_deployment_budget(registry, ["qwen_lora"])
    assert report.total_loaded_parameters == 4_050_000_000
    # Spec §17's base-only convention is reported alongside, not instead.
    assert report.base_only_parameters == 4_000_000_000
    assert report.adapter_parameters == 50_000_000


def test_calibration_parameters_are_included_even_if_small() -> None:
    registry = _registry(_verified("big", 1_000_000_000), _verified("calibration", 6_542))
    report = compute_deployment_budget(registry, ["big", "calibration"])
    assert report.total_loaded_parameters == 1_000_006_542
    assert any(c.component_id == "calibration" and c.counted_parameters == 6_542
               for c in report.components)


def test_a_planned_entry_may_not_carry_a_guessed_count() -> None:
    with pytest.raises(ParameterBudgetError, match="guessed count"):
        CandidateModel(component_id="x", architecture_layer="L5", training_stage="S3",
                       status=STATUS_PLANNED, total_parameters=600_000_000)


def test_verified_flag_requires_a_real_counting_method() -> None:
    with pytest.raises(ParameterBudgetError, match="cannot be marked verified"):
        CandidateModel(
            component_id="x", architecture_layer="L3", training_stage="S1",
            status=STATUS_IMPLEMENTED, total_parameters=10,
            parameter_count_method=METHOD_PUBLISHED_ESTIMATE,
            parameter_count_verified=True)


def test_empty_deployment_selection_is_rejected() -> None:
    with pytest.raises(ParameterBudgetError, match="at least one component"):
        compute_deployment_budget(_registry(_verified("a", 1)), [])


def test_count_parameters_is_programmatic_not_an_estimate() -> None:
    from mednorm_vi.training.phase2.l4_training import DEFAULT_FEATURE_ORDER, build_l4_mlp

    total, trainable = count_parameters(build_l4_mlp(len(DEFAULT_FEATURE_ORDER), 64))
    assert total == trainable == 6_542


def test_s2_head_parameter_count_is_programmatic() -> None:
    total, trainable = count_parameters(build_s2_assertion_head())
    assert total == trainable == S2_HEAD_PARAMETERS
    assert s2_head_parameter_count() == S2_HEAD_PARAMETERS


# --- the tracked registry files -------------------------------------------


def test_tracked_registry_covers_every_planned_stage() -> None:
    registry = load_candidate_registry(REGISTRY_PATH)
    ids = {component.component_id for component in registry.components}
    for expected in ("e3_vihealthbert_span_type",
                     "e5_xlmr_mrc_ner", "l4_learned_resolver_v2", "s2_assertion_head",
                     "s3_retrieval_dense", "s4_reranker", "s5_qwen_lora_critic",
                     "s6_calibration_meta_model"):
        assert expected in ids
    # Audit 0051: a retired component is not a planned stage. The registry exists
    # so a deployment can select from it, and E4 can never be selected.
    assert "e4_phobert_w2ner" not in ids


def test_tracked_registry_records_the_verified_vihealthbert_count() -> None:
    component = load_candidate_registry(REGISTRY_PATH).by_id("e3_vihealthbert_span_type")
    assert component.total_parameters == VIHEALTHBERT_PARAMETERS
    assert component.parameter_count_verified is True
    assert component.parameter_count_method == METHOD_COUNTED_FROM_CONFIG


def test_tracked_registry_records_the_reproducible_s2_head_count() -> None:
    component = load_candidate_registry(REGISTRY_PATH).by_id("s2_assertion_head")
    assert component.total_parameters == VIHEALTHBERT_PARAMETERS
    assert component.adapter_parameters == S2_HEAD_PARAMETERS
    assert component.shares_weights_with == "vihealthbert_base"
    assert component.parameter_count_verified is True
    assert component.parameter_count_method == METHOD_COUNTED_FROM_CONFIG


def test_tracked_registry_leaves_uncounted_models_unverified() -> None:
    registry = load_candidate_registry(REGISTRY_PATH)
    for component_id in ("e5_xlmr_mrc_ner", "s5_qwen_lora_critic"):
        component = registry.by_id(component_id)
        assert component.parameter_count_verified is False
        assert component.total_parameters is None


def test_the_spec_8852b_estimate_is_never_hardcoded_as_fact() -> None:
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "8852000000" not in text
    assert "8_852_000_000" not in text
    source = (REPO / "src" / "mednorm_vi" / "governance" / "parameter_budget.py"
              ).read_text(encoding="utf-8")
    assert "8852000000" not in source


def test_tracked_deployment_template_passes_the_gate() -> None:
    registry = load_candidate_registry(REGISTRY_PATH)
    name, selected = load_deployment_selection(DEPLOYMENT_PATH)
    report = compute_deployment_budget(registry, selected, manifest_name=name)
    assert report.within_budget is True
    assert report.total_loaded_parameters == VIHEALTHBERT_PARAMETERS + 6_542


# ---------------------------------------------------------------------------
# A. E5 readiness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _corpus_available(), reason="governed corpus not present")
def test_e5_reports_exactly_which_types_have_supervision() -> None:
    report = scan_supervised_types("train", CORPUS / "splits" / "train.jsonl")
    assert report.supervised_types == ("DIAGNOSIS", "MEDICATION", "SYMPTOM")
    assert report.unsupervised_types == ("TEST_NAME", "TEST_RESULT")
    assert report.entity_counts["TEST_NAME"] == 0
    assert report.entity_counts["TEST_RESULT"] == 0
    payload = report.as_dict()
    assert payload["laboratory_labels_fabricated"] is False
    # Queries are still built for all five architecture types.
    assert len(payload["queries_built_for"]) == 5


def test_e5_refuses_internal_test() -> None:
    with pytest.raises(E5TrainingContractError, match="internal_test"):
        scan_supervised_types("internal_test", "anything.jsonl")


def test_e5_accumulation_accounting_is_derived() -> None:
    plan = plan_e5_accumulation(11_720, micro_batch_size=1, accumulation_steps=8,
                                epochs=3)
    assert plan.optimizer_steps_per_epoch == -(-11_720 // 8)
    assert plan.expected_optimizer_steps == plan.optimizer_steps_per_epoch * 3
    assert plan.expected_backward_passes == 11_720 * 3
    assert plan.effective_batch_size == 8


def test_e5_precision_matches_the_e4_policy() -> None:
    assert resolve_e5_precision("bf16", device_type="cuda",
                                bf16_supported=True).mode == "bf16"
    fallback = resolve_e5_precision("bf16", device_type="cuda", bf16_supported=False)
    assert fallback.mode == "fp16" and fallback.use_grad_scaler is True
    assert resolve_e5_precision("bf16", device_type="cpu").mode == "fp32"


def test_e5_weight_format_is_resolved_from_a_listing_when_available() -> None:
    resolved = resolve_e5_weight_format(
        "a" * 40, repository_files=["config.json", "model.safetensors"])
    assert resolved.use_safetensors is True
    bin_only = resolve_e5_weight_format(
        "a" * 40, repository_files=["config.json", "pytorch_model.bin"])
    assert bin_only.use_safetensors is False
    with pytest.raises(E5TrainingContractError, match="publishes neither"):
        resolve_e5_weight_format("a" * 40, repository_files=["config.json"])


def test_e5_resolved_config_declares_supervised_and_unsupervised_types() -> None:
    config = build_e5_full_resolved_config(
        mode="full", model_revision="a" * 40, tokenizer_revision="a" * 40, seed=1,
        max_length=384, max_span_chars=200, allow_overlaps=False,
        weight_format=resolve_e5_weight_format("a" * 40),
        accumulation=plan_e5_accumulation(100, micro_batch_size=1,
                                          accumulation_steps=8, epochs=1),
        precision=resolve_e5_precision("bf16", device_type="cuda",
                                       bf16_supported=True),
        supervised_types=("DIAGNOSIS", "MEDICATION", "SYMPTOM"),
        unsupervised_types=("TEST_NAME", "TEST_RESULT"))
    assert config["supervised_entity_types"] == ["DIAGNOSIS", "MEDICATION", "SYMPTOM"]
    assert config["unsupervised_entity_types"] == ["TEST_NAME", "TEST_RESULT"]
    assert config["laboratory_labels_fabricated"] is False
    assert config["internal_test_accessed"] is False
    assert config["query_hash"]


def test_e5_checkpoint_custody_and_resume_compatibility() -> None:
    payload = {
        "mode": "full", "model_state": {}, "optimizer_state": {}, "scaler_state": {},
        "epoch": 1, "optimizer_steps": 10, "best_metric": 0.1,
        "e5_input_contract_version": "e5-mrc-start-end-v1",
        "e5_checkpoint_schema_version": "phase2-e5-checkpoint-v1",
        "query_revision": "mrc-type-queries-v1", "query_hash": "abc",
        "config_sha256": "0" * 64, "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40, "precision_mode": "bf16",
        "optimizer_signature": "AdamW-lr1e-05-wd0-clip1",
        "accumulation_signature": "micro1-accum8-effective8"}
    assert_e5_checkpoint_custody(payload)
    expected = {field: payload[field] for field in E5_RESUME_COMPATIBILITY_FIELDS}
    assert_compatible_e5_resume(payload, expected=expected)
    with pytest.raises(E5TrainingContractError, match="precision_mode"):
        assert_compatible_e5_resume(payload, expected={**expected,
                                                       "precision_mode": "fp16"})
    incomplete = {key: value for key, value in payload.items()
                  if key != "optimizer_state"}
    with pytest.raises(E5TrainingContractError, match="missing resume state"):
        assert_e5_checkpoint_custody(incomplete)


@pytest.mark.skipif(not _corpus_available(), reason="governed corpus not present")
def test_e5_readiness_status_is_not_trained() -> None:
    report = e5_readiness_report(
        scan_supervised_types("train", CORPUS / "splits" / "train.jsonl"))
    payload = report.as_dict()
    assert payload["trained"] is False
    assert "READY_FOR_COLAB_SMOKE" in payload["status"]
    assert payload["internal_test_accessed"] is False


# ---------------------------------------------------------------------------
# B. S2 assertion readiness
# ---------------------------------------------------------------------------


def test_missing_supervision_is_unknown_never_false() -> None:
    labels = extract_assertion_labels({"start": 0, "end": 1, "text": "x"})
    for label in ASSERTION_LABEL_ORDER:
        assert labels.state_for(label) == UNKNOWN
    assert labels.has_any_supervision is False
    # Targets are None (masked), not 0.0 (a confident negative).
    assert labels.target_vector() == (None, None, None)
    assert labels.loss_mask() == (0, 0, 0)


def test_partial_supervision_leaves_the_rest_unknown() -> None:
    labels = extract_assertion_labels({"assertions": {"isNegated": True}})
    assert labels.state_for("isNegated") == SUPERVISED_TRUE
    assert labels.state_for("isHistorical") == UNKNOWN
    assert labels.state_for("isFamily") == UNKNOWN
    assert labels.target_vector() == (1.0, None, None)
    assert labels.loss_mask() == (1, 0, 0)
    assert labels.supervised_label_count == 1


def test_explicit_false_is_distinct_from_unknown() -> None:
    labels = extract_assertion_labels(
        {"assertions": {"isNegated": False, "isFamily": None}})
    assert labels.state_for("isNegated") == SUPERVISED_FALSE
    assert labels.state_for("isFamily") == UNKNOWN
    assert labels.target_vector() == (0.0, None, None)


def test_non_boolean_supervision_is_rejected() -> None:
    with pytest.raises(S2AssertionContractError, match="boolean or null"):
        extract_assertion_labels({"assertions": {"isNegated": "yes"}})
    with pytest.raises(S2AssertionContractError, match="must be a mapping"):
        extract_assertion_labels({"assertions": ["isNegated"]})


def test_trusted_and_weak_provenance_are_separate_channels() -> None:
    trusted = AssertionLabels({"isNegated": SUPERVISED_TRUE}, PROVENANCE_TRUSTED)
    weak = AssertionLabels({"isNegated": SUPERVISED_TRUE}, PROVENANCE_WEAK)
    assert trusted.provenance != weak.provenance
    assert trusted.as_dict()["provenance"] == "trusted"
    assert weak.as_dict()["provenance"] == "weak"
    # Governed extraction never produces a weak label implicitly.
    assert extract_assertion_labels({"assertions": {"isNegated": True}}).provenance == (
        PROVENANCE_TRUSTED)
    with pytest.raises(S2AssertionContractError, match="unknown provenance"):
        AssertionLabels({"isNegated": SUPERVISED_TRUE}, "guessed")


def test_only_assertion_eligible_types_produce_examples() -> None:
    row = {
        "document_id": "d1", "example_id": "d1", "text": "sốt cao WBC 12",
        "entities": [
            {"start": 0, "end": 3, "target_type": "SYMPTOM", "text": "sốt"},
            {"start": 8, "end": 11, "target_type": "TEST_NAME", "text": "WBC"},
            {"start": 12, "end": 14, "target_type": "TEST_RESULT", "text": "12"},
        ]}
    examples, ineligible = build_assertion_examples(row)
    assert [example.entity_type for example in examples] == ["SYMPTOM"]
    assert ineligible == 2
    assert set(ASSERTION_INELIGIBLE_TYPES) == {"TEST_NAME", "TEST_RESULT"}
    assert set(ASSERTION_ELIGIBLE_TYPES) == {"MEDICATION", "DIAGNOSIS", "SYMPTOM"}


def test_an_ineligible_type_cannot_be_forced_into_an_example() -> None:
    with pytest.raises(S2AssertionContractError, match="not assertion-eligible"):
        AssertionExample(
            document_id="d", example_id="e", source_dataset="s",
            entity_type="TEST_NAME", start=0, end=3, mention_text="WBC",
            labels=AssertionLabels.all_unknown())


def test_the_offset_invariant_is_enforced() -> None:
    row = {"document_id": "d", "example_id": "e", "text": "sốt cao",
           "entities": [{"start": 0, "end": 3, "target_type": "SYMPTOM",
                         "text": "WRONG"}]}
    with pytest.raises(S2AssertionContractError, match="does not\\s+slice out"):
        build_assertion_examples(row)


def test_document_level_leakage_is_detected() -> None:
    def example(document: str) -> AssertionExample:
        return AssertionExample(
            document_id=document, example_id=f"{document}#0", source_dataset="s",
            entity_type="SYMPTOM", start=0, end=3, mention_text="sốt",
            labels=AssertionLabels.all_unknown(), group_id=document)

    assert_no_document_leakage([example("d1")], [example("d2")])
    with pytest.raises(S2AssertionContractError, match="document leakage"):
        assert_no_document_leakage([example("d1")], [example("d1")])


def test_label_order_is_deterministic() -> None:
    assert ASSERTION_LABEL_ORDER == ("isNegated", "isHistorical", "isFamily")
    from mednorm_vi.schemas.constants import ASSERTION_LABELS as SCHEMA_LABELS

    assert set(ASSERTION_LABEL_ORDER) == set(SCHEMA_LABELS)


def test_multi_label_metrics_ignore_unknown_positions() -> None:
    # Row 1: isNegated supervised-true and predicted; the rest unknown.
    metrics = multi_label_metrics(
        predictions=[[1, 1, 0], [0, 0, 0]],
        targets=[[1.0, None, None], [0.0, None, None]])
    assert metrics["supervised_positions"] == 2
    assert metrics["per_label"]["isNegated"]["tp"] == 1
    # The predicted isHistorical on row 1 is NOT scored as a false positive,
    # because there is no supervision to score it against.
    assert metrics["per_label"]["isHistorical"]["supervised"] == 0
    assert metrics["per_label"]["isHistorical"]["fp"] == 0
    assert metrics["micro"]["tp"] == 1
    assert metrics["micro"]["f1"] == pytest.approx(1.0)


def test_multi_label_metrics_reject_malformed_rows() -> None:
    with pytest.raises(S2AssertionContractError, match="row count mismatch"):
        multi_label_metrics(predictions=[[1, 0, 0]], targets=[])
    with pytest.raises(S2AssertionContractError, match="one value per assertion label"):
        multi_label_metrics(predictions=[[1, 0]], targets=[[1.0, None]])


def test_s2_refuses_internal_test() -> None:
    with pytest.raises(S2AssertionContractError, match="internal_test"):
        scan_assertion_supervision("internal_test", "anything.jsonl")


def test_s2_resolved_config_checkpoint_schema_and_artifact_validator(
    tmp_path: Path,
) -> None:
    resolved = build_s2_resolved_config(
        mode="smoke",
        model_id="demdecuong/vihealthbert-base-word",
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        seed=1,
        max_length=192,
        micro_batch_size=2,
        accumulation_steps=4,
        epochs=1,
        learning_rate=2e-5,
        progress={"heartbeat_first_n": 10, "heartbeat_every_n": 100},
    )
    config_hash = s2_config_sha256(resolved)
    payload = build_s2_checkpoint_payload(
        mode="smoke",
        config_sha256=config_hash,
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        parameter_count=VIHEALTHBERT_PARAMETERS + S2_HEAD_PARAMETERS,
    )
    validate_s2_checkpoint_payload(
        payload, expected_mode="smoke", expected_config_sha256=config_hash)
    artifact = tmp_path / "s2_artifact"
    (artifact / "checkpoints").mkdir(parents=True)
    (artifact / "logs").mkdir()
    manifest = {
        **resolved,
        "config_sha256": config_hash,
        "completed_epochs": 1,
        "optimizer_steps": 1,
        "parameter_count": VIHEALTHBERT_PARAMETERS + S2_HEAD_PARAMETERS,
        "internal_test_accessed": False,
    }
    (artifact / "resolved_config.json").write_text(
        json.dumps({**resolved, "config_sha256": config_hash}), encoding="utf-8")
    (artifact / "validation_metrics.json").write_text(
        json.dumps({"macro_f1": 0.0, "internal_test_accessed": False}), encoding="utf-8")
    (artifact / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    (artifact / "logs" / "training_history.jsonl").write_text(
        json.dumps({"epoch": 1, "mode": "smoke"}) + "\n", encoding="utf-8")
    for name in ("best.pt", "latest.pt"):
        (artifact / "checkpoints" / name).write_text(
            json.dumps(payload), encoding="utf-8")

    report = validate_s2_artifact(artifact, mode="smoke")
    assert report.ok is True
    assert set(report.checkpoint_hashes) == {"best.pt", "latest.pt"}

    bad_payload = {**payload, "internal_test_accessed": True}
    with pytest.raises(S2AssertionContractError, match="internal_test_accessed"):
        validate_s2_checkpoint_payload(
            bad_payload, expected_mode="smoke", expected_config_sha256=config_hash)


def test_a_split_without_supervision_is_not_trainable(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({
        "document_id": "d1", "example_id": "d1", "source_dataset": "s",
        "text": "sốt cao",
        "entities": [{"start": 0, "end": 3, "target_type": "SYMPTOM", "text": "sốt"}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    report, examples = scan_assertion_supervision("train", path)
    assert report.eligible_mentions == 1
    assert report.examples_with_any_supervision == 0
    assert report.trainable is False
    assert len(examples) == 1
    with pytest.raises(S2AssertionContractError, match="fabricating labels"):
        assert_trainable(report)


def test_a_split_with_supervision_is_trainable(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({
        "document_id": "d1", "example_id": "d1", "source_dataset": "s",
        "text": "sốt cao",
        "entities": [{"start": 0, "end": 3, "target_type": "SYMPTOM", "text": "sốt",
                      "assertions": {"isNegated": True}}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    report, _examples = scan_assertion_supervision("train", path)
    assert report.examples_with_any_supervision == 1
    assert report.trusted_examples == 1
    assert report.trainable is True
    assert_trainable(report)


@pytest.mark.skipif(not _corpus_available(), reason="governed corpus not present")
def test_the_real_governed_corpus_has_no_assertion_supervision() -> None:
    """The decisive measured fact: S2 is code-ready but data-blocked."""
    coverage = json.loads(
        (CORPUS / "manifests" / "annotation_coverage.json").read_text(encoding="utf-8"))
    report, _examples = scan_assertion_supervision(
        "train", CORPUS / "splits" / "train.jsonl", coverage_manifest=coverage)
    assert report.eligible_mentions == 11_720
    assert report.examples_with_any_supervision == 0
    assert report.trusted_examples == 0
    assert report.trainable is False
    for label in ASSERTION_LABEL_ORDER:
        counts = report.supervision_counts[label]
        assert counts[UNKNOWN] == 11_720
        assert counts[SUPERVISED_TRUE] == 0
        assert counts[SUPERVISED_FALSE] == 0
    # Every governed source declares assertions: false.
    assert set(report.coverage_manifest_declares_assertions.values()) == {False}


# ---------------------------------------------------------------------------
# F. Notebook parsing
# ---------------------------------------------------------------------------


def _notebook_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


@pytest.mark.parametrize(
    ("relative_path", "required_text"),
    (
        ("notebooks/MedNorm_E5_XLMR_MRC_NER_Training.ipynb",
         "I_AUTHORIZE_E5_FULL_TRAINING"),
        ("notebooks/MedNorm_S2_Assertion.ipynb",
         "I_AUTHORIZE_S2_FULL_TRAINING"),
    ),
)
def test_e5_and_s2_notebook_cells_parse(relative_path: str, required_text: str) -> None:
    path = REPO / relative_path
    text = path.read_text(encoding="utf-8")
    assert required_text in text
    assert "internal_test" in text
    code_cells = _notebook_code_cells(path)
    assert code_cells
    for index, source in enumerate(code_cells, start=1):
        ast.parse(source, filename=f"{relative_path}#cell{index}")


# ---------------------------------------------------------------------------
# H. Protected paths and repository hygiene
# ---------------------------------------------------------------------------

# Audit 0042 pinned the modules that a parallel E4 workstream must not touch.
# Audit 0045 moved that subject once when the E4 implementation was replaced;
# Audit 0051 deleted E4 entirely, so four of the seven pinned paths no longer
# exist and `training/phase2/artifacts.py` legitimately changed (E4 was removed
# from it).
#
# What is pinned now is what the guard was always really protecting: the exact
# character-offset evaluator that every reported mention number depends on, and
# the architecture PDF. A silent edit to either would invalidate measurements
# rather than merely break a build, which is why they are pinned by digest and
# not by a test of their behaviour.
PROTECTED_SHA256: dict[str, str] = {
    "src/mednorm_vi/evaluation/exact_mention.py":
        "7b2ba8fd72afdde715f90ac321c85cf3ea1d6e88e3eb98701903b463f14e07f0",
    "docs/MedNorm-VI_Architecture.pdf":
        "0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b",
}


@pytest.mark.parametrize("relative_path", sorted(PROTECTED_SHA256))
def test_protected_paths_are_byte_for_byte_unchanged(relative_path: str) -> None:
    digest = hashlib.sha256((REPO / relative_path).read_bytes()).hexdigest()
    assert digest == PROTECTED_SHA256[relative_path], (
        f"{relative_path} changed; every reported measurement depends on it "
        "staying byte-identical")


def test_no_model_checkpoint_cache_or_archive_is_tracked_in_git() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip"))
        assert not line.startswith(
            ("artifacts/", "weights/", "caches/", "checkpoint/", ".claude/"))
        assert Path(line).name not in {"CLAUDE.md", "AGENTS.md"}


def test_new_configs_are_valid_yaml_and_forbid_internal_test() -> None:
    e5 = yaml.safe_load(
        (REPO / "configs" / "training" / "phase2_e5_xlmr_mrc_ner_colab.yaml").read_text(
            encoding="utf-8"))
    assert e5["data"]["internal_test_allowed"] is False
    assert e5["training"]["gradient_accumulation_steps"] == 8
    assert e5["training"]["max_grad_norm"] == 1.0
    assert e5["progress"]["heartbeat_every_n_train_samples"] == 100
    assert e5["supervision"]["supervised_entity_types"] == [
        "DIAGNOSIS", "MEDICATION", "SYMPTOM"]
    assert e5["supervision"]["unsupervised_entity_types"] == ["TEST_NAME", "TEST_RESULT"]
    s2 = yaml.safe_load(
        (REPO / "configs" / "training" / "phase2_s2_assertion_colab.yaml").read_text(
            encoding="utf-8"))
    assert s2["data"]["internal_test_allowed"] is False
    assert s2["governed_supervision"]["trainable"] is False
    assert s2["labels"]["unknown_supervision_is_masked_not_false"] is True
    assert s2["provenance"]["allow_weak_supervision"] is False
    assert s2["artifacts"]["checkpoint_schema_version"] == "phase2-s2-checkpoint-v1"
    assert s2["head"]["verified_parameter_count"] == S2_HEAD_PARAMETERS
