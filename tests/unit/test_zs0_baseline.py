"""ZS0 zero-shot baseline and E4 retirement (Audit 0048).

Two things are locked down here. First, that E4 is genuinely out of the active
stack — off in every profile, absent from every ledger, unable to reach Stage 3
or 4 — while its source and audits survive. Second, that ZS0 is actually
zero-shot: it can neither load a fitted checkpoint nor invent a mention, an
assertion label or an ontology code.

No test trains, opens internal_test, or runs organizer inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tokenize
import zipfile
from pathlib import Path

import pytest
import yaml

from mednorm_vi.governance.e4_retirement import (
    E4_FEATURE_FLAG,
    E4_FORBIDDEN_STAGES,
    RETIRED_FROM_ACTIVE_STACK,
    STAGE2_FINAL_RESULT,
    E4RetirementError,
    E4RetirementRecord,
    assert_e4_absent_from_ledger,
    assert_e4_disabled,
    assert_no_e4_checkpoint_required,
    assert_stage_not_forbidden,
)
from mednorm_vi.lattice.models import (
    EXPERT_GLINER,
    EXPERT_MEDICATION_GRAMMAR,
    ExpertSpanProposal,
)
from mednorm_vi.schemas.constants import ASSERTION_LABELS, ORGANIZER_LABEL_BY_TYPE
from mednorm_vi.zs0 import (
    MAX_ACTIVE_PARAMETERS,
    ORGANIZER_AUTHORIZATION,
    ArmResult,
    BudgetExceeded,
    Candidate,
    LedgerEntry,
    OntologySnapshot,
    OrganizerPreconditions,
    SubmissionError,
    UnverifiedComponent,
    ZS0ProfileError,
    all_arms,
    assert_organizer_inference_allowed,
    assert_zs0_components_allowed,
    assert_zs0_feature_flags,
    build_ledger,
    convert_gliner_spans,
    qwen_proposals,
    resolve,
    resolve_occurrence,
    select_arm,
    validate_package,
)
from mednorm_vi.zs0.assertions import (
    AssertionDecision,
    decide_deterministically,
    parse_qwen_assertions,
)
from mednorm_vi.zs0.linking import (
    LinkingError,
    assert_codes_in_snapshot,
    generate_candidates,
    link_mention,
    select_from_candidates,
)
from mednorm_vi.zs0.profile import (
    FORBIDDEN_COMPONENTS,
    ZS0_A,
    ZS0_B,
    ZS0_C,
    assert_no_external_api,
    assert_no_training_executed,
    build_arm,
)
from mednorm_vi.zs0.proposals import (
    REJECT_AMBIGUOUS_OCCURRENCE,
    REJECT_NOT_A_SUBSTRING,
    REJECT_OUT_OF_RANGE,
    REJECT_UNSUPPORTED_TYPE,
    ProposalRejected,
    RejectionLedger,
)
from mednorm_vi.zs0.resolver import ResolverThresholds, emitted
from mednorm_vi.zs0.submission import (
    OUTPUT_FILENAMES,
    validate_output_payload,
)

REPO = Path(__file__).resolve().parents[2]
ZS0_PACKAGE = REPO / "src" / "mednorm_vi" / "zs0"
NOTEBOOK = REPO / "notebooks" / "MedNorm_ZS0_Baseline_Submission.ipynb"
ZS0_CONFIG = REPO / "configs" / "pipeline" / "zs0_baseline.yaml"
LEDGER_CONFIG = REPO / "configs" / "models" / "zs0_parameter_ledger.yaml"

ACTIVE_PIPELINE_CONFIGS = tuple(
    sorted(REPO.glob("configs/pipeline/*.yaml")))


def _notebook_code() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", []))
                     for c in payload["cells"] if c.get("cell_type") == "code")


def _executable_tokens(path: Path) -> str:
    kept: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if token.string.strip():
                kept.append(token.string)
    return " " + " ".join(kept) + " "


# ---------------------------------------------------------------------------
# E4 retirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_path", ACTIVE_PIPELINE_CONFIGS)
def test_e4_is_disabled_in_every_pipeline_profile(config_path: Path) -> None:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    flags = document.get("feature_flags", {})
    assert flags.get(E4_FEATURE_FLAG, False) is False, config_path.name
    assert_e4_disabled(flags, profile=config_path.name)


@pytest.mark.parametrize("config_path", ACTIVE_PIPELINE_CONFIGS)
def test_no_profile_requires_an_e4_checkpoint(config_path: Path) -> None:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for key in ("full_requires_checkpoints", "specialist_requires_checkpoints"):
        assert_no_e4_checkpoint_required(
            document.get(key, []) or [], profile=f"{config_path.name}:{key}")


def test_enabling_e4_is_refused() -> None:
    with pytest.raises(E4RetirementError, match=RETIRED_FROM_ACTIVE_STACK):
        assert_e4_disabled({E4_FEATURE_FLAG: True}, profile="hypothetical")


@pytest.mark.parametrize("stage", E4_FORBIDDEN_STAGES)
def test_e4_stage_three_and_four_can_never_run(stage: str) -> None:
    with pytest.raises(E4RetirementError, match="must never run"):
        assert_stage_not_forbidden(stage)
    # Stage 2 is history, not forbidden — it is what produced the evidence.
    assert_stage_not_forbidden("tiny_recipe_ablation")


def test_e4_is_absent_from_the_active_parameter_ledger() -> None:
    document = yaml.safe_load(LEDGER_CONFIG.read_text(encoding="utf-8"))
    component_ids = [c["component_id"] for c in document["components"]]
    assert_e4_absent_from_ledger(component_ids)
    excluded = {c["component_id"] for c in document["excluded"]}
    assert "e4_phobert_w2ner" in excluded
    with pytest.raises(E4RetirementError, match="loads no weights"):
        assert_e4_absent_from_ledger([*component_ids, "e4_phobert_w2ner"])


def test_the_stage_two_result_is_recorded_exactly() -> None:
    result = STAGE2_FINAL_RESULT
    assert result["examples"] == 12
    assert result["gold_mentions"] == 22
    assert result["all_required_types_present"] is True
    assert result["epochs_per_recipe"] == 200
    assert result["optimizer_steps_per_recipe"] == 600
    assert result["save_reload_reproduction_passed"] is True
    assert result["recipes"]["reference_ce"]["best_exact_f1"] == 0.3448
    assert result["recipes"]["group_balanced_ce"]["best_exact_f1"] == 0.7333
    assert result["recipes"]["hard_negative_ce"]["best_exact_f1"] == 0.3704
    assert result["any_recipe_met_the_gate"] is False
    assert result["selected_recipe"] is None
    assert result["stage3_blocked"] is True
    assert result["stage4_blocked"] is True
    # A completed run, not a runtime failure.
    assert result["run_validity"] == "VALID_COMPLETED_RUN"


def test_e4_source_and_audits_are_preserved() -> None:
    record = E4RetirementRecord()
    assert record.status == RETIRED_FROM_ACTIVE_STACK
    for path in record.preserved:
        assert (REPO / path).exists(), path
    payload = record.as_dict()
    assert payload["source_deleted"] is False
    assert payload["audits_modified"] is False
    assert payload["architecture_pdf_modified"] is False


# ---------------------------------------------------------------------------
# ZS0 is genuinely zero-shot
# ---------------------------------------------------------------------------


def test_zs0_profile_disables_every_trained_expert() -> None:
    document = yaml.safe_load(ZS0_CONFIG.read_text(encoding="utf-8"))
    flags = document["feature_flags"]
    assert flags["enable_e3_vihealthbert"] is False
    assert flags["enable_e4_phobert_w2ner"] is False
    assert flags["enable_e5_xlmr_mrc"] is False
    assert flags["enable_l4_learned_v2"] is False
    assert flags["enable_e6_gliner"] is True
    assert flags["enable_e7_qwen_proposer"] is True
    assert_zs0_feature_flags(flags)
    assert document["full_requires_checkpoints"] == []


@pytest.mark.parametrize("component", sorted(FORBIDDEN_COMPONENTS))
def test_every_forbidden_component_is_refused_by_name(component: str) -> None:
    with pytest.raises(ZS0ProfileError, match="forbidden in ZS0"):
        assert_zs0_components_allowed([component])


def test_zs0_cannot_load_e4_or_the_fine_tuned_e3_or_a_random_e5_head() -> None:
    for component in ("E4_phobert_w2ner", "E3_vihealthbert_span_type",
                      "E5_xlmr_mrc_ner"):
        with pytest.raises(ZS0ProfileError):
            assert_zs0_components_allowed([component])


def test_no_zs0_arm_contains_a_trained_expert() -> None:
    for arm in all_arms():
        assert_zs0_components_allowed(arm.components)
        assert arm.as_dict()["uses_fine_tuned_weights"] is False
        assert arm.as_dict()["uses_random_task_head"] is False
    assert build_arm(ZS0_A).components == (
        "E1_medication_grammar", "E2_laboratory_parser")
    assert "E6_gliner_open_type" in build_arm(ZS0_B).components
    assert "E7_qwen_cascade" in build_arm(ZS0_C).components


def test_the_qwen_proposer_arm_presupposes_gliner() -> None:
    from mednorm_vi.zs0.profile import ZS0Arm
    with pytest.raises(ZS0ProfileError, match="presupposes GLiNER"):
        ZS0Arm(name=ZS0_C, uses_gliner=False, uses_qwen_proposer=True)


def test_zs0_refuses_training_and_external_apis() -> None:
    assert_no_training_executed({"backward_passes": 0, "optimizer_steps": 0})
    with pytest.raises(ZS0ProfileError, match="inference-only"):
        assert_no_training_executed({"backward_passes": 1})
    with pytest.raises(ZS0ProfileError, match="optimizer or scheduler"):
        assert_no_training_executed({"optimizer_constructed": True})
    assert_no_external_api({})
    with pytest.raises(ZS0ProfileError, match="external model API"):
        assert_no_external_api({"openai_api_key": "sk-x"})


def test_the_zs0_package_never_trains() -> None:
    for path in sorted(ZS0_PACKAGE.glob("*.py")):
        tokens = _executable_tokens(path)
        for forbidden in (" backward ( ", " AdamW ", " optim ", " zero_grad ( "):
            assert forbidden not in tokens, f"{path.name}: {forbidden}"


def test_the_zs0_package_never_opens_internal_test() -> None:
    """Match usage, not prose.

    A comment stating the frozen split is never opened must not fail this check;
    what must be absent is any path or split argument naming it.
    """
    for path in sorted(ZS0_PACKAGE.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "internal_test.jsonl" not in source, path.name
        assert 'split="internal_test"' not in source, path.name
        assert '"internal_test"' not in _executable_tokens(path), path.name


# ---------------------------------------------------------------------------
# Proposals preserve exact offsets and never invent text
# ---------------------------------------------------------------------------

TEXT = "Bệnh nhân sốt cao và ho khan ."


def test_gliner_proposals_preserve_exact_substrings_and_offsets() -> None:
    accepted, ledger = convert_gliner_spans(
        document_id="d1", original_text=TEXT,
        raw_spans=[{"start": 10, "end": 17, "label": "symptom", "score": 0.9}],
        model_revision="rev", checkpoint_sha256="0" * 64)
    assert len(accepted) == 1
    proposal = accepted[0]
    assert TEXT[proposal.start:proposal.end] == proposal.text == "sốt cao"
    assert proposal.type_scores == {"SYMPTOM": 0.9}
    assert ledger.total == 0


def test_a_gliner_span_that_does_not_reproduce_its_text_is_rejected() -> None:
    _accepted, ledger = convert_gliner_spans(
        document_id="d1", original_text=TEXT,
        raw_spans=[{"start": 900, "end": 910, "label": "symptom", "score": 0.9}],
        model_revision="rev", checkpoint_sha256="0" * 64)
    assert ledger.counts[REJECT_OUT_OF_RANGE] == 1
    assert ledger.as_dict()["contains_clinical_text"] is False


def test_an_unmappable_gliner_label_is_rejected() -> None:
    _accepted, ledger = convert_gliner_spans(
        document_id="d1", original_text=TEXT,
        raw_spans=[{"start": 10, "end": 17, "label": "vehicle", "score": 0.9}],
        model_revision="rev", checkpoint_sha256="0" * 64)
    assert ledger.counts[REJECT_UNSUPPORTED_TYPE] == 1


def test_qwen_may_only_return_literal_substrings() -> None:
    accepted, ledger = qwen_proposals(
        document_id="d1", original_text=TEXT, segment_start=0, segment_text=TEXT,
        payload=json.dumps({"mentions": [
            {"text": "sốt cao", "type": "SYMPTOM"},
            {"text": "viêm phổi cấp", "type": "DIAGNOSIS"},
        ]}),
        model_revision="rev", checkpoint_sha256="0" * 64)
    assert len(accepted) == 1
    assert accepted[0].text == "sốt cao"
    assert TEXT[accepted[0].start:accepted[0].end] == "sốt cao"
    assert ledger.counts[REJECT_NOT_A_SUBSTRING] == 1


def test_a_repeated_substring_fails_closed_without_an_anchor() -> None:
    with pytest.raises(ProposalRejected) as excinfo:
        resolve_occurrence("sốt và sốt", "sốt")
    assert excinfo.value.reason == REJECT_AMBIGUOUS_OCCURRENCE
    # An anchor that identifies exactly one occurrence resolves it.
    assert resolve_occurrence("sốt và sốt cao", "sốt", anchor="sốt cao") == (7, 10)
    # An anchor that is itself ambiguous does not.
    with pytest.raises(ProposalRejected):
        resolve_occurrence("a x a x", "a", anchor="a x")


def test_qwen_cannot_emit_an_unsupported_entity_type() -> None:
    _accepted, ledger = qwen_proposals(
        document_id="d1", original_text=TEXT, segment_start=0, segment_text=TEXT,
        payload=json.dumps({"mentions": [{"text": "sốt cao", "type": "PROCEDURE"}]}),
        model_revision="rev", checkpoint_sha256="0" * 64)
    assert ledger.counts[REJECT_UNSUPPORTED_TYPE] == 1


def test_malformed_qwen_output_is_rejected_not_repaired() -> None:
    for payload in ("not json at all", json.dumps({"wrong": []}),
                    json.dumps({"mentions": "a string"})):
        accepted, ledger = qwen_proposals(
            document_id="d1", original_text=TEXT, segment_start=0,
            segment_text=TEXT, payload=payload, model_revision="rev",
            checkpoint_sha256="0" * 64)
        assert accepted == ()
        assert ledger.total >= 1


def test_the_rejection_ledger_never_holds_clinical_text() -> None:
    ledger = RejectionLedger()
    ledger.record(REJECT_NOT_A_SUBSTRING)
    serialized = json.dumps(ledger.as_dict(), ensure_ascii=False)
    assert TEXT not in serialized
    assert ledger.as_dict()["contains_clinical_text"] is False


# ---------------------------------------------------------------------------
# The conservative resolver
# ---------------------------------------------------------------------------


def _proposal(start: int, end: int, entity_type: str, score: float, expert: str,
              ordinal: int = 1):
    return ExpertSpanProposal(
        document_id="d1", start=start, end=end, text=TEXT[start:end],
        type_scores={entity_type: score}, local_score=score, expert_id=expert,
        proposal_id=f"{expert}-{ordinal}", original_start=start, original_end=end)


def test_the_resolver_preserves_every_contributing_provenance() -> None:
    resolved = resolve([
        _proposal(10, 17, "SYMPTOM", 0.9, EXPERT_GLINER, 1),
        _proposal(10, 17, "SYMPTOM", 0.8, EXPERT_MEDICATION_GRAMMAR, 2),
    ])
    assert len(resolved) == 1
    assert set(resolved[0].provenance) == {EXPERT_GLINER, EXPERT_MEDICATION_GRAMMAR}
    assert len(resolved[0].proposal_ids) == 2
    assert resolved[0].agreeing_experts == 2


def test_identical_text_at_different_positions_stays_two_mentions() -> None:
    """Text-only dedup would silently delete one of them (spec §5 case C7)."""
    text = "sốt và sốt"
    proposals = [
        ExpertSpanProposal(
            document_id="d", start=s, end=s + 3, text="sốt",
            type_scores={"SYMPTOM": 0.9}, local_score=0.9,
            expert_id=EXPERT_GLINER, proposal_id=f"p{s}",
            original_start=s, original_end=s + 3)
        for s in (0, 7)
    ]
    assert text[0:3] == text[7:10] == "sốt"
    resolved = resolve(proposals)
    assert len(resolved) == 2
    assert [(m.start, m.end) for m in resolved] == [(0, 3), (7, 10)]


def test_the_resolver_is_deterministic_and_sorted() -> None:
    proposals = [
        _proposal(21, 28, "SYMPTOM", 0.9, EXPERT_GLINER, 1),
        _proposal(10, 17, "SYMPTOM", 0.9, EXPERT_GLINER, 2),
    ]
    first = resolve(proposals)
    second = resolve(list(reversed(proposals)))
    assert [(m.start, m.end) for m in first] == [(10, 17), (21, 28)]
    assert [m.as_dict() for m in first] == [m.as_dict() for m in second]


def test_the_resolver_abstains_on_a_lone_low_confidence_proposal() -> None:
    resolved = resolve(
        [_proposal(10, 17, "SYMPTOM", 0.10, EXPERT_GLINER)],
        thresholds=ResolverThresholds(min_single_expert_score=0.5))
    assert resolved[0].abstained
    assert emitted(resolved) == ()


def test_resolver_thresholds_are_bounded_and_recorded() -> None:
    payload = ResolverThresholds().as_dict()
    assert payload["threshold_search_performed"] is False
    tracked = yaml.safe_load(
        (REPO / "configs" / "resolution" / "zs0_conservative_v1.yaml").read_text(
            encoding="utf-8"))
    for key in ("min_single_expert_score", "min_agreement_score",
                "max_wrong_type_risk", "near_complete_overlap_iou"):
        assert tracked[key] == payload[key]


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def test_assertion_output_is_constrained_to_the_three_labels() -> None:
    deterministic = AssertionDecision(labels=(), source="deterministic_no_cue")
    rejected = parse_qwen_assertions(
        json.dumps({"assertions": ["isFabricated"]}), deterministic=deterministic)
    assert rejected.labels == ()
    assert rejected.source == "qwen_rejected_unsupported_label"
    accepted = parse_qwen_assertions(
        json.dumps({"assertions": ["isNegated"]}), deterministic=deterministic)
    assert accepted.labels == ("isNegated",)
    assert set(accepted.labels) <= ASSERTION_LABELS


def test_insufficient_assertion_evidence_stays_empty_not_false() -> None:
    decision = decide_deterministically("Bệnh nhân sốt cao .", mention_start=10)
    assert decision.labels == ()
    assert decision.as_dict()[
        "empty_means_insufficient_evidence_not_false"] is True


def test_a_negation_cue_in_scope_is_detected() -> None:
    text = "không sốt cao"
    decision = decide_deterministically(text, mention_start=text.index("sốt"))
    assert "isNegated" in decision.labels


def test_an_out_of_scope_cue_is_uncertain_rather_than_asserted() -> None:
    text = "không " + ("x" * 200) + " sốt cao"
    decision = decide_deterministically(text, mention_start=text.index("sốt"))
    assert decision.labels == ()
    assert decision.uncertain is True


def test_malformed_qwen_assertions_fall_back_to_the_deterministic_decision() -> None:
    deterministic = AssertionDecision(
        labels=("isHistorical",), source="deterministic_cue_in_scope")
    fallback = parse_qwen_assertions("{oops", deterministic=deterministic)
    assert fallback.labels == ("isHistorical",)
    assert fallback.uncertain is True


# ---------------------------------------------------------------------------
# Linking: retrieved, never generated
# ---------------------------------------------------------------------------

SNAPSHOT = OntologySnapshot(
    ontology="ICD10", snapshot_id="tt06-2026",
    codes=frozenset({"J18.9", "R50.9", "I10"}),
    aliases={"viêm phổi": ("J18.9",)})


def test_qwen_cannot_introduce_a_code_outside_the_candidate_set() -> None:
    offered = (Candidate(code="J18.9", ontology="ICD10", score=0.8,
                         source="lexical", rank=1),)
    selected, rejected = select_from_candidates(
        json.dumps({"selected": ["I10"]}), offered, snapshot=SNAPSHOT)
    # I10 exists in the snapshot but was NOT retrieved for this mention.
    assert selected == ()
    assert rejected == ("I10",)


def test_every_emitted_code_exists_in_the_snapshot() -> None:
    assert_codes_in_snapshot(["J18.9"], snapshot=SNAPSHOT)
    with pytest.raises(LinkingError, match="never generates one"):
        assert_codes_in_snapshot(["Z99.9"], snapshot=SNAPSHOT)


def test_retrieval_drops_codes_absent_from_the_snapshot() -> None:
    candidates = generate_candidates(
        mention_text="sốt", entity_type="DIAGNOSIS", snapshot=SNAPSHOT,
        lexical=[("R50.9", 0.7), ("Z99.9", 0.9)])
    assert [c.code for c in candidates] == ["R50.9"]


def test_an_exact_alias_hit_wins_outright() -> None:
    result = link_mention(
        mention_text="Viêm Phổi", entity_type="DIAGNOSIS", snapshot=SNAPSHOT,
        lexical=[("R50.9", 0.99)])
    assert result.codes() == ("J18.9",)
    assert result.fallback == "exact_alias"


def test_empty_candidates_beat_invented_ones() -> None:
    result = link_mention(
        mention_text="không có gì", entity_type="DIAGNOSIS", snapshot=SNAPSHOT)
    assert result.codes() == ()
    assert result.fallback == "no_candidate_retrieved"
    assert result.as_dict()["codes_were_generated_by_a_model"] is False


def test_cross_ontology_linking_is_refused() -> None:
    with pytest.raises(LinkingError, match="cross-ontology"):
        generate_candidates(
            mention_text="paracetamol", entity_type="MEDICATION", snapshot=SNAPSHOT)


def test_a_type_without_candidates_returns_none() -> None:
    result = link_mention(
        mention_text="sốt", entity_type="SYMPTOM", snapshot=SNAPSHOT)
    assert result.ontology is None
    assert result.codes() == ()


# ---------------------------------------------------------------------------
# Parameter ledger
# ---------------------------------------------------------------------------


def _entry(component_id: str, **overrides) -> LedgerEntry:
    fields = {
        "component_id": component_id, "model_id": "m", "revision": "r",
        "checkpoint_sha256": "0" * 64, "parameter_count": 1_000_000,
        "loaded_during_inference": True, "count_method": "counted_from_config",
        "count_verified": True,
    }
    fields.update(overrides)
    return LedgerEntry(**fields)


def test_the_ledger_fails_closed_above_nine_billion() -> None:
    with pytest.raises(BudgetExceeded, match="exceeds"):
        build_ledger([_entry("big", parameter_count=MAX_ACTIVE_PARAMETERS + 1)])
    report = build_ledger([_entry("ok", parameter_count=MAX_ACTIVE_PARAMETERS)])
    assert report.within_budget
    assert report.remaining_margin == 0


def test_the_ledger_fails_closed_on_an_unverified_active_component() -> None:
    with pytest.raises(UnverifiedComponent, match="fails closed"):
        build_ledger([_entry("unknown", parameter_count=None,
                             count_method="unknown", count_verified=False)])


def test_an_unverified_inactive_component_does_not_block() -> None:
    report = build_ledger([
        _entry("active"),
        _entry("shelved", parameter_count=None, loaded_during_inference=False,
               count_method="unknown", count_verified=False)])
    assert report.within_budget
    assert report.active_total == 1_000_000


def test_a_lora_deployment_counts_the_base_plus_its_adapters() -> None:
    report = build_ledger([
        _entry("lora", parameter_count=1_700_000_000, adapter_parameters=9_000)])
    assert report.active_total == 1_700_009_000
    assert report.as_dict()["adapter_only_counting_used"] is False


def test_a_shared_backbone_is_counted_once() -> None:
    report = build_ledger([
        _entry("qwen_proposer", parameter_count=1_700_000_000,
               shared_instance_key="qwen"),
        _entry("qwen_assertions", parameter_count=1_700_000_000,
               adapter_parameters=5_000, shared_instance_key="qwen")])
    # 1.7B once, plus the second duty's adapter only.
    assert report.active_total == 1_700_005_000
    assert report.shared_bases_counted_once == ("qwen",)


def test_the_tracked_ledger_ships_unverified_so_the_gate_fails_closed() -> None:
    document = yaml.safe_load(LEDGER_CONFIG.read_text(encoding="utf-8"))
    assert document["max_active_parameters"] == MAX_ACTIVE_PARAMETERS
    neural = [c for c in document["components"]
              if c.get("loaded_during_inference") and c.get("model_id") != "deterministic"]
    assert neural, "the ledger should list the pretrained models"
    for component in neural:
        assert component.get("count_verified") is False
        assert component.get("parameter_count") is None


# ---------------------------------------------------------------------------
# Arm selection and the organizer gate
# ---------------------------------------------------------------------------


def _arm_result(arm: str, **overrides) -> ArmResult:
    fields = {
        "arm": arm, "exact_f1": 0.40, "exact_precision": 0.45,
        "exact_recall": 0.36, "wrong_type_count": 5,
        "malformed_proposal_count": 0, "offset_violations": 0,
        "runtime_seconds": 100.0, "peak_vram_gib": 6.0,
        "active_parameters": 1_000_000,
    }
    fields.update(overrides)
    return ArmResult(**fields)


def test_arm_selection_takes_the_highest_admissible_exact_f1() -> None:
    selected, report = select_arm([
        _arm_result(ZS0_A, exact_f1=0.30),
        _arm_result(ZS0_B, exact_f1=0.52),
        _arm_result(ZS0_C, exact_f1=0.44)])
    assert selected is not None and selected.arm == ZS0_B
    assert report["selected_arm"] == ZS0_B


def test_an_arm_with_an_offset_violation_is_never_selected() -> None:
    selected, report = select_arm([
        _arm_result(ZS0_A, exact_f1=0.30),
        _arm_result(ZS0_C, exact_f1=0.99, offset_violations=3)])
    assert selected is not None and selected.arm == ZS0_A
    assert report["excluded_for_offset_violations"] == [ZS0_C]


def test_a_negligible_f1_gap_does_not_buy_a_larger_model() -> None:
    """Within tolerance, fewer wrong types and fewer parameters win."""
    selected, _report = select_arm([
        _arm_result(ZS0_A, exact_f1=0.500, wrong_type_count=2,
                    active_parameters=0),
        _arm_result(ZS0_C, exact_f1=0.503, wrong_type_count=9,
                    active_parameters=2_000_000_000)])
    assert selected is not None and selected.arm == ZS0_A


def test_no_admissible_arm_means_no_submission() -> None:
    selected, report = select_arm([_arm_result(ZS0_A, offset_violations=1)])
    assert selected is None
    assert "no submission may be built" in report["reason"]


def _preconditions(**overrides) -> OrganizerPreconditions:
    fields = {
        "selected_arm": ZS0_B, "active_trained_experts": (),
        "training_executed": False,
        "ledger": build_ledger([_entry("gliner")]),
        "local_gates_passed": True,
        "discovered_documents": tuple(f"{i}.txt" for i in range(1, 101)),
        "external_api_configured": False, "seed": 1, "manifest_recorded": True,
    }
    fields.update(overrides)
    return OrganizerPreconditions(**fields)


def test_organizer_inference_requires_the_exact_authorization_string() -> None:
    with pytest.raises(SubmissionError, match="exact authorization string"):
        assert_organizer_inference_allowed(
            enabled=True, confirmation="yes please",
            preconditions=_preconditions())
    with pytest.raises(SubmissionError, match="disabled"):
        assert_organizer_inference_allowed(
            enabled=False, confirmation=ORGANIZER_AUTHORIZATION,
            preconditions=_preconditions())
    assert_organizer_inference_allowed(
        enabled=True, confirmation=ORGANIZER_AUTHORIZATION,
        preconditions=_preconditions())


@pytest.mark.parametrize("override,fragment", [
    ({"selected_arm": None}, "no ZS0 arm was selected"),
    ({"active_trained_experts": ("enable_e3_vihealthbert",)}, "trained experts"),
    ({"training_executed": True}, "training code was executed"),
    ({"local_gates_passed": False}, "local schema/offset/ontology gate"),
    ({"external_api_configured": True}, "external model API"),
    ({"manifest_recorded": False}, "no run manifest"),
    ({"discovered_documents": tuple(f"{i}.txt" for i in range(1, 100))},
     "expected 100"),
])
def test_each_unmet_precondition_refuses_organizer_inference(
    override: dict, fragment: str,
) -> None:
    with pytest.raises(SubmissionError, match="refused"):
        assert_organizer_inference_allowed(
            enabled=True, confirmation=ORGANIZER_AUTHORIZATION,
            preconditions=_preconditions(**override))
    assert any(fragment in problem
               for problem in _preconditions(**override).failures())


def test_a_duplicate_discovered_document_refuses_inference() -> None:
    duplicated = tuple(["1.txt"] + [f"{i}.txt" for i in range(1, 100)])
    failures = _preconditions(discovered_documents=duplicated).failures()
    assert any("more than once" in problem for problem in failures)


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_packaging_requires_exactly_one_hundred_root_level_json_files(
    tmp_path: Path,
) -> None:
    good = tmp_path / "output.zip"
    with zipfile.ZipFile(good, "w") as archive:
        for name in OUTPUT_FILENAMES:
            archive.writestr(name, "{}")
    assert validate_package(good) == ()

    short = tmp_path / "short.zip"
    with zipfile.ZipFile(short, "w") as archive:
        for name in OUTPUT_FILENAMES[:99]:
            archive.writestr(name, "{}")
    assert any("missing" in problem for problem in validate_package(short))

    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        for name in OUTPUT_FILENAMES:
            archive.writestr(f"output/{name}", "{}")
    assert any("root" in problem for problem in validate_package(nested))


def test_output_validation_enforces_the_offset_invariant() -> None:
    text = "Bệnh nhân sốt cao ."
    label = ORGANIZER_LABEL_BY_TYPE["SYMPTOM"]
    good = json.dumps({"entities": [
        {"type": label, "start_pos": 10, "end_pos": 17, "text": "sốt cao"}]})
    assert validate_output_payload("1.json", good, original_text=text) == ()
    bad = json.dumps({"entities": [
        {"type": label, "start_pos": 10, "end_pos": 17, "text": "ho khan"}]})
    problems = validate_output_payload("1.json", bad, original_text=text)
    assert any("does not equal original_text" in p for p in problems)


def test_output_validation_rejects_a_disallowed_type_and_duplicates() -> None:
    text = "Bệnh nhân sốt cao ."
    label = ORGANIZER_LABEL_BY_TYPE["SYMPTOM"]
    payload = json.dumps({"entities": [
        {"type": "SYMPTOM", "start_pos": 10, "end_pos": 17, "text": "sốt cao"},
        {"type": label, "start_pos": 10, "end_pos": 17, "text": "sốt cao"},
        {"type": label, "start_pos": 10, "end_pos": 17, "text": "sốt cao"},
    ]})
    problems = validate_output_payload("1.json", payload, original_text=text)
    assert any("is not allowed" in p for p in problems)
    assert any("duplicate entity record" in p for p in problems)


def test_output_validation_rejects_a_code_outside_the_snapshot() -> None:
    text = "viêm phổi"
    label = ORGANIZER_LABEL_BY_TYPE["DIAGNOSIS"]
    payload = json.dumps({"entities": [
        {"type": label, "start_pos": 0, "end_pos": 9, "text": "viêm phổi",
         "candidates": ["Z99.9"]}]})
    problems = validate_output_payload(
        "1.json", payload, original_text=text,
        allowed_codes={label: frozenset({"J18.9"})})
    assert any("not in the locked ontology snapshot" in p for p in problems)


# ---------------------------------------------------------------------------
# The notebook
# ---------------------------------------------------------------------------


def test_the_notebook_ships_organizer_inference_disabled() -> None:
    code = _notebook_code()
    assert "RUN_ORGANIZER_INFERENCE = False" in code
    assert 'CONFIRM_ORGANIZER_INFERENCE = ""' in code
    assert "assert_organizer_inference_allowed(" in code
    # The literal lives in one place: the profile module.
    assert ORGANIZER_AUTHORIZATION not in code


def test_the_notebook_has_all_eight_stages() -> None:
    code = _notebook_code()
    for stage in ("STAGE 0", "STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4",
                  "STAGE 5", "STAGE 6", "STAGE 7"):
        assert stage in code, stage


def test_the_notebook_never_touches_internal_test_and_asserts_e4_retired() -> None:
    code = _notebook_code()
    # internal_test appears only as the declaration that it was NOT accessed,
    # and in comments saying so; it is never a path or a split argument.
    assert "internal_test.jsonl" not in code
    assert 'split="internal_test"' not in code
    assert '"internal_test_accessed": False' in code
    assert "assert_e4_disabled(" in code
    assert "assert_no_e4_checkpoint_required(" in code
    assert "assert_e4_absent_from_ledger(" in code


def test_the_notebook_never_trains() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c.get("source", []))
                     for c in payload["cells"] if c.get("cell_type") == "code")
    for forbidden in (".backward(", "torch.optim", "AdamW", "optimizer.step("):
        assert forbidden not in code, forbidden


def test_every_notebook_code_cell_parses_once_magics_are_stripped() -> None:
    import ast
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "\n".join(
            "" if line.lstrip().startswith(("%", "!")) else line
            for line in "".join(cell.get("source", [])).splitlines())
        ast.parse(source, filename=f"zs0_cell_{index}")


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


def test_no_assistant_control_file_is_tracked() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    for path in tracked:
        assert Path(path).name not in {"CLAUDE.md", "AGENTS.md"}, path
        assert not path.startswith(".claude/"), path


def test_the_architecture_pdf_is_unchanged() -> None:
    digest = hashlib.sha256(
        (REPO / "docs" / "MedNorm-VI_Architecture.pdf").read_bytes()).hexdigest()
    assert digest == (
        "0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b")


# ---------------------------------------------------------------------------
# Stage 3 actually runs (Audit 0049)
# ---------------------------------------------------------------------------
#
# The Audit-0048 notebook shipped Stage 3 as a placeholder: it set
# ARM_RESULTS = [] and printed a note asking the reader to populate it. Stage 4's
# guard then fired correctly on an empty list. These tests pin the behaviour that
# was missing, and the invariants that keep a broken run from looking like a
# measured one.

from mednorm_vi.zs0.backends import (  # noqa: E402
    GLINER_PROMPT_LABELS,
    BackendError,
    GLiNERBackend,
    QwenBackend,
    build_mention_prompt,
    extract_json_object,
    parse_completion,
)
from mednorm_vi.zs0.runner import (  # noqa: E402
    ArmExecutionError,
    ArmSources,
    Document,
    assert_stage3_complete,
    run_all_arms,
    run_arm,
)

DOC_TEXT = "Bệnh nhân sốt cao và ho khan ."
GOLD = ((10, 17, "SYMPTOM"),)


def _document() -> Document:
    return Document(document_id="d1", text=DOC_TEXT, gold=GOLD)


def _det_hit(document: Document):
    return [ExpertSpanProposal(
        document_id=document.document_id, start=10, end=17,
        text=document.text[10:17], type_scores={"SYMPTOM": 0.9}, local_score=0.9,
        expert_id=EXPERT_MEDICATION_GRAMMAR, proposal_id="det-1",
        original_start=10, original_end=17)]


def _det_none(document: Document):
    return []


def _gliner_hit(text: str):
    return [{"start": 21, "end": 28, "label": "symptom", "score": 0.8}]


def _qwen_hit(text: str) -> str:
    return json.dumps({"mentions": [{"text": "ho khan", "type": "SYMPTOM"}]})


def _sources_for(arm_name: str) -> ArmSources:
    if arm_name == ZS0_A:
        return ArmSources(deterministic=_det_hit)
    if arm_name == ZS0_B:
        return ArmSources(deterministic=_det_hit, gliner=_gliner_hit)
    return ArmSources(deterministic=_det_hit, gliner=_gliner_hit, qwen=_qwen_hit)


def test_stage_three_produces_exactly_three_real_arm_results() -> None:
    arms = list(all_arms())
    results = run_all_arms(
        arms, [_document()], {a.name: _sources_for(a.name) for a in arms})
    assert len(results) == 3
    assert {r.arm for r in results} == {ZS0_A, ZS0_B, ZS0_C}
    assert_stage3_complete(results, arms)


def test_stage_four_receives_exactly_three_results() -> None:
    """The condition whose absence produced 'Stage 3 produced no arm results'."""
    arms = list(all_arms())
    results = run_all_arms(
        arms, [_document()], {a.name: _sources_for(a.name) for a in arms})
    selected, report = select_arm(results)
    assert len(report["results"]) == 3
    assert selected is not None


def test_an_arm_with_zero_predictions_still_returns_a_result() -> None:
    """Zero predictions is a measurement; a missing result is a bug."""
    result = run_arm(build_arm(ZS0_A), [_document()],
                     ArmSources(deterministic=_det_none))
    assert result.arm == ZS0_A
    assert result.exact_f1 == 0.0
    assert result.exact_precision == 0.0
    assert result.exact_recall == 0.0
    assert result.offset_violations == 0
    assert result.admissible


def test_every_arm_returns_a_result_even_when_all_of_them_predict_nothing() -> None:
    arms = list(all_arms())
    empty = {
        ZS0_A: ArmSources(deterministic=_det_none),
        ZS0_B: ArmSources(deterministic=_det_none, gliner=lambda t: []),
        ZS0_C: ArmSources(deterministic=_det_none, gliner=lambda t: [],
                          qwen=lambda t: json.dumps({"mentions": []})),
    }
    results = run_all_arms(arms, [_document()], empty)
    assert len(results) == 3
    assert all(r.exact_f1 == 0.0 for r in results)


def test_a_missing_backend_fails_loudly_and_names_the_arm() -> None:
    with pytest.raises(ArmExecutionError) as excinfo:
        run_arm(build_arm(ZS0_B), [_document()],
                ArmSources(deterministic=_det_hit))   # no GLiNER
    assert excinfo.value.arm == ZS0_B
    assert "GLiNER" in excinfo.value.prerequisite
    with pytest.raises(ArmExecutionError) as excinfo:
        run_arm(build_arm(ZS0_C), [_document()],
                ArmSources(deterministic=_det_hit, gliner=_gliner_hit))  # no Qwen
    assert excinfo.value.arm == ZS0_C
    assert "Qwen" in excinfo.value.prerequisite


def test_an_arm_exception_propagates_and_is_never_swallowed() -> None:
    def exploding(document):
        raise RuntimeError("model went missing")

    with pytest.raises(ArmExecutionError) as excinfo:
        run_arm(build_arm(ZS0_A), [_document()],
                ArmSources(deterministic=exploding))
    assert excinfo.value.arm == ZS0_A
    assert "deterministic E1/E2 path failed" in excinfo.value.prerequisite
    assert "model went missing" in excinfo.value.detail


def test_a_failing_arm_stops_the_whole_run_rather_than_shortening_it() -> None:
    """Continuing past a broken arm would leave Stage 4 a short list again."""
    arms = list(all_arms())
    sources = {a.name: _sources_for(a.name) for a in arms}
    sources[ZS0_B] = ArmSources(deterministic=_det_hit)   # missing GLiNER
    with pytest.raises(ArmExecutionError) as excinfo:
        run_all_arms(arms, [_document()], sources)
    assert excinfo.value.arm == ZS0_B


def test_run_all_arms_refuses_an_arm_with_no_sources() -> None:
    arms = list(all_arms())
    with pytest.raises(ArmExecutionError, match="no ArmSources"):
        run_all_arms(arms, [_document()], {ZS0_A: _sources_for(ZS0_A)})


def test_an_arm_refuses_to_run_on_no_documents() -> None:
    with pytest.raises(ArmExecutionError, match="no governed validation document"):
        run_arm(build_arm(ZS0_A), [], ArmSources(deterministic=_det_hit))


def test_stage3_postcondition_catches_a_short_or_wrong_result_set() -> None:
    arms = list(all_arms())
    partial = [run_arm(build_arm(ZS0_A), [_document()], _sources_for(ZS0_A))]
    with pytest.raises(ArmExecutionError, match="expected"):
        assert_stage3_complete(partial, arms)


def test_arm_metrics_are_real_and_reflect_the_predictions() -> None:
    perfect = run_arm(build_arm(ZS0_A), [_document()],
                      ArmSources(deterministic=_det_hit))
    assert perfect.exact_precision == 1.0
    assert perfect.exact_recall == 1.0
    assert perfect.exact_f1 == 1.0
    # Adding a second correct span raises recall on a two-gold document.
    two_gold = Document(document_id="d1", text=DOC_TEXT,
                        gold=((10, 17, "SYMPTOM"), (21, 28, "SYMPTOM")))
    half = run_arm(build_arm(ZS0_A), [two_gold],
                   ArmSources(deterministic=_det_hit))
    assert half.exact_recall == 0.5
    both = run_arm(build_arm(ZS0_B), [two_gold],
                   ArmSources(deterministic=_det_hit, gliner=_gliner_hit))
    assert both.exact_recall == 1.0


def test_a_right_span_with_the_wrong_type_is_counted_as_wrong_type() -> None:
    def wrong_type(document):
        return [ExpertSpanProposal(
            document_id=document.document_id, start=10, end=17,
            text=document.text[10:17], type_scores={"DIAGNOSIS": 0.9},
            local_score=0.9, expert_id=EXPERT_MEDICATION_GRAMMAR,
            proposal_id="det-1", original_start=10, original_end=17)]

    result = run_arm(build_arm(ZS0_A), [_document()],
                     ArmSources(deterministic=wrong_type))
    assert result.wrong_type_count == 1
    assert result.exact_f1 == 0.0


def test_malformed_model_output_is_counted_not_crashed_on() -> None:
    result = run_arm(
        build_arm(ZS0_C), [_document()],
        ArmSources(deterministic=_det_none, gliner=lambda t: [],
                   qwen=lambda t: "not json at all"))
    assert result.malformed_proposal_count >= 1
    assert result.exact_f1 == 0.0


# ---------------------------------------------------------------------------
# Stage 3 is independent of organizer inference
# ---------------------------------------------------------------------------


def test_stage_three_does_not_reference_organizer_inference() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])))
             for c in payload["cells"] if c.get("cell_type") == "code"]
    stage3 = [c for c in cells if "STAGE 3" in c]
    assert len(stage3) == 1
    source = stage3[0]
    assert "RUN_ORGANIZER_INFERENCE" not in source
    assert "CONFIRM_ORGANIZER_INFERENCE" not in source
    assert "assert_organizer_inference_allowed" not in source


def test_stage_three_is_no_longer_a_placeholder() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])))
             for c in payload["cells"] if c.get("cell_type") == "code"]
    source = next(c for c in cells if "STAGE 3" in c)
    # The exact placeholder that shipped in Audit 0048.
    assert "populate ARM_RESULTS with one ArmResult per arm" not in source
    assert "ARM_RESULTS = []" not in source
    # And what must be there instead.
    assert "run_all_arms(" in source
    assert "assert_stage3_complete(" in source
    assert "GLiNERBackend(" in source
    assert "QwenBackend(" in source
    assert "run_phase1b(" in source


def test_stage_three_has_no_try_except_around_an_arm() -> None:
    """A caught arm error would reproduce the short-list failure silently."""
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])))
             for c in payload["cells"] if c.get("cell_type") == "code"]
    source = next(c for c in cells if "STAGE 3" in c)
    assert "run_all_arms(" in source
    run_index = source.index("run_all_arms(")
    assert "try:" not in source[max(0, run_index - 400):run_index]


def test_arm_results_is_assigned_exactly_once_from_the_runner() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])))
             for c in payload["cells"] if c.get("cell_type") == "code"]
    assignments = [line.strip() for cell in cells for line in cell.splitlines()
                   if line.strip().startswith("ARM_RESULTS =")]
    assert assignments == ["ARM_RESULTS = run_all_arms("]


# ---------------------------------------------------------------------------
# The real backends
# ---------------------------------------------------------------------------


def test_the_backends_import_without_their_heavy_dependencies() -> None:
    """The module must import on a laptop with neither gliner nor a GPU."""
    assert GLiNERBackend().loaded is False
    assert QwenBackend().loaded is False


def test_an_unloaded_backend_refuses_to_predict_or_count() -> None:
    with pytest.raises(BackendError, match="not loaded"):
        GLiNERBackend().predict("text")
    with pytest.raises(BackendError, match="not loaded"):
        GLiNERBackend().parameter_count()
    with pytest.raises(BackendError, match="not loaded"):
        QwenBackend().generate("prompt")
    with pytest.raises(BackendError, match="not loaded"):
        QwenBackend().parameter_count()


def test_the_gliner_prompt_labels_map_into_the_five_btc_types() -> None:
    from mednorm_vi.zs0.proposals import map_gliner_label
    mapped = {map_gliner_label(label) for label in GLINER_PROMPT_LABELS}
    assert mapped <= set(ORGANIZER_LABEL_BY_TYPE)
    assert len(mapped) == 5


def test_the_qwen_prompt_forbids_invention_and_names_the_types() -> None:
    prompt = build_mention_prompt("Bệnh nhân sốt cao .")
    assert "verbatim" in prompt
    assert "never paraphrase" in prompt
    for entity_type in ORGANIZER_LABEL_BY_TYPE:
        assert entity_type in prompt
    assert "anchor" in prompt
    assert "Bệnh nhân sốt cao ." in prompt


def test_a_fenced_completion_is_recovered_without_repairing_its_content() -> None:
    fenced = '```json\n{"mentions": [{"text": "sốt", "type": "SYMPTOM"}]}\n```'
    recovered = parse_completion(fenced)
    assert json.loads(recovered)["mentions"][0]["text"] == "sốt"
    # A nested object still balances correctly.
    nested = 'Here: {"mentions": [{"text": "a", "type": "SYMPTOM"}]} done'
    assert json.loads(parse_completion(nested))["mentions"][0]["text"] == "a"
    # Genuinely malformed text is handed through, not repaired.
    assert json.loads(json.dumps(extract_json_object("no object here"))) == (
        "no object here")


def test_the_backends_never_train() -> None:
    from mednorm_vi.zs0 import backends as backends_module
    tokens = _executable_tokens(Path(backends_module.__file__))
    for forbidden in (" backward ( ", " AdamW ", " optim ", " zero_grad ( "):
        assert forbidden not in tokens, forbidden
    source = Path(backends_module.__file__).read_text(encoding="utf-8")
    assert "torch.no_grad()" in source
    assert ".eval()" in source
    # Greedy decoding only: a sampled model is not reproducible.
    assert "do_sample=False" in source


def test_no_forbidden_expert_is_reachable_from_the_runner() -> None:
    from mednorm_vi.zs0 import runner as runner_module
    source = Path(runner_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("vihealthbert", "phobert_w2ner", "xlmr_mrc",
                      "EXPERT_VIHEALTHBERT", "EXPERT_PHOBERT_W2NER",
                      "EXPERT_XLMR_MRC"):
        assert forbidden not in source, forbidden


def test_the_notebook_stage_three_activates_no_forbidden_expert() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])))
             for c in payload["cells"] if c.get("cell_type") == "code"]
    source = next(c for c in cells if "STAGE 3" in c)
    for forbidden in ("vihealthbert", "phobert_w2ner", "xlmr_mrc",
                      "backward", "AdamW", "torch.optim"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# Stage-3 imports and call signatures (Audit 0050)
# ---------------------------------------------------------------------------
#
# The first real Colab run died on `ImportError: cannot import name
# 'load_l1_config'`. That name is a LOCAL ALIAS internal callers give to
# `document_intelligence.load_config`; it is not a symbol anywhere in the tree.
# Three more mismatches were hiding behind it. These tests execute the real
# import block and check every call shape against the real implementations, so
# the next such bug is caught here rather than on a GPU.

import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402


def _stage3_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [("".join(c.get("source", [])))
             for c in payload["cells"] if c.get("cell_type") == "code"]
    return next(c for c in cells if "STAGE 3" in c)


def _stage3_import_block() -> str:
    """The contiguous import statements at the top of the Stage-3 cell."""
    collected: list[str] = []
    buffer: list[str] = []
    depth = 0
    for line in _stage3_source().splitlines():
        if line.strip().startswith(("from ", "import ")) or depth:
            buffer.append(line)
            depth += line.count("(") - line.count(")")
            if depth == 0:
                collected.append("\n".join(buffer))
                buffer = []
    return "\n".join(collected)


def test_the_stage_three_import_block_executes_against_the_real_package(
    tmp_path: Path,
) -> None:
    """Run the real imports in a clean subprocess, no mocks.

    A clean interpreter is the point: importing them inside this process would
    pass on names another test already imported.
    """
    block = _stage3_import_block()
    assert "from mednorm_vi" in block
    script = tmp_path / "stage3_imports.py"
    script.write_text(block + "\nprint('ok')\n", encoding="utf-8")
    completed = _subprocess.run(
        [_sys.executable, str(script)],
        cwd=str(REPO), env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)
    assert completed.returncode == 0, (
        f"Stage-3 import block failed:\n{completed.stderr}")


def test_stage_three_imports_the_real_public_names() -> None:
    """The exact names that were wrong, pinned so they cannot regress."""
    block = _stage3_import_block()
    # `load_l1_config` is an alias, not a symbol. It must never reappear.
    assert "load_l1_config" not in block
    assert "from mednorm_vi.document_intelligence import" in block
    assert "load_config" in block
    # Phase1BConfig lives in deterministic_baseline, not phase1c_foundation.
    assert "from mednorm_vi.phase1c_foundation import" not in block
    assert "from mednorm_vi.deterministic_baseline import" in block
    assert "Phase1BConfig" in block and "run_phase1b" in block


def test_every_stage_three_imported_name_exists_in_its_stated_module() -> None:
    """Parse the block and resolve each name against the real module."""
    import ast
    import importlib

    tree = ast.parse(_stage3_import_block())
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{node.module} does not export {alias.name}")
            checked += 1
    assert checked >= 15, f"only {checked} names checked; the block shrank"


def test_stage_three_call_signatures_match_the_real_implementations() -> None:
    import inspect

    from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b
    from mednorm_vi.document_intelligence import analyze_document, load_config

    # load_config(path) -> (L1Config, SectionLexicon); Stage 3 unpacks two.
    assert len(inspect.signature(load_config).parameters) == 1
    assert "tuple[L1Config, SectionLexicon]" in str(
        inspect.signature(load_config).return_annotation)
    # Phase1BConfig.load takes exactly three positional configs.
    assert list(inspect.signature(Phase1BConfig.load).parameters) == [
        "router_config", "medication_config", "laboratory_config"]
    # analyze_document takes config/lexicon keyword-only; Stage 3 uses keywords.
    parameters = inspect.signature(analyze_document).parameters
    assert parameters["config"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["lexicon"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "config=L1_CONFIG, lexicon=L1_LEXICON" in _stage3_source()
    # run_phase1b(graph, config)
    assert list(inspect.signature(run_phase1b).parameters) == ["graph", "config"]
    # run_all_arms' keyword-only budget argument, as Stage 3 passes it.
    assert inspect.signature(run_all_arms).parameters[
        "active_parameters_by_arm"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "active_parameters_by_arm=ACTIVE_BY_ARM" in _stage3_source()


def test_stage_three_reads_the_real_span_proposal_fields() -> None:
    """SpanProposal has proposed_types/local_score, not entity_type/score."""
    import dataclasses

    from mednorm_vi.mention_factory.models import SpanProposal

    fields = {f.name for f in dataclasses.fields(SpanProposal)}
    assert {"proposed_types", "local_score", "source_specialist"} <= fields
    assert "entity_type" not in fields
    assert "score" not in fields
    source = _stage3_source()
    assert "proposal.proposed_types" in source
    assert "proposal.local_score" in source
    assert "proposal.source_specialist" in source
    assert "proposal.entity_type" not in source
    assert "proposal.score" not in source


def test_stage_three_maps_organizer_labels_into_internal_enums() -> None:
    """E1/E2 emit "THUỐC"; ExpertSpanProposal requires "MEDICATION"."""
    from mednorm_vi.lattice.models import LatticeError
    from mednorm_vi.schemas.constants import (
        ORGANIZER_LABEL_BY_TYPE,
        TYPE_BY_ORGANIZER_LABEL,
    )

    organizer_label = ORGANIZER_LABEL_BY_TYPE["MEDICATION"]
    assert TYPE_BY_ORGANIZER_LABEL[organizer_label] == "MEDICATION"
    # The lattice genuinely refuses the organizer label, which is why the map
    # is required rather than cosmetic.
    with pytest.raises(LatticeError, match="unsupported proposed type"):
        ExpertSpanProposal(
            document_id="d", start=0, end=3, text="abc",
            type_scores={organizer_label: 1.0}, local_score=1.0,
            expert_id=EXPERT_MEDICATION_GRAMMAR, proposal_id="p",
            original_start=0, original_end=3)
    assert "TYPE_BY_ORGANIZER_LABEL" in _stage3_source()


def test_stage_three_picks_the_expert_from_the_specialist_not_the_type() -> None:
    """Inferring E1 vs E2 from the entity type would be a guess."""
    source = _stage3_source()
    assert 'proposal.source_specialist == "laboratory"' in source
    assert 'proposal.entity_type in ("TEST_NAME", "TEST_RESULT")' not in source


def test_the_deterministic_path_runs_end_to_end_on_a_real_document(
    tmp_path: Path,
) -> None:
    """E1 + E2 over a real document, converted exactly as Stage 3 converts it.

    Deterministic only: no pretrained model is downloaded or loaded.
    """
    from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b
    from mednorm_vi.document_intelligence import analyze_document, load_config
    from mednorm_vi.lattice.models import EXPERT_LABORATORY_PARSER
    from mednorm_vi.schemas.constants import ENTITY_TYPES, TYPE_BY_ORGANIZER_LABEL

    config, lexicon = load_config(str(REPO / "configs/document_intelligence/base.yaml"))
    phase1b = Phase1BConfig.load(
        str(REPO / "configs/case_router/base.yaml"),
        str(REPO / "configs/medication/grammar_v1.yaml"),
        str(REPO / "configs/laboratory/parser_v1.yaml"))
    text = "Thuốc: Paracetamol 500mg uống 2 viên/ngày ."
    source_path = tmp_path / "doc.txt"
    source_path.write_text(text, encoding="utf-8")

    graph = analyze_document(source_path, config=config, lexicon=lexicon)
    result = run_phase1b(graph, phase1b)

    converted = []
    for ordinal, proposal in enumerate(result.proposals, start=1):
        start, end = int(proposal.start), int(proposal.end)
        if text[start:end] != proposal.text:
            continue
        internal = tuple(
            t if t in ENTITY_TYPES else TYPE_BY_ORGANIZER_LABEL.get(t, "")
            for t in proposal.proposed_types)
        internal = tuple(t for t in internal if t)
        if not internal:
            continue
        score = float(proposal.local_score)
        converted.append(ExpertSpanProposal(
            document_id="d1", start=start, end=end, text=proposal.text,
            type_scores={t: score / len(internal) for t in internal},
            local_score=score,
            expert_id=(EXPERT_LABORATORY_PARSER
                       if proposal.source_specialist == "laboratory"
                       else EXPERT_MEDICATION_GRAMMAR),
            proposal_id=f"zs0-det-d1-{ordinal:04d}",
            original_start=start, original_end=end))

    assert converted, "E1/E2 produced no convertible proposal on a medication line"
    for proposal in converted:
        assert text[proposal.start:proposal.end] == proposal.text
        assert set(proposal.type_scores) <= ENTITY_TYPES
    # And the resolver consumes them without raising.
    assert len(resolve(converted)) >= 1


def test_no_pretrained_model_is_loaded_by_the_local_tests() -> None:
    """Lazy backend loading is what keeps these tests laptop-safe."""
    assert GLiNERBackend().loaded is False
    assert QwenBackend().loaded is False
    source = Path(GLiNERBackend.__module__.replace(".", "/") + ".py")
    del source  # the import-time behaviour is what matters, asserted above
