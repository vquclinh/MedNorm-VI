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
    from mednorm_vi.lattice.models import ExpertSpanProposal
    return ExpertSpanProposal(
        document_id="d1", start=start, end=end, text=TEXT[start:end],
        type_scores={entity_type: score}, local_score=score, expert_id=expert,
        proposal_id=f"{expert}-{ordinal}", original_start=start, original_end=end)


def test_the_resolver_preserves_every_contributing_provenance() -> None:
    from mednorm_vi.lattice.models import EXPERT_GLINER, EXPERT_MEDICATION_GRAMMAR
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
    from mednorm_vi.lattice.models import EXPERT_GLINER
    text = "sốt và sốt"
    from mednorm_vi.lattice.models import ExpertSpanProposal
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
    from mednorm_vi.lattice.models import EXPERT_GLINER
    proposals = [
        _proposal(21, 28, "SYMPTOM", 0.9, EXPERT_GLINER, 1),
        _proposal(10, 17, "SYMPTOM", 0.9, EXPERT_GLINER, 2),
    ]
    first = resolve(proposals)
    second = resolve(list(reversed(proposals)))
    assert [(m.start, m.end) for m in first] == [(10, 17), (21, 28)]
    assert [m.as_dict() for m in first] == [m.as_dict() for m in second]


def test_the_resolver_abstains_on_a_lone_low_confidence_proposal() -> None:
    from mednorm_vi.lattice.models import EXPERT_GLINER
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
