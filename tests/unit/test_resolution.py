"""Deterministic L4 resolver foundation (Phase 1C-A).

Covers TEST_NAME/TEST_RESULT independence, repeated-occurrence preservation, the
offset invariant, boundary selection, overlap resolution, has_result retention,
and the guarantee that no ontology candidate is emitted.
"""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b
from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.models import RelationProposal, SpanProposal
from mednorm_vi.resolution import ResolverConfig, resolve, validate_result

REPO = Path(__file__).resolve().parents[2]
RCFG = ResolverConfig.load(REPO / "configs" / "resolution" / "resolver_v1.yaml")


def _sp(pid: str, start: int, end: int, text: str, otype: str, *, specialist: str,
        rule: str, bgid: str | None = None, parse_ref: str = "p1") -> SpanProposal:
    return SpanProposal(
        proposal_id=pid, document_id="doc", start=start, end=end, text=text,
        proposed_types=(otype,), source_specialist=specialist, source_node_id="n1",
        source_routes=("C2",), local_score=0.8, matched_rule=rule, parse_ref=parse_ref,
        boundary_group_id=bgid)


# --- TEST_NAME / TEST_RESULT independence ---

def test_unpaired_test_name_retained() -> None:
    text = "WBC value here"
    props = [_sp("n", 0, 3, "WBC", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
                 rule="lab:test_name:key_value")]
    r = resolve("doc", text, props, [], RCFG)
    assert validate_result(r, text).ok
    accepted = [h.text for h in r.accepted()]
    assert "WBC" in accepted  # a name with no relation survives


def test_unpaired_test_result_retained() -> None:
    text = "the value 7.8 stands alone"
    props = [_sp("v", 10, 13, "7.8", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
                 rule="lab:test_result:value_only:key_value", bgid="rg1")]
    r = resolve("doc", text, props, [], RCFG)
    assert validate_result(r, text).ok
    assert "7.8" in [h.text for h in r.accepted()]


def test_relation_is_optional_evidence() -> None:
    # A name + a result, WITH a relation: both accepted and the relation is kept.
    text = "WBC 5.0"
    props = [
        _sp("n", 0, 3, "WBC", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_name:key_value"),
        _sp("v", 4, 7, "5.0", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rg1"),
    ]
    rel = RelationProposal(
        relation_id="r1", document_id="doc", relation_type="has_result",
        source_proposal_id="n", target_proposal_id="v", score=0.9, pairing_cost=0.1,
        pair_group_id="pg1", is_primary=True, target_boundary_group_id="rg1")
    r = resolve("doc", text, props, [rel], RCFG)
    by_type = {h.entity_type: h for h in r.accepted()}
    assert "TÊN_XÉT_NGHIỆM" in by_type and "KẾT_QUẢ_XÉT_NGHIỆM" in by_type
    assert by_type["TÊN_XÉT_NGHIỆM"].has_result_pair_group_ids == ("pg1",)


def test_result_kinds_numeric_qualitative_descriptive() -> None:
    text = "A pos B trace-amount"
    props = [
        _sp("q", 2, 5, "pos", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rgq"),
        _sp("d", 8, 20, "trace-amount", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rgd"),
    ]
    r = resolve("doc", text, props, [], RCFG)
    assert {h.text for h in r.accepted()} == {"pos", "trace-amount"}


# --- repeated occurrences are distinct concepts ---

def test_repeated_occurrence_retained() -> None:
    text = "WBC and WBC again"
    props = [
        _sp("n1", 0, 3, "WBC", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_name:key_value", parse_ref="p1"),
        _sp("n2", 8, 11, "WBC", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_name:key_value", parse_ref="p2"),
    ]
    r = resolve("doc", text, props, [], RCFG)
    accepted = r.accepted()
    assert len(accepted) == 2  # not deduplicated by text
    assert {h.position for h in accepted} == {(0, 3), (8, 11)}


# --- boundary selection ---

def test_medication_full_boundary_with_retained_alternatives() -> None:
    text = "amlodipine 10 mg po daily"
    kinds = [("m0", 0, 10, "amlodipine", "name_only"),
             ("m1", 0, 16, "amlodipine 10 mg", "name_strength"),
             ("m2", 0, 19, "amlodipine 10 mg po", "name_strength_route"),
             ("m3", 0, 25, "amlodipine 10 mg po daily", "full")]
    props = [_sp(pid, s, e, t, "THUỐC", specialist="medication",
                 rule=f"med_grammar:{k}", bgid="mp1") for pid, s, e, t, k in kinds]
    r = resolve("doc", text, props, [], RCFG)
    accepted = r.accepted()
    assert len(accepted) == 1
    h = accepted[0]
    assert h.text == "amlodipine 10 mg po daily"  # confirmed golden full span
    assert h.boundary_evidence.chosen_kind == "full"
    assert len(h.retained_alternatives) == 3


def test_name_only_policy_selects_narrow() -> None:
    text = "amlodipine 10 mg"
    props = [
        _sp("m0", 0, 10, "amlodipine", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="mp1"),
        _sp("m1", 0, 16, "amlodipine 10 mg", "THUỐC", specialist="medication",
            rule="med_grammar:name_strength", bgid="mp1"),
    ]
    cfg = ResolverConfig(medication_boundary="name_only")
    r = resolve("doc", text, props, [], cfg)
    assert r.accepted()[0].text == "amlodipine"


# --- overlap resolution ---

def test_same_type_overlap_suppressed_deterministically() -> None:
    text = "amlodipine besylate"
    props = [
        _sp("a", 0, 10, "amlodipine", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="ga", parse_ref="pa"),
        _sp("b", 0, 19, "amlodipine besylate", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="gb", parse_ref="pb"),
    ]
    r = resolve("doc", text, props, [], RCFG)
    assert len(r.accepted()) == 1 and len(r.rejected()) == 1
    assert r.rejected()[0].rejection_reason is not None


def test_cross_type_overlap_both_kept() -> None:
    # TEST_NAME overlapping TEST_RESULT are different types -> both survive.
    text = "glucose7.8"
    props = [
        _sp("n", 0, 7, "glucose", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_name:key_value"),
        _sp("v", 0, 10, "glucose7.8", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rg"),
    ]
    r = resolve("doc", text, props, [], RCFG)
    assert len(r.accepted()) == 2


# --- invariants ---

def test_offset_invariant_and_no_ontology_candidate() -> None:
    cfg = Phase1BConfig.load(
        REPO / "configs" / "case_router" / "base.yaml",
        REPO / "configs" / "medication" / "grammar_v1.yaml",
        REPO / "configs" / "laboratory" / "parser_v1.yaml")
    g = analyze_text("Thuốc:\n1. amlodipine 10 mg po daily\nWBC: 14.43; Glucose: 7.8 mmol/L\n")
    phase1b = run_phase1b(g, cfg)
    r = resolve(g.document_id, g.original_text, list(phase1b.proposals),
                list(phase1b.relations), RCFG)
    assert validate_result(r, g.original_text).ok
    for h in r.hypotheses:
        assert g.original_text[h.start:h.end] == h.text  # offset invariant
        assert not any(k.lower() in {"rxcui", "icd10", "candidate", "code"} for k in h.features)


def test_resolution_deterministic() -> None:
    text = "amlodipine 10 mg"
    props = [
        _sp("m0", 0, 10, "amlodipine", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="mp1"),
        _sp("m1", 0, 16, "amlodipine 10 mg", "THUỐC", specialist="medication",
            rule="med_grammar:name_strength", bgid="mp1"),
    ]
    a = resolve("doc", text, props, [], RCFG)
    b = resolve("doc", text, props, [], RCFG)
    assert [(h.hypothesis_id, h.position, h.status) for h in a.hypotheses] == \
           [(h.hypothesis_id, h.position, h.status) for h in b.hypotheses]
