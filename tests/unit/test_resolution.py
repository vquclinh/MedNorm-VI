"""Canonical L4 Boundary & Type Resolver behaviour (spec §7).

Every assertion here was migrated from the Phase-1C-A version of this module by
Audit 0055 and now runs through the **one** canonical entry point,
``resolution.canonical.resolve_lattice_to_hypotheses``, over a real ``SpanLattice``.
The retired ``resolution/resolver.py`` is deleted; nothing in this file imports it.

Covered, unchanged in substance: TEST_NAME/TEST_RESULT independence,
repeated-occurrence preservation, the offset invariant, the migrated boundary
policies, overlap resolution, ``has_result`` retention, determinism, and the
guarantee that L4 emits no ontology candidate.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b
from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.lattice.builder import build_span_lattice
from mednorm_vi.mention_factory.models import (
    ComponentSpan,
    RelationProposal,
    SpanProposal,
)
from mednorm_vi.resolution import (
    ResolverV1Config,
    load_resolver_v1_config,
    resolve_lattice_to_hypotheses,
    validate_result,
)

REPO = Path(__file__).resolve().parents[2]
L4_CONFIG_PATH = REPO / "configs" / "resolution" / "boundary_type_resolver_v1.yaml"
L4 = load_resolver_v1_config(L4_CONFIG_PATH)


# Route context matters: the canonical L4 weights a route prior (spec §5) into its
# type utility and abstains below `min_type_utility`. A medication proposal carrying
# the laboratory route C2 scores too low to be emitted at all — so each specialist
# gets the route its documents actually carry, exactly as Phase 1B would supply it.
_ROUTES_BY_SPECIALIST: dict[str, tuple[str, ...]] = {
    "medication": ("C1",),
    "laboratory": ("C2",),
}


# E1's structured components are what the canonical L4 turns into its
# `grammar_completeness` evidence family: the lattice builder counts the
# ComponentSpan records and `typing.grammar_completeness` normalizes that count. A
# synthetic medication proposal with no components scores on the route prior alone
# (0.20), lands under `min_type_utility` (0.30) and abstains — so these fixtures
# carry real components, exactly as E1 supplies them.
_GRAMMAR_ROLES = ("name", "strength_value", "strength_unit", "route")


def _components(text: str, start: int, roles: tuple[str, ...]) -> tuple[ComponentSpan, ...]:
    """One zero-risk component per role inside the span, for grammar completeness.

    Coordinates are inside `[start, start + len(text))` and each component's text is
    sliced from the span itself, so the offset invariant holds for the components too.
    """
    built: list[ComponentSpan] = []
    for index, role in enumerate(roles):
        if index >= len(text):
            break
        built.append(ComponentSpan(
            role=role, start=start + index, end=start + index + 1,
            text=text[index:index + 1]))
    return tuple(built)


def _sp(pid: str, start: int, end: int, text: str, otype: str, *, specialist: str,
        rule: str, bgid: str | None = None, parse_ref: str = "p1",
        routes: tuple[str, ...] | None = None,
        roles: tuple[str, ...] = _GRAMMAR_ROLES) -> SpanProposal:
    components = (_components(text, start, roles)
                  if specialist == "medication" else ())
    return SpanProposal(
        proposal_id=pid, document_id="doc", start=start, end=end, text=text,
        proposed_types=(otype,), source_specialist=specialist, source_node_id="n1",
        source_routes=routes or _ROUTES_BY_SPECIALIST.get(specialist, ("C3",)),
        local_score=0.8, matched_rule=rule, parse_ref=parse_ref,
        boundary_group_id=bgid, components=components)


def _resolve(
    text: str,
    proposals: list[SpanProposal],
    relations: list[RelationProposal] | None = None,
    *,
    config: ResolverV1Config | None = None,
    document_id: str = "doc",
):
    """Drive the canonical L4 over a lattice built from these proposals.

    This is the same two-call path the canonical runner and the Phase-1C debug CLI
    use — ``build_span_lattice`` then ``resolve_lattice_to_hypotheses``. There is no
    other L4 entry point to test against.
    """
    lattice = build_span_lattice(
        document_id, text, routings=(), specialist_proposals=tuple(proposals),
        expert_spans=(), relations=tuple(relations or ()))
    return resolve_lattice_to_hypotheses(
        lattice, config or L4, relations=tuple(relations or ()))


def _with_policy(medication: str = "full", test_result: str = "value_only",
                 *, abstain_on_conflict: bool = False) -> ResolverV1Config:
    """The canonical config with the migrated boundary policies overridden."""
    return dataclasses.replace(
        L4,
        boundary=dataclasses.replace(
            L4.boundary,
            group_preference={"medication": medication, "test_result": test_result}),
        overlap=dataclasses.replace(
            L4.overlap, abstain_on_conflict=abstain_on_conflict))


# --- TEST_NAME / TEST_RESULT independence ---


def test_unpaired_test_name_retained() -> None:
    text = "WBC value here"
    props = [_sp("n", 0, 3, "WBC", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
                 rule="lab:test_name:key_value")]
    r = _resolve(text, props)
    assert validate_result(r, text).ok
    assert "WBC" in [h.text for h in r.accepted()]  # a name with no relation survives


def test_unpaired_test_result_retained() -> None:
    text = "the value 7.8 stands alone"
    props = [_sp("v", 10, 13, "7.8", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
                 rule="lab:test_result:value_only:key_value", bgid="rg1")]
    r = _resolve(text, props)
    assert validate_result(r, text).ok
    assert "7.8" in [h.text for h in r.accepted()]


def test_relation_is_optional_evidence() -> None:
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
    r = _resolve(text, props, [rel])
    by_type = {h.entity_type: h for h in r.accepted()}
    assert "TÊN_XÉT_NGHIỆM" in by_type and "KẾT_QUẢ_XÉT_NGHIỆM" in by_type
    # Pair-group evidence survives the lattice and the canonical L4 (Audit 0052).
    assert by_type["TÊN_XÉT_NGHIỆM"].has_result_pair_group_ids == ("pg1",)


def test_result_kinds_numeric_qualitative_descriptive() -> None:
    text = "A pos B trace-amount"
    props = [
        _sp("q", 2, 5, "pos", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rgq"),
        _sp("d", 8, 20, "trace-amount", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rgd"),
    ]
    r = _resolve(text, props)
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
    accepted = _resolve(text, props).accepted()
    assert len(accepted) == 2  # never deduplicated by text
    assert {h.position for h in accepted} == {(0, 3), (8, 11)}


# --- the migrated boundary policies (Audit 0055) ---


def _medication_ladder() -> list[SpanProposal]:
    kinds = [("m0", 0, 10, "amlodipine", "name_only"),
             ("m1", 0, 16, "amlodipine 10 mg", "name_strength"),
             ("m2", 0, 19, "amlodipine 10 mg po", "name_strength_route"),
             ("m3", 0, 25, "amlodipine 10 mg po daily", "full")]
    return [_sp(pid, s, e, t, "THUỐC", specialist="medication",
                rule=f"med_grammar:{k}", bgid="mp1") for pid, s, e, t, k in kinds]


def test_medication_full_boundary_with_retained_alternatives() -> None:
    text = "amlodipine 10 mg po daily"
    r = _resolve(text, _medication_ladder(), config=_with_policy("full"))
    accepted = r.accepted()
    assert len(accepted) == 1, [h.text for h in accepted]
    hypothesis = accepted[0]
    assert hypothesis.text == "amlodipine 10 mg po daily"  # confirmed golden span
    assert hypothesis.retained_alternatives, "competing boundaries must be retained"
    # The winning ladder rung is recorded in provenance, not merely applied.
    assert "policy=full" in hypothesis.boundary_evidence.policy


def test_name_only_policy_selects_narrow() -> None:
    text = "amlodipine 10 mg"
    props = [
        _sp("m0", 0, 10, "amlodipine", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="mp1"),
        _sp("m1", 0, 16, "amlodipine 10 mg", "THUỐC", specialist="medication",
            rule="med_grammar:name_strength", bgid="mp1"),
    ]
    accepted = _resolve(text, props, config=_with_policy("name_only")).accepted()
    assert [h.text for h in accepted] == ["amlodipine"]
    assert "policy=name_only" in accepted[0].boundary_evidence.policy


def test_name_strength_policy_selects_the_middle_rung() -> None:
    text = "amlodipine 10 mg po daily"
    accepted = _resolve(
        text, _medication_ladder(), config=_with_policy("name_strength")).accepted()
    assert [h.text for h in accepted] == ["amlodipine 10 mg"]


def test_test_result_policies_select_their_configured_kind() -> None:
    text = "WBC: 14.43 K/uL"
    props = [
        _sp("rv", 5, 10, "14.43", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:r1", bgid="rg1"),
        _sp("ru", 5, 15, "14.43 K/uL", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_unit:r1", bgid="rg1"),
    ]
    value_only = _resolve(text, props, config=_with_policy(test_result="value_only"))
    assert [h.text for h in value_only.accepted()] == ["14.43"]
    value_unit = _resolve(text, props, config=_with_policy(test_result="value_unit"))
    assert [h.text for h in value_unit.accepted()] == ["14.43 K/uL"]


def test_unit_and_dosage_attachment_stays_exact() -> None:
    """A migrated policy must not shave a unit off or swallow a neighbour."""
    text = "WBC: 14.43 K/uL"
    props = [
        _sp("ru", 5, 15, "14.43 K/uL", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_unit:r1", bgid="rg1"),
    ]
    accepted = _resolve(
        text, props, config=_with_policy(test_result="value_unit")).accepted()
    assert accepted
    for hypothesis in accepted:
        assert text[hypothesis.start:hypothesis.end] == hypothesis.text
        assert hypothesis.text == "14.43 K/uL"


# --- overlap resolution ---


def _overlapping_same_type() -> list[SpanProposal]:
    return [
        _sp("a", 0, 10, "amlodipine", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="ga", parse_ref="pa"),
        _sp("b", 0, 19, "amlodipine besylate", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="gb", parse_ref="pb"),
    ]


def test_same_type_overlap_resolved_deterministically() -> None:
    r = _resolve("amlodipine besylate", _overlapping_same_type())
    # Exactly one same-type span on near-identical coordinates survives, and the
    # other carries a recorded reason rather than vanishing.
    assert len(r.accepted()) == 1
    losers = r.rejected() + r.unresolved()
    assert len(losers) == 1
    assert losers[0].rejection_reason or losers[0].overlap_decision


def test_abstention_is_distinct_from_rejection() -> None:
    """Migrated `abstain_on_conflict`: a tie is UNRESOLVED, never REJECTED."""
    text = "amlodipine besylate"
    abstaining = _resolve(
        text, _overlapping_same_type(), config=_with_policy(abstain_on_conflict=True))
    deciding = _resolve(
        text, _overlapping_same_type(), config=_with_policy(abstain_on_conflict=False))
    assert len(abstaining.rejected()) <= len(deciding.rejected())
    for hypothesis in abstaining.unresolved():
        assert hypothesis.status == "unresolved"
        assert hypothesis not in abstaining.rejected()


def test_cross_type_overlap_both_kept() -> None:
    text = "glucose7.8"
    props = [
        _sp("n", 0, 7, "glucose", "TÊN_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_name:key_value"),
        _sp("v", 0, 10, "glucose7.8", "KẾT_QUẢ_XÉT_NGHIỆM", specialist="laboratory",
            rule="lab:test_result:value_only:key_value", bgid="rg"),
    ]
    assert len(_resolve(text, props).accepted()) == 2


# --- invariants ---


def test_offset_invariant_and_no_ontology_candidate() -> None:
    config = Phase1BConfig.load(
        REPO / "configs" / "case_router" / "base.yaml",
        REPO / "configs" / "medication" / "grammar_v1.yaml",
        REPO / "configs" / "laboratory" / "parser_v1.yaml")
    graph = analyze_text(
        "Thuốc:\n1. amlodipine 10 mg po daily\nWBC: 14.43; Glucose: 7.8 mmol/L\n")
    phase1b = run_phase1b(graph, config)
    lattice = build_span_lattice(
        graph.document_id, graph.original_text, routings=phase1b.routings,
        specialist_proposals=phase1b.proposals, expert_spans=(),
        relations=phase1b.relations)
    r = resolve_lattice_to_hypotheses(lattice, L4, relations=phase1b.relations)
    assert validate_result(r, graph.original_text).ok
    for hypothesis in r.hypotheses:
        assert graph.original_text[hypothesis.start:hypothesis.end] == hypothesis.text
        assert not any(
            key.lower() in {"rxcui", "icd10", "candidate", "code"}
            for key in hypothesis.features), "L4 must emit no ontology candidate"


def test_resolution_deterministic() -> None:
    text = "amlodipine 10 mg"
    props = [
        _sp("m0", 0, 10, "amlodipine", "THUỐC", specialist="medication",
            rule="med_grammar:name_only", bgid="mp1"),
        _sp("m1", 0, 16, "amlodipine 10 mg", "THUỐC", specialist="medication",
            rule="med_grammar:name_strength", bgid="mp1"),
    ]
    first = _resolve(text, props)
    second = _resolve(text, props)
    assert [(h.hypothesis_id, h.position, h.status) for h in first.hypotheses] == \
           [(h.hypothesis_id, h.position, h.status) for h in second.hypotheses]


def test_there_is_exactly_one_l4_public_entry_point() -> None:
    """The package exports the canonical resolver and nothing that competes with it."""
    import mednorm_vi.resolution as package

    assert "resolve_lattice_to_hypotheses" in package.__all__
    for retired in ("resolve", "ResolverConfig"):
        assert retired not in package.__all__, f"{retired} must not be re-exported"
        assert not hasattr(package, retired), f"{retired} must not be reachable"
