"""Audit 0054: structured RxNorm linking, ICD hierarchy, L6 edges + consistency,
L7 locked-option escalation, the run manifest, and the code-linking contract.

Section A uses **miniature synthetic graph fixtures** for expected decisions, and the
live locked snapshots only for schema, membership and determinism. That split is
deliberate: a decision test pinned to a 82,429-concept snapshot would break whenever the
snapshot is refreshed, and a schema test against a toy graph would prove nothing about
the real artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.confidence_cascade.cascade import CascadeDecision
from mednorm_vi.confidence_cascade.escalation import (
    ACCEPT,
    COND_ASSERTION_UNCERTAIN,
    COND_GRAPH_CONTRADICTION,
    REFUSE_INVENTED_VALUE,
    REFUSE_UNKNOWN_OPTION,
    REFUSE_WRONG_SUBJECT,
    REJECT,
    TIER_CRITIC,
    UNRESOLVED_DISPOSITION,
    EscalationDecision,
    build_escalation_request,
    build_locked_options,
    deterministic_fallback,
    resolve_escalation,
    run_cascade_escalation,
    validate_escalation_decision,
)
from mednorm_vi.evaluation.code_linking import (
    ADJUDICATED_MULTIPLE,
    ADJUDICATED_NONE,
    ADJUDICATED_SINGLE,
    DISPUTED,
    ERR_BROADER,
    ERR_EXACT,
    ERR_SPURIOUS,
    ERR_STALE_CODE,
    ERR_WRONG_ONTOLOGY,
    PENDING,
    REQUIRED_ANNOTATION_ARTIFACT,
    UNMEASURABLE,
    CodeLinkingGoldRecord,
    CodeLinkingPredictionRecord,
    GoldRecordInvalid,
    GoldSetUnavailable,
    candidate_jaccard,
    candidate_quality_status,
    categorize_errors,
    evaluate_code_linking,
    load_gold_records,
    recall_at_k,
)
from mednorm_vi.evidence_graph.consistency import (
    CONTRADICTED,
    NOT_APPLICABLE,
    RULE_ASSERTION_TYPE,
    RULE_DUPLICATE_CANDIDATE,
    RULE_OVERLAP,
    RULE_UNSAFE_TREATS,
    SUPPORTED,
    UNRESOLVED,
    evaluate_consistency,
    ontology_for,
)
from mednorm_vi.evidence_graph.graph import (
    REL_HAS_RESULT,
    REL_IN_SECTION,
    REL_MODIFIED_BY,
    REL_OVERLAPS,
    REL_SAME_SURFACE,
    REL_TREATS,
    build_evidence_graph,
)
from mednorm_vi.inference.config import PipelineConfig
from mednorm_vi.inference.manifest import (
    RunManifest,
    build_run_manifest,
    role_safe_path,
    write_run_manifest,
)
from mednorm_vi.inference.packaging import ZIP_FIXED_TIMESTAMP
from mednorm_vi.inference.pipeline import run_document
from mednorm_vi.kb.indexing.retrieval import LocalIndex, load_index
from mednorm_vi.linking.icd10 import (
    DROP_NO_LEXICAL_SUPPORT,
    DROP_UNSUPPORTED_SPECIFICITY,
    KEEP_BROADER_FALLBACK,
    KEEP_EXACT_NAME,
    TIER_EXACT_NAME,
    link_icd10,
    link_icd10_hierarchical,
)
from mednorm_vi.linking.icd10_hierarchy import (
    REL_ANCESTOR,
    ancestor_path,
    assess_specificity,
    children,
    declared_specificity,
    graph_depth,
    hierarchy_health,
    parents,
    siblings,
)
from mednorm_vi.linking.models import LinkedCandidate, LinkerResult
from mednorm_vi.linking.rxnorm import (
    CONFLICT,
    DROP_DOSE_FORM_CONFLICT,
    DROP_NON_TERMINAL_TTY,
    DROP_RELEASE_CONFLICT,
    DROP_STRENGTH_CONFLICT,
    NOT_STATED,
    TIER_INGREDIENT_ONLY,
    compare_candidate,
    link_rxnorm,
    link_rxnorm_structured,
)
from mednorm_vi.linking.rxnorm_graph import (
    TTY_INGREDIENT,
    graph_health,
    traverse_ingredient_chain,
    tty_of,
)
from mednorm_vi.linking.structured_medication import (
    INN_TO_RXNORM_NAME,
    bridged_ingredient_names,
    build_structured_mention,
    canonical_unit,
    strength_in_mg,
    unit_family,
)
from mednorm_vi.resolution.models import (
    STATUS_ACCEPTED,
    BoundaryAlternative,
    BoundaryEvidence,
    EntityHypothesis,
    TypeEvidence,
)
from mednorm_vi.specialists.assertion import AssertionDecision

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "configs" / "pipeline" / "full_v1.yaml"
FIX = REPO / "tests" / "fixtures" / "phase1b"


def _config() -> PipelineConfig:
    return PipelineConfig.load(PROFILE)


def _hypothesis(
    text: str,
    entity_type: str = "THUỐC",
    *,
    hypothesis_id: str = "h1",
    start: int = 0,
    components: tuple[dict[str, object], ...] = (),
    alternatives: tuple[BoundaryAlternative, ...] = (),
    proposed_types: tuple[str, ...] = (),
    pair_groups: tuple[str, ...] = (),
    score: float = 0.9,
) -> EntityHypothesis:
    return EntityHypothesis(
        hypothesis_id=hypothesis_id,
        document_id="d1",
        start=start,
        end=start + len(text),
        text=text,
        entity_type=entity_type,
        status=STATUS_ACCEPTED,
        chosen_proposal_id="p1",
        source_proposal_ids=("p1",),
        boundary_evidence=BoundaryEvidence("policy", "full", "p1", ()),
        type_evidence=TypeEvidence(entity_type, "E1", proposed_types or (entity_type,)),
        retained_alternatives=alternatives,
        components=components,
        has_result_pair_group_ids=pair_groups,
        score=score,
    )


def _component(role: str, text: str, start: int = 0) -> dict[str, object]:
    return {
        "role": role,
        "text": text,
        "start": start,
        "end": start + len(text),
        "normalized": text,
        "detail": "",
    }


def _mini_rxnorm_index() -> LocalIndex:
    """A miniature governed RxNorm graph with the real schema shape.

    Same structure as the locked snapshot — `dict[id, list[id]]` adjacency with no
    relation labels, TTY in `metadata` — so a decision test exercises the real
    traversal logic without depending on 82,429 concepts.
    """
    records = {
        "IN1": {
            "concept_id": "IN1",
            "canonical_name": "testosterol",
            "aliases": ["testosterolum"],
            "metadata": {"tty": "IN", "suppress": "N"},
        },
        "SCDC1": {
            "concept_id": "SCDC1",
            "canonical_name": "testosterol 500 MG",
            "aliases": [],
            "metadata": {"tty": "SCDC", "suppress": "N"},
        },
        "SCDC2": {
            "concept_id": "SCDC2",
            "canonical_name": "testosterol 250 MG",
            "aliases": [],
            "metadata": {"tty": "SCDC", "suppress": "N"},
        },
        "SCD1": {
            "concept_id": "SCD1",
            "canonical_name": "testosterol 500 MG Extended Release Oral Tablet",
            "aliases": [],
            "metadata": {"tty": "SCD", "suppress": "N"},
        },
        "SCD2": {
            "concept_id": "SCD2",
            "canonical_name": "testosterol 500 MG Oral Capsule",
            "aliases": [],
            "metadata": {"tty": "SCD", "suppress": "N"},
        },
        "SBD1": {
            "concept_id": "SBD1",
            "canonical_name": "testosterol 500 MG Oral Capsule [Brandex]",
            "aliases": [],
            "metadata": {"tty": "SBD", "suppress": "N"},
        },
        "SCDG1": {
            "concept_id": "SCDG1",
            "canonical_name": "testosterol Oral Product",
            "aliases": [],
            "metadata": {"tty": "SCDG", "suppress": "N"},
        },
        "SUP1": {
            "concept_id": "SUP1",
            "canonical_name": "testosterol 500 MG obsolete",
            "aliases": [],
            "metadata": {"tty": "SCD", "suppress": "Y"},
        },
    }
    graph = {
        "IN1": ["SCDC1", "SCDC2", "SCDG1"],
        "SCDC1": ["IN1", "SCD1", "SCD2"],
        "SCDC2": ["IN1"],
        "SCD1": ["SCDC1"],
        "SCD2": ["SCDC1", "SBD1"],
        "SBD1": ["SCD2"],
        "SCDG1": ["IN1"],
        "SUP1": [],
    }
    exact = {"testosterol": ["IN1"], "testosterolum": ["IN1"]}
    return LocalIndex(
        index_type="rxnorm",
        source_snapshot_id="mini-rxnorm-fixture",
        records=records,
        exact=exact,
        exact_ascii=dict(exact),
        ngrams={},
        sparse_terms={"testosterol": list(records)},
        graph=graph,
    )


def _mini_icd_index() -> LocalIndex:
    """A miniature ICD-10 hierarchy with the real schema shape."""
    records = {
        "X10": {
            "concept_id": "X10",
            "canonical_name": "viem phoi",
            "aliases": [],
            "metadata": {
                "dotted_code": "X10",
                "specificity": "0",
                "chapter": "X",
                "block": "X00-X99",
            },
        },
        "X100": {
            "concept_id": "X100",
            "canonical_name": "viem phoi thuy duoi phai",
            "aliases": [],
            "metadata": {
                "dotted_code": "X10.0",
                "specificity": "1",
                "chapter": "X",
                "block": "X00-X99",
            },
        },
        "X101": {
            "concept_id": "X101",
            "canonical_name": "viem phoi do virus",
            "aliases": [],
            "metadata": {
                "dotted_code": "X10.1",
                "specificity": "1",
                "chapter": "X",
                "block": "X00-X99",
            },
        },
        "X9": {
            "concept_id": "X9",
            "canonical_name": "benh khac",
            "aliases": [],
            "metadata": {"dotted_code": "X9", "specificity": "0"},
        },
    }
    graph = {"X10": ["X100", "X101"], "X100": ["X10"], "X101": ["X10"]}
    # X9 is deliberately retrievable but ABSENT from the graph — the locked snapshot
    # has 774 such records, so the code must handle one without raising.
    exact = {"viem phoi": ["X10"], "viem phoi do virus": ["X101"], "benh khac": ["X9"]}
    return LocalIndex(
        index_type="icd10_vi",
        source_snapshot_id="mini-icd-fixture",
        records=records,
        exact=exact,
        exact_ascii=dict(exact),
        ngrams={},
        sparse_terms={"viem": ["X10", "X100", "X101"], "phoi": ["X10", "X100", "X101"]},
        graph=graph,
    )


# ---------------------------------------------------------------------------
# A. Structured medication representation (spec §10.1)
# ---------------------------------------------------------------------------


def test_structured_mention_consumes_e1_components() -> None:
    hypothesis = _hypothesis(
        "testosterol 500 mg ER po",
        components=(
            _component("name", "testosterol"),
            _component("strength_value", "500", 12),
            _component("strength_unit", "mg", 16),
            _component("release", "ER", 19),
            _component("route", "po", 22),
        ),
    )
    mention = build_structured_mention(hypothesis)
    assert mention.ingredient is not None
    assert mention.ingredient.text == "testosterol"
    assert mention.strength_mg == 500.0
    assert mention.release is not None
    assert "dose_form" in mention.unresolved_fields


def test_missing_evidence_is_not_negative_evidence() -> None:
    """A field the mention never states must be absent, not falsely present."""
    mention = build_structured_mention(
        _hypothesis("aspirin", components=(_component("name", "aspirin"),))
    )
    assert mention.dose_form is None
    assert "dose_form" in mention.unresolved_fields
    assert mention.incoherent_fields == ()


def test_a_half_stated_strength_is_recorded_as_incoherent() -> None:
    mention = build_structured_mention(
        _hypothesis(
            "aspirin 81",
            components=(_component("name", "aspirin"), _component("strength_value", "81", 8)),
        )
    )
    assert any("value_unit_pair_incomplete" in f for f in mention.incoherent_fields)


def test_unit_normalization_is_conservative() -> None:
    assert canonical_unit("MG") == "mg"
    assert canonical_unit("UI") == "unit"
    assert canonical_unit("µg") == "mcg"
    assert canonical_unit("furlongs") is None
    assert unit_family("mg") == "mass"
    assert unit_family("ml") == "volume"
    assert strength_in_mg(1.0, "g") == 1000.0
    assert strength_in_mg(1.0, "ml") is None


def test_retrieval_queries_the_ingredient_not_only_the_surface() -> None:
    mention = build_structured_mention(
        _hypothesis(
            "testosterol 500 mg po daily",
            components=(_component("name", "testosterol"), _component("strength_value", "500", 12)),
        )
    )
    terms = mention.query_terms()
    assert "testosterol" in terms
    assert terms[-1] == "testosterol 500 mg po daily"


def test_the_inn_bridge_widens_lookup_without_conferring_authority() -> None:
    """`paracetamol` occurs in ZERO records of the locked snapshot; RxNorm says
    `acetaminophen`.

    3A disabled the bridge outright, which meant the most common Vietnamese
    medication surface linked to nothing at all. Audit 0058 governs it instead: the
    bridge is ``RETRIEVAL_ONLY``, so it may widen *lookup* while candidacy is still
    decided by the searchable index and the mention's own evidence. It returns names,
    never codes, so it cannot confer authority even in principle.
    """
    assert bridged_ingredient_names("paracetamol") == ("acetaminophen",)
    assert bridged_ingredient_names("aspirin") == (), "only governed surfaces expand"
    assert not any(name.isdigit() for name in bridged_ingredient_names("paracetamol"))
    assert len(INN_TO_RXNORM_NAME) <= 20, (
        "a long hand-written drug-name table is a safety hazard, not a convenience"
    )


# ---------------------------------------------------------------------------
# B. RxNorm traversal + hard negatives, on the miniature governed graph
# ---------------------------------------------------------------------------


def test_traversal_walks_ingredient_to_component_to_drug_to_brand() -> None:
    index = _mini_rxnorm_index()
    paths, notes = traverse_ingredient_chain(index, ["IN1"])
    reached = {p.concept_id: p for p in paths}
    assert {"IN1", "SCDC1", "SCD1", "SCD2", "SBD1"} <= set(reached)
    assert reached["SCD2"].depth == 2
    assert reached["SBD1"].depth == 3
    assert "IN:IN1->SCDC:SCDC1" in reached["SBD1"].as_text()
    assert notes == ()


def test_traversal_needs_an_ingredient_seed_and_says_so() -> None:
    index = _mini_rxnorm_index()
    _paths, notes = traverse_ingredient_chain(index, ["SCD1"])
    assert "no_ingredient_level_seed" in notes


def test_a_strength_conflict_is_a_hard_negative() -> None:
    index = _mini_rxnorm_index()
    mention = build_structured_mention(
        _hypothesis(
            "testosterol 500 mg",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
            ),
        )
    )
    comparisons = {c.field_name: c for c in compare_candidate(mention, index, "SCDC2")}
    assert comparisons["strength"].verdict == CONFLICT
    report = link_rxnorm_structured(
        _hypothesis(
            "testosterol 500 mg",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
            ),
        ),
        index,
    )
    assert any(
        d.concept_id == "SCDC2" and d.reason == DROP_STRENGTH_CONFLICT for d in report.suppressed
    )


def test_a_unit_family_conflict_is_a_hard_negative() -> None:
    index = _mini_rxnorm_index()
    mention = build_structured_mention(
        _hypothesis(
            "testosterol 500 ml",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "ml", 16),
            ),
        )
    )
    comparisons = {c.field_name: c for c in compare_candidate(mention, index, "SCDC1")}
    assert comparisons["unit"].verdict == CONFLICT, "mass vs volume is a category error"


def test_a_dose_form_conflict_is_a_hard_negative() -> None:
    index = _mini_rxnorm_index()
    mention = build_structured_mention(
        _hypothesis(
            "testosterol 500 mg capsule",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
                _component("dose_form", "capsule", 19),
            ),
        )
    )
    report = link_rxnorm_structured(
        _hypothesis(
            "testosterol 500 mg capsule",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
                _component("dose_form", "capsule", 19),
            ),
        ),
        index,
    )
    reasons = {d.concept_id: d.reason for d in report.suppressed}
    assert reasons.get("SCD1") == DROP_DOSE_FORM_CONFLICT
    assert "SCD2" in {d.concept_id for d in report.retained}
    assert mention.dose_form is not None


def test_a_release_conflict_is_a_hard_negative() -> None:
    index = _mini_rxnorm_index()
    report = link_rxnorm_structured(
        _hypothesis(
            "testosterol 500 mg DR",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
                _component("release", "DR", 19),
            ),
        ),
        index,
    )
    reasons = {d.concept_id: d.reason for d in report.suppressed}
    assert reasons.get("SCD1") == DROP_RELEASE_CONFLICT


def test_a_suppressed_concept_is_never_offered() -> None:
    index = _mini_rxnorm_index()
    report = link_rxnorm_structured(
        _hypothesis("testosterol", components=(_component("name", "testosterol"),)), index
    )
    assert "SUP1" not in {d.concept_id for d in report.retained}


def test_a_grouper_tty_is_evidence_not_a_candidate() -> None:
    index = _mini_rxnorm_index()
    report = link_rxnorm_structured(
        _hypothesis("testosterol", components=(_component("name", "testosterol"),)), index
    )
    assert any(
        d.concept_id == "SCDG1" and d.reason == DROP_NON_TERMINAL_TTY for d in report.suppressed
    )


def test_an_incomplete_mention_falls_back_to_the_ingredient() -> None:
    index = _mini_rxnorm_index()
    report = link_rxnorm_structured(
        _hypothesis("testosterol", components=(_component("name", "testosterol"),)), index
    )
    ingredient = next(d for d in report.retained if d.concept_id == "IN1")
    assert ingredient.tier == TIER_INGREDIENT_ONLY
    assert all(
        c.verdict == NOT_STATED
        for c in ingredient.comparisons
        if c.field_name in {"dose_form", "release", "route"}
    )


def test_no_rxcui_is_invented() -> None:
    index = _mini_rxnorm_index()
    report = link_rxnorm_structured(
        _hypothesis(
            "testosterol 500 mg",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
            ),
        ),
        index,
    )
    for decision in report.decisions:
        assert index.exists(decision.concept_id), decision.concept_id


def test_every_decision_records_its_provenance() -> None:
    index = _mini_rxnorm_index()
    report = link_rxnorm_structured(
        _hypothesis(
            "testosterol 500 mg",
            components=(
                _component("name", "testosterol"),
                _component("strength_value", "500", 12),
                _component("strength_unit", "mg", 16),
            ),
        ),
        index,
    )
    for decision in report.retained:
        payload = decision.as_dict()
        assert payload["tier"] and payload["reason"] and payload["snapshot_id"]
        assert "comparisons" in payload


def test_rxnorm_linking_is_deterministic() -> None:
    index = _mini_rxnorm_index()
    hypothesis = _hypothesis(
        "testosterol 500 mg",
        components=(
            _component("name", "testosterol"),
            _component("strength_value", "500", 12),
            _component("strength_unit", "mg", 16),
        ),
    )
    first = link_rxnorm_structured(hypothesis, index).as_dict()
    second = link_rxnorm_structured(hypothesis, index).as_dict()
    assert first == second


def test_wrong_index_type_is_refused() -> None:
    assert link_rxnorm(_hypothesis("x"), _mini_icd_index()).warnings == ("wrong_index_type",)


# --- live-snapshot tests: schema, membership and determinism only -----------


def test_live_rxnorm_snapshot_has_the_expected_schema() -> None:
    index = load_index(_config().rxnorm_index)
    assert index.index_type == "rxnorm"
    assert index.graph, "the RxNorm graph must be materialized"
    sample = next(iter(index.graph.items()))
    assert isinstance(sample[1], list), "adjacency is dict[str, list[str]]"
    assert all(isinstance(n, str) for n in sample[1]), (
        "edges are plain concept ids: the builder discarded relation labels"
    )
    health = graph_health(index)
    assert health["graph_nodes"] > 10_000
    assert health["graph_nodes_absent_from_records"] == 0


def test_live_rxnorm_ingredients_carry_tty() -> None:
    index = load_index(_config().rxnorm_index)
    ingredient_ids = [
        cid for cid in list(index.graph)[:2000] if tty_of(index, cid) in TTY_INGREDIENT
    ]
    assert ingredient_ids, "no IN/PIN/MIN concept found in a 2,000-node sample"


# ---------------------------------------------------------------------------
# C. ICD-10 hierarchy + specificity (spec §9.3)
# ---------------------------------------------------------------------------


def test_hierarchy_direction_comes_from_code_length() -> None:
    index = _mini_icd_index()
    assert parents(index, "X100") == ("X10",)
    assert children(index, "X10") == ("X100", "X101")
    assert ancestor_path(index, "X100") == ("X10",)
    assert graph_depth("X10") == 0
    assert graph_depth("X100") == 1


def test_a_missing_graph_node_is_reported_not_raised() -> None:
    index = _mini_icd_index()
    assert ancestor_path(index, "X9") == ()
    assert children(index, "X9") == ()
    report = link_icd10_hierarchical(_hypothesis("benh khac", "CHẨN_ĐOÁN"), index)
    assert "anchor_absent_from_graph:X9" in report.expansion_notes
    kept = {d.code for d in report.retained}
    assert "X9" in kept, "a graph-less record must still be linkable by name"
    decision = next(d for d in report.retained if d.code == "X9")
    assert decision.hierarchy is not None
    assert decision.hierarchy.in_graph is False
    assert decision.hierarchy.ancestor_path == ()


def test_declared_specificity_is_only_depth() -> None:
    """`metadata.specificity` == len(code) - 3 throughout the locked snapshot."""
    index = load_index(_config().icd_index)
    for code in list(index.records)[:500]:
        declared = declared_specificity(index, code)
        assert declared == graph_depth(code), code


def test_an_exact_general_diagnosis_wins_its_own_tier() -> None:
    index = _mini_icd_index()
    report = link_icd10_hierarchical(_hypothesis("viem phoi", "CHẨN_ĐOÁN"), index)
    best = report.retained[0]
    assert best.code == "X10"
    assert best.tier == TIER_EXACT_NAME
    assert best.reason == KEEP_EXACT_NAME


def test_an_explicitly_specific_diagnosis_keeps_the_specific_code() -> None:
    index = _mini_icd_index()
    report = link_icd10_hierarchical(_hypothesis("viem phoi do virus", "CHẨN_ĐOÁN"), index)
    kept = {d.code for d in report.retained}
    assert "X101" in kept, "the mention states the detail X101 adds"


def test_an_unsupported_over_specific_descendant_is_suppressed() -> None:
    index = _mini_icd_index()
    report = link_icd10_hierarchical(_hypothesis("viem phoi", "CHẨN_ĐOÁN"), index)
    suppressed = {d.code: d for d in report.suppressed}
    assert "X100" in suppressed, "'thuy duoi phai' is not in the mention"
    decision = suppressed["X100"]
    assert decision.reason in {DROP_UNSUPPORTED_SPECIFICITY, DROP_NO_LEXICAL_SUPPORT}
    if decision.reason == DROP_UNSUPPORTED_SPECIFICITY:
        assert decision.missing_detail, "the missing tokens must be recorded"


def test_the_broader_code_is_the_conservative_fallback() -> None:
    index = _mini_icd_index()
    report = link_icd10_hierarchical(_hypothesis("viem phoi do virus", "CHẨN_ĐOÁN"), index)
    ancestors = [d for d in report.retained if d.reason == KEEP_BROADER_FALLBACK]
    assert ancestors or "X10" in {d.code for d in report.retained}
    if ancestors:
        assert ancestors[0].hierarchy is not None
        assert ancestors[0].hierarchy.relationship == REL_ANCESTOR


def test_specificity_assessment_records_the_missing_tokens() -> None:
    index = _mini_icd_index()
    assessment = assess_specificity(index, "X100", "X10", "viem phoi")
    assert not assessment.justified
    assert assessment.missing_tokens
    supported = assess_specificity(index, "X101", "X10", "viem phoi do virus")
    assert supported.justified


def test_sibling_competition_is_recorded() -> None:
    index = _mini_icd_index()
    assert siblings(index, "X100") == ("X101",)
    report = link_icd10_hierarchical(_hypothesis("viem phoi", "CHẨN_ĐOÁN"), index)
    assert isinstance(report.has_sibling_competition, bool)


def test_depth_alone_never_outranks_a_parent() -> None:
    index = _mini_icd_index()
    report = link_icd10_hierarchical(_hypothesis("viem phoi", "CHẨN_ĐOÁN"), index)
    kept = [d.code for d in report.retained]
    assert kept[0] == "X10", f"a deeper code must not win merely for being deeper: got {kept}"


def test_no_icd_code_is_invented() -> None:
    index = load_index(_config().icd_index)
    report = link_icd10_hierarchical(_hypothesis("Viêm phổi, không xác định", "CHẨN_ĐOÁN"), index)
    for decision in report.decisions:
        assert index.exists(decision.code), decision.code


def test_icd_linking_is_deterministic_on_the_live_snapshot() -> None:
    index = load_index(_config().icd_index)
    hypothesis = _hypothesis("Bệnh tả", "CHẨN_ĐOÁN")
    assert (
        link_icd10_hierarchical(hypothesis, index).as_dict()
        == link_icd10_hierarchical(hypothesis, index).as_dict()
    )


def test_live_icd_hierarchy_is_incomplete_and_that_is_recorded() -> None:
    """410 codes have no parent record; the code must survive that, not assume it away."""
    health = hierarchy_health(load_index(_config().icd_index))
    assert health["codes_without_parent_record"] > 0
    assert health["records_absent_from_graph"] > 0


def test_icd_wrong_index_type_is_refused() -> None:
    assert link_icd10(_hypothesis("x", "CHẨN_ĐOÁN"), _mini_rxnorm_index()).warnings == (
        "wrong_index_type",
    )


# ---------------------------------------------------------------------------
# D. L6 edges (spec §11)
# ---------------------------------------------------------------------------


def test_has_result_comes_from_e2_pair_groups() -> None:
    name = _hypothesis("WBC", "TÊN_XÉT_NGHIỆM", hypothesis_id="n1", start=0, pair_groups=("g1",))
    value = _hypothesis(
        "14.43", "KẾT_QUẢ_XÉT_NGHIỆM", hypothesis_id="v1", start=5, pair_groups=("g1",)
    )
    graph = build_evidence_graph("d1", (name, value), (), ())
    edges = graph.edges_of(REL_HAS_RESULT)
    assert len(edges) == 1
    assert edges[0].source_id == "n1" and edges[0].target_id == "v1"
    assert any(p.startswith("pair_group:") for p in edges[0].provenance)


def test_modified_by_comes_only_from_structured_components() -> None:
    with_components = _hypothesis(
        "aspirin 81 mg",
        components=(_component("name", "aspirin"), _component("strength_value", "81", 8)),
    )
    graph = build_evidence_graph("d1", (with_components,), (), ())
    roles = {
        p.removeprefix("role:")
        for e in graph.edges_of(REL_MODIFIED_BY)
        for p in e.provenance
        if p.startswith("role:")
    }
    assert roles == {"strength_value"}, "the head 'name' is not a modifier"
    bare = build_evidence_graph("d1", (_hypothesis("aspirin"),), (), ())
    assert bare.edges_of(REL_MODIFIED_BY) == ()


def test_overlaps_comes_from_verified_intervals() -> None:
    left = _hypothesis("aspirin 81", hypothesis_id="a", start=0)
    right = _hypothesis("81 mg", hypothesis_id="b", start=8)
    graph = build_evidence_graph("d1", (left, right), (), ())
    edges = graph.edges_of(REL_OVERLAPS)
    assert len(edges) == 1
    assert any(p.startswith("same_type:") for p in edges[0].provenance)


def test_same_surface_links_repeats_without_merging_coordinates() -> None:
    first = _hypothesis("aspirin", hypothesis_id="a", start=0)
    second = _hypothesis("Aspirin", hypothesis_id="b", start=50)
    graph = build_evidence_graph("d1", (first, second), (), ())
    edges = graph.edges_of(REL_SAME_SURFACE)
    assert len(edges) == 1
    assert "left:0:7" in edges[0].provenance
    assert "right:50:57" in edges[0].provenance


def test_treats_requires_explicit_evidence_and_co_occurrence_is_not_enough() -> None:
    from mednorm_vi.document_intelligence.models import DocumentGraph

    class _Doc:
        original_text = "aspirin va viem phoi cung xuat hien trong ho so nay"
        nodes: tuple[object, ...] = ()
        document_id = "d1"

    medication = _hypothesis("aspirin", "THUỐC", hypothesis_id="m", start=0)
    diagnosis = _hypothesis("viem phoi", "CHẨN_ĐOÁN", hypothesis_id="dx", start=11)
    graph = build_evidence_graph("d1", (medication, diagnosis), (), (), document=_Doc())  # type: ignore[arg-type]
    assert graph.edges_of(REL_TREATS) == (), "co-occurrence must not assert treatment"
    assert ("m", "dx") in graph.declined_treats
    assert DocumentGraph is not None  # import kept meaningful


def test_treats_is_asserted_when_the_text_says_so() -> None:
    class _Doc:
        original_text = "aspirin dieu tri viem phoi"
        nodes: tuple[object, ...] = ()
        document_id = "d1"

    medication = _hypothesis("aspirin", "THUỐC", hypothesis_id="m", start=0)
    diagnosis = _hypothesis("viem phoi", "CHẨN_ĐOÁN", hypothesis_id="dx", start=17)
    graph = build_evidence_graph("d1", (medication, diagnosis), (), (), document=_Doc())  # type: ignore[arg-type]
    edges = graph.edges_of(REL_TREATS)
    assert len(edges) == 1
    assert edges[0].provenance == ("trigger:dieu tri",)


def test_in_section_comes_from_l1_and_appears_on_real_documents() -> None:
    result = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    assert result.evidence_graph.edges_of(REL_IN_SECTION), "L1 section provenance must reach L6"


def test_edges_are_not_duplicated_and_the_hash_is_stable() -> None:
    hypothesis = _hypothesis("aspirin 81 mg", components=(_component("strength_value", "81", 8),))
    first = build_evidence_graph("d1", (hypothesis,), (), ())
    second = build_evidence_graph("d1", (hypothesis,), (), ())
    assert first.graph_hash == second.graph_hash
    keys = [e.key for e in first.edges]
    assert len(keys) == len(set(keys)), "duplicate edges present"


def test_every_edge_carries_provenance_and_a_tier() -> None:
    result = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    for edge in result.evidence_graph.edges:
        assert edge.evidence_source
        assert edge.confidence_tier
        assert edge.rule


# ---------------------------------------------------------------------------
# E. Typed consistency contract (spec §11)
# ---------------------------------------------------------------------------


def test_the_ontology_lookup_translates_organizer_labels() -> None:
    """The bug this guards against withheld 100% of output. See `ontology_for`."""
    assert ontology_for("THUỐC") == "RXNORM"
    assert ontology_for("CHẨN_ĐOÁN") == "ICD10"
    assert ontology_for("TRIỆU_CHỨNG") is None


def test_uncertainty_is_never_encoded_as_false() -> None:
    unpaired = _hypothesis("WBC", "TÊN_XÉT_NGHIỆM", hypothesis_id="n1")
    graph = build_evidence_graph("d1", (unpaired,), (), ())
    report = evaluate_consistency("d1", graph, (unpaired,))
    verdicts = {d.verdict for d in report.decisions}
    assert UNRESOLVED in verdicts
    assert all(v in {SUPPORTED, CONTRADICTED, UNRESOLVED, NOT_APPLICABLE} for v in verdicts)


def test_an_assertion_on_an_ineligible_type_is_fatal() -> None:
    hypothesis = _hypothesis("14.43", "KẾT_QUẢ_XÉT_NGHIỆM", hypothesis_id="v1")
    assertion = AssertionDecision("v1", ("isNegated",), {})
    graph = build_evidence_graph("d1", (hypothesis,), (assertion,), ())
    report = evaluate_consistency("d1", graph, (hypothesis,), (assertion,))
    fatal = [i for i in report.fatal_issues if i.rule == RULE_ASSERTION_TYPE]
    assert fatal and fatal[0].verdict == CONTRADICTED


def test_a_duplicate_candidate_code_is_fatal() -> None:
    hypothesis = _hypothesis("aspirin", hypothesis_id="m1")
    link = LinkerResult(
        "m1",
        (
            LinkedCandidate("111", 1.0, ("exact",), "snap"),
            LinkedCandidate("111", 0.9, ("exact",), "snap"),
        ),
    )
    graph = build_evidence_graph("d1", (hypothesis,), (), (link,))
    report = evaluate_consistency("d1", graph, (hypothesis,), (), (link,))
    fatal = [i for i in report.fatal_issues if i.rule == RULE_DUPLICATE_CANDIDATE]
    assert fatal
    assert "111" in report.blocked_candidate_codes("m1")


def test_same_type_overlap_survival_is_fatal() -> None:
    left = _hypothesis("aspirin 81", hypothesis_id="a", start=0)
    right = _hypothesis("81 mg", hypothesis_id="b", start=8)
    graph = build_evidence_graph("d1", (left, right), (), ())
    report = evaluate_consistency("d1", graph, (left, right))
    assert any(i.rule == RULE_OVERLAP and i.fatal for i in report.issues)


def test_a_declined_treats_pair_is_reported_not_dropped() -> None:
    class _Doc:
        original_text = "aspirin va viem phoi"
        nodes: tuple[object, ...] = ()
        document_id = "d1"

    medication = _hypothesis("aspirin", "THUỐC", hypothesis_id="m", start=0)
    diagnosis = _hypothesis("viem phoi", "CHẨN_ĐOÁN", hypothesis_id="dx", start=11)
    graph = build_evidence_graph("d1", (medication, diagnosis), (), (), document=_Doc())  # type: ignore[arg-type]
    report = evaluate_consistency("d1", graph, (medication, diagnosis))
    assert any(i.rule == RULE_UNSAFE_TREATS and i.verdict == UNRESOLVED for i in report.issues)


def test_the_consistency_report_carries_no_clinical_text() -> None:
    result = run_document(FIX / "synthetic_medication_list.txt", _config(), mode="deterministic")
    assert result.consistency is not None
    serialized = json.dumps(result.consistency.as_dict(), ensure_ascii=False)
    source = (FIX / "synthetic_medication_list.txt").read_text(encoding="utf-8")
    for token in source.split():
        if len(token) > 4 and token.isalpha():
            assert token not in serialized, f"leaked {token!r}"
    assert result.consistency.as_dict()["contains_clinical_text"] is False


def test_the_consistency_report_serializes_deterministically() -> None:
    result = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    again = run_document(FIX / "synthetic_medical_document.txt", _config(), mode="deterministic")
    assert result.consistency is not None and again.consistency is not None
    assert result.consistency.report_hash == again.consistency.report_hash


# ---------------------------------------------------------------------------
# F. L7 locked-option escalation (spec §12.1, P7)
# ---------------------------------------------------------------------------


def test_l7_imports_no_model_stack() -> None:
    """Checks the import statements, not the prose.

    An earlier version of this test scanned the whole source, which failed because the
    module *docstring* names the things it must not import. Parsing the AST asks the
    question that was actually meant.
    """
    import ast
    import inspect

    import mednorm_vi.confidence_cascade.escalation as module

    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("transformers", "torch", "requests", "urllib", "socket"):
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.") for name in imported
        ), f"L7 escalation imports {forbidden}"
    assert not any("llm" in name.split(".") for name in imported), (
        "L7 escalation must not import the model-loading package"
    )


def test_locked_options_are_built_only_from_existing_alternatives() -> None:
    hypothesis = _hypothesis(
        "aspirin 81 mg",
        alternatives=(BoundaryAlternative("p2", 0, 7, "aspirin", "name_only"),),
        proposed_types=("THUỐC",),
    )
    link = LinkerResult("h1", (LinkedCandidate("111", 1.0, ("exact",), "snap"),))
    options = build_locked_options(hypothesis, link_result=link, ontology="RXNORM")
    assert len(options.boundaries) == 2
    assert options.offered_codes() == frozenset({"111"})
    assert {o.entity_type for o in options.types} == {"THUỐC"}, (
        "a type nobody proposed is not an alternative"
    )


def test_a_decision_outside_the_locked_set_is_refused() -> None:
    request = build_escalation_request(_hypothesis("aspirin"), tier=TIER_CRITIC)
    invented = EscalationDecision(
        subject_id="h1", disposition=ACCEPT, tier=TIER_CRITIC, candidate_option_ids=("c:99999",)
    )
    result = validate_escalation_decision(request, invented)
    assert not result.accepted
    assert REFUSE_INVENTED_VALUE in result.refusals


def test_a_decision_for_the_wrong_subject_is_refused() -> None:
    request = build_escalation_request(_hypothesis("aspirin"))
    result = validate_escalation_decision(
        request, EscalationDecision(subject_id="other", disposition=ACCEPT)
    )
    assert REFUSE_WRONG_SUBJECT in result.refusals


def test_an_unknown_boundary_option_is_refused() -> None:
    request = build_escalation_request(_hypothesis("aspirin"))
    result = validate_escalation_decision(
        request,
        EscalationDecision(subject_id="h1", disposition=ACCEPT, boundary_option_id="b:invented"),
    )
    assert REFUSE_UNKNOWN_OPTION in result.refusals


def test_entry_conditions_record_both_halves() -> None:
    assertion = AssertionDecision("h1", (), {}, uncertain=True)
    request = build_escalation_request(_hypothesis("aspirin"), assertion=assertion)
    assert COND_ASSERTION_UNCERTAIN in request.triggered_conditions
    assert request.skipped_conditions, "conditions that did not fire must be recorded"
    assert set(request.triggered_conditions) & set(request.skipped_conditions) == set()


def test_the_no_model_fallback_is_explicit_and_never_a_silent_accept() -> None:
    assertion = AssertionDecision("h1", (), {}, uncertain=True)
    request = build_escalation_request(_hypothesis("aspirin"), assertion=assertion)
    decision = deterministic_fallback(request)
    assert decision.disposition == UNRESOLVED_DISPOSITION
    assert "no_decision_source" in decision.reasons
    quiet = build_escalation_request(_hypothesis("aspirin"))
    assert deterministic_fallback(quiet).disposition == ACCEPT


def test_a_refused_source_decision_degrades_to_the_fallback() -> None:
    class _TestDecisionSource:
        """A FIXTURE decision source. Not a model, not an LLM — a test double."""

        name = "test_fixture_decision_source"

        def decide(self, request):  # type: ignore[no-untyped-def]
            return EscalationDecision(
                subject_id=request.subject_id,
                disposition=ACCEPT,
                tier=TIER_CRITIC,
                candidate_option_ids=("c:invented",),
            )

    assertion = AssertionDecision("h1", (), {}, uncertain=True)
    request = build_escalation_request(_hypothesis("aspirin"), assertion=assertion)
    decision, validation = resolve_escalation(request, _TestDecisionSource())
    assert not validation.accepted
    assert decision.disposition == UNRESOLVED_DISPOSITION
    assert any("source_refused" in r for r in decision.reasons)


def test_a_cascade_rejection_is_authoritative() -> None:
    hypothesis = _hypothesis("aspirin")
    report = run_cascade_escalation(
        "d1", (hypothesis,), cascade=(CascadeDecision("h1", False, 0.0, ("below_threshold",)),)
    )
    decision = report.decision_for("h1")
    assert decision is not None and decision.disposition == REJECT


def test_a_graph_contradiction_triggers_escalation() -> None:
    left = _hypothesis("aspirin 81", hypothesis_id="a", start=0)
    right = _hypothesis("81 mg", hypothesis_id="b", start=8)
    graph = build_evidence_graph("d1", (left, right), (), ())
    consistency = evaluate_consistency("d1", graph, (left, right))
    request = build_escalation_request(left, consistency=consistency)
    assert COND_GRAPH_CONTRADICTION in request.triggered_conditions


# ---------------------------------------------------------------------------
# G. L8 consumes L6 and L7
# ---------------------------------------------------------------------------


def test_a_fatal_contradiction_cannot_be_silently_accepted() -> None:
    from mednorm_vi.metric_decoder import decode_entities

    hypothesis = _hypothesis("14.43", "KẾT_QUẢ_XÉT_NGHIỆM", hypothesis_id="v1")
    assertion = AssertionDecision("v1", ("isNegated",), {})
    graph = build_evidence_graph("d1", (hypothesis,), (assertion,), ())
    consistency = evaluate_consistency("d1", graph, (hypothesis,), (assertion,))
    cascade = (CascadeDecision("v1", True, 0.9, ("resolver_accepted",)),)
    without = decode_entities((hypothesis,), cascade, (assertion,), ())
    with_report = decode_entities((hypothesis,), cascade, (assertion,), (), consistency=consistency)
    assert len(without) == 1, "the Audit-0053 behaviour is unchanged without a report"
    assert with_report == (), "a fatal contradiction must withhold the entity"


def test_a_blocked_code_is_dropped_with_a_consistency_reason() -> None:
    from mednorm_vi.metric_decoder import decode_entities
    from mednorm_vi.metric_decoder.decoder import DROP_GRAPH_CONTRADICTION

    hypothesis = _hypothesis("aspirin", hypothesis_id="m1")
    link = LinkerResult(
        "m1",
        (
            LinkedCandidate("111", 1.0, ("sparse",), "snap"),
            LinkedCandidate("111", 0.9, ("sparse",), "snap"),
            LinkedCandidate("222", 0.8, ("sparse",), "snap"),
        ),
    )
    graph = build_evidence_graph("d1", (hypothesis,), (), (link,))
    consistency = evaluate_consistency("d1", graph, (hypothesis,), (), (link,))
    decoded = decode_entities(
        (hypothesis,), (CascadeDecision("m1", True, 0.9, ()),), (), (link,), consistency=consistency
    )
    assert decoded, "a duplicate code must not withhold the whole entity"
    assert "111" not in decoded[0].candidates
    assert any(d.reason == DROP_GRAPH_CONTRADICTION for d in decoded[0].candidate_decisions)


def test_consistency_evidence_appears_in_decoder_reason_codes() -> None:
    result = run_document(FIX / "synthetic_medication_list.txt", _config(), mode="deterministic")
    assert result.consistency is not None
    assert result.predictions, "the fixture must still produce entities"


def test_an_l7_reject_withholds_the_entity() -> None:
    from mednorm_vi.metric_decoder import decode_entities

    hypothesis = _hypothesis("aspirin", hypothesis_id="m1")
    escalation = run_cascade_escalation(
        "d1", (hypothesis,), cascade=(CascadeDecision("m1", False, 0.0, ("below_threshold",)),)
    )
    decoded = decode_entities(
        (hypothesis,), (CascadeDecision("m1", True, 0.9, ()),), (), (), escalation=escalation
    )
    assert decoded == ()


def test_no_fixed_top_k_returns() -> None:
    """No *hidden* fixed top-K — the Audit-0053 guard, made profile-aware.

    This test was written to catch `decode_expected_jaccard`'s unconditional
    `candidates[:10]`: a constant candidate-set size meant a hardcoded slice nobody had
    declared. Audit 0060 made the cap a **governed profile**, and `competition_top1`
    legitimately produces a constant size of 1 — which the original assertion would have
    reported as the very bug it was written to catch.

    So the property is now checked where it is meaningful and strengthened where it is
    not. Under `competition_adaptive` the size must still vary with the evidence, exactly
    as before. Under `competition_top1` the size must respect the *declared*
    `max_candidates`, and the same input under adaptive must produce a strictly larger
    set — proving the narrowing is policy-driven and reversible rather than a slice
    baked into the decoder.
    """
    from dataclasses import replace

    from mednorm_vi.kb.competition.topk import COMPETITION_TOP1

    base = _config()
    adaptive = run_document(
        FIX / "synthetic_medication_list.txt",
        replace(base, decoding_profile="competition_adaptive"),
        mode="deterministic",
    )
    adaptive_sizes = {len(p.candidates or ()) for p in adaptive.predictions}
    assert len(adaptive_sizes) > 1, (
        f"candidate-set size collapsed to a constant under adaptive: {adaptive_sizes}"
    )

    top1 = run_document(
        FIX / "synthetic_medication_list.txt",
        replace(base, decoding_profile="competition_top1"),
        mode="deterministic",
    )
    top1_sizes = {len(p.candidates or ()) for p in top1.predictions}
    assert max(top1_sizes) <= COMPETITION_TOP1.max_candidates
    assert max(adaptive_sizes) > max(top1_sizes), (
        "top-1 must be a narrowing of adaptive, not an independently hardcoded slice"
    )


def test_the_calibrated_decoder_still_fails_closed() -> None:
    from mednorm_vi.metric_decoder import (
        CalibratedDecoderUnavailable,
        decode_expected_jaccard_calibrated,
    )

    with pytest.raises(CalibratedDecoderUnavailable):
        decode_expected_jaccard_calibrated()


# ---------------------------------------------------------------------------
# H. Inference run manifest
# ---------------------------------------------------------------------------


def test_role_safe_path_removes_the_home_directory() -> None:
    safe = role_safe_path("/home/someone/checkpoint/s1/best.pt", role="e3_checkpoint")
    assert safe == "role:e3_checkpoint/best.pt"
    assert "someone" not in safe


def test_a_manifest_is_not_written_unless_requested(tmp_path: Path) -> None:
    manifest = build_run_manifest((), mode="deterministic", config=_config())
    assert isinstance(manifest, RunManifest)
    assert not list(tmp_path.iterdir()), "constructing a manifest must not write"


def test_a_manifest_reports_missing_checkpoints_and_degradation() -> None:
    manifest = build_run_manifest(
        (),
        mode="full",
        config=_config(),
        readiness=type(
            "R",
            (),
            {
                "status": "NOT_READY",
                "errors": ("missing_checkpoint:x",),
                "expected_experts": ("e",),
                "degraded": ("d",),
            },
        )(),
    )
    payload = manifest.as_dict()
    assert payload["readiness"]["status"] == "NOT_READY"
    assert payload["readiness"]["fail_closed"] is True
    assert payload["readiness"]["errors"] == ["missing_checkpoint:x"]


def test_a_manifest_records_an_l9_stop() -> None:
    manifest = build_run_manifest(
        (),
        mode="deterministic",
        config=_config(),
        l9_issue_counts={"kb.candidate_not_in_snapshot": 3},
        l9_stopped=True,
    )
    payload = manifest.as_dict()
    assert payload["l9_stopped_the_run"] is True
    assert payload["aggregates"]["l9_issues"]["kb.candidate_not_in_snapshot"] == 3


def test_manifest_serialization_is_stable_and_hashable(tmp_path: Path) -> None:
    manifest = build_run_manifest((), mode="deterministic", config=_config())
    first = write_run_manifest(manifest, tmp_path / "m.json").read_text(encoding="utf-8")
    second = manifest.to_json()
    assert first == second
    assert json.loads(first)["manifest_schema_version"]
    assert len(manifest.manifest_hash) == 64


def test_a_manifest_refuses_to_claim_it_holds_text() -> None:
    manifest = build_run_manifest((), mode="deterministic", config=_config())
    assert manifest.as_dict()["contains_clinical_text"] is False


def test_the_output_zip_is_byte_reproducible() -> None:
    assert ZIP_FIXED_TIMESTAMP == (1980, 1, 1, 0, 0, 0), (
        "a fixed entry timestamp is what makes the archive hash stable"
    )


# ---------------------------------------------------------------------------
# I. Code-linking evaluation contract — synthetic miniature records only
# ---------------------------------------------------------------------------


def _gold(
    codes: set[str],
    adjudication: str = ADJUDICATED_SINGLE,
    *,
    start: int = 0,
    ontology: str = "ICD10",
) -> CodeLinkingGoldRecord:
    return CodeLinkingGoldRecord(
        document_id="d1",
        start=start,
        end=start + 5,
        mention_text="xxxxx",
        entity_type="DIAGNOSIS",
        ontology=ontology,
        snapshot_id="snap-1",
        acceptable_codes=frozenset(codes),
        adjudication=adjudication,
        annotators=("a1", "a2"),
        adjudication_rule="majority",
        split_sha256="deadbeef",
    )


def _prediction(
    codes: tuple[str, ...],
    *,
    start: int = 0,
    ontology: str = "ICD10",
) -> CodeLinkingPredictionRecord:
    return CodeLinkingPredictionRecord(
        document_id="d1",
        start=start,
        end=start + 5,
        entity_type="DIAGNOSIS",
        ontology=ontology,
        snapshot_id="snap-1",
        predicted_codes=codes,
    )


def test_candidate_quality_is_unmeasurable_without_gold() -> None:
    assert candidate_quality_status() == UNMEASURABLE
    assert candidate_quality_status("/nonexistent/gold.jsonl") == UNMEASURABLE


def test_scoring_refuses_rather_than_returning_zero() -> None:
    with pytest.raises(GoldSetUnavailable):
        evaluate_code_linking((), (_prediction(("A00",)),))


def test_an_unsettled_annotation_is_not_gold(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(
            {
                "document_id": "d1",
                "start": 0,
                "end": 5,
                "ontology": "ICD10",
                "snapshot_id": "s",
                "adjudication": PENDING,
                "acceptable_codes": ["A00"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldRecordInvalid):
        load_gold_records(path)
    path.write_text(
        json.dumps(
            {
                "document_id": "d1",
                "start": 0,
                "end": 5,
                "ontology": "ICD10",
                "snapshot_id": "s",
                "adjudication": DISPUTED,
                "acceptable_codes": ["A00"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldRecordInvalid):
        load_gold_records(path)


def test_a_missing_gold_file_raises_the_sentinel_message(tmp_path: Path) -> None:
    with pytest.raises(GoldSetUnavailable) as raised:
        load_gold_records(tmp_path / "absent.jsonl")
    assert UNMEASURABLE in str(raised.value)


def test_the_schema_is_strict() -> None:
    with pytest.raises(GoldRecordInvalid):
        _gold({"A00", "A01"}, ADJUDICATED_SINGLE)  # SINGLE must carry exactly one
    with pytest.raises(GoldRecordInvalid):
        _gold({"A00"}, ADJUDICATED_NONE)  # NONE must carry no code
    with pytest.raises(GoldRecordInvalid):
        CodeLinkingGoldRecord(
            document_id="d",
            start=5,
            end=5,
            mention_text="",
            entity_type="",
            ontology="ICD10",
            snapshot_id="s",
            acceptable_codes=frozenset({"A"}),
            adjudication=ADJUDICATED_SINGLE,
        )
    with pytest.raises(GoldRecordInvalid):
        CodeLinkingGoldRecord(
            document_id="d",
            start=0,
            end=1,
            mention_text="",
            entity_type="",
            ontology="SNOMED",
            snapshot_id="s",
            acceptable_codes=frozenset({"A"}),
            adjudication=ADJUDICATED_SINGLE,
        )


def test_zero_one_and_many_acceptable_codes_are_all_valid() -> None:
    assert _gold(set(), ADJUDICATED_NONE).usable_as_gold
    assert _gold({"A00"}, ADJUDICATED_SINGLE).usable_as_gold
    assert _gold({"A00", "A01"}, ADJUDICATED_MULTIPLE).usable_as_gold


def test_recall_at_k_respects_k() -> None:
    gold = (_gold({"A02"}),)
    predictions = (_prediction(("A00", "A01", "A02")),)
    assert recall_at_k(gold, predictions, k=1).recall == 0.0
    assert recall_at_k(gold, predictions, k=5).recall == 1.0


def test_recall_excludes_correctly_empty_gold() -> None:
    result = recall_at_k((_gold(set(), ADJUDICATED_NONE),), (), k=5)
    assert result.eligible == 0


def test_jaccard_rewards_a_correct_withholding() -> None:
    result = candidate_jaccard((_gold(set(), ADJUDICATED_NONE),), (_prediction(()),))
    assert result.mean_jaccard == 1.0
    assert result.exact_set_matches == 1


def test_exact_candidate_set_match_is_counted() -> None:
    result = candidate_jaccard((_gold({"A00"}),), (_prediction(("A00",)),))
    assert result.exact_set_matches == 1
    assert result.mean_jaccard == 1.0


def test_error_categories_separate_broader_from_wrong() -> None:
    buckets = {
        c.category: c.count
        for c in categorize_errors(
            (_gold({"A001"}),), (_prediction(("A00",)),), ancestors_of={"A001": frozenset({"A00"})}
        )
    }
    assert buckets[ERR_BROADER] == 1
    assert buckets[ERR_EXACT] == 0


def test_error_categories_flag_wrong_ontology_and_stale_codes() -> None:
    wrong = categorize_errors((_gold({"A00"}),), (_prediction(("111",), ontology="RXNORM"),))
    assert {c.category: c.count for c in wrong}[ERR_WRONG_ONTOLOGY] == 1
    stale = categorize_errors(
        (_gold({"A00"}),), (_prediction(("ZZZ",)),), snapshot_members={"ICD10": frozenset({"A00"})}
    )
    assert {c.category: c.count for c in stale}[ERR_STALE_CODE] == 1


def test_spurious_candidates_where_gold_is_none_are_flagged() -> None:
    buckets = {
        c.category: c.count
        for c in categorize_errors((_gold(set(), ADJUDICATED_NONE),), (_prediction(("A00",)),))
    }
    assert buckets[ERR_SPURIOUS] == 1


def test_a_full_report_is_only_constructible_from_usable_gold() -> None:
    report = evaluate_code_linking(
        (_gold({"A00"}), _gold(set(), ADJUDICATED_NONE, start=10)),
        (_prediction(("A00",)), _prediction((), start=10)),
    )
    payload = report.as_dict()
    assert payload["gold_records"] == 2
    assert payload["candidate_jaccard"]["mean_jaccard"] == 1.0
    assert payload["snapshot_ids"] == ["snap-1"]


def test_the_required_human_artifact_is_documented() -> None:
    spec = REQUIRED_ANNOTATION_ARTIFACT
    assert spec["minimum_scope"]["documents"]
    assert any("language model" in item for item in spec["explicitly_forbidden"])
    assert any(
        "two independent clinical annotators" in item for item in spec["process_requirements"]
    )


def test_no_gold_data_is_shipped_in_the_repository() -> None:
    """The contract must not arrive with fabricated labels attached."""
    candidates = (
        list((REPO / "data").rglob("*code_linking*gold*")) if (REPO / "data").is_dir() else []
    )
    assert not candidates, f"unexpected gold-looking artifacts: {candidates}"
