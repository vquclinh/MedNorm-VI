"""Milestone 3B competition KB policy, candidate v3 and runtime top-k (Audit 0058).

Behaviour, not organizer output. The organizer's examples inspire the *shapes* tested
here — an ingredient-only mention, an ingredient with a strength, a combination drug,
a specific/unspecified diagnosis pair — but no organizer text, span or code is
reproduced, and nothing asserts a gold answer. What is asserted is that the governance
boundaries hold: a retrieval aid cannot emit a code, a closure node cannot become a
candidate, an advisory hierarchy edge cannot introduce one, and L8 cannot invent one.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mednorm_vi.confidence_cascade.cascade import CascadeDecision
from mednorm_vi.kb.competition.crosswalk import (
    CONCEPT_DEFINING_TTYS,
    normalize_crosswalk,
    read_crosswalk_evidence,
    retrieval_expansion_names,
)
from mednorm_vi.kb.competition.eligibility import (
    ELIGIBLE,
    NOT_FINAL_ELIGIBLE,
    NOT_IN_SNAPSHOT,
    ONTOLOGY_RXNORM,
    load_candidate_eligibility,
)
from mednorm_vi.kb.competition.icd10_view import (
    ADVISORY_HIERARCHY_WEIGHT,
    EXACT_MATCH_WEIGHT,
    HIERARCHY_ADVISORY,
    NAME_SUSPECT,
    advisory_hierarchy_may_offer,
    build_icd10_competition_view,
    is_structurally_valid_code,
    name_quality_penalty,
)
from mednorm_vi.kb.competition.policy import (
    CLINICALLY_APPROVED,
    COMPETITION_RUNTIME_ELIGIBLE,
    RETRIEVAL_ONLY,
    ClinicalApprovalForbidden,
    assert_automated_decision,
    may_emit_final_candidate,
)
from mednorm_vi.kb.competition.rxnorm_view import (
    CONCEPT_CLOSURE_ONLY,
    CONCEPT_SEARCHABLE,
    build_rxnorm_competition_view,
    concept_may_emit_final_candidate,
)
from mednorm_vi.kb.competition.topk import (
    DROP_BEYOND_TOPK,
    DROP_NOT_CLOSE,
    KEEP_CLOSE_TIE,
    KEEP_SPECIFICITY_PAIR,
    MAX_CANDIDATES,
    RankedCandidate,
    apply_competition_topk,
    is_specific_unspecified_pair,
)
from mednorm_vi.kb.indexing.retrieval import LocalIndex
from mednorm_vi.linking.models import LinkedCandidate, LinkerResult
from mednorm_vi.linking.rxnorm import (
    TIER_INGREDIENT_ONLY,
    TIER_STRUCTURED_EXACT,
    link_rxnorm,
    link_rxnorm_structured,
)
from mednorm_vi.metric_decoder import decode_entities
from mednorm_vi.metric_decoder.decoder import DROP_COMPETITION_TOPK
from mednorm_vi.resolution.models import (
    STATUS_ACCEPTED,
    BoundaryEvidence,
    EntityHypothesis,
    TypeEvidence,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "kb" / "competition"
CROSSWALK_V2 = (
    REPO
    / "data"
    / "derived"
    / "kb_freeze"
    / "0057_candidate_v2"
    / "rxnorm_prescribable_2026_07_06"
    / "inn_rxnorm_crosswalk_v1.csv"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hypothesis(
    text: str,
    entity_type: str = "THUỐC",
    *,
    hypothesis_id: str = "h1",
    components: tuple[dict[str, object], ...] = (),
) -> EntityHypothesis:
    return EntityHypothesis(
        hypothesis_id=hypothesis_id,
        document_id="d1",
        start=0,
        end=len(text),
        text=text,
        entity_type=entity_type,
        status=STATUS_ACCEPTED,
        chosen_proposal_id="p1",
        source_proposal_ids=("p1",),
        boundary_evidence=BoundaryEvidence("policy", "full", "p1", ()),
        type_evidence=TypeEvidence(entity_type, "E1", (entity_type,)),
        components=components,
        score=0.9,
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


def _competition_rxnorm_index() -> LocalIndex:
    """A miniature RxNorm graph shaped like the locked snapshot.

    One ingredient with two strengths, a branded form, a combination product and a
    grouper, so strength/dose-form/combination behaviour can be exercised without
    loading 82,429 concepts.
    """
    records = {
        "IN9": {
            "concept_id": "IN9",
            "canonical_name": "amlodipol",
            "aliases": ["amlodipolum"],
            "metadata": {"tty": "IN", "suppress": "N"},
        },
        "PIN9": {
            "concept_id": "PIN9",
            "canonical_name": "amlodipol besylate",
            "aliases": [],
            "metadata": {"tty": "PIN", "suppress": "N"},
        },
        "SCDC5": {
            "concept_id": "SCDC5",
            "canonical_name": "amlodipol 5 MG",
            "aliases": [],
            "metadata": {"tty": "SCDC", "suppress": "N"},
        },
        "SCDC10": {
            "concept_id": "SCDC10",
            "canonical_name": "amlodipol 10 MG",
            "aliases": [],
            "metadata": {"tty": "SCDC", "suppress": "N"},
        },
        "SCD5": {
            "concept_id": "SCD5",
            "canonical_name": "amlodipol 5 MG Oral Tablet",
            "aliases": [],
            "metadata": {"tty": "SCD", "suppress": "N"},
        },
        "SCD10": {
            "concept_id": "SCD10",
            "canonical_name": "amlodipol 10 MG Oral Tablet",
            "aliases": [],
            "metadata": {"tty": "SCD", "suppress": "N"},
        },
        "SBD10": {
            "concept_id": "SBD10",
            "canonical_name": "amlodipol 10 MG Oral Tablet [Brandol]",
            "aliases": [],
            "metadata": {"tty": "SBD", "suppress": "N"},
        },
        "INC1": {
            "concept_id": "INC1",
            "canonical_name": "sulfamethoxazin",
            "aliases": [],
            "metadata": {"tty": "IN", "suppress": "N"},
        },
        "INC2": {
            "concept_id": "INC2",
            "canonical_name": "trimethoprix",
            "aliases": [],
            "metadata": {"tty": "IN", "suppress": "N"},
        },
        "MIN1": {
            "concept_id": "MIN1",
            "canonical_name": "sulfamethoxazin / trimethoprix",
            "aliases": ["cotrimox"],
            "metadata": {"tty": "MIN", "suppress": "N"},
        },
        "SCDG9": {
            "concept_id": "SCDG9",
            "canonical_name": "amlodipol Oral Product",
            "aliases": [],
            "metadata": {"tty": "SCDG", "suppress": "N"},
        },
    }
    graph = {
        "IN9": ["SCDC5", "SCDC10", "SCDG9", "PIN9"],
        "PIN9": ["IN9"],
        "SCDC5": ["IN9", "SCD5"],
        "SCDC10": ["IN9", "SCD10"],
        "SCD5": ["SCDC5"],
        "SCD10": ["SCDC10", "SBD10"],
        "SBD10": ["SCD10"],
        "SCDG9": ["IN9"],
        "INC1": ["MIN1"],
        "INC2": ["MIN1"],
        "MIN1": ["INC1", "INC2"],
    }
    exact = {
        "amlodipol": ["IN9"],
        "amlodipolum": ["IN9"],
        "cotrimox": ["MIN1"],
        "sulfamethoxazin / trimethoprix": ["MIN1"],
    }
    return LocalIndex(
        index_type="rxnorm",
        source_snapshot_id="competition-mini-fixture",
        records=records,
        exact=exact,
        exact_ascii=dict(exact),
        ngrams={},
        sparse_terms={
            "amlodipol": ["IN9", "PIN9", "SCDC5", "SCDC10", "SCD5", "SCD10", "SBD10", "SCDG9"],
            "cotrimox": ["MIN1"],
        },
        graph=graph,
    )


def _crosswalk_rows() -> list[dict[str, str]]:
    with CROSSWALK_V2.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# A. Governance policy: no automated path may claim clinical approval
# ---------------------------------------------------------------------------


def test_an_automated_builder_can_never_record_clinical_approval() -> None:
    with pytest.raises(ClinicalApprovalForbidden):
        assert_automated_decision(CLINICALLY_APPROVED)
    for allowed in ("RETRIEVAL_ONLY", "EXCLUDED", "UNMAPPED"):
        assert assert_automated_decision(allowed) == allowed
    with pytest.raises(ValueError):
        assert_automated_decision("APPROVED_BY_VIBES")


def test_only_approved_or_eligible_roles_may_emit_a_final_candidate() -> None:
    assert may_emit_final_candidate(COMPETITION_RUNTIME_ELIGIBLE)
    assert may_emit_final_candidate(CLINICALLY_APPROVED)
    assert not may_emit_final_candidate(RETRIEVAL_ONLY)
    assert not may_emit_final_candidate("EXCLUDED")
    assert not may_emit_final_candidate("UNMAPPED")


# ---------------------------------------------------------------------------
# B. Canonical crosswalk normalization (Audit 0058 §3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_46_atom_rows_collapse_to_12_canonical_bridges() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    assert adjudication.source_row_count == 46
    assert adjudication.bridge_count == 12
    assert adjudication.unique_normalized_surfaces == 12
    assert adjudication.unique_rxcuis == 11


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_repeated_casing_and_tty_atoms_do_not_create_duplicate_bridges() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    keys = [bridge.key for bridge in adjudication.bridges]
    assert len(keys) == len(set(keys)), "the (surface, rxcui) pair must be unique"

    adrenaline = adjudication.bridge_for("adrenaline")
    assert len(adrenaline) == 1, "six evidence atoms describe one bridge"
    bridge = adrenaline[0]
    assert bridge.supporting_atom_count == 6
    # The evidence really did differ only in casing and term type.
    assert len({item.target_name for item in bridge.evidence}) > 1
    assert bridge.target_rxcui == "3992"


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_two_local_surfaces_may_point_to_one_rxcui_as_separate_bridges() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    sharing = sorted(
        bridge.normalized_local_surface
        for bridge in adjudication.bridges
        if bridge.target_rxcui == "10831"
    )
    assert sharing == ["co-trimoxazole", "trimethoprim-sulfamethoxazole"]
    for bridge in adjudication.bridges:
        if bridge.target_rxcui == "10831":
            assert "rxcui_shared_with_another_surface" in bridge.ambiguity_flags


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_observed_tty_is_a_collection_not_a_fabricated_concept_type() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    bridge = adjudication.bridge_for("adrenaline")[0]
    assert bridge.observed_tty == ("IN", "SU", "TMSY")
    assert all("|" not in tty for tty in bridge.observed_tty)
    # The row encoding may join them for storage; the semantics never do.
    row = bridge.as_row(snapshot_id="s")
    assert row["observed_tty"] == "IN|SU|TMSY"
    assert CONCEPT_DEFINING_TTYS & set(bridge.observed_tty)


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_provenance_and_source_evidence_are_retained_per_bridge() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    for bridge in adjudication.bridges:
        assert bridge.supporting_provenance_ids, bridge.normalized_local_surface
        assert bridge.source_evidence, bridge.normalized_local_surface
        assert bridge.supporting_atom_count == len(bridge.evidence)
        assert bridge.preferred_target_name


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_every_bridge_is_retrieval_only_and_none_is_clinically_approved() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    assert adjudication.decision_counts == {RETRIEVAL_ONLY: 12}
    assert CLINICALLY_APPROVED not in adjudication.decision_counts
    assert not any(bridge.may_emit_final_candidate for bridge in adjudication.bridges)


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_a_retrieval_only_bridge_expands_retrieval_but_emits_no_code() -> None:
    adjudication = normalize_crosswalk(_crosswalk_rows())
    # It *does* widen lookup: this is the whole point of keeping the bridge.
    assert retrieval_expansion_names(adjudication, "paracetamol") == ("ACETAMINOPHEN",)
    assert retrieval_expansion_names(adjudication, "PARACETAMOL") == ("ACETAMINOPHEN",)
    # And it returns names, never codes, so a caller cannot mistake one for a candidate.
    for name in retrieval_expansion_names(adjudication, "paracetamol"):
        assert not name.isdigit()
    bridge = adjudication.bridge_for("paracetamol")[0]
    assert bridge.target_rxcui == "161"
    assert not bridge.may_emit_final_candidate


@pytest.mark.skipif(not CROSSWALK_V2.is_file(), reason="0057 candidate v2 not present locally")
def test_canonical_bridge_hashes_are_identical_across_rebuilds() -> None:
    rows = _crosswalk_rows()
    first = normalize_crosswalk(rows)
    second = normalize_crosswalk(list(reversed(rows)))
    assert first.canonical_content_sha256 == second.canonical_content_sha256, (
        "grouping must not depend on source row order"
    )
    assert [b.key for b in first.bridges] == [b.key for b in second.bridges]


def test_a_surface_reaching_two_rxcuis_keeps_both_bridges_and_is_flagged() -> None:
    rows = [
        {
            "local_inn_surface": "ambiguol",
            "normalized_local_surface": "ambiguol",
            "rxnorm_name": "ambiguol",
            "rxcui": "111",
            "rxnorm_canonical_name": "Ambiguol A",
            "rxnorm_tty": "IN",
            "provenance_ids": "p1",
            "evidence_source": "fixture",
        },
        {
            "local_inn_surface": "ambiguol",
            "normalized_local_surface": "ambiguol",
            "rxnorm_name": "ambiguol",
            "rxcui": "222",
            "rxnorm_canonical_name": "Ambiguol B",
            "rxnorm_tty": "IN",
            "provenance_ids": "p2",
            "evidence_source": "fixture",
        },
    ]
    adjudication = normalize_crosswalk(rows)
    assert adjudication.bridge_count == 2, "competing targets are preserved, not resolved"
    for bridge in adjudication.bridges:
        assert "surface_maps_to_multiple_rxcuis" in bridge.ambiguity_flags
        assert bridge.competing_rxcuis
        assert bridge.automated_decision == RETRIEVAL_ONLY


def test_a_surface_with_no_target_is_unmapped_not_invented() -> None:
    adjudication = normalize_crosswalk(
        [
            {
                "local_inn_surface": "nowhereol",
                "normalized_local_surface": "nowhereol",
                "rxnorm_name": "nowhereol",
                "rxcui": "",
                "rxnorm_canonical_name": "",
                "rxnorm_tty": "",
                "provenance_ids": "p1",
                "evidence_source": "fixture",
            }
        ]
    )
    bridge = adjudication.bridges[0]
    assert bridge.automated_decision == "UNMAPPED"
    assert not bridge.may_emit_final_candidate


def test_crosswalk_evidence_parses_pipe_joined_term_types() -> None:
    evidence = read_crosswalk_evidence(
        [
            {
                "local_inn_surface": "x",
                "normalized_local_surface": "x",
                "rxcui": "1",
                "rxnorm_tty": "IN|SU|TMSY",
                "rxnorm_canonical_name": "X",
                "rxnorm_name": "x",
                "provenance_ids": "p",
                "evidence_source": "f",
            }
        ]
    )
    assert evidence[0].observed_tty == ("IN", "SU", "TMSY")
    assert evidence[0].names_concept


# ---------------------------------------------------------------------------
# C. ICD competition policy (Audit 0058 §4)
# ---------------------------------------------------------------------------


def _icd_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    concepts = [
        {
            "code": "A00",
            "display_code": "A00",
            "lookup_code": "a00",
            "original_source_name": "Bệnh tả",
            "quality_flags": "",
            "provenance_id": "p1",
        },
        {
            "code": "A000",
            "display_code": "A00.0",
            "lookup_code": "a000",
            "original_source_name": "+ bệnh kẽ ống",
            "quality_flags": "suspect_pdf_fragment:leading_marker",
            "provenance_id": "p2",
        },
        {
            "code": "NOTACODE!",
            "display_code": "NOTACODE!",
            "lookup_code": "notacode!",
            "original_source_name": "structurally invalid",
            "quality_flags": "",
            "provenance_id": "p3",
        },
    ]
    edges = [
        {
            "parent_code": "A00",
            "child_code": "A000",
            "source_evidence": "legacy_code_prefix_inference",
            "provenance_id": "e1",
        },
        {
            "parent_code": "A00",
            "child_code": "NOTACODE!",
            "source_evidence": "legacy_code_prefix_inference",
            "provenance_id": "e2",
        },
    ]
    return concepts, edges


def test_a_suspect_name_stays_searchable_and_is_only_penalised() -> None:
    concepts, edges = _icd_rows()
    view = build_icd10_competition_view(concepts, edges)
    suspect = next(c for c in view.concepts if c.code == "A000")
    assert suspect.name_quality == NAME_SUSPECT
    assert suspect.searchable, "a fragment label does not make the code wrong"
    assert suspect.runtime_role == COMPETITION_RUNTIME_ELIGIBLE
    assert suspect.ranking_penalty > 0
    clean = next(c for c in view.concepts if c.code == "A00")
    assert clean.ranking_penalty == 0.0
    # The name is preserved exactly, never repaired.
    assert suspect.source_name == "+ bệnh kẽ ống"


def test_structural_validity_not_name_quality_decides_searchability() -> None:
    concepts, edges = _icd_rows()
    view = build_icd10_competition_view(concepts, edges)
    invalid = next(c for c in view.concepts if c.code == "NOTACODE!")
    assert not invalid.structurally_valid
    assert not invalid.searchable
    assert invalid.exclusion_reason == "code_does_not_match_icd10_structure"
    assert is_structurally_valid_code("A00") and is_structurally_valid_code("A001")
    assert not is_structurally_valid_code("00A") and not is_structurally_valid_code("")


def test_advisory_icd_hierarchy_cannot_independently_offer_a_candidate() -> None:
    concepts, edges = _icd_rows()
    view = build_icd10_competition_view(concepts, edges)
    assert view.counts["hierarchy_may_offer_candidate"] == 0
    assert view.counts["hierarchy_source_supported"] == 0
    for edge in view.edges:
        assert edge.evidence_class == HIERARCHY_ADVISORY
        assert not edge.source_authoritative
        assert not edge.may_offer_candidate
        assert edge.ranking_weight == ADVISORY_HIERARCHY_WEIGHT
    assert not advisory_hierarchy_may_offer(HIERARCHY_ADVISORY)


def test_exact_lexical_evidence_dominates_advisory_hierarchy() -> None:
    assert EXACT_MATCH_WEIGHT > ADVISORY_HIERARCHY_WEIGHT * 100
    # A suspect name is penalised far less than an exact match is worth, so an exact
    # hit on a fragment-named code still beats a weak hit on a clean one.
    assert name_quality_penalty(NAME_SUSPECT) < EXACT_MATCH_WEIGHT


def test_an_edge_to_a_non_searchable_endpoint_is_not_carried() -> None:
    concepts, edges = _icd_rows()
    view = build_icd10_competition_view(concepts, edges)
    assert all(edge.child_code != "NOTACODE!" for edge in view.edges)
    assert view.counts["hierarchy_edges"] == 1


# ---------------------------------------------------------------------------
# D. RxNorm searchable/closure policy and endpoint closure (§5, §6)
# ---------------------------------------------------------------------------


def test_atom_level_relations_are_resolved_not_counted_as_missing(tmp_path: Path) -> None:
    """The Audit-0057 correction: an AUI-level row is not an endpoint-less relation."""
    report = build_rxnorm_competition_view(
        prescribable_rrf=FIXTURES / "prescribable",
        full_rrf=FIXTURES / "full",
        output_dir=tmp_path,
        snapshot_id="fixture",
    )
    assert report.prescribable_aui_level_rows > 0
    assert report.prescribable_aui_unresolvable == 0
    assert report.prescribable_missing_endpoint_cui_check > 0, "the naive check flags them"
    assert report.prescribable_missing_endpoint_after_resolution == 0, "resolution clears them"


def test_every_retained_relation_has_both_endpoints_after_closure(tmp_path: Path) -> None:
    report = build_rxnorm_competition_view(
        prescribable_rrf=FIXTURES / "prescribable",
        full_rrf=FIXTURES / "full",
        output_dir=tmp_path,
        snapshot_id="fixture",
    )
    assert report.missing_endpoints_after_closure == 0
    concepts = {
        row["rxcui"]
        for row in csv.DictReader(
            (tmp_path / "rxnorm_competition_concepts_v1.csv").open(encoding="utf-8")
        )
    }
    for row in csv.DictReader(
        (tmp_path / "rxnorm_competition_relations_v1.csv").open(encoding="utf-8")
    ):
        assert row["source_rxcui"] in concepts
        assert row["target_rxcui"] in concepts
        assert row["endpoint_status"] == "ok"
        assert row["direction"] == "source_to_target"
        assert row["rela"], "relation labels must survive"


def test_closure_only_concepts_are_traversable_but_never_final_candidates(
    tmp_path: Path,
) -> None:
    build_rxnorm_competition_view(
        prescribable_rrf=FIXTURES / "prescribable",
        full_rrf=FIXTURES / "full",
        output_dir=tmp_path,
        snapshot_id="fixture",
    )
    rows = list(
        csv.DictReader((tmp_path / "rxnorm_competition_concepts_v1.csv").open(encoding="utf-8"))
    )
    closure = [r for r in rows if r["membership"] == CONCEPT_CLOSURE_ONLY]
    assert closure, "the fixture must exercise a real closure node"
    for row in closure:
        assert row["searchable"] == "false"
        assert row["graph_endpoint"] == "true", "separate flags, not one collapsed flag"
        assert row["runtime_role"] == RETRIEVAL_ONLY
        assert not concept_may_emit_final_candidate(row["membership"])
    assert all(concept_may_emit_final_candidate(CONCEPT_SEARCHABLE) for _ in [0])


def test_a_relation_with_no_searchable_endpoint_is_excluded(tmp_path: Path) -> None:
    report = build_rxnorm_competition_view(
        prescribable_rrf=FIXTURES / "prescribable",
        full_rrf=FIXTURES / "full",
        output_dir=tmp_path,
        snapshot_id="fixture",
    )
    assert report.excluded_no_searchable_endpoint > 0
    concepts = {
        row["rxcui"]
        for row in csv.DictReader(
            (tmp_path / "rxnorm_competition_concepts_v1.csv").open(encoding="utf-8")
        )
    }
    # 900002/900003 relate only to each other, so neither may enter the closure.
    assert "900002" not in concepts
    assert "900003" not in concepts


def test_non_structure_relations_are_excluded_with_a_recorded_count(tmp_path: Path) -> None:
    report = build_rxnorm_competition_view(
        prescribable_rrf=FIXTURES / "prescribable",
        full_rrf=FIXTURES / "full",
        output_dir=tmp_path,
        snapshot_id="fixture",
    )
    assert report.excluded_non_structure_rows > 0
    labels = set(report.relation_label_distribution)
    assert "has_inactive_ingredient" not in labels
    assert {"has_ingredient", "tradename_of"} <= labels


def test_in_pin_min_scd_sbd_semantics_are_not_collapsed(tmp_path: Path) -> None:
    build_rxnorm_competition_view(
        prescribable_rrf=FIXTURES / "prescribable",
        full_rrf=FIXTURES / "full",
        output_dir=tmp_path,
        snapshot_id="fixture",
    )
    by_cui = {
        row["rxcui"]: row
        for row in csv.DictReader(
            (tmp_path / "rxnorm_competition_concepts_v1.csv").open(encoding="utf-8")
        )
    }
    assert by_cui["200001"]["tty_values"] == "IN"
    assert by_cui["200002"]["tty_values"] == "SCD"
    assert by_cui["200004"]["tty_values"] == "SBD"
    assert by_cui["200006"]["tty_values"] == "MIN"
    assert by_cui["900001"]["tty_values"] == "PIN"


def test_the_rxnorm_view_rebuilds_identically(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for target in (first, second):
        build_rxnorm_competition_view(
            prescribable_rrf=FIXTURES / "prescribable",
            full_rrf=FIXTURES / "full",
            output_dir=target,
            snapshot_id="fixture",
        )
    for name in ("rxnorm_competition_concepts_v1.csv", "rxnorm_competition_relations_v1.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


# ---------------------------------------------------------------------------
# E. Final-candidate eligibility gate
# ---------------------------------------------------------------------------


def _eligibility_csv(path: Path) -> Path:
    fields = [
        "schema_version",
        "ontology",
        "code",
        "snapshot_id",
        "canonical_display_name",
        "source_concept_type",
        "runtime_role",
        "final_candidate_eligible",
        "deterministic_ordering_key",
        "exclusion_reason",
    ]
    rows = [
        ("rxnorm", "200002", COMPETITION_RUNTIME_ELIGIBLE, "true", ""),
        ("rxnorm", "900001", RETRIEVAL_ONLY, "false", "closure_only_not_searchable"),
        ("icd10_vi", "A00", COMPETITION_RUNTIME_ELIGIBLE, "true", ""),
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ontology, code, role, eligible, reason in rows:
            writer.writerow(
                {
                    "schema_version": "candidate-representation-schema-v1",
                    "ontology": ontology,
                    "code": code,
                    "snapshot_id": "fixture-v3",
                    "canonical_display_name": code,
                    "source_concept_type": "",
                    "runtime_role": role,
                    "final_candidate_eligible": eligible,
                    "deterministic_ordering_key": f"{ontology}:{code}",
                    "exclusion_reason": reason,
                }
            )
    return path


def test_the_eligibility_gate_refuses_a_closure_only_code(tmp_path: Path) -> None:
    gate = load_candidate_eligibility(_eligibility_csv(tmp_path / "index.csv"))
    assert gate.may_emit(ONTOLOGY_RXNORM, "200002")
    assert not gate.may_emit(ONTOLOGY_RXNORM, "900001")
    assert gate.reason_for(ONTOLOGY_RXNORM, "900001") == NOT_FINAL_ELIGIBLE
    assert gate.reason_for(ONTOLOGY_RXNORM, "404404") == NOT_IN_SNAPSHOT
    assert gate.reason_for(ONTOLOGY_RXNORM, "200002") == ELIGIBLE
    assert gate.runtime_role(ONTOLOGY_RXNORM, "900001") == RETRIEVAL_ONLY


def test_the_eligibility_gate_preserves_order_and_reports_refusals(tmp_path: Path) -> None:
    gate = load_candidate_eligibility(_eligibility_csv(tmp_path / "index.csv"))
    kept, refused = gate.filter_candidates(ONTOLOGY_RXNORM, ["200002", "900001", "404404"])
    assert kept == ("200002",)
    assert refused == (("900001", NOT_FINAL_ELIGIBLE), ("404404", NOT_IN_SNAPSHOT))


# ---------------------------------------------------------------------------
# F. Medication linking behaviour (§11)
# ---------------------------------------------------------------------------


def test_an_ingredient_only_mention_falls_back_to_the_ingredient_concept() -> None:
    index = _competition_rxnorm_index()
    hypothesis = _hypothesis("amlodipol", components=(_component("name", "amlodipol"),))
    report = link_rxnorm_structured(hypothesis, index)
    retained = report.retained
    assert retained, "an ingredient-only mention must still link"
    assert retained[0].concept_id == "IN9"
    assert retained[0].tier == TIER_INGREDIENT_ONLY
    assert report.structured.strength_value is None


def test_an_ingredient_plus_strength_prefers_the_matching_product() -> None:
    index = _competition_rxnorm_index()
    hypothesis = _hypothesis(
        "amlodipol 10 mg",
        components=(
            _component("name", "amlodipol"),
            _component("strength_value", "10", 10),
            _component("strength_unit", "mg", 13),
        ),
    )
    report = link_rxnorm_structured(hypothesis, index)
    retained = report.retained
    assert retained[0].tier == TIER_STRUCTURED_EXACT
    assert retained[0].concept_id in {"SCD10", "SCDC10", "SBD10"}
    kept = {d.concept_id for d in retained}
    assert "SCD5" not in kept, "the 5 MG product conflicts on strength"


def test_the_same_ingredient_at_two_strengths_ranks_different_concepts() -> None:
    index = _competition_rxnorm_index()

    def best(value: str) -> str:
        hypothesis = _hypothesis(
            f"amlodipol {value} mg",
            components=(
                _component("name", "amlodipol"),
                _component("strength_value", value, 10),
                _component("strength_unit", "mg", 13),
            ),
        )
        return link_rxnorm_structured(hypothesis, index).retained[0].concept_id

    assert best("5") != best("10"), "strength must change the answer"
    assert best("5") in {"SCD5", "SCDC5"}
    assert best("10") in {"SCD10", "SCDC10", "SBD10"}


def test_a_dose_form_conflict_suppresses_an_otherwise_matching_product() -> None:
    index = _competition_rxnorm_index()
    hypothesis = _hypothesis(
        "amlodipol 10 mg vien nang",
        components=(
            _component("name", "amlodipol"),
            _component("strength_value", "10", 10),
            _component("strength_unit", "mg", 13),
            _component("dose_form", "vien nang", 16),
        ),
    )
    report = link_rxnorm_structured(hypothesis, index)
    suppressed = {d.concept_id for d in report.suppressed}
    # Every product in this fixture is an Oral Tablet, so a capsule mention conflicts.
    assert "SCD10" in suppressed


def test_a_combination_mention_is_not_reduced_to_one_ingredient() -> None:
    index = _competition_rxnorm_index()
    hypothesis = _hypothesis("cotrimox", components=(_component("name", "cotrimox"),))
    report = link_rxnorm_structured(hypothesis, index)
    kept = [d.concept_id for d in report.retained]
    assert "MIN1" in kept, "the combination concept must be reachable"
    combination = next(d for d in report.retained if d.concept_id == "MIN1")
    assert combination.is_combination, "two attached ingredients must be visible"
    assert "INC1" not in kept and "INC2" not in kept, (
        "a combination must not silently collapse to one of its ingredients"
    )


def test_a_grouper_tty_is_evidence_and_never_a_candidate() -> None:
    index = _competition_rxnorm_index()
    hypothesis = _hypothesis("amlodipol", components=(_component("name", "amlodipol"),))
    report = link_rxnorm_structured(hypothesis, index)
    assert "SCDG9" not in {d.concept_id for d in report.retained}


def test_medication_candidate_ordering_is_deterministic() -> None:
    index = _competition_rxnorm_index()
    hypothesis = _hypothesis(
        "amlodipol 10 mg",
        components=(
            _component("name", "amlodipol"),
            _component("strength_value", "10", 10),
            _component("strength_unit", "mg", 13),
        ),
    )
    runs = [tuple(c.code for c in link_rxnorm(hypothesis, index).candidates) for _ in range(5)]
    assert len(set(runs)) == 1, f"ordering drifted across runs: {set(runs)}"


# ---------------------------------------------------------------------------
# G. Competition top-k policy (§7)
# ---------------------------------------------------------------------------


def test_medication_defaults_to_top_1() -> None:
    selection = apply_competition_topk(
        "THUỐC",
        (
            RankedCandidate("A", 10.0, "structured_exact"),
            RankedCandidate("B", 8.0, "structured_exact"),
        ),
    )
    assert selection.codes == ("A",)
    assert any(d.reason == DROP_NOT_CLOSE for d in selection.decisions)


def test_medication_emits_a_second_candidate_only_on_a_near_exact_tie() -> None:
    selection = apply_competition_topk(
        "THUỐC",
        (
            RankedCandidate("A", 10.0, "structured_exact"),
            RankedCandidate("B", 9.95, "structured_exact"),
        ),
    )
    assert selection.codes == ("A", "B")
    assert any(d.reason == KEEP_CLOSE_TIE for d in selection.decisions)


def test_diagnosis_emits_a_second_candidate_on_a_supported_specificity_pair() -> None:
    selection = apply_competition_topk(
        "CHẨN_ĐOÁN",
        (
            RankedCandidate("A09", 10.0, "supported_specific"),
            RankedCandidate("A090", 8.5, "supported_specific"),
        ),
    )
    assert selection.codes == ("A09", "A090")
    assert any(d.reason == KEEP_SPECIFICITY_PAIR for d in selection.decisions)
    assert is_specific_unspecified_pair("A09", "A09.0")


def test_an_unrelated_low_confidence_diagnosis_candidate_is_suppressed() -> None:
    selection = apply_competition_topk(
        "CHẨN_ĐOÁN",
        (
            RankedCandidate("A09", 10.0, "supported_specific"),
            RankedCandidate("J189", 8.5, "supported_specific"),
        ),
    )
    assert selection.codes == ("A09",), "a distant code is not a specificity pair"


def test_a_weaker_tier_runner_up_is_never_a_tie() -> None:
    selection = apply_competition_topk(
        "THUỐC",
        (
            RankedCandidate("A", 10.0, "structured_exact"),
            RankedCandidate("B", 10.0, "lexical"),
        ),
    )
    assert selection.codes == ("A",)


def test_types_that_take_no_candidates_receive_none() -> None:
    for label in ("TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
        selection = apply_competition_topk(label, (RankedCandidate("X", 10.0, "lexical"),))
        assert selection.codes == ()


def test_top_k_never_exceeds_the_ceiling() -> None:
    many = tuple(RankedCandidate(f"C{i}", 10.0, "supported_specific") for i in range(10))
    selection = apply_competition_topk("CHẨN_ĐOÁN", many)
    assert len(selection.codes) <= MAX_CANDIDATES
    assert any(d.reason == DROP_BEYOND_TOPK for d in selection.decisions)


def test_top_k_is_a_subsequence_and_can_never_introduce_a_code() -> None:
    offered = (
        RankedCandidate("A", 10.0, "supported_specific"),
        RankedCandidate("B", 9.9, "supported_specific"),
        RankedCandidate("C", 1.0, "supported_specific"),
    )
    selection = apply_competition_topk("CHẨN_ĐOÁN", offered)
    assert set(selection.codes) <= {c.code for c in offered}
    order = [c.code for c in offered]
    assert list(selection.codes) == [c for c in order if c in set(selection.codes)]


# ---------------------------------------------------------------------------
# H. L8/L9 invariants under the competition policy (§7, §10)
# ---------------------------------------------------------------------------


def _decode(codes: tuple[tuple[str, float], ...], entity_type: str, **kwargs: object) -> object:
    hypothesis = _hypothesis("x", entity_type, hypothesis_id="m1")
    link = LinkerResult(
        "m1",
        tuple(LinkedCandidate(code, score, ("sparse",), "snap") for code, score in codes),
    )
    return decode_entities(
        (hypothesis,), (CascadeDecision("m1", True, 0.9, ()),), (), (link,), **kwargs
    )


def test_l8_cannot_introduce_a_code_absent_from_the_offered_set() -> None:
    offered = (("111", 10.0), ("222", 9.99), ("333", 9.98))
    decoded = _decode(offered, "CHẨN_ĐOÁN")
    emitted = set(decoded[0].candidates)  # type: ignore[index]
    assert emitted <= {code for code, _ in offered}


def test_the_competition_topk_narrows_l8_and_records_why() -> None:
    offered = (("111", 10.0), ("222", 9.0), ("333", 8.9))
    narrowed = _decode(offered, "THUỐC")
    wide = _decode(offered, "THUỐC", competition_topk=False)
    assert len(narrowed[0].candidates) < len(wide[0].candidates)  # type: ignore[index]
    assert len(narrowed[0].candidates) <= MAX_CANDIDATES  # type: ignore[index]
    assert any(
        d.reason == DROP_COMPETITION_TOPK
        for d in narrowed[0].candidate_decisions  # type: ignore[index]
    )
    # Opting out reproduces the Audit-0054 behaviour exactly.
    assert set(narrowed[0].candidates) <= set(wide[0].candidates)  # type: ignore[index]


def test_decoding_is_deterministic_across_repeated_runs() -> None:
    offered = (("111", 10.0), ("222", 9.99), ("333", 8.0))
    runs = {tuple(_decode(offered, "CHẨN_ĐOÁN")[0].candidates) for _ in range(5)}  # type: ignore[index]
    assert len(runs) == 1


def test_l9_offered_set_enforcement_is_unchanged() -> None:
    """L9 must still fail a code that exists in the snapshot but was never offered."""
    from mednorm_vi.validator.kb_membership import (
        LockedSnapshots,
        validate_document_candidates,
    )

    class _Snapshot:
        index_type = "rxnorm"
        source_snapshot_id = "snap"

        def exists(self, concept_id: str) -> bool:
            return concept_id in {"111", "222"}

    entities = [{"text": "x", "type": "THUỐC", "candidates": ["222"], "position": [0, 1]}]
    unoffered = validate_document_candidates(
        entities,
        LockedSnapshots(rxnorm=_Snapshot()),
        document_id="1.json",
        offered_codes_by_index={0: ["111"]},
    )
    assert not unoffered.ok, "a snapshot-valid but unoffered code must still fail"
    assert any("offered" in issue.code or "offered" in issue.message for issue in unoffered.issues)

    offered = validate_document_candidates(
        entities,
        LockedSnapshots(rxnorm=_Snapshot()),
        document_id="1.json",
        offered_codes_by_index={0: ["222"]},
    )
    assert offered.ok, "a code that was offered and exists must still pass"


# ---------------------------------------------------------------------------
# I. Built snapshot invariants (only when the local v3 snapshot exists)
# ---------------------------------------------------------------------------

V3 = REPO / "data" / "derived" / "kb_freeze" / "0058_competition_candidate_v3"


@pytest.mark.skipif(not V3.is_dir(), reason="competition candidate v3 not built locally")
def test_the_built_v3_snapshot_reports_a_valid_competition_candidate() -> None:
    report = json.loads((V3 / "competition_validation_report_v1.json").read_text(encoding="utf-8"))
    assert report["validation_status"] == "VALID_COMPETITION_CANDIDATE"
    assert report["violations"] == []
    assert report["crosswalk"]["source_rows"] == 46
    assert report["crosswalk"]["canonical_bridges"] == 12
    assert report["crosswalk"]["decision_counts"] == {RETRIEVAL_ONLY: 12}
    assert report["rxnorm_closure"]["missing_endpoints_after_closure"] == 0
    assert report["icd"]["hierarchy_may_offer_candidate"] == 0


@pytest.mark.skipif(not V3.is_dir(), reason="competition candidate v3 not built locally")
def test_the_built_v3_manifest_records_both_digests_per_artifact() -> None:
    manifest = json.loads(
        (V3 / "competition_snapshot_manifest_v1.json").read_text(encoding="utf-8")
    )
    assert manifest["validation_status"] == "VALID_COMPETITION_CANDIDATE"
    assert manifest["artifacts"]
    for artifact in manifest["artifacts"]:
        assert len(artifact["artifact_sha256"]) == 64
        assert len(artifact["canonical_content_sha256"]) == 64
        assert artifact["rows"] >= 0
    # v2 is input only and must never be named as an output location.
    assert "0057_candidate_v2" in manifest["predecessor_snapshot_dir"]


V3_RX_INDEX = REPO / "indices" / "candidate" / "rxnorm" / "competition-v3" / "index.json"


@pytest.mark.skipif(not V3_RX_INDEX.is_file(), reason="v3 runtime index not built locally")
def test_the_retrieval_only_bridge_closes_the_paracetamol_gap() -> None:
    """`paracetamol` is absent from RxNorm, so without expansion it links to nothing."""
    from mednorm_vi.kb.indexing.retrieval import load_index
    from mednorm_vi.linking.rxnorm_graph import is_closure_only

    index = load_index(V3_RX_INDEX)
    hypothesis = _hypothesis("paracetamol", components=(_component("name", "paracetamol"),))
    report = link_rxnorm_structured(hypothesis, index)
    assert report.retained, "the bridge must make an unlinkable INN surface linkable"
    assert report.retained[0].concept_id == "161", "acetaminophen is the correct concept"
    # The expansion is recorded, never hidden.
    assert any(note.startswith("inn_bridge:paracetamol") for note in report.traversal_notes)
    # And the bridge did not hand over a code: 161 was reached through the index.
    assert "exact" in report.retained[0].retrieval_sources
    assert not any(is_closure_only(index, d.concept_id) for d in report.retained)


@pytest.mark.skipif(not V3_RX_INDEX.is_file(), reason="v3 runtime index not built locally")
def test_no_closure_only_concept_is_retrievable_in_the_v3_index() -> None:
    from mednorm_vi.kb.indexing.retrieval import load_index, search_index
    from mednorm_vi.linking.rxnorm_graph import is_closure_only, name_of

    index = load_index(V3_RX_INDEX)
    closure = [cid for cid in list(index.records)[:20000] if is_closure_only(index, cid)][:10]
    assert closure, "the v3 index must contain closure-only records"
    for concept_id in closure:
        hits = search_index(index, name_of(index, concept_id), limit=25)
        reached = {hit.concept_id for hit in hits}
        assert concept_id not in reached, "a closure node must not be retrievable"


def test_l9_refuses_a_closure_only_code_even_if_the_linker_offered_it() -> None:
    """The gate must not trust the linker — that is the component whose bug it catches."""
    from mednorm_vi.validator.kb_membership import (
        LockedSnapshots,
        validate_document_candidates,
    )

    class _Snapshot:
        index_type = "rxnorm"
        source_snapshot_id = "competition-v3"
        records = {
            "200002": {"concept_id": "200002", "metadata": {"membership": CONCEPT_SEARCHABLE}},
            "900001": {"concept_id": "900001", "metadata": {"membership": CONCEPT_CLOSURE_ONLY}},
        }

        def exists(self, concept_id: str) -> bool:
            return concept_id in self.records

    def _check(code: str) -> bool:
        entities = [{"text": "x", "type": "THUỐC", "candidates": [code], "position": [0, 1]}]
        return validate_document_candidates(
            entities,
            LockedSnapshots(rxnorm=_Snapshot()),
            document_id="1.json",
            offered_codes_by_index={0: [code]},
        ).ok

    assert _check("200002"), "a searchable concept must pass"
    assert not _check("900001"), "a closure-only endpoint must fail even when offered"


def test_a_snapshot_that_states_no_membership_permits_everything() -> None:
    """Silence must not mean exclusion, or the legacy indices would emit nothing."""
    from mednorm_vi.validator.kb_membership import (
        LockedSnapshots,
        validate_document_candidates,
    )

    class _LegacySnapshot:
        index_type = "rxnorm"
        source_snapshot_id = "legacy"
        records = {"111": {"concept_id": "111", "metadata": {"tty": "IN"}}}

        def exists(self, concept_id: str) -> bool:
            return concept_id in self.records

    entities = [{"text": "x", "type": "THUỐC", "candidates": ["111"], "position": [0, 1]}]
    result = validate_document_candidates(
        entities,
        LockedSnapshots(rxnorm=_LegacySnapshot()),
        document_id="1.json",
        offered_codes_by_index={0: ["111"]},
    )
    assert result.ok


@pytest.mark.skipif(not V3.is_dir(), reason="competition candidate v3 not built locally")
def test_the_built_v3_candidate_index_refuses_closure_only_codes() -> None:
    gate = load_candidate_eligibility(V3 / "competition_candidate_index_v1.csv")
    closure = [
        row
        for row in csv.DictReader(
            (V3 / "rxnorm_competition_concepts_v1.csv").open(encoding="utf-8")
        )
        if row["membership"] == CONCEPT_CLOSURE_ONLY
    ][:50]
    assert closure, "the built snapshot must contain closure-only concepts"
    for row in closure:
        assert not gate.may_emit(ONTOLOGY_RXNORM, row["rxcui"])
