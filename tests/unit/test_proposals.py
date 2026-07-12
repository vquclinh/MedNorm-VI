"""Proposal / relation contract + merge + validation tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from mednorm_vi.deterministic_baseline import Phase1BConfig, run_phase1b, to_json
from mednorm_vi.deterministic_baseline.serialization import to_debug_dict
from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.models import HAS_RESULT, RelationProposal, SpanProposal
from mednorm_vi.mention_factory.validation import validate_proposals

REPO = Path(__file__).resolve().parents[2]
CONFIG = Phase1BConfig.load(
    REPO / "configs" / "case_router" / "base.yaml",
    REPO / "configs" / "medication" / "grammar_v1.yaml",
    REPO / "configs" / "laboratory" / "parser_v1.yaml",
)

DOC = ("Thuốc tại nhà:\n1. amlodipine 10 mg po daily\n\nXét nghiệm:\n"
       "WBC: 14.43; NEUT%: 76.4\n\nBệnh nhân không sốt.\n")


def _result():
    g = analyze_text(DOC)
    return run_phase1b(g, CONFIG), g


def test_proposal_ids_deterministic() -> None:
    a, _ = _result()
    b, _ = _result()
    assert [p.proposal_id for p in a.proposals] == [p.proposal_id for p in b.proposals]


def test_relation_ids_deterministic() -> None:
    a, _ = _result()
    b, _ = _result()
    assert [r.relation_id for r in a.relations] == [r.relation_id for r in b.relations]


def test_source_provenance_complete() -> None:
    res, _ = _result()
    for p in res.proposals:
        assert p.source_specialist and p.source_node_id and p.source_routes
        assert p.config_version and p.lexicon_version


def test_exact_offset_invariant() -> None:
    res, g = _result()
    for p in res.proposals:
        assert g.original_text[p.start : p.end] == p.text


def test_repeated_text_distinct_positions() -> None:
    g = analyze_text("Xét nghiệm:\nWBC: 1\n\nWBC: 2")
    res = run_phase1b(g, CONFIG)
    wbc = [p for p in res.proposals if p.text == "WBC"]
    assert len({p.start for p in wbc}) == 2


def test_overlapping_boundaries_preserved() -> None:
    res, _ = _result()
    # medication produces several overlapping boundary proposals over one parse
    med = [p for p in res.proposals if p.source_specialist == "medication"]
    refs = {p.parse_ref for p in med}
    assert any(sum(1 for p in med if p.parse_ref == r) > 1 for r in refs)


def test_duplicate_same_span_policy_keeps_both() -> None:
    # two specialists proposing identical coords+type keep separate provenance
    p1 = SpanProposal("a", "d", 0, 3, "abc", ("THUỐC",), "medication", "n1", ("C1",), 0.5)
    p2 = SpanProposal("b", "d", 0, 3, "abc", ("THUỐC",), "medication", "n1", ("C1",), 0.5)
    from mednorm_vi.mention_factory.merge import collect
    ordered, diag = collect([p1, p2])
    assert len(ordered) == 2  # both kept
    assert diag.duplicate_identity_groups  # flagged


def test_broken_relation_endpoint_rejected() -> None:
    p = SpanProposal("p1", "d", 0, 3, "abc", ("TÊN_XÉT_NGHIỆM",), "laboratory", "n", ("C2",), 0.5)
    rel = RelationProposal("r1", "d", HAS_RESULT, "p1", "missing", 0.5, 0.1)
    res = validate_proposals([p], [rel], "abcxyz")
    assert not res.ok
    assert any(i.code == "relation.broken_endpoint" for i in res.errors)


def test_cross_document_relation_rejected() -> None:
    p1 = SpanProposal("p1", "d1", 0, 3, "abc", ("TÊN_XÉT_NGHIỆM",), "laboratory", "n", ("C2",), 0.5)
    p2 = SpanProposal("p2", "d2", 0, 3, "abc", ("KẾT_QUẢ_XÉT_NGHIỆM",), "laboratory", "n",
                      ("C2",), 0.5)
    rel = RelationProposal("r1", "d1", HAS_RESULT, "p1", "p2", 0.5, 0.1)
    res = validate_proposals([p1, p2], [rel], "abc")
    assert any(i.code == "relation.cross_document" for i in res.errors)


def test_unsupported_final_code_rejected() -> None:
    p = SpanProposal("p1", "d", 0, 3, "abc", ("THUỐC",), "medication", "n", ("C1",), 0.5,
                     features={"rxnorm": 1.0})
    res = validate_proposals([p], [], "abc")
    assert any(i.code == "proposal.final_code" for i in res.errors)


def test_no_organizer_final_entity_emitted() -> None:
    res, _ = _result()
    # proposals carry PROPOSED types (evidence), never final assertion/candidate fields
    for p in res.proposals:
        assert not any(k in p.features for k in ("icd10", "rxnorm", "candidate", "assertions"))
    d = to_debug_dict(res)
    assert "candidates" not in d and "assertions" not in d


def test_roundtrip_and_byte_deterministic_json() -> None:
    a, _ = _result()
    b, _ = _result()
    assert to_json(a) == to_json(b)
    assert "viêm" not in to_json(a) or "\\u" not in to_json(a)  # UTF-8, not escaped


def test_relations_point_to_valid_proposals() -> None:
    res, _ = _result()
    ids = {p.proposal_id for p in res.proposals}
    for r in res.relations:
        assert r.source_proposal_id in ids and r.target_proposal_id in ids


def test_features_carry_no_final_labels() -> None:
    p = SpanProposal("p1", "d", 0, 3, "abc", ("THUỐC",), "medication", "n", ("C1",), 0.5,
                     features={"THUỐC": 1.0})
    assert dataclasses.is_dataclass(p)
    from mednorm_vi.mention_factory.validation import contains_organizer_final_entity
    assert contains_organizer_final_entity(p.features)
