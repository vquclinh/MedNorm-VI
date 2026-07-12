"""has_result evidence preserved across result boundary alternatives (Area 2).

One logical name->result pairing emits one relation per result boundary
alternative (value-only and value+unit), all sharing a ``pair_group_id`` and
each targeting a member of the same result ``boundary_group_id``. So whichever
boundary L4 selects later, a valid has_result endpoint still exists. Logical
pairs are counted by distinct ``pair_group_id`` — not inflated by alternatives.
"""

from __future__ import annotations

from pathlib import Path

from _routing_helpers import forced_routings

from mednorm_vi.document_intelligence import analyze_text
from mednorm_vi.mention_factory.laboratory import load_lab_lexicon, parse_graph

REPO = Path(__file__).resolve().parents[2]
LEX = load_lab_lexicon(REPO / "configs" / "laboratory" / "parser_v1.yaml")


def _run(text: str):
    g = analyze_text(text)
    return parse_graph(g, forced_routings(g, ("C2",)), LEX, "lab-p1"), g


def test_value_only_and_value_plus_unit_alternatives_exist() -> None:
    res, _ = _run("Glucose: 7.8 mmol/L")
    by_id = {p.proposal_id: p for p in res.proposals}
    # The result parse exposes both boundary alternatives sharing a group id.
    results = [p for p in res.proposals if "KẾT_QUẢ_XÉT_NGHIỆM" in p.proposed_types]
    texts = {p.text for p in results}
    assert {"7.8", "7.8 mmol/L"} <= texts
    groups = {p.boundary_group_id for p in results}
    assert len(groups) == 1 and None not in groups  # one logical result parse

    # One relation per alternative, all in the same pair group.
    assert {r.pair_group_id for r in res.relations} == {res.relations[0].pair_group_id}
    targets = {by_id[r.target_proposal_id].text for r in res.relations}
    assert {"7.8", "7.8 mmol/L"} <= targets
    # Exactly one primary (the value-only alternative).
    primaries = [r for r in res.relations if r.is_primary]
    assert len(primaries) == 1
    assert by_id[primaries[0].target_proposal_id].text == "7.8"


def test_either_boundary_is_a_valid_endpoint() -> None:
    # Simulate L4 selecting the value+unit boundary later: a has_result relation
    # still targets that exact proposal, so the pairing survives the choice.
    res, _ = _run("Glucose: 7.8 mmol/L")
    by_id = {p.proposal_id: p for p in res.proposals}
    chosen = next(p for p in res.proposals if p.text == "7.8 mmol/L")
    assert any(r.target_proposal_id == chosen.proposal_id for r in res.relations)
    assert any(r.target_boundary_group_id == chosen.boundary_group_id
               for r in res.relations)
    # And every relation endpoint is a real proposal (no dangling references).
    for r in res.relations:
        assert r.source_proposal_id in by_id and r.target_proposal_id in by_id


def test_logical_pairs_not_inflated_by_alternatives() -> None:
    res, _ = _run("Glucose: 7.8 mmol/L")
    logical = {r.pair_group_id for r in res.relations}
    assert len(logical) == 1  # one logical pair
    assert len(res.relations) >= 2  # but multiple concrete alternatives


def test_no_cross_row_relation_propagation() -> None:
    # Two independent rows: a name never links to the other row's result group.
    res, _ = _run("Xét nghiệm:\nWBC: 14.43\n\nGlucose: 7.8 mmol/L")
    by_id = {p.proposal_id: p.text for p in res.proposals}
    pairs = {(by_id[r.source_proposal_id], by_id[r.target_proposal_id])
             for r in res.relations}
    assert not any(src == "WBC" and tgt.startswith("7.8") for src, tgt in pairs)
    assert not any(src == "Glucose" and tgt == "14.43" for src, tgt in pairs)


def _relation_key(res) -> list[tuple]:
    return [(r.relation_id, r.pair_group_id, r.is_primary,
             r.target_boundary_group_id, r.pairing_cost) for r in res.relations]


def test_deterministic_relation_serialization() -> None:
    a, _ = _run("Glucose: 7.8 mmol/L; WBC: 14.43")
    b, _ = _run("Glucose: 7.8 mmol/L; WBC: 14.43")
    assert _relation_key(a) == _relation_key(b)
