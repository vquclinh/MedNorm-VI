"""Hierarchy-aware specificity arbitration invariants (Audit 0070 §10).

Every fixture is synthetic. No clinical record, no private annotation and no public-test text
appears here. Concept labels are either real ICD category names or invented placeholders; the
tests assert *policy invariants*, not medical facts.

The family under test mirrors the two real shapes Audit 0069 exposed - a parent whose
repaired title carries words the mention lacks, and a parent whose children add qualifiers
the mention lacks - without depending on either specific code.
"""

from __future__ import annotations

from typing import Any

import pytest

from mednorm_vi.kb.indexing import evidence as ev
from mednorm_vi.kb.indexing.retrieval import LocalIndex
from mednorm_vi.linking import icd10_specificity as sp

PARENT = "X10"
CHILD_SPECIFIC = "X100"
CHILD_OTHER = "X108"
CHILD_UNSPEC = "X109"
SIBLING = "X101"
GRANDCHILD = "X1001"
OUTSIDER = "Y20"


def _index(titles: dict[str, str]) -> LocalIndex:
    payload: dict[str, Any] = {
        "index_type": "icd10_vi",
        "source_snapshot_id": "test",
        "records": {c: {"concept_id": c, "canonical_name": n} for c, n in titles.items()},
        "exact": {},
        "exact_ascii": {},
        "ngrams": {},
        "sparse_terms": {},
        "graph": {},
        "exact_canonical": {"seed": ["X10"]},
    }
    return LocalIndex(**payload)


def _features(
    index: LocalIndex,
    codes: dict[str, tuple[str, ...]],
    mention: str,
    context: str = "",
) -> dict[str, sp.HierarchyFeatures]:
    return {
        code: sp.compute_features(
            index,
            code,
            channels=channels,
            lexical_score=1.0,
            mention_text=mention,
            context_text=context,
        )
        for code, channels in codes.items()
    }


def _arbitrate(order: list[str], features: dict[str, sp.HierarchyFeatures]) -> list[str]:
    return sp.arbitrate_order(order, features, policy=sp.POLICY_HIERARCHY_AWARE)


# ------------------------------------------------------------- parent versus child shapes


def test_parent_with_unsupported_words_loses_to_exact_child() -> None:
    """The B06/B069 shape: repair gave the parent words the mention does not contain."""
    index = _index({PARENT: "Bệnh rubella kèm sởi đức", CHILD_UNSPEC: "Bệnh rubella"})
    features = _features(
        index, {PARENT: ("sparse_accent",), CHILD_UNSPEC: ("sparse_accent",)}, "bệnh rubella"
    )
    assert _arbitrate([PARENT, CHILD_UNSPEC], features) == [CHILD_UNSPEC, PARENT]


def test_child_adding_an_unstated_qualifier_loses_to_parent() -> None:
    """The Q01/Q018 shape: the child asserts a site the text never states."""
    index = _index({PARENT: "Dị tật thoát vị não", CHILD_OTHER: "Dị tật thoát vị não vùng trán"})
    features = _features(
        index,
        {CHILD_OTHER: ("sparse_accent",), PARENT: ("sparse_accent",)},
        "dị tật thoát vị não",
    )
    assert _arbitrate([CHILD_OTHER, PARENT], features) == [PARENT, CHILD_OTHER]


def test_context_supporting_the_qualifier_lets_the_child_win() -> None:
    """CASE A: the qualifier is stated, just not inside the mention span."""
    index = _index({PARENT: "Dị tật thoát vị não", CHILD_SPECIFIC: "Dị tật thoát vị não vùng trán"})
    features = _features(
        index,
        {PARENT: ("sparse_accent",), CHILD_SPECIFIC: ("sparse_accent",)},
        "dị tật thoát vị não",
        context="tổn thương vùng trán",
    )
    assert _arbitrate([PARENT, CHILD_SPECIFIC], features) == [CHILD_SPECIFIC, PARENT]


def test_generic_mention_keeps_the_parent() -> None:
    index = _index({PARENT: "Dị tật thoát vị não", CHILD_SPECIFIC: "Dị tật thoát vị não vùng trán"})
    features = _features(
        index,
        {PARENT: ("sparse_accent",), CHILD_SPECIFIC: ("sparse_accent",)},
        "dị tật thoát vị não",
    )
    assert _arbitrate([CHILD_SPECIFIC, PARENT], features) == [PARENT, CHILD_SPECIFIC]


def test_unspecified_child_ties_and_resolves_upward_to_the_category() -> None:
    """CASE C: `.9` adds only stopwords, so it cannot out-argue its own category."""
    index = _index(
        {PARENT: "Dị tật thoát vị não", CHILD_UNSPEC: "Dị tật thoát vị não, không xác định"}
    )
    features = _features(
        index,
        {CHILD_UNSPEC: ("sparse_accent",), PARENT: ("sparse_accent",)},
        "dị tật thoát vị não",
    )
    assert _arbitrate([CHILD_UNSPEC, PARENT], features)[0] == PARENT


def test_other_child_never_wins_on_lexical_similarity_alone() -> None:
    """CASE E: `.8` carries a site qualifier; lexical closeness must not buy it the slot."""
    index = _index({PARENT: "Dị tật thoát vị não", CHILD_OTHER: "Dị tật thoát vị não vị trí khác"})
    features = _features(
        index,
        {CHILD_OTHER: ("sparse_accent",), PARENT: ("sparse_accent",)},
        "dị tật thoát vị não",
    )
    assert _arbitrate([CHILD_OTHER, PARENT], features)[0] == PARENT


def test_two_equally_plausible_siblings_do_not_get_picked_arbitrarily() -> None:
    """CASE D: neither sibling is stated, so the category represents the family."""
    index = _index(
        {
            PARENT: "Dị tật thoát vị não",
            CHILD_SPECIFIC: "Dị tật thoát vị não vùng trán",
            SIBLING: "Dị tật thoát vị não vùng chẩm",
        }
    )
    features = _features(
        index,
        {
            CHILD_SPECIFIC: ("sparse_accent",),
            SIBLING: ("sparse_accent",),
            PARENT: ("sparse_accent",),
        },
        "dị tật thoát vị não",
    )
    assert _arbitrate([CHILD_SPECIFIC, SIBLING, PARENT], features)[0] == PARENT


def test_hierarchy_distance_greater_than_one_is_still_arbitrated() -> None:
    index = _index({PARENT: "Bệnh rubella kèm sởi đức", GRANDCHILD: "Bệnh rubella"})
    features = _features(
        index, {PARENT: ("sparse_accent",), GRANDCHILD: ("sparse_accent",)}, "bệnh rubella"
    )
    assert features[GRANDCHILD].parent_child_distance == 2
    assert _arbitrate([PARENT, GRANDCHILD], features) == [GRANDCHILD, PARENT]


# --------------------------------------------------------------------- evidence dominance


def test_a_weak_fuzzy_child_never_defeats_an_exact_parent() -> None:
    """Audit-0069 tiers stay dominant: hierarchy cannot overrule stronger evidence."""
    index = _index({PARENT: "Dị tật thoát vị não", CHILD_SPECIFIC: "Dị tật thoát vị não vùng trán"})
    features = _features(
        index,
        {PARENT: ("exact_canonical",), CHILD_SPECIFIC: ("sparse",)},
        "dị tật thoát vị não vùng trán",
    )
    assert _arbitrate([PARENT, CHILD_SPECIFIC], features) == [PARENT, CHILD_SPECIFIC]


def test_an_exact_child_may_defeat_an_exact_parent_when_specificity_is_supported() -> None:
    index = _index({PARENT: "Dị tật thoát vị não", CHILD_SPECIFIC: "Dị tật thoát vị não vùng trán"})
    features = _features(
        index,
        {PARENT: ("exact_canonical",), CHILD_SPECIFIC: ("exact_canonical",)},
        "dị tật thoát vị não vùng trán",
    )
    assert _arbitrate([PARENT, CHILD_SPECIFIC], features) == [CHILD_SPECIFIC, PARENT]


@pytest.mark.parametrize("weaker", ["exact_alias", "exact_synonym", "sparse_accent", "sparse"])
def test_evidence_tier_dominates_every_hierarchy_signal(weaker: str) -> None:
    index = _index({PARENT: "Bệnh rubella kèm sởi đức", CHILD_UNSPEC: "Bệnh rubella"})
    features = _features(
        index, {PARENT: ("exact_canonical",), CHILD_UNSPEC: (weaker,)}, "bệnh rubella"
    )
    assert _arbitrate([PARENT, CHILD_UNSPEC], features)[0] == PARENT


# ------------------------------------------------------------------------- containment


def test_arbitration_never_moves_a_family_past_another_family() -> None:
    """Containment is why the accent false friend cannot come back through this door."""
    index = _index(
        {
            OUTSIDER: "Bệnh khác hoàn toàn",
            PARENT: "Bệnh rubella kèm sởi đức",
            CHILD_UNSPEC: "Bệnh rubella",
        }
    )
    features = _features(
        index,
        {
            OUTSIDER: ("sparse_accent",),
            PARENT: ("sparse_accent",),
            CHILD_UNSPEC: ("sparse_accent",),
        },
        "bệnh rubella",
    )
    result = _arbitrate([OUTSIDER, PARENT, CHILD_UNSPEC], features)
    assert result[0] == OUTSIDER
    assert result[1] == CHILD_UNSPEC


def test_arbitration_preserves_the_candidate_set_exactly() -> None:
    index = _index({PARENT: "Bệnh rubella kèm sởi đức", CHILD_UNSPEC: "Bệnh rubella"})
    features = _features(
        index, {PARENT: ("sparse_accent",), CHILD_UNSPEC: ("sparse_accent",)}, "bệnh rubella"
    )
    order = [PARENT, CHILD_UNSPEC]
    result = _arbitrate(order, features)
    assert sorted(result) == sorted(order)
    assert len(result) == len(set(result))


def test_arbitration_is_deterministic() -> None:
    index = _index({PARENT: "Bệnh rubella kèm sởi đức", CHILD_UNSPEC: "Bệnh rubella"})
    features = _features(
        index, {PARENT: ("sparse_accent",), CHILD_UNSPEC: ("sparse_accent",)}, "bệnh rubella"
    )
    first = _arbitrate([PARENT, CHILD_UNSPEC], features)
    assert all(_arbitrate([PARENT, CHILD_UNSPEC], features) == first for _ in range(5))


def test_h0_is_an_exact_no_op() -> None:
    index = _index({PARENT: "Bệnh rubella kèm sởi đức", CHILD_UNSPEC: "Bệnh rubella"})
    features = _features(
        index, {PARENT: ("sparse_accent",), CHILD_UNSPEC: ("sparse_accent",)}, "bệnh rubella"
    )
    order = [PARENT, CHILD_UNSPEC]
    assert sp.arbitrate_order(order, features, policy=sp.POLICY_NONE) == order


def test_only_three_policies_exist() -> None:
    assert sp.POLICIES == (sp.POLICY_NONE, sp.POLICY_CONSERVATIVE, sp.POLICY_HIERARCHY_AWARE)


def test_policy_document_records_the_key_and_invariants() -> None:
    document = sp.policy_document(sp.POLICY_HIERARCHY_AWARE)
    assert document["representative_key"][0] == "evidence_tier_index"
    assert document["representative_key"][1] == "unsupported_extra"
    assert len(document["cases"]) == 5


# ------------------------------------------------- untiered indexes and RxNorm are inert


def test_untiered_index_is_untouched_by_the_policy() -> None:
    """competition-v3 and RxNorm carry no tier postings, so the linker never arbitrates."""
    from pathlib import Path

    from mednorm_vi.kb.indexing.retrieval import load_index

    root = Path(__file__).resolve().parents[2]
    for relative in (
        "indices/candidate/icd10_vi/competition-v3/index.json",
        "indices/candidate/rxnorm/competition-v3/index.json",
    ):
        path = root / relative
        if not path.is_file():  # pragma: no cover - artifacts are gitignored
            pytest.skip(f"{relative} not present in this checkout")
        assert not load_index(path).tiered


def test_soi_still_resolves_to_measles_under_arbitration() -> None:
    """The Audit-0069 guarantee must survive Audit 0070 unchanged."""
    from pathlib import Path

    from mednorm_vi.kb.indexing.retrieval import load_index
    from mednorm_vi.linking.icd10 import link_icd10
    from mednorm_vi.resolution.models import (
        BoundaryEvidence,
        EntityHypothesis,
        TypeEvidence,
    )

    root = Path(__file__).resolve().parents[2]
    path = root / "indices/candidate/icd10_vi/competition-v4.1/index.json"
    if not path.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("competition-v4.1 not built in this checkout")
    index = load_index(path)
    hypothesis = EntityHypothesis(
        hypothesis_id="p",
        document_id="d",
        start=0,
        end=3,
        text="sởi",
        entity_type="CHẨN_ĐOÁN",
        status="accepted",
        chosen_proposal_id="g",
        source_proposal_ids=("g",),
        boundary_evidence=BoundaryEvidence(
            policy="g", chosen_kind="g", chosen_proposal_id="g", considered_kinds=("g",)
        ),
        type_evidence=TypeEvidence(
            entity_type="CHẨN_ĐOÁN", source_specialist="g", proposed_types=("CHẨN_ĐOÁN",)
        ),
        score=1.0,
    )
    result = link_icd10(hypothesis, index, hierarchy_policy=sp.POLICY_HIERARCHY_AWARE)
    assert result.candidates[0].code == "B05"


def test_evidence_tier_order_is_unchanged_by_this_milestone() -> None:
    assert ev.TIER_ORDER == ("A", "B", "C", "D", "E", "F")
