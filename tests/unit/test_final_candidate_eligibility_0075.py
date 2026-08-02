"""Final-candidate eligibility boundary (Audit 0075 hotfix). No weights, no network.

L9 refused `corrected_dense_0075` with `kb.candidate_not_final_eligible` on two documents.
Cause: dense retrieval scores every RxNorm document, including the 129,520 `closure_only`
records that exist only so the ingredient/product walk has somewhere to land. `exists()` says
they are in the snapshot, which is not the same as "the runtime may emit them".
"""

from __future__ import annotations

from typing import Any

from mednorm_vi.kb.indexing.retrieval import LocalIndex
from mednorm_vi.linking.semantic.hybrid import (
    final_candidate_eligible,
    sanitize_final_candidates,
)

SEARCHABLE = "111"
CLOSURE = "222"
SEARCHABLE_2 = "333"


def rx_index() -> LocalIndex:
    def record(cui: str, membership: str) -> dict[str, Any]:
        return {
            "concept_id": cui,
            "canonical_name": f"drug {cui}",
            "metadata": {
                "membership": membership,
                "runtime_role": "COMPETITION_RUNTIME_ELIGIBLE"
                if membership == "searchable"
                else "RETRIEVAL_ONLY",
            },
        }

    return LocalIndex(
        index_type="rxnorm",
        source_snapshot_id="t",
        records={
            SEARCHABLE: record(SEARCHABLE, "searchable"),
            CLOSURE: record(CLOSURE, "closure_only"),
            SEARCHABLE_2: record(SEARCHABLE_2, "searchable"),
        },
        exact={},
        exact_ascii={},
        ngrams={},
        sparse_terms={},
        graph={},
    )


def icd_index() -> LocalIndex:
    return LocalIndex(
        index_type="icd10_vi",
        source_snapshot_id="t",
        records={"A00": {"concept_id": "A00", "canonical_name": "x", "metadata": {}}},
        exact={},
        exact_ascii={},
        ngrams={},
        sparse_terms={},
        graph={},
    )


def test_final_eligible_candidate_is_preserved_unchanged() -> None:
    assert sanitize_final_candidates([SEARCHABLE], rx_index()) == (SEARCHABLE,)
    assert final_candidate_eligible(rx_index(), SEARCHABLE)


def test_closure_only_candidate_is_removed() -> None:
    """The exact class that triggered kb.candidate_not_final_eligible."""
    assert not final_candidate_eligible(rx_index(), CLOSURE)
    assert sanitize_final_candidates([CLOSURE], rx_index()) == ()


def test_mixed_list_preserves_order_of_survivors_and_dedups() -> None:
    mixed = [SEARCHABLE, CLOSURE, SEARCHABLE_2, SEARCHABLE, CLOSURE]
    assert sanitize_final_candidates(mixed, rx_index()) == (SEARCHABLE, SEARCHABLE_2)


def test_all_invalid_becomes_empty_list() -> None:
    assert sanitize_final_candidates([CLOSURE, CLOSURE, "not-in-kb"], rx_index()) == ()


def test_concept_absent_from_the_snapshot_is_removed() -> None:
    assert sanitize_final_candidates(["999999"], rx_index()) == ()


def test_icd_is_unaffected_because_every_icd_concept_is_searchable() -> None:
    assert sanitize_final_candidates(["A00"], icd_index()) == ("A00",)


def test_missing_index_emits_nothing_rather_than_trusting_the_id() -> None:
    assert sanitize_final_candidates([SEARCHABLE], None) == ()


def test_boundary_is_applied_at_the_single_emission_point() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src/mednorm_vi/linking/semantic/hybrid.py"
    ).read_text(encoding="utf-8")
    assert "codes = sanitize_final_candidates(order[:limit], governed)" in source
    # Reuses the lexical linker's predicate rather than defining a second one.
    assert "from ..rxnorm_graph import is_closure_only" in source


def test_l9_still_refuses_a_closure_only_candidate_end_to_end() -> None:
    """L9 is the backstop and stays unchanged; the fix is upstream, not a replacement.

    Feeding L9 the exact raw output that failed the run must still produce
    `kb.candidate_not_final_eligible` - proving the validator was never weakened.
    """
    from mednorm_vi.validator.kb_membership import (
        LockedSnapshots,
        validate_entity_candidates,
    )

    snapshots = LockedSnapshots(icd10=icd_index(), rxnorm=rx_index())
    entity = {
        "text": "thuốc",
        "type": "THUỐC",
        "position": [0, 5],
        "assertions": [],
        "candidates": [CLOSURE],
    }
    result = validate_entity_candidates(entity, snapshots)
    codes = " ".join(getattr(i, "code", str(i)) for i in result.issues)
    assert "candidate_not_final_eligible" in codes

    # And the sanitized list that our boundary produces passes cleanly.
    cleaned = dict(entity)
    cleaned["candidates"] = list(sanitize_final_candidates([CLOSURE, SEARCHABLE], rx_index()))
    assert cleaned["candidates"] == [SEARCHABLE]
    assert not validate_entity_candidates(cleaned, snapshots).issues
