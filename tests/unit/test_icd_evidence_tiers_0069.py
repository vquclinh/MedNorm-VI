"""Evidence-tier and accent false-friend regressions (Audit 0069 §5, §9).

Every fixture here is synthetic. No clinical record, no private annotation and no public-test
text appears in this file - the two Vietnamese concepts used (measles and calculus) are ICD
category names, chosen because they are the pair that actually failed: `sởi` and `sỏi` differ
by one diacritic, and before Audit 0069 the accent-stripped channel let the stone outrank the
measles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mednorm_vi.kb.icd10.repair.row_reconstruction import reconstruct_row, segments
from mednorm_vi.kb.icd10.repair.title_recovery import (
    accept_recovered_title,
    has_repeated_phrase,
    is_damaged_title,
    recover_titles_v2,
    trim_embedded_note,
)
from mednorm_vi.kb.indexing import evidence as ev
from mednorm_vi.kb.indexing.normalization import accent_marked_ngrams, accent_marked_tokens
from mednorm_vi.kb.indexing.retrieval import LocalIndex, search_index

MEASLES = "B05"
CALCULUS = "N20"
CANCER = "C25"


def _index(**postings: dict[str, list[str]]) -> LocalIndex:
    base: dict[str, Any] = {
        "index_type": "icd10_vi",
        "source_snapshot_id": "test",
        "records": {
            MEASLES: {"concept_id": MEASLES, "canonical_name": "Bệnh sởi"},
            CALCULUS: {"concept_id": CALCULUS, "canonical_name": "Sỏi thận và niệu quản"},
            CANCER: {"concept_id": CANCER, "canonical_name": "U ác tính ở tụy"},
        },
        "exact": {},
        "exact_ascii": {},
        "ngrams": {},
        "sparse_terms": {},
        "graph": {},
    }
    base.update(postings)
    return LocalIndex(**base)


# --------------------------------------------------------------------------- tier ordering


def test_tier_order_is_the_documented_a_to_f() -> None:
    assert ev.TIER_ORDER == ("A", "B", "C", "D", "E", "F")


def test_every_channel_maps_to_a_tier() -> None:
    for channel, tier in ev.CHANNEL_TIERS.items():
        assert tier in ev.TIER_ORDER, channel


@pytest.mark.parametrize(
    ("stronger", "weaker"),
    [
        ("exact_canonical", "exact_alias"),
        ("exact_alias", "exact_synonym"),
        ("exact_synonym", "sparse_accent"),
        ("sparse_accent", "exact_ascii"),
        ("exact_ascii", "sparse"),
    ],
)
def test_a_lower_tier_never_outranks_a_higher_one_whatever_the_score(
    stronger: str, weaker: str
) -> None:
    """The whole point of the contract: fuzzy score cannot buy its way past a tier."""
    strong = ev.rank_key(concept_id="A", channels=(stronger,), lexical_score=0.001)
    weak = ev.rank_key(concept_id="B", channels=(weaker,), lexical_score=10_000.0)
    assert strong < weak


def test_ranking_is_total_so_repeats_are_deterministic() -> None:
    first = ev.rank_key(concept_id="A01", channels=("sparse",), lexical_score=3.0)
    second = ev.rank_key(concept_id="A02", channels=("sparse",), lexical_score=3.0)
    assert first < second


def test_untiered_index_keeps_its_previous_score_ordering() -> None:
    """competition-v3 and RxNorm must not move; their rank reduces to (-score, id)."""
    high = ev.rank_key(concept_id="B", channels=("exact",), lexical_score=9.0, tiered=False)
    low = ev.rank_key(concept_id="A", channels=("exact",), lexical_score=1.0, tiered=False)
    assert high < low
    assert high[0] == ev.UNTIERED_RANK


# ------------------------------------------------------------------- accent false friends


def test_accent_marked_forms_exclude_accent_free_padding() -> None:
    """`"  s"` carries no accent information; indexing it collapses the accent tier."""
    grams = accent_marked_ngrams("sởi")
    assert "sởi" in grams
    assert "  s" not in grams and "i  " not in grams
    assert accent_marked_tokens("sởi") == ("sởi",)
    assert accent_marked_tokens("covid") == ()


def test_soi_measles_outranks_soi_calculus_on_the_accent_channel() -> None:
    """The Audit-0067 failure, as a test: `sởi` must not return a stone at rank 1."""
    index = _index(
        sparse_accent={"sởi": [MEASLES], "sỏi": [CALCULUS]},
        sparse_terms={"soi": [MEASLES, CALCULUS]},
        exact_canonical={"bệnh sởi": [MEASLES]},
    )
    hits = search_index(index, "sởi")
    assert hits[0].concept_id == MEASLES
    assert hits[0].tier == ev.TIER_D_ACCENT_FUZZY
    calculus = next(h for h in hits if h.concept_id == CALCULUS)
    assert calculus.tier == ev.TIER_F_UNACCENTED_FUZZY


def test_accented_exact_beats_accent_stripped_exact() -> None:
    index = _index(
        exact_canonical={"bệnh sởi": [MEASLES]},
        exact_ascii={"benh soi": [MEASLES, CALCULUS]},
    )
    hits = search_index(index, "Bệnh sởi")
    assert hits[0].concept_id == MEASLES
    assert hits[0].tier == ev.TIER_A_EXACT_CANONICAL


def test_accent_stripped_query_still_retrieves_but_ranks_recall_only() -> None:
    """An unaccented mention must still find the concept - just without strong evidence."""
    index = _index(exact_canonical={"bệnh sởi": [MEASLES]}, exact_ascii={"benh soi": [MEASLES]})
    hits = search_index(index, "benh soi")
    assert [h.concept_id for h in hits] == [MEASLES]
    assert hits[0].tier == ev.TIER_E_UNACCENTED_EXACT


def test_governed_synonym_beats_fuzzy_spelling() -> None:
    index = _index(
        exact_synonym={"ung thư tụy": [CANCER]},
        ngrams={"ung": [CALCULUS], " un": [CALCULUS], "ng ": [CALCULUS]},
    )
    hits = search_index(index, "ung thư tụy")
    assert hits[0].concept_id == CANCER
    assert hits[0].tier == ev.TIER_C_SEMANTIC_SYNONYM


def test_exact_canonical_survives_many_fuzzy_competitors() -> None:
    """One exact match against a crowd of trigram matches - the v4 ranking failure."""
    crowd = [f"X{n:02d}" for n in range(40)]
    index = _index(
        exact_canonical={"bệnh sởi": [MEASLES]},
        ngrams={f"g{n}": crowd for n in range(60)},
    )
    hits = search_index(index, "bệnh sởi")
    assert hits[0].concept_id == MEASLES


# ------------------------------------------------------------------------ title recovery


def test_wrapped_title_is_rejoined_from_its_own_column() -> None:
    """A title split across physical lines is rejoined; C14.2's wrap is the real shape."""
    lines = [
        "1000 II   C00-D48 Neoplasms   C14.2   C142   Waldeyer ring   U ác tính ở vòng bạch",
        "                                             and other       huyết Waldeyer",
        "",
    ]
    recovered, unrecovered = recover_titles_v2(lines, {"C142": "- Bao gồm:"})
    assert recovered["C142"].recovered_title == "U ác tính ở vòng bạch huyết Waldeyer"
    assert recovered["C142"].confidence == "exact_anchor_wrapped_x_aligned"
    assert len(recovered["C142"].source_lines) == 2
    assert not unrecovered


def test_a_parent_never_inherits_a_child_title() -> None:
    """B05 must not take B05.9's name: the anchor pair identifies whose row it is."""
    lines = [
        "1000 I   A00-B99 Diseases   B05.9   B059   Measles   Bệnh sởi không biến chứng",
        "",
    ]
    recovered, unrecovered = recover_titles_v2(lines, {"B05": "- Bao gồm: bệnh"})
    assert "B05" not in recovered
    assert unrecovered == {"B05": "- Bao gồm: bệnh"}


def test_column_gap_stops_a_join_instead_of_resuming() -> None:
    """Resuming across a gap is how a title acquires another row's words."""
    lines = [
        "1000 I    A00-B99 Diseases     B05     B05    Measles       Bệnh sởi",
        "                               other text",
        "                                                            KHÔNG THUỘC HÀNG NÀY",
        "",
    ]
    recovered, _ = recover_titles_v2(lines, {"B05": "- Bao gồm: bệnh"})
    assert recovered["B05"].recovered_title == "Bệnh sởi"


def test_instruction_text_is_never_accepted_as_a_title() -> None:
    assert accept_recovered_title("Bao gồm: bệnh sởi", joined_lines=1) == ""
    assert accept_recovered_title(
        "Các phân loại ký tự thứ năm sau đây được sử dụng", joined_lines=3
    ) == ""


def test_embedded_note_is_trimmed_off_a_title() -> None:
    """I25 is stored with a glued note that `is_damaged_title` cannot see from the start."""
    assert trim_embedded_note("Bệnh tim thiếu máu cục bộ Loại trừ: bệnh") == (
        "Bệnh tim thiếu máu cục bộ"
    )


def test_interleaved_columns_are_rejected() -> None:
    assert has_repeated_phrase("Bệnh lao phổi, được khẳng - Bệnh lao phổi, định bằng soi")
    assert not has_repeated_phrase("U ác tính ở trực tràng")


def test_a_clean_title_is_never_rewritten() -> None:
    for title in ("Bệnh sởi", "U ác tính ở thực quản", "Gút [thống phong]"):
        assert not is_damaged_title(title)


def test_over_long_joins_are_refused() -> None:
    assert accept_recovered_title("Bệnh sởi rất dài và dài hơn nữa", joined_lines=14) == ""


def test_reconstruct_row_preserves_line_provenance() -> None:
    anchor = "1000 I   A00   A00   Cholera   Bệnh tả"
    lines = [anchor, " " * anchor.index("Bệnh tả") + "nặng", ""]
    columns = reconstruct_row(lines, 0)
    assert [c.x for c in columns] == [x for x, _ in segments(anchor)]
    joined = next(c for c in columns if c.text.startswith("Bệnh tả"))
    assert joined.source_lines == (1, 2)


# ----------------------------------------------------------------- shipped index contract


def test_v41_ships_its_retrieval_policy_and_stays_dotless() -> None:
    root = Path(__file__).resolve().parents[2]
    policy = root / "indices/candidate/icd10_vi/competition-v4.1/retrieval_policy.json"
    if not policy.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("competition-v4.1 not built in this checkout")
    document = json.loads(policy.read_text(encoding="utf-8"))
    assert document["policy_id"] == ev.policy_document()["policy_id"]
    assert [t["tier"] for t in document["tiers"]] == list(ev.TIER_ORDER)


def test_competition_v3_carries_no_tier_postings() -> None:
    """v3 is the rollback: it must stay untiered so its ordering cannot drift."""
    root = Path(__file__).resolve().parents[2]
    index = root / "indices/candidate/icd10_vi/competition-v3/index.json"
    if not index.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("competition-v3 not present in this checkout")
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert not any(
        payload.get(key)
        for key in ("exact_canonical", "exact_alias", "exact_synonym", "sparse_accent")
    )
