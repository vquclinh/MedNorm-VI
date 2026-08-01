"""Structured RxNorm recovery and compatibility invariants (Audit 0074).

Model-free and network-free. Fixtures are synthetic drug records; no clinical text and no
public-test material appear here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.kb.rxnorm import structured as st
from mednorm_vi.kb.rxnorm.rrf_reader import _SAT
from mednorm_vi.linking.semantic import rxnorm_structured as rs

ROOT = Path(__file__).resolve().parents[2]


def drug(**kwargs: object) -> st.StructuredDrug:
    base = {"rxcui": "1", "tty": "SCD", "name": "d"}
    base.update(kwargs)
    return st.StructuredDrug(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ RRF column contract


def test_rxnsat_columns_match_the_official_layout() -> None:
    """RXCUI|LUI|SUI|RXAUI|STYPE|CODE|ATUI|SATUI|ATN|SAB|ATV|SUPPRESS|CVF."""
    assert _SAT["ATN"] == 8
    assert _SAT["SAB"] == 9
    assert _SAT["ATV"] == 10
    assert _SAT["SAB"] < _SAT["ATV"], "ATV and SAB were transposed before Audit 0074"


# ------------------------------------------------------------------------ normalization


@pytest.mark.parametrize(
    ("text", "expected"),
    [("500 MG", "500mg"), ("500mg", "500mg"), ("0.50 mg", "0.5mg"), ("12,5 MG", "12.5mg")],
)
def test_strength_normalization_is_case_and_format_insensitive(text: str, expected: str) -> None:
    parsed = st.parse_strengths(text)
    assert parsed and parsed[0].key == expected


def test_units_are_never_converted() -> None:
    """`500 MG` and `0.5 G` are the same dose but the source does not say so."""
    mg = st.parse_strengths("500 MG")[0].key
    g = st.parse_strengths("0.5 G")[0].key
    assert mg != g


def test_multiple_strength_components_are_kept_in_order() -> None:
    parsed = st.parse_strengths("12.5 MG / 5 ML")
    assert [p.key for p in parsed] == ["12.5mg", "5ml"]


def test_absent_strength_yields_nothing_not_a_guess() -> None:
    assert st.parse_strengths("") == ()
    assert st.parse_strengths("amoxicillin capsule") == ()


# ---------------------------------------------------------------------- semantic text


def test_semantic_text_includes_recovered_structure() -> None:
    text = st.semantic_text(
        drug(
            name="amoxicillin 500 MG Oral Capsule",
            ingredients=("amoxicillin",),
            strengths=st.parse_strengths("500 MG"),
            dose_forms=("Oral Capsule",),
            brands=("Amoxil",),
        )
    )
    for expected in (
        "ingredient: amoxicillin",
        "strength: 500 MG",
        "dose form: Oral Capsule",
        "brand: Amoxil",
        "TTY: SCD",
    ):
        assert expected in text


def test_semantic_text_omits_absent_fields() -> None:
    text = st.semantic_text(drug(name="aspirin"))
    for absent in ("ingredient:", "strength:", "dose form:", "brand:"):
        assert absent not in text


def test_semantic_text_is_deterministic() -> None:
    d = drug(ingredients=("a",), strengths=st.parse_strengths("5 MG"))
    assert len({st.semantic_text(d) for _ in range(5)}) == 1


# ------------------------------------------------------------------ compatibility rules


def test_exact_strength_is_an_exact_match() -> None:
    mention = rs.parse_mention("amoxicillin 500mg")
    verdict = rs.assess(mention, drug(strengths=st.parse_strengths("500 MG")))
    assert verdict.strength == rs.MATCH_EXACT


def test_wrong_strength_is_contradicted_not_merely_unsupported() -> None:
    """The high-value signal: the source says 250 mg, the note says 500 mg."""
    mention = rs.parse_mention("amoxicillin 500mg")
    verdict = rs.assess(mention, drug(strengths=st.parse_strengths("250 MG")))
    assert verdict.strength == rs.MATCH_CONTRADICTED
    assert verdict.contradicted


def test_unrecorded_strength_is_unsupported_not_contradicted() -> None:
    """Silence in the source is not disagreement."""
    verdict = rs.assess(rs.parse_mention("amoxicillin 500mg"), drug())
    assert verdict.strength == rs.MATCH_UNSUPPORTED
    assert not verdict.contradicted


def test_mention_without_a_strength_contradicts_nothing() -> None:
    verdict = rs.assess(
        rs.parse_mention("amoxicillin"), drug(strengths=st.parse_strengths("500 MG"))
    )
    assert verdict.strength == rs.MATCH_UNSUPPORTED


def test_vietnamese_form_word_supports_the_matching_dose_form() -> None:
    verdict = rs.assess(rs.parse_mention("amoxicillin viên nang"), drug(rxterm_form="Capsule"))
    assert verdict.form == rs.MATCH_SUPPORTED


def test_vietnamese_form_word_contradicts_a_different_dose_form() -> None:
    verdict = rs.assess(rs.parse_mention("amoxicillin viên nén"), drug(rxterm_form="Capsule"))
    assert verdict.form == rs.MATCH_CONTRADICTED


def test_missing_structured_record_is_never_a_contradiction() -> None:
    verdict = rs.assess(rs.parse_mention("amoxicillin 500mg"), None)
    assert not verdict.contradicted
    assert "no_structured_record" in verdict.reasons


# ------------------------------------------------------------------------- reordering


def test_contradicted_strength_is_demoted_below_a_compatible_candidate() -> None:
    structured = {
        "wrong": drug(rxcui="wrong", strengths=st.parse_strengths("250 MG")),
        "right": drug(rxcui="right", strengths=st.parse_strengths("500 MG")),
    }
    assert rs.reorder(["wrong", "right"], "amoxicillin 500mg", structured) == ["right", "wrong"]


def test_reordering_preserves_the_candidate_set_exactly() -> None:
    structured = {c: drug(rxcui=c) for c in ("a", "b", "c")}
    order = ["b", "a", "c"]
    assert sorted(rs.reorder(order, "x", structured)) == sorted(order)


def test_reordering_without_stated_structure_is_a_no_op() -> None:
    """A bare ingredient mention states nothing to arbitrate, so order must not move."""
    structured = {
        c: drug(rxcui=c, strengths=st.parse_strengths(f"{n} MG"))
        for n, c in enumerate(("a", "b", "c"), start=1)
    }
    order = ["b", "a", "c"]
    assert rs.reorder(order, "amoxicillin", structured) == order


def test_reordering_is_deterministic() -> None:
    structured = {c: drug(rxcui=c, strengths=st.parse_strengths("250 MG")) for c in ("a", "b")}
    first = rs.reorder(["a", "b"], "amoxicillin 500mg", structured)
    assert all(rs.reorder(["a", "b"], "amoxicillin 500mg", structured) == first for _ in range(5))


def test_lexical_similarity_cannot_promote_a_contradicted_candidate() -> None:
    """Ordering is lexicographic; there is no score to accumulate past a contradiction."""
    contradicted = rs.rank_key("x", rs.Compatibility(worst=rs.MATCH_CONTRADICTED), base_rank=0)
    supported = rs.rank_key("y", rs.Compatibility(worst=rs.MATCH_UNSUPPORTED), base_rank=999)
    assert supported < contradicted


# ------------------------------------------------------- governed artifacts, if present


def test_recovered_fields_are_declared_and_bounded() -> None:
    assert st.ATN_STRENGTH in st.RECOVERED_ATTRIBUTES
    assert st.REL_HAS_INGREDIENT in st.RECOVERED_RELATIONS
    assert len(st.RECOVERED_ATTRIBUTES) <= 12
    assert len(st.RECOVERED_RELATIONS) <= 8


def test_structured_manifest_records_provenance_and_no_inference() -> None:
    path = ROOT / "artifacts/rxnorm_structured/0074/manifest.json"
    if not path.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("structured artifact not built in this checkout")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["external_sources_used"] == []
    assert manifest["inferred_fields"] == []
    assert manifest["contains_clinical_text"] is False
    assert manifest["governed_concepts"] == manifest["records_written"] == 211_949
    assert manifest["archive_sha256"] and manifest["governed_index_sha256"]


def test_ab_report_preserves_candidate_sets_and_never_uses_leaderboard() -> None:
    path = ROOT / "artifacts/rxnorm_structured/0074/ab_report.json"
    if not path.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("A/B report not built in this checkout")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["candidate_set_preserved"] is True
    assert report["public_leaderboard_signal_used"] is False
    assert report["null_policy"].startswith("shadow")


def test_icd_path_is_untouched_by_this_milestone() -> None:
    source = (ROOT / "src/mednorm_vi/linking/semantic/rxnorm_structured.py").read_text(
        encoding="utf-8"
    )
    assert "ICD" not in source.replace("ICD is untouched.", "")
