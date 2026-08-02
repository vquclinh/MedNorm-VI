"""Strict merge-only lattice guards (experiment 0078). No weights, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mednorm_vi.reasoner.merge_only import (
    MERGE_TO_UNION,
    apply_merge,
    build_merge_clusters,
)
from mednorm_vi.reasoner.prompt import MERGE_ONLY_POLICY, merge_only_prompt

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_merge_only_0078.py"
spec = importlib.util.spec_from_file_location("m0078", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

NOTE = "Trẻ Thiếu men G6PD bẩm sinh, viêm tuyến mồ hôi. Có cục máu đông sau 22 tuần, 3 ngày."


def at(text: str, entity_type: str, assertions: list[str] | None = None) -> dict:
    start = NOTE.index(text)
    return {
        "text": text,
        "type": entity_type,
        "position": [start, start + len(text)],
        "assertions": assertions if assertions is not None else [],
    }


# --------------------------------------------------- the 0077 regressions are impossible


@pytest.mark.parametrize(
    "text,entity_type",
    [
        ("máu đông", "CHẨN_ĐOÁN"),
        ("22", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ("3", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ("viêm tuyến mồ hôi", "CHẨN_ĐOÁN"),
    ],
)
def test_single_proposals_are_never_offered_to_the_model(text: str, entity_type: str) -> None:
    """'22' -> '22 tuần' cannot happen: a lone proposal never enters a cluster."""
    clusters, untouched = build_merge_clusters(NOTE, [at(text, entity_type)])
    assert clusters == []
    assert [p["text"] for p in untouched] == [text]


def test_lone_proposals_are_emitted_byte_identically() -> None:
    proposals = [at("máu đông", "CHẨN_ĐOÁN"), at("22", "KẾT_QUẢ_XÉT_NGHIỆM")]
    clusters, untouched = build_merge_clusters(NOTE, proposals)
    assert clusters == []
    assert untouched == proposals


def test_no_expansion_operation_exists_in_the_module() -> None:
    source = (ROOT / "src/mednorm_vi/reasoner/merge_only.py").read_text(encoding="utf-8")
    for banned in ("EXPANSION_TOKEN_RADIUS", "_snap(", "token_radius", "PROVENANCE_EXPANDED"):
        assert banned not in source


# ------------------------------------------------------------------- the intended merge


def test_fragments_produce_exactly_one_union_option() -> None:
    clusters, _ = build_merge_clusters(NOTE, [at("Thiếu", "CHẨN_ĐOÁN"), at("G6PD", "CHẨN_ĐOÁN")])
    assert len(clusters) == 1
    assert clusters[0].union_text == "Thiếu men G6PD"
    assert NOTE[clusters[0].union_start : clusters[0].union_end] == clusters[0].union_text


def test_union_is_exactly_min_start_to_max_end() -> None:
    proposals = [at("Thiếu men", "CHẨN_ĐOÁN"), at("G6PD", "CHẨN_ĐOÁN")]
    clusters, _ = build_merge_clusters(NOTE, proposals)
    assert clusters[0].union_start == min(p["position"][0] for p in proposals)
    assert clusters[0].union_end == max(p["position"][1] for p in proposals)


def test_merge_replaces_fragments_and_keeps_type() -> None:
    clusters, _ = build_merge_clusters(NOTE, [at("Thiếu", "CHẨN_ĐOÁN"), at("G6PD", "CHẨN_ĐOÁN")])
    out, merged = apply_merge(clusters[0], MERGE_TO_UNION, NOTE)
    assert merged and [e["text"] for e in out] == ["Thiếu men G6PD"]
    assert out[0]["type"] == "CHẨN_ĐOÁN"


@pytest.mark.parametrize("choice", [None, "", "KEEP_ORIGINALS", "MERGE", "yes", "merge_to_union"])
def test_anything_but_exact_merge_keeps_originals(choice: str | None) -> None:
    clusters, _ = build_merge_clusters(NOTE, [at("Thiếu", "CHẨN_ĐOÁN"), at("G6PD", "CHẨN_ĐOÁN")])
    out, merged = apply_merge(clusters[0], choice, NOTE)
    assert not merged
    assert [e["text"] for e in out] == ["Thiếu", "G6PD"]


# ------------------------------------------------------------- frozen type / assertions


def test_mixed_types_are_never_clustered() -> None:
    clusters, untouched = build_merge_clusters(
        NOTE, [at("22", "KẾT_QUẢ_XÉT_NGHIỆM"), at("tuần", "TÊN_XÉT_NGHIỆM")]
    )
    assert clusters == []
    assert len(untouched) == 2


def test_disagreeing_assertions_block_the_merge() -> None:
    """Rather than guess a flag for the merged span, keep the originals."""
    clusters, untouched = build_merge_clusters(
        NOTE,
        [at("Thiếu", "CHẨN_ĐOÁN", ["isNegated"]), at("G6PD", "CHẨN_ĐOÁN", [])],
    )
    assert clusters == []
    assert [p["text"] for p in untouched] == ["Thiếu", "G6PD"]


def test_agreeing_assertions_are_inherited_unchanged() -> None:
    clusters, _ = build_merge_clusters(
        NOTE,
        [at("Thiếu", "CHẨN_ĐOÁN", ["isHistorical"]), at("G6PD", "CHẨN_ĐOÁN", ["isHistorical"])],
    )
    assert clusters[0].assertions == ("isHistorical",)
    out, _ = apply_merge(clusters[0], MERGE_TO_UNION, NOTE)
    assert runner.finalize(out[0], clusters[0].assertions)["assertions"] == ["isHistorical"]


def test_no_merge_across_a_clause_boundary() -> None:
    clusters, _ = build_merge_clusters(
        NOTE, [at("G6PD", "CHẨN_ĐOÁN"), at("viêm tuyến mồ hôi", "CHẨN_ĐOÁN")]
    )
    assert clusters == []


# --------------------------------------------------------------- serialization contract


def test_candidates_forced_empty_and_lab_types_carry_no_assertions() -> None:
    diagnosis = runner.finalize({"text": "x", "type": "CHẨN_ĐOÁN", "position": [0, 1]}, ())
    assert diagnosis["candidates"] == [] and diagnosis["assertions"] == []
    lab = runner.finalize({"text": "x", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [0, 1]}, ())
    assert "assertions" not in lab and "candidates" not in lab


def test_prompt_offers_only_two_answers_and_freezes_everything_else() -> None:
    clusters, _ = build_merge_clusters(NOTE, [at("Thiếu", "CHẨN_ĐOÁN"), at("G6PD", "CHẨN_ĐOÁN")])
    text = merge_only_prompt(NOTE, clusters)
    assert MERGE_ONLY_POLICY in text
    assert "MERGE_TO_UNION" in text
    assert "KEEP_ORIGINALS" in text
    assert "cannot change text, offsets, types or assertions" in text
    assert "Thiếu men G6PD" in text


@pytest.mark.parametrize("reply", ["", "junk", '{"clusters": 1}', "{oops"])
def test_unusable_replies_fall_back_to_keep(reply: str) -> None:
    assert runner.parse_choices(reply) is None


def test_diagnostics_pin_the_forbidden_counters_to_zero() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for field in (
        '"boundary_expansions": 0',
        '"single_proposal_changes": 0',
        '"type_changes": 0',
        '"assertion_changes": 0',
    ):
        assert field in source
    for field in (
        "eligible_multi_fragment_clusters",
        "clusters_merged",
        "clusters_kept",
        "fragments_removed",
        "merged_entities_created",
        "parse_fallbacks",
        "merges",
    ):
        assert field in source
