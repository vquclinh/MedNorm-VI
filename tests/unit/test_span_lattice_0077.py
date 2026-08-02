"""Finite span-lattice corrector guards (experiment 0077). No weights, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mednorm_vi.reasoner.lattice import (
    MAX_CANDIDATES_PER_CLUSTER,
    apply_selection,
    build_clusters,
)
from mednorm_vi.reasoner.prompt import LATTICE_POLICY, lattice_prompt

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_span_lattice_0077.py"
spec = importlib.util.spec_from_file_location("l0077", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

NOTE = "Trẻ Thiếu men G6PD bẩm sinh, viêm tuyến mồ hôi."


def at(text: str, entity_type: str, needle: str | None = None) -> dict:
    needle = needle or text
    start = NOTE.index(needle)
    return {
        "text": text,
        "type": entity_type,
        "position": [start, start + len(needle)],
        "assertions": [],
    }


FRAGMENTS = [
    at("Thiếu", "CHẨN_ĐOÁN"),
    at("Thiếu men", "CHẨN_ĐOÁN"),
    at("men", "CHẨN_ĐOÁN"),
    at("G6PD", "CHẨN_ĐOÁN"),
    at("viêm tuyến mồ hôi", "CHẨN_ĐOÁN"),
]


def test_every_candidate_is_a_literal_slice_of_the_source() -> None:
    """Arbitrary text is unrepresentable: candidates are produced by slicing."""
    for cluster in build_clusters(NOTE, FRAGMENTS):
        for candidate in cluster.candidates:
            assert NOTE[candidate.start : candidate.end] == candidate.text


def test_fragments_produce_a_merged_option() -> None:
    clusters = build_clusters(NOTE, FRAGMENTS)
    fragment_cluster = clusters[0]
    texts = [c.text for c in fragment_cluster.candidates]
    assert "Thiếu men G6PD" in texts


def test_unrelated_concept_is_not_merged_across_a_comma() -> None:
    clusters = build_clusters(NOTE, FRAGMENTS)
    assert len(clusters) == 2
    assert [p["text"] for p in clusters[1].proposals] == ["viêm tuyến mồ hôi"]


def test_no_candidate_crosses_clause_punctuation() -> None:
    for cluster in build_clusters(NOTE, FRAGMENTS):
        for candidate in cluster.candidates:
            assert "," not in candidate.text.rstrip(",")


def test_mixed_types_never_cluster_together() -> None:
    source = "sốt cao 39 độ"
    proposals = [
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3]},
        {"text": "39 độ", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [8, 13]},
    ]
    clusters = build_clusters(source, proposals)
    assert len(clusters) == 2
    assert {c.entity_type for c in clusters} == {"TRIỆU_CHỨNG", "KẾT_QUẢ_XÉT_NGHIỆM"}


def test_type_is_frozen_from_e3() -> None:
    clusters = build_clusters(NOTE, FRAGMENTS)
    out = apply_selection(clusters[0], ["c0_4"], NOTE)
    assert all(e["type"] == "CHẨN_ĐOÁN" for e in out)
    for cluster in clusters:
        assert all(c.entity_type == cluster.entity_type for c in cluster.candidates)


def test_lattice_is_capped() -> None:
    for cluster in build_clusters(NOTE, FRAGMENTS):
        assert len(cluster.candidates) <= MAX_CANDIDATES_PER_CLUSTER


@pytest.mark.parametrize("selection", [None, [], ["not_a_real_id"], ["c9_9"]])
def test_anything_unrecognised_keeps_originals(selection: list[str] | None) -> None:
    clusters = build_clusters(NOTE, FRAGMENTS)
    out = apply_selection(clusters[0], selection, NOTE)
    assert [e["text"] for e in out] == [p["text"] for p in clusters[0].proposals]


def test_selecting_the_merge_replaces_the_fragments() -> None:
    clusters = build_clusters(NOTE, FRAGMENTS)
    merged = next(c for c in clusters[0].candidates if c.text == "Thiếu men G6PD")
    out = apply_selection(clusters[0], [merged.candidate_id], NOTE)
    assert [e["text"] for e in out] == ["Thiếu men G6PD"]
    assert out[0]["position"] == [merged.start, merged.end]


def test_assertion_inheritance_is_unanimous_and_conservative() -> None:
    proposals = (
        {"text": "a", "type": "CHẨN_ĐOÁN", "assertions": ["isNegated", "isHistorical"]},
        {"text": "b", "type": "CHẨN_ĐOÁN", "assertions": ["isNegated"]},
    )
    entities = [{"text": "ab", "type": "CHẨN_ĐOÁN", "position": [0, 2]}]
    runner.inherit_assertions(entities, proposals)
    assert entities[0]["assertions"] == ["isNegated"]  # only the unanimous flag
    assert entities[0]["candidates"] == []


def test_lab_types_get_no_assertions_and_no_candidates() -> None:
    entities = [{"text": "x", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [0, 1]}]
    runner.inherit_assertions(entities, ({"text": "x", "type": "KẾT_QUẢ_XÉT_NGHIỆM"},))
    assert "assertions" not in entities[0]
    assert "candidates" not in entities[0]


def test_prompt_freezes_type_and_forbids_text_or_offsets() -> None:
    clusters = build_clusters(NOTE, FRAGMENTS)
    text = lattice_prompt(NOTE, clusters)
    assert LATTICE_POLICY in text
    assert "FIXED" in text
    assert "You cannot write text or offsets" in text
    assert "Prefer KEEP_ORIGINALS" in text
    assert "c0_4" in text


@pytest.mark.parametrize("reply", ["", "junk", '{"clusters": "no"}', "{oops"])
def test_unusable_replies_fall_back(reply: str) -> None:
    assert runner.parse_clusters(reply) is None


def test_parse_extracts_cluster_selections() -> None:
    parsed = runner.parse_clusters('{"clusters":[{"cluster_id":0,"select":["c0_4"]}]}')
    assert parsed == {0: ["c0_4"]}


def test_diagnostics_are_computed_from_serialized_output() -> None:
    """0076 reported 9 type changes while the files showed 7; type is now frozen at 0."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "computed from the SERIALIZED output" in source
    assert '"type_changes": 0' in source
    assert '"type_frozen": True' in source
    for field in (
        "clusters_changed",
        "fragments_removed",
        "merged_entities_created",
        "boundary_expansions",
        "parse_fallbacks",
        "changed_cluster_examples",
    ):
        assert field in source


def test_no_4b_models_or_candidate_retrieval() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for banned in ("Qwen3Reranker(", "build_pool(", "load_index(", "S1_doc_vectors"):
        assert banned not in source
    assert 'if "semantic_s1_0072" in (' in source
