"""Phase 2A corpus-analysis pipeline — integration tests (synthetic corpus).

Runs the real L1 pipeline + router over synthetic fixtures and asserts the
Phase 2A contract: descriptive statistics only, deterministic reports, and no
change to L1 preprocessing behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.corpus_analysis import (
    Phase2AConfig,
    analyze_corpus_dir,
    determinism_hash,
    load_corpus,
    run_corpus_analysis,
    to_dict,
    to_json,
    to_markdown,
)
from mednorm_vi.corpus_analysis.cli import main as cli_main
from mednorm_vi.document_intelligence import analyze_text

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "corpus_analysis" / "input"
CONFIG = Phase2AConfig.load(
    REPO / "configs" / "corpus_analysis" / "analysis_v1.yaml",
    REPO / "configs" / "case_router" / "base.yaml",
    REPO / "configs" / "document_intelligence" / "base.yaml")
ANALYSIS = analyze_corpus_dir(FIX, CONFIG)


def test_every_document_analyzed_and_l1_valid() -> None:
    assert ANALYSIS.n_documents == 5
    assert all(d.l1_valid for d in ANALYSIS.documents)
    assert [d.source_name for d in ANALYSIS.documents] == ["1", "2", "3", "10", "100"]


def test_lengths_measured() -> None:
    dist = ANALYSIS.distributions
    assert dist["characters"].total > 0
    assert dist["characters"].n == 5
    doc1 = next(d for d in ANALYSIS.documents if d.source_name == "1")
    assert doc1.length.characters == len((FIX / "1.txt").read_text(encoding="utf-8"))


def test_routing_units_and_cases_counted() -> None:
    kinds = dict(ANALYSIS.unit_kind_frequencies)
    assert kinds.get("list_item", 0) > 0  # med/lab/imaging fixtures are lists
    cases = dict(ANALYSIS.case_frequencies)
    assert cases  # at least one structural case fired
    # C6/C7 are derived from specialist proposals and must NOT be computed here
    assert "C6" not in cases and "C7" not in cases


def test_structure_distributions() -> None:
    doc1 = next(d for d in ANALYSIS.documents if d.source_name == "1")
    assert doc1.structure.list_items >= 3
    assert doc1.profile in ("list_dominant", "mixed", "narrative_dominant")


def test_layout_families_detected_across_corpus() -> None:
    assert ANALYSIS.docs_with["lab_layout"] >= 1
    assert ANALYSIS.docs_with["medication_layout"] >= 1
    assert ANALYSIS.docs_with["imaging_layout"] >= 1


def test_sections_and_headings_collected() -> None:
    assert ANALYSIS.distributions["sections"].total >= 1
    # headings are structure labels taken verbatim from the document
    assert isinstance(ANALYSIS.heading_frequencies, tuple)


# --- determinism ---

def test_analysis_deterministic() -> None:
    a = analyze_corpus_dir(FIX, CONFIG)
    b = analyze_corpus_dir(FIX, CONFIG)
    assert determinism_hash(a) == determinism_hash(b)
    assert to_json(a) == to_json(b)
    assert to_markdown(a) == to_markdown(b)


def test_report_has_no_timestamp_fields() -> None:
    payload = to_dict(ANALYSIS)
    blob = json.dumps(payload).lower()
    for volatile in ("timestamp", "generated_at", "created_at", "datetime", "\"now\""):
        assert volatile not in blob


def test_json_report_roundtrips() -> None:
    payload = json.loads(to_json(ANALYSIS))
    assert payload["n_documents"] == 5
    assert payload["schema_version"] == "corpus-analysis-2a-1"
    assert len(payload["documents"]) == 5


# --- scope contract: descriptive only ---

def test_scope_declares_no_prediction_or_extraction() -> None:
    scope = to_dict(ANALYSIS)["scope"]
    assert scope["descriptive_only"] is True
    assert scope["entity_extraction"] is False
    assert scope["prediction"] is False
    assert scope["linking"] is False
    assert scope["ml_or_llm"] is False
    assert scope["l1_behavior_modified"] is False


def test_no_entity_or_candidate_fields_in_report() -> None:
    # The `scope` block legitimately names these concepts to declare them false,
    # so it is excluded; the DATA must not carry any of them.
    payload = to_dict(ANALYSIS)
    payload.pop("scope")
    blob = json.dumps(payload).lower()
    for forbidden in ("rxcui", "icd10", "candidates", "proposals", "entities",
                      "assertions", "predict"):
        assert forbidden not in blob


def test_l1_behavior_unchanged_by_analysis() -> None:
    # Analysing a document must not alter what L1 produces for the same text.
    text = (FIX / "1.txt").read_text(encoding="utf-8")
    before = analyze_text(text, config=CONFIG.l1_config, lexicon=CONFIG.l1_lexicon)
    run_corpus_analysis(load_corpus(FIX), CONFIG)
    after = analyze_text(text, config=CONFIG.l1_config, lexicon=CONFIG.l1_lexicon)
    assert before.document_id == after.document_id
    assert before.original_text == after.original_text
    assert len(before.nodes) == len(after.nodes)
    assert before.config_hash == after.config_hash


def test_offsets_preserved_for_headings() -> None:
    # Heading text is sliced from original_text; it must match verbatim.
    for doc in load_corpus(FIX):
        graph = analyze_text(doc.text, config=CONFIG.l1_config, lexicon=CONFIG.l1_lexicon)
        analysis = next(d for d in ANALYSIS.documents if d.source_name == doc.source_name)
        for heading in analysis.sections.headings:
            assert heading in graph.original_text


# --- CLI ---

def test_cli_writes_deterministic_reports(tmp_path: Path) -> None:
    out1, out2 = tmp_path / "a", tmp_path / "b"
    args = ["--input-dir", str(FIX)]
    assert cli_main([*args, "--output-dir", str(out1)]) == 0
    assert cli_main([*args, "--output-dir", str(out2)]) == 0
    assert (out1 / "report.json").read_bytes() == (out2 / "report.json").read_bytes()
    assert (out1 / "report.md").read_bytes() == (out2 / "report.md").read_bytes()


def test_cli_errors_on_empty_corpus(tmp_path: Path) -> None:
    assert cli_main(["--input-dir", str(tmp_path)]) == 2


# --- optional: the real organizer corpus, only if it is present locally ---

def test_real_public_corpus_if_present() -> None:
    corpus = REPO / "data" / "organizer_test" / "input"
    if not corpus.is_dir() or not list(corpus.glob("*.txt")):
        pytest.skip("organizer public corpus not present locally (git-ignored)")
    result = analyze_corpus_dir(corpus, CONFIG)
    assert result.n_documents == 100
    assert all(d.l1_valid for d in result.documents)
    assert determinism_hash(result) == determinism_hash(analyze_corpus_dir(corpus, CONFIG))
