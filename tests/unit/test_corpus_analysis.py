"""Phase 2A corpus analysis — unit tests over SYNTHETIC fixtures.

These tests never touch the organizer corpus (`data/**` is git-ignored and absent
in CI); they use small synthetic documents under
``tests/fixtures/corpus_analysis/``.
"""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.corpus_analysis import (
    analyze_layouts,
    corpus_sha256,
    discover_documents,
    histogram,
    load_analysis_config,
    load_corpus,
    summarize,
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "corpus_analysis" / "input"
CFG = load_analysis_config(REPO / "configs" / "corpus_analysis" / "analysis_v1.yaml")


# --- loader ---

def test_documents_sorted_numerically_not_lexically() -> None:
    names = [p.stem for p in discover_documents(FIX)]
    assert names == ["1", "2", "3", "10", "100"]  # not 1,10,100,2,3


def test_load_corpus_reads_verbatim() -> None:
    docs = load_corpus(FIX)
    assert len(docs) == 5
    one = next(d for d in docs if d.source_name == "1")
    assert "metoprolol 25mg po bid" in one.text
    assert one.text == Path(one.path).read_text(encoding="utf-8")  # unmodified


def test_corpus_hash_deterministic_and_content_defined() -> None:
    a, b = load_corpus(FIX), load_corpus(FIX)
    assert corpus_sha256(a) == corpus_sha256(b)
    assert len(corpus_sha256(a)) == 64


# --- distributions ---

def test_summarize_basic() -> None:
    d = summarize([1, 2, 3, 4])
    assert d.n == 4 and d.total == 10 and d.minimum == 1 and d.maximum == 4
    assert d.mean == 2.5 and d.median == 2.5


def test_summarize_empty_is_zeroed() -> None:
    d = summarize([])
    assert d.n == 0 and d.total == 0 and d.median == 0.0


def test_histogram_buckets_and_overflow() -> None:
    h = histogram([10, 100, 5000], (250, 500))
    assert h == [("1-250", 2), ("251-500", 0), (">500", 1)]


# --- layouts (descriptive shape only) ---

def test_lab_layout_signals() -> None:
    lines = ["- Glucose: 7.8 mmol/L (3.9-6.4)", "- HIV: âm tính",
             "- bảng công thức máu bình thường"]
    s = analyze_layouts(lines, CFG)
    assert s.lab_numeric_with_unit >= 1
    assert s.lab_reference_range == 1
    assert s.lab_qualitative == 1
    assert s.lab_normality_phrase == 1
    assert s.lab_lines >= 2


def test_medication_layout_signals() -> None:
    s = analyze_layouts(["- metoprolol 25mg po bid"], CFG)
    assert s.med_strength == 1 and s.med_route == 1 and s.med_frequency == 1
    assert s.med_full_pattern == 1 and s.med_lines == 1


def test_imaging_layout_trailing_parenthetical() -> None:
    s = analyze_layouts(["- Tim to (chụp x-quang ngực)"], CFG)
    assert s.imaging_modality == 1 and s.imaging_lines == 1
    assert s.imaging_trailing_parenthetical == 1


def test_plain_narrative_has_no_layout_signals() -> None:
    s = analyze_layouts(["Bệnh nhân đau ngực và khó thở."], CFG)
    assert s.lab_lines == 0 and s.med_lines == 0 and s.imaging_lines == 0


def test_blank_lines_ignored() -> None:
    assert analyze_layouts(["", "   ", "\t"], CFG).lab_lines == 0
