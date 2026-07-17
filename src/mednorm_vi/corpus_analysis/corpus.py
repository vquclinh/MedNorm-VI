"""Deterministic corpus-level aggregation (Phase 2A).

Aggregates per-document measurements into distributions and frequency tables.
Every collection is ordered deterministically (count desc, then key asc) so the
reports are byte-reproducible across runs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from .config import AnalysisConfig
from .distributions import histogram, summarize
from .models import CorpusAnalysis, DocumentAnalysis, LayoutStats


def _ranked(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Frequency table ordered by count desc, then key asc (deterministic)."""
    return tuple(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def aggregate(
    documents: list[DocumentAnalysis], *, corpus_id: str, corpus_sha256: str,
    config: AnalysisConfig, l1_config_hash: str,
) -> CorpusAnalysis:
    """Aggregate per-document analyses into a deterministic corpus view."""
    headings: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    unit_kinds: Counter[str] = Counter()
    cases: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    warnings: Counter[str] = Counter()

    for d in documents:
        headings.update(d.sections.headings)
        categories.update(d.sections.categories)
        unit_kinds.update(d.routing.units_by_kind)
        cases.update({k: v for k, v in d.routing.cases.items() if v})
        profiles[d.profile] += 1
        warnings.update(d.warnings)

    dists = {
        "characters": summarize([d.length.characters for d in documents]),
        "bytes_utf8": summarize([d.length.bytes_utf8 for d in documents]),
        "lines": summarize([d.length.lines for d in documents]),
        "non_blank_lines": summarize([d.length.non_blank_lines for d in documents]),
        "max_line_length": summarize([d.length.max_line_length for d in documents]),
        "tokens": summarize([d.length.tokens for d in documents]),
        "sentences": summarize([d.length.sentences for d in documents]),
        "sections": summarize([d.sections.sections for d in documents]),
        "list_items": summarize([d.structure.list_items for d in documents]),
        "table_rows": summarize([d.structure.table_rows for d in documents]),
        "key_value_rows": summarize([d.structure.key_value_rows for d in documents]),
        "routable_units": summarize(
            [sum(d.routing.units_by_kind.values()) for d in documents]),
    }

    # layout totals across the corpus (field-by-field, deterministic order)
    layout_totals: dict[str, int] = {}
    for field_name in sorted(asdict(LayoutStats()).keys()):
        layout_totals[field_name] = sum(getattr(d.layout, field_name) for d in documents)

    # document-level presence counts ("how many documents have any X")
    docs_with = {
        "sections": sum(1 for d in documents if d.sections.sections),
        "list_items": sum(1 for d in documents if d.structure.list_items),
        "numbered_items": sum(1 for d in documents if d.structure.numbered_items),
        "key_value_rows": sum(1 for d in documents if d.structure.key_value_rows),
        "table_like_rows": sum(1 for d in documents if d.structure.table_like_rows),
        "lab_layout": sum(1 for d in documents if d.layout.lab_lines),
        "lab_reference_range": sum(1 for d in documents if d.layout.lab_reference_range),
        "medication_layout": sum(1 for d in documents if d.layout.med_lines),
        "medication_full_pattern": sum(1 for d in documents if d.layout.med_full_pattern),
        "imaging_layout": sum(1 for d in documents if d.layout.imaging_lines),
        "imaging_trailing_parenthetical": sum(
            1 for d in documents if d.layout.imaging_trailing_parenthetical),
        "l1_warnings": sum(1 for d in documents if d.warnings),
    }

    return CorpusAnalysis(
        corpus_id=corpus_id, n_documents=len(documents), corpus_sha256=corpus_sha256,
        analysis_version=config.analysis_version, config_hash=config.config_hash,
        l1_config_hash=l1_config_hash, documents=tuple(documents), distributions=dists,
        heading_frequencies=_ranked(headings), category_frequencies=_ranked(categories),
        unit_kind_frequencies=_ranked(unit_kinds), case_frequencies=_ranked(cases),
        profile_frequencies=_ranked(profiles),
        length_histogram=tuple(histogram([d.length.characters for d in documents],
                                         config.length_buckets)),
        layout_totals=layout_totals, docs_with=docs_with,
        warnings=tuple(f"{code}:{count}" for code, count in _ranked(warnings)))


__all__ = ["aggregate"]
