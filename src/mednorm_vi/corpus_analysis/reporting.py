"""Deterministic JSON + Markdown reporting for Phase 2A.

Reports are byte-reproducible: no timestamps, no wall-clock, no set iteration —
identity comes from the corpus content hash and the config hashes. Reports are
written under ``reports/corpus_analysis/`` (git-ignored) because they summarise
organizer INPUT documents, which are never committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .models import CorpusAnalysis, DocumentAnalysis

SCHEMA_VERSION = "corpus-analysis-2a-1"


def document_to_dict(d: DocumentAnalysis) -> dict[str, Any]:
    return {
        "document_id": d.document_id,
        "source_name": d.source_name,
        "content_sha256": d.content_sha256,
        "profile": d.profile,
        "l1_valid": d.l1_valid,
        "length": asdict(d.length),
        "sections": {
            "sections": d.sections.sections,
            "categorized": d.sections.categorized,
            "uncategorized": d.sections.uncategorized,
            "categories": list(d.sections.categories),
            "headings": list(d.sections.headings),
        },
        "structure": asdict(d.structure),
        "routing": {
            "units_by_kind": dict(sorted(d.routing.units_by_kind.items())),
            "cases": dict(sorted(d.routing.cases.items())),
            "multi_case_units": d.routing.multi_case_units,
            "unrouted_units": d.routing.unrouted_units,
        },
        "layout": asdict(d.layout),
        "warnings": list(d.warnings),
    }


def to_dict(analysis: CorpusAnalysis) -> dict[str, Any]:
    """Full deterministic report payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": analysis.corpus_id,
        "n_documents": analysis.n_documents,
        "corpus_sha256": analysis.corpus_sha256,
        "analysis_version": analysis.analysis_version,
        "config_hash": analysis.config_hash,
        "l1_config_hash": analysis.l1_config_hash,
        "scope": {
            "descriptive_only": True,
            "entity_extraction": False,
            "prediction": False,
            "linking": False,
            "ml_or_llm": False,
            "l1_behavior_modified": False,
            "derived_cases_c6_c7": "not computed (specialists deliberately not run)",
        },
        "distributions": {k: analysis.distributions[k].as_dict()
                          for k in sorted(analysis.distributions)},
        "length_histogram": [list(x) for x in analysis.length_histogram],
        "profile_frequencies": [list(x) for x in analysis.profile_frequencies],
        "unit_kind_frequencies": [list(x) for x in analysis.unit_kind_frequencies],
        "case_frequencies": [list(x) for x in analysis.case_frequencies],
        "category_frequencies": [list(x) for x in analysis.category_frequencies],
        "heading_frequencies": [list(x) for x in analysis.heading_frequencies],
        "layout_totals": dict(sorted(analysis.layout_totals.items())),
        "docs_with": dict(sorted(analysis.docs_with.items())),
        "l1_warnings": list(analysis.warnings),
        "documents": [document_to_dict(d) for d in analysis.documents],
    }


def to_json(analysis: CorpusAnalysis) -> str:
    return json.dumps(to_dict(analysis), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def determinism_hash(analysis: CorpusAnalysis) -> str:
    canonical = json.dumps(to_dict(analysis), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _table(rows: list[tuple[str, int]], header: tuple[str, str], limit: int = 0) -> list[str]:
    shown = rows[:limit] if limit else rows
    out = [f"| {header[0]} | {header[1]} |", "| --- | ---: |"]
    out += [f"| {k} | {v} |" for k, v in shown]
    if limit and len(rows) > limit:
        out.append(f"| _… {len(rows) - limit} more_ | |")
    return out


def to_markdown(analysis: CorpusAnalysis, *, top_headings: int = 25) -> str:
    a = analysis
    lines: list[str] = [
        "# Phase 2A — Public Corpus Analysis",
        "",
        "**Descriptive statistics only.** No prediction, no entity extraction, no",
        "linking, no LLM/ML. Every document was parsed through the existing L1",
        "pipeline with its existing config — no preprocessing behavior was changed.",
        "Route cases are **processing cases**, not entity types or predictions;",
        "derived cases C6/C7 are not computed (the mention specialists are not run).",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| corpus_id | `{a.corpus_id}` |",
        f"| documents | {a.n_documents} |",
        f"| corpus_sha256 | `{a.corpus_sha256}` |",
        f"| analysis_version | `{a.analysis_version}` |",
        f"| analysis config_hash | `{a.config_hash[:16]}…` |",
        f"| L1 config_hash | `{a.l1_config_hash[:16]}…` |",
        f"| determinism_hash | `{determinism_hash(a)}` |",
        "",
        "## Document length distributions",
        "",
        "| measure | n | total | min | p25 | median | p75 | p90 | max | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(a.distributions):
        dist = a.distributions[key]
        lines.append(
            f"| {key} | {dist.n} | {dist.total} | {dist.minimum} | {dist.p25} | "
            f"{dist.median} | {dist.p75} | {dist.p90} | {dist.maximum} | {dist.mean} |")

    lines += ["", "## Length histogram (characters)", ""]
    lines += _table(list(a.length_histogram), ("bucket", "documents"))

    lines += ["", "## Structural profile", ""]
    lines += _table(list(a.profile_frequencies), ("profile", "documents"))

    lines += ["", "## Canonical routable units (L1/L2 structure)", ""]
    lines += _table(list(a.unit_kind_frequencies), ("unit kind", "count"))

    lines += ["", "## Processing cases (C1-C5; not predictions)", ""]
    lines += _table(list(a.case_frequencies), ("case", "units"))

    lines += ["", "## Section categories (L1)", ""]
    lines += (_table(list(a.category_frequencies), ("category", "count"))
              if a.category_frequencies else ["_none detected_"])

    lines += ["", f"## Section headings (top {top_headings})", ""]
    lines += (_table(list(a.heading_frequencies), ("heading", "count"), limit=top_headings)
              if a.heading_frequencies else ["_none detected_"])

    lines += ["", "## Layout totals (line shape, not entities)", ""]
    lines += _table(sorted(a.layout_totals.items()), ("signal", "lines"))

    lines += ["", "## Documents containing each feature", ""]
    lines += _table(sorted(a.docs_with.items()), ("feature", "documents"))

    if a.warnings:
        lines += ["", "## L1 warnings (code:count)", ""]
        lines += [f"- `{w}`" for w in a.warnings]

    lines += ["", "## Per-document summary", "",
              "| doc | chars | lines | sections | list items | kv rows | units | profile |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for doc in a.documents:
        units = sum(doc.routing.units_by_kind.values())
        lines.append(
            f"| {doc.source_name} | {doc.length.characters} | {doc.length.lines} | "
            f"{doc.sections.sections} | {doc.structure.list_items} | "
            f"{doc.structure.key_value_rows} | {units} | {doc.profile} |")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA_VERSION", "document_to_dict", "to_dict", "to_json", "to_markdown",
           "determinism_hash"]
