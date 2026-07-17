"""Phase 2A — public corpus analysis (descriptive statistics only).

Parses the organizer's PUBLIC INPUT documents through the EXISTING L1 pipeline
(no preprocessing behavior changed) and measures document lengths, section
headings, canonical routable-unit frequencies, list/table/key-value structure,
laboratory layouts, medication-list patterns, imaging/report styles, and overall
document structure.

Explicitly NOT here: prediction, entity extraction, ontology linking, assertion
classification, LLM/ML, training, or synthetic-data generation. Route cases are
processing cases, never entity types or predictions. Organizer input documents
have no ground truth and are never labeled or committed.
"""

from __future__ import annotations

from .config import AnalysisConfig, load_analysis_config
from .corpus import aggregate
from .distributions import histogram, summarize
from .document import analyze_document
from .layouts import analyze_layouts
from .loader import CorpusDocument, corpus_sha256, discover_documents, load_corpus
from .models import (
    CorpusAnalysis,
    Distribution,
    DocumentAnalysis,
    LayoutStats,
    LengthStats,
    RoutingStats,
    SectionStats,
    StructureStats,
)
from .pipeline import Phase2AConfig, analyze_corpus_dir, run_corpus_analysis
from .reporting import determinism_hash, to_dict, to_json, to_markdown

__all__ = [
    "AnalysisConfig",
    "load_analysis_config",
    "CorpusDocument",
    "discover_documents",
    "load_corpus",
    "corpus_sha256",
    "Distribution",
    "LengthStats",
    "SectionStats",
    "StructureStats",
    "RoutingStats",
    "LayoutStats",
    "DocumentAnalysis",
    "CorpusAnalysis",
    "summarize",
    "histogram",
    "analyze_layouts",
    "analyze_document",
    "aggregate",
    "Phase2AConfig",
    "run_corpus_analysis",
    "analyze_corpus_dir",
    "to_dict",
    "to_json",
    "to_markdown",
    "determinism_hash",
]
