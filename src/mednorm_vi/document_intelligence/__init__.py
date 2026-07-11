"""L1 — Document Intelligence (spec section 4).

Converts an exact UTF-8 clinical document into a deterministic
:class:`DocumentGraph` while preserving reversible mappings to the immutable
``original_text``. L1 identifies STRUCTURE only ("possible section header",
"list-like", "table-like"); it never claims a medical entity was detected and
never emits organizer predictions. ``original_text[start:end] == node.text`` for
every structural node.

High-level entry points:
    analyze_text(text, ...)     -> DocumentGraph  (in-memory string)
    analyze_document(path, ...) -> DocumentGraph  (file on disk)
"""

from __future__ import annotations

from pathlib import Path

from .builder import build_document_graph, default_config, load_config
from .io import DocumentLoadError, LoadedDocument, from_text, load_document
from .models import (
    DocumentGraph,
    DocumentWarning,
    L1Config,
    NodeKind,
    NormalizedDocument,
    StructuralNode,
)
from .sections import SectionLexicon


def analyze_text(
    text: str,
    *,
    config: L1Config | None = None,
    lexicon: SectionLexicon | None = None,
    source_path: str | None = None,
) -> DocumentGraph:
    """Build a DocumentGraph from an in-memory string (default config if omitted)."""
    if config is None or lexicon is None:
        default_cfg, default_lex = default_config()
        config = config or default_cfg
        lexicon = lexicon or default_lex
    document = from_text(text, source_path=source_path)
    return build_document_graph(document, config, lexicon)


def analyze_document(
    path: str | Path,
    *,
    config: L1Config | None = None,
    lexicon: SectionLexicon | None = None,
) -> DocumentGraph:
    """Load a file exactly and build its DocumentGraph."""
    if config is None or lexicon is None:
        default_cfg, default_lex = default_config()
        config = config or default_cfg
        lexicon = lexicon or default_lex
    document = load_document(path, encoding=config.encoding, bom_policy=config.bom_policy)
    return build_document_graph(document, config, lexicon)


__all__ = [
    "analyze_text",
    "analyze_document",
    "build_document_graph",
    "load_config",
    "default_config",
    "load_document",
    "from_text",
    "LoadedDocument",
    "DocumentLoadError",
    "DocumentGraph",
    "StructuralNode",
    "NormalizedDocument",
    "DocumentWarning",
    "NodeKind",
    "L1Config",
    "SectionLexicon",
]
