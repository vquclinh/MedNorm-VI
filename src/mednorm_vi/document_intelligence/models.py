"""Typed L1 data model: config, structural nodes, normalized views, DocumentGraph.

Reuses ``Span``/``SpanProvenance`` (schemas.spans) and ``NormalizedView``
(schemas.document). A single flexible :class:`StructuralNode` represents every
structural kind so ordering, serialization, and validation stay uniform. Node
identity is never text-only — it is an id plus absolute coordinates, so repeated
surface forms at different positions remain distinct nodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..schemas.document import NormalizedView
from ..schemas.spans import Span
from .alignment import CharAlignment
from .io import DocumentSource


class NodeKind(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    LINE = "line"
    SENTENCE = "sentence"
    LIST_ITEM = "list_item"
    LIST_MARKER = "list_marker"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    DELIMITER = "delimiter"
    TOKEN = "token"


@dataclass(frozen=True, slots=True)
class StructuralNode:
    """One structural node with absolute ``[start, end)`` coordinates.

    Only the fields relevant to a node's ``kind`` are populated. L1 labels
    structure ("possible section header", "table_like row"); it never asserts a
    medical entity.
    """

    node_id: str
    kind: NodeKind
    start: int
    end: int
    parent_id: str | None = None
    children: tuple[str, ...] = ()
    # section-specific
    category: str | None = None
    header_start: int | None = None
    header_end: int | None = None
    confidence: float | None = None
    matched_rule: str | None = None
    prior_label: str | None = None
    prior_strength: float | None = None
    # list-specific
    marker_start: int | None = None
    marker_end: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    indent: int | None = None
    depth: int | None = None
    # row-specific
    row_kind: str | None = None  # "table_like" | "key_value_like"
    # token-specific
    is_punctuation: bool | None = None
    whitespace_before: bool | None = None
    token_category: str | None = None
    normalized_text: str | None = None
    line_id: str | None = None
    segment_id: str | None = None
    section_id: str | None = None
    # generic
    provenance: str | None = None
    warnings: tuple[str, ...] = ()
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)

    def text(self, original_text: str) -> str:
        return original_text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class NormalizationStageRecord:
    """Provenance for one normalization stage applied to build a view."""

    name: str
    version: str
    input_length: int
    output_length: int
    transformation_count: int
    config_hash: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """A named normalized view with its composed alignment and stage provenance."""

    name: str
    text: str
    alignment: CharAlignment
    stages: tuple[NormalizationStageRecord, ...] = ()

    def as_schema_view(self) -> NormalizedView:
        form = "+".join(s.name for s in self.stages) or "identity"
        return NormalizedView(
            text=self.text,
            alignment=self.alignment.to_offset_alignment(),
            normalization_form=form,
        )


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class DocumentWarning:
    """A structural warning (uncertain interpretation), never a hard failure."""

    code: str
    message: str
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class L1Config:
    """Resolved L1 configuration (see configs/document_intelligence/base.yaml)."""

    version: int
    builder_version: str
    encoding: str
    bom_policy: str
    normalization_views: dict[str, tuple[str, ...]]
    matching_view: str
    sentence_terminators: tuple[str, ...]
    newline_is_boundary: bool
    ordered_marker_patterns: tuple[str, ...]
    bullet_markers: tuple[str, ...]
    kv_min_colon_pairs: int
    kv_regex: str
    table_delimiters: tuple[str, ...]
    section_lexicon_file: str
    fuzzy_threshold: float
    require_structural_evidence: bool
    max_header_chars: int
    colon_bonus: bool
    blank_bonus: bool
    emit_delimiter_tokens: bool
    strict: bool
    require_full_line_coverage: bool
    id_zero_pad: int
    config_hash: str

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> L1Config:
        io_cfg = data.get("io", {}) or {}
        norm = data.get("normalization", {}) or {}
        views_raw = norm.get("views", {}) or {}
        sent = data.get("sentences", {}) or {}
        lists_cfg = data.get("lists", {}) or {}
        table = data.get("table_rows", {}) or {}
        sec = data.get("sections", {}) or {}
        toks = data.get("tokens", {}) or {}
        val = data.get("validation", {}) or {}
        det = data.get("determinism", {}) or {}
        return L1Config(
            version=int(data.get("version", 1)),
            builder_version=str(data.get("builder_version", "l1-1a")),
            encoding=str(io_cfg.get("encoding", "utf-8")),
            bom_policy=str(io_cfg.get("bom_policy", "preserve")),
            normalization_views={
                str(k): tuple(str(s) for s in v) for k, v in views_raw.items()
            },
            matching_view=str(norm.get("matching_view", "accent")),
            sentence_terminators=tuple(sent.get("terminators", [".", "!", "?", "…", ";"])),
            newline_is_boundary=bool(sent.get("newline_is_boundary", True)),
            ordered_marker_patterns=tuple(lists_cfg.get("ordered_markers", [])),
            bullet_markers=tuple(lists_cfg.get("bullet_markers", [])),
            kv_min_colon_pairs=int(table.get("min_colon_pairs", 1)),
            kv_regex=str(table.get("key_value_regex", r"^[^:]{1,60}:\s*\S")),
            table_delimiters=tuple(table.get("delimiters", [":", ";", "\t"])),
            section_lexicon_file=str(sec.get("lexicon", "section_lexicon_v1.yaml")),
            fuzzy_threshold=float(sec.get("fuzzy_threshold", 0.86)),
            require_structural_evidence=bool(sec.get("require_structural_evidence", True)),
            max_header_chars=int(sec.get("max_header_chars", 64)),
            colon_bonus=bool(sec.get("colon_terminated_bonus", True)),
            blank_bonus=bool(sec.get("blank_line_context_bonus", True)),
            emit_delimiter_tokens=bool(toks.get("emit_delimiter_tokens", True)),
            strict=bool(val.get("strict", True)),
            require_full_line_coverage=bool(val.get("require_full_line_coverage", True)),
            id_zero_pad=int(det.get("id_zero_pad", 4)),
            config_hash=_hash_mapping(data),
        )


def _hash_mapping(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentGraph:
    """Immutable L1 output: original text + normalized views + structural nodes.

    Invariant: ``original_text`` is never mutated; every node's span slices back
    to its exact original substring; normalized views never provide output
    coordinates.
    """

    document_id: str
    source: DocumentSource
    original_text: str
    normalized: dict[str, NormalizedDocument]
    nodes: tuple[StructuralNode, ...]
    warnings: tuple[DocumentWarning, ...]
    builder_version: str
    config_hash: str

    def substring(self, node: StructuralNode) -> str:
        return self.original_text[node.start : node.end]

    def nodes_of_kind(self, kind: NodeKind) -> tuple[StructuralNode, ...]:
        return tuple(n for n in self.nodes if n.kind is kind)

    def by_id(self) -> dict[str, StructuralNode]:
        return {n.node_id: n for n in self.nodes}

    def children_of(self, node_id: str) -> tuple[StructuralNode, ...]:
        index = self.by_id()
        parent = index.get(node_id)
        if parent is None:
            return ()
        return tuple(index[c] for c in parent.children if c in index)


__all__ = [
    "NodeKind",
    "StructuralNode",
    "NormalizationStageRecord",
    "NormalizedDocument",
    "Severity",
    "DocumentWarning",
    "L1Config",
    "DocumentGraph",
]
