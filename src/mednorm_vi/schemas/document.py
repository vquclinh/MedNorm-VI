"""L1 document contracts (spec §4).

**NON-RUNTIME, except ``NormalizedView``** (Audit 0052).

``DocumentGraph`` in this module is the spec's five-field sketch. The runtime graph
is ``document_intelligence.models.DocumentGraph`` (eight fields, with builder version
and config hash), which is what ``analyze_document`` returns and what every layer
consumes. This copy is kept as documentation of §4's contract; nothing constructs it.

``NormalizedView`` *is* runtime — ``document_intelligence.models`` imports it from
here, which is why this module is not simply deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spans import OffsetAlignment, Span


@dataclass(frozen=True, slots=True)
class NormalizedView:
    """A normalized copy of the original text plus its reversible alignment.

    ``text`` here is for matching/retrieval/abbreviation-expansion ONLY. It must
    never be used to emit output spans; ``alignment`` maps any normalized index
    back to authoritative original coordinates.
    """

    text: str
    alignment: OffsetAlignment
    normalization_form: str = "NFC"


@dataclass(frozen=True, slots=True)
class SegmentNode:
    """A sentence, list item, or table-like row within a section.

    ``span`` indexes into ``original_text`` (absolute). ``local_offset`` is the
    absolute index of this segment's start, so a segment-local index ``i`` maps
    to absolute ``local_offset + i``.
    """

    segment_id: str
    span: Span
    local_offset: int
    kind: str = "sentence"  # sentence | list_item | table_row | cell | ...
    section_id: str | None = None

    def to_absolute(self, local_index: int) -> int:
        return self.local_offset + local_index


@dataclass(frozen=True, slots=True)
class SectionNode:
    """A document section carrying a discourse prior (evidence, not a rule).

    E.g. a "pre-admission medication list" section carries an ``isHistorical``
    prior with some ``prior_strength``; a local sentence may override it.
    """

    section_id: str
    span: Span
    matched_pattern: str | None = None
    prior_label: str | None = None  # e.g. "isHistorical", "isFamily", "DIAGNOSIS"
    prior_strength: float = 0.0
    segment_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DocumentGraph:
    """Immutable per-document structure produced by L1.

    Invariant: ``original_text`` is never mutated after construction. All nodes
    store absolute offsets into it.
    """

    document_id: str
    original_text: str
    normalized_view: NormalizedView | None = None
    sections: tuple[SectionNode, ...] = field(default_factory=tuple)
    segments: tuple[SegmentNode, ...] = field(default_factory=tuple)

    def substring(self, span: Span) -> str:
        return span.slice_of(self.original_text)
