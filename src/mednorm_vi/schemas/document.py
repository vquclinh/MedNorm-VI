"""L1 Document Intelligence contracts (spec section 4).

The ``DocumentGraph`` is the immutable backbone every later layer reads. It
holds the authoritative ``original_text``, a reversible ``NormalizedView``, and
a tree/graph of section and segment nodes — each carrying absolute offsets into
``original_text``.

Contracts only; no parsing logic here (see ``document_intelligence`` package).
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
