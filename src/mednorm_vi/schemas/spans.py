"""Span coordinate contracts.

Every span-bearing object in MedNorm-VI must distinguish three coordinate
systems (spec section 4):

* **absolute** — indexes into ``original_text``; the ONLY source of emitted
  ``text``/``position``. Immutable, authoritative.
* **local** — indexes within the containing segment (optional).
* **normalized** — indexes into ``normalized_view`` used for matching/retrieval
  (optional). Normalized coordinates must never become output coordinates.

Nothing here loads a model. These are pure typed contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character interval ``[start, end)``.

    ``start``/``end`` are code-point indexes into some reference text. The
    end-exclusive convention matches the organizer's ``position`` field.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} must be >= start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def is_empty(self) -> bool:
        return self.end == self.start

    def overlaps(self, other: Span) -> bool:
        """True if the two half-open intervals share at least one index."""
        return self.start < other.end and other.start < self.end

    def slice_of(self, text: str) -> str:
        """Return ``text[start:end]`` — the substring this span designates."""
        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class SpanCoordinates:
    """The (absolute, local, normalized) coordinate triplet for a span.

    ``absolute`` is required and authoritative. ``local`` and ``normalized`` are
    optional context. Output ``position`` is always derived from ``absolute``.
    """

    absolute: Span
    local: Span | None = None
    normalized: Span | None = None

    def output_position(self) -> tuple[int, int]:
        """The ``[start, end)`` pair to emit — always from absolute coordinates."""
        return (self.absolute.start, self.absolute.end)


@dataclass(frozen=True, slots=True)
class OffsetAlignment:
    """Reversible mapping between ``original_text`` and ``normalized_view``.

    Contract only. ``original_to_normalized[i]`` is the normalized index that
    original code point ``i`` maps to (and vice-versa). For the identity case
    (normalization changed nothing structural, e.g. NFC on already-composed
    text), leave both maps empty and identity mapping is assumed.

    The essential guarantee: every normalized code point maps back to an original
    code point, so a match found on ``normalized_view`` can be re-expressed in
    authoritative absolute coordinates.
    """

    original_length: int
    normalized_length: int
    original_to_normalized: tuple[int, ...] = field(default_factory=tuple)
    normalized_to_original: tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_identity(self) -> bool:
        return not self.original_to_normalized and not self.normalized_to_original

    def to_original(self, normalized_index: int) -> int:
        """Map a ``normalized_view`` index back to an ``original_text`` index."""
        if self.is_identity:
            return normalized_index
        return self.normalized_to_original[normalized_index]

    def to_normalized(self, original_index: int) -> int:
        """Map an ``original_text`` index to a ``normalized_view`` index."""
        if self.is_identity:
            return original_index
        return self.original_to_normalized[original_index]


@dataclass(frozen=True, slots=True)
class SpanProvenance:
    """Where a span proposal came from and how confident its source was.

    Mirrors spec section 4: every span proposal stores ``source_model``,
    ``local_score``, ``route``, normalized form, and (implicitly, via the owning
    object) original coordinates.
    """

    source_model: str
    local_score: float
    route: str | None = None
    normalized_form: str | None = None
    rule_ids: tuple[str, ...] = field(default_factory=tuple)
    features: dict[str, float] = field(default_factory=dict)
