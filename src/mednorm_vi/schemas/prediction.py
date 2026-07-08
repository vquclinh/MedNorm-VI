"""Final output contract (spec sections 1, 16, 19.1).

``EntityPrediction`` is the metric-decoder output that L9 validates and packages
into ``output/N.json``.

Two serializations exist:

* ``to_internal_dict`` / ``to_output_dict`` — INTERNAL normalized shape with the
  English enum ``type`` and always-present ``assertions``/``candidates`` lists.
  Used inside the pipeline; NOT the submission format.
* ``to_submission_dict`` — ORGANIZER-FACING shape: ``type`` serializes to the
  exact Vietnamese label, and ONLY the fields allowed for that type are emitted
  (see ``ORGANIZER_FIELDS_BY_TYPE``).

``position`` is derived from ABSOLUTE coordinates only. The invariant
``original_text[start:end] == text`` is enforced by ``validator.offsets``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ORGANIZER_FIELDS_BY_TYPE, ORGANIZER_LABEL_BY_TYPE
from .spans import SpanCoordinates


@dataclass(frozen=True, slots=True)
class EntityPrediction:
    """A single final entity for submission.

    Attributes map onto the organizer JSON schema. ``coords`` retains the full
    triplet internally; ``position`` / ``to_output_dict`` expose only the
    authoritative absolute span.
    """

    text: str
    type: str  # one of ENTITY_TYPES
    coords: SpanCoordinates
    assertions: tuple[str, ...] = field(default_factory=tuple)  # subset of ASSERTION_LABELS
    candidates: tuple[str, ...] = field(default_factory=tuple)  # ICD-10 codes or RxCUIs

    @property
    def start(self) -> int:
        return self.coords.absolute.start

    @property
    def end(self) -> int:
        return self.coords.absolute.end

    @property
    def position(self) -> tuple[int, int]:
        return self.coords.output_position()

    def to_internal_dict(self) -> dict[str, object]:
        """Serialize to the INTERNAL normalized shape (English enum ``type``,
        always-present ``assertions``/``candidates`` lists). Not for submission."""
        start, end = self.position
        return {
            "text": self.text,
            "type": self.type,
            "assertions": list(self.assertions),
            "candidates": list(self.candidates),
            "position": [start, end],
        }

    # Backwards-compatible alias for the internal shape.
    def to_output_dict(self) -> dict[str, object]:
        """Deprecated alias of :meth:`to_internal_dict` (internal shape)."""
        return self.to_internal_dict()

    def to_submission_dict(self) -> dict[str, object]:
        """Serialize to the ORGANIZER-FACING JSON shape.

        ``type`` becomes the exact Vietnamese label and only the fields allowed
        for this type are emitted, in the organizer field order. Raises
        ``ValueError`` for an unknown internal type (caller should validate
        first). Vietnamese text is preserved verbatim; write with
        ``json.dumps(..., ensure_ascii=False)`` to keep UTF-8 characters.
        """
        if self.type not in ORGANIZER_LABEL_BY_TYPE:
            raise ValueError(f"unknown internal entity type {self.type!r}")
        start, end = self.position
        available: dict[str, object] = {
            "text": self.text,
            "type": ORGANIZER_LABEL_BY_TYPE[self.type],
            "assertions": list(self.assertions),
            "candidates": list(self.candidates),
            "position": [start, end],
        }
        allowed = ORGANIZER_FIELDS_BY_TYPE[self.type]
        return {name: available[name] for name in allowed}
