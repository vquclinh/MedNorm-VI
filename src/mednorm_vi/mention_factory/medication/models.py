"""Medication grammar data contracts (Phase 1B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ComponentSpan


@dataclass(frozen=True, slots=True)
class MedicationParse:
    """A structured medication parse (proposal source; not a final entity).

    ``components`` hold exact original offsets for each recognized field. A parsed
    field does NOT imply it belongs to the final organizer span (L4 decides).
    """

    parse_id: str
    document_id: str
    source_node_id: str
    name_start: int
    name_end: int
    name_text: str
    name_known: bool
    components: tuple[ComponentSpan, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def roles(self) -> set[str]:
        return {c.role for c in self.components}


@dataclass(frozen=True, slots=True)
class MedicationBoundaryCandidate:
    """One proposed boundary over the same parse (L4 picks finals)."""

    kind: str  # name_only | name_strength | name_strength_form | name_strength_route | full
    start: int
    end: int
    text: str


__all__ = ["MedicationParse", "MedicationBoundaryCandidate"]
