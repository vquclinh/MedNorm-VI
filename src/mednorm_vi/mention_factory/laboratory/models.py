"""Laboratory parser data contracts (Phase 1B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ComponentSpan


@dataclass(frozen=True, slots=True)
class LaboratoryCell:
    """One extracted lab cell (name/value/unit/flag/reference_range) with offsets."""

    role: str
    start: int
    end: int
    text: str
    normalized: str | None = None


@dataclass(frozen=True, slots=True)
class LaboratoryParse:
    """A structured lab row/fragment parse (proposal source, not a final entity)."""

    parse_id: str
    document_id: str
    source_node_id: str
    source_kind: str  # key_value | table_like | semicolon_row | narrative
    cells: tuple[LaboratoryCell, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TestResultPair:
    """A candidate TEST_NAME -> TEST_RESULT pairing with a deterministic cost."""

    name_component: ComponentSpan
    result_component: ComponentSpan
    cost: float
    same_row: bool
    unit_match: bool


__all__ = ["LaboratoryCell", "LaboratoryParse", "TestResultPair"]
