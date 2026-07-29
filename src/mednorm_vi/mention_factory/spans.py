"""Expert-independent span contracts for L3 (spec §4, §6).

One entity-span type, shared by every mention expert and by the training paths
that supervise them. It previously lived inside a single expert's module, which
meant other experts imported that expert to get a contract that was never
specific to it; the contract lives here instead.

The invariant is spec §4's, and it is checked rather than assumed:

    original_text[start:end] == text        end-exclusive, over code points
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.constants import ENTITY_TYPES

SPAN_CONTRACT_VERSION = "mention-span-v1"

# The canonical type order for any dense per-type tensor, grid or label vocabulary
# an expert builds. Fixed so a checkpoint's label axis means the same thing in
# every expert and across every run.
DEFAULT_TYPE_ORDER: tuple[str, ...] = (
    "DIAGNOSIS",
    "MEDICATION",
    "SYMPTOM",
    "TEST_NAME",
    "TEST_RESULT",
)


class SpanContractError(ValueError):
    """Raised when a span violates the offset or type contract. Never repaired."""


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """A gold or decoded contiguous entity span in original character offsets."""

    start: int
    end: int
    entity_type: str
    text: str

    def validate_against(self, original_text: str) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise SpanContractError(f"unsupported entity type {self.entity_type!r}")
        if self.end <= self.start:
            raise SpanContractError(f"invalid entity offsets {self.start}:{self.end}")
        if original_text[self.start : self.end] != self.text:
            raise SpanContractError("entity text is not an exact original_text slice")


def assert_type_order_complete(type_order: tuple[str, ...]) -> None:
    """A label axis must cover exactly the five organizer types, no more."""
    if set(type_order) != set(ENTITY_TYPES):
        raise SpanContractError(
            f"type order {list(type_order)} does not cover exactly the five "
            f"entity types {sorted(ENTITY_TYPES)}")
    if len(type_order) != len(set(type_order)):
        raise SpanContractError(f"type order {list(type_order)} repeats a type")


__all__ = [
    "DEFAULT_TYPE_ORDER",
    "SPAN_CONTRACT_VERSION",
    "EntitySpan",
    "SpanContractError",
    "assert_type_order_complete",
]
