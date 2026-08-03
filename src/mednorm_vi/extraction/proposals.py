"""Proposals from every source, aligned to exact source offsets (0081).

A proposal is a *candidate mention*, not an entity. Three independent sources produce them -
the E3 mention expert, Qwen reading the document, and whole-phrase matches against the
governed KB - and they are deliberately kept separable so the diagnostics can say which
source found what, and so a variant can be built from a subset.

Everything here is offsets and literal source slices. `Proposal.text` is always
`source[start:end]`; there is no code path that stores text a document does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..validation.organizer import ORGANIZER_TYPES
from .alignment import Alignment, DocumentView, align

SOURCE_E3 = "e3"
SOURCE_QWEN = "qwen_context"
SOURCE_ALIAS = "governed_alias"
SOURCES: tuple[str, ...] = (SOURCE_E3, SOURCE_QWEN, SOURCE_ALIAS)

REJECT_UNKNOWN_TYPE = "unknown_type"


@dataclass(frozen=True, slots=True)
class Proposal:
    """One aligned candidate mention. `text` is a literal slice of the source."""

    start: int
    end: int
    text: str
    type: str
    sources: frozenset[str] = field(default_factory=frozenset)
    line_id: str = ""
    how: str = ""
    #: True when this span strictly contains a same-type proposal the seed expert produced.
    #: That is the fragment-completion case: the seed found part of a concept and another
    #: source found the whole of it.
    subsumes_seed: bool = False

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def agreed(self) -> bool:
        """Found independently by more than one source."""
        return len(self.sources) > 1

    def contains(self, other: Proposal) -> bool:
        return self.start <= other.start and other.end <= self.end and self.span != other.span

    def overlaps(self, other: Proposal) -> bool:
        return self.start < other.end and other.start < self.end

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start, "end": self.end, "text": self.text, "type": self.type,
            "sources": sorted(self.sources), "line_id": self.line_id, "how": self.how,
            "subsumes_seed": self.subsumes_seed,
        }


@dataclass
class ProposalPool:
    """Aligned proposals plus the tally of everything that was refused."""

    proposals: list[Proposal] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def add(self, proposal: Proposal) -> None:
        """Merge into an existing identical span+type, unioning the sources."""
        for index, existing in enumerate(self.proposals):
            if existing.span == proposal.span and existing.type == proposal.type:
                self.proposals[index] = replace(
                    existing, sources=existing.sources | proposal.sources
                )
                return
        self.proposals.append(proposal)

    def mark_seed_completions(self) -> int:
        """Flag every proposal that strictly contains a same-type seed proposal."""
        seeds = [p for p in self.proposals if SOURCE_E3 in p.sources]
        marked = 0
        for index, proposal in enumerate(self.proposals):
            if SOURCE_E3 in proposal.sources:
                continue
            if any(
                proposal.contains(seed) and proposal.type == seed.type for seed in seeds
            ):
                self.proposals[index] = replace(proposal, subsumes_seed=True)
                marked += 1
        return marked

    def sorted(self) -> list[Proposal]:
        """Document order, longest first at the same start - stable for the lattice."""
        return sorted(self.proposals, key=lambda p: (p.start, -p.end, p.type))

    def by_source(self, source: str) -> list[Proposal]:
        return [p for p in self.proposals if source in p.sources]


def propose(
    pool: ProposalPool,
    document: DocumentView,
    *,
    source: str,
    line_id: str,
    text: str,
    entity_type: str,
    occurrence: int = 0,
) -> Alignment:
    """Align one raw proposal and record it, or count the reason it was refused."""
    if entity_type not in ORGANIZER_TYPES:
        pool.reject(REJECT_UNKNOWN_TYPE)
        return Alignment(reason=REJECT_UNKNOWN_TYPE)
    alignment = align(document, line_id, text, occurrence)
    if not alignment.ok:
        pool.reject(alignment.reason)
        return alignment
    pool.add(
        Proposal(
            start=alignment.start, end=alignment.end,
            text=document.slice(alignment.start, alignment.end),
            type=entity_type, sources=frozenset({source}),
            line_id=alignment.line_id, how=alignment.how,
        )
    )
    return alignment


def propose_from_offsets(
    pool: ProposalPool,
    document: DocumentView,
    *,
    source: str,
    start: int,
    end: int,
    entity_type: str,
) -> bool:
    """For sources that already own exact offsets (E3). Still re-sliced from the source."""
    if entity_type not in ORGANIZER_TYPES:
        pool.reject(REJECT_UNKNOWN_TYPE)
        return False
    if not (0 <= start < end <= len(document.source)):
        pool.reject("offsets_out_of_range")
        return False
    line = document.line_of(start)
    pool.add(
        Proposal(
            start=start, end=end, text=document.slice(start, end), type=entity_type,
            sources=frozenset({source}), line_id=line.line_id if line else "",
            how="exact_offsets",
        )
    )
    return True


__all__ = [
    "REJECT_UNKNOWN_TYPE",
    "SOURCES",
    "SOURCE_ALIAS",
    "SOURCE_E3",
    "SOURCE_QWEN",
    "Proposal",
    "ProposalPool",
    "propose",
    "propose_from_offsets",
]
