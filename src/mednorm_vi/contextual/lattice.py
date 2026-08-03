"""Finite overlap/boundary lattice over aligned proposals (0081).

Everything reaching the verifier is a finite, indexed list of spans that already exist in the
source. The lattice groups proposals that compete for the same piece of text and offers each
group's alternatives together, so the model chooses **between** spans rather than writing
one.

One bounded boundary alternative is allowed: when two proposals in a group are separated by
at most one ordinary word, the 0079 safe-bridge rule may offer their union. That rule was
validated against every merge the 0078 run produced and refuses anything crossing
punctuation, connectors or an unexpected capital. There is no other expansion - no token
radius, no left growth, no free-form completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..reasoner.safe_bridge import evaluate_merge
from .document import DocumentView
from .proposals import Proposal

BRIDGE_SOURCE = "safe_bridge_union"


@dataclass(frozen=True, slots=True)
class LatticeGroup:
    """Competing alternatives for one region of the document."""

    start: int
    end: int
    options: tuple[Proposal, ...]

    @property
    def single(self) -> bool:
        return len(self.options) == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start, "end": self.end,
            "options": [o.as_dict() for o in self.options],
        }


@dataclass
class Lattice:
    groups: list[LatticeGroup] = field(default_factory=list)
    bridges_offered: int = 0

    @property
    def option_count(self) -> int:
        return sum(len(g.options) for g in self.groups)


def _cluster(proposals: list[Proposal]) -> list[list[Proposal]]:
    """Transitively overlapping proposals belong to the same decision."""
    clusters: list[list[Proposal]] = []
    for proposal in sorted(proposals, key=lambda p: (p.start, -p.end)):
        if clusters and proposal.start < max(p.end for p in clusters[-1]):
            clusters[-1].append(proposal)
        else:
            clusters.append([proposal])
    return clusters


def _bridge(
    document: DocumentView, left: Proposal, right: Proposal
) -> Proposal | None:
    """The union of two nearby same-type fragments, when the 0079 rule allows it."""
    if left.type != right.type or left.end > right.start:
        return None
    verdict = evaluate_merge(
        document.source, [[left.start, left.end], [right.start, right.end]]
    )
    if not verdict.accepted:
        return None
    start, end = left.start, right.end
    line = document.line_of(start)
    return Proposal(
        start=start, end=end, text=document.slice(start, end), type=left.type,
        sources=left.sources | right.sources | {BRIDGE_SOURCE},
        line_id=line.line_id if line else "", how=BRIDGE_SOURCE,
    )


def build_lattice(document: DocumentView, proposals: list[Proposal]) -> Lattice:
    """Group competing proposals and offer at most the safe boundary unions."""
    lattice = Lattice()
    for cluster in _cluster(proposals):
        options: list[Proposal] = list(cluster)
        for left in cluster:
            for right in cluster:
                if left.span >= right.span:
                    continue
                union = _bridge(document, left, right)
                if union is None:
                    continue
                if any(o.span == union.span and o.type == union.type for o in options):
                    continue
                options.append(union)
                lattice.bridges_offered += 1
        ordered = tuple(sorted(options, key=lambda p: (p.start, -p.end, p.type)))
        lattice.groups.append(
            LatticeGroup(
                start=min(o.start for o in ordered), end=max(o.end for o in ordered),
                options=ordered,
            )
        )
    return lattice


__all__ = ["BRIDGE_SOURCE", "Lattice", "LatticeGroup", "build_lattice"]
