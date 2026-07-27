"""Hard invariants for the unified L3 span lattice.

Every rule here **raises**. Spec §16 is explicit: "Never silently repair output."
A lattice that violates an invariant is a defect in the expert that produced it,
and hiding it would make every downstream number untrustworthy.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import LatticeError, SpanLattice, SpanProposal


def validate_proposal(proposal: SpanProposal, original_text: str) -> None:
    """Enforce spec §4 for one lattice node."""
    if proposal.start < 0:
        raise LatticeError(f"span start {proposal.start} is negative")
    if proposal.end <= proposal.start:
        raise LatticeError(
            f"span [{proposal.start}, {proposal.end}) is empty or inverted; "
            "offsets are end-exclusive and spans must be non-empty")
    if proposal.end > len(original_text):
        raise LatticeError(
            f"span end {proposal.end} exceeds the text length {len(original_text)}")
    if original_text[proposal.start:proposal.end] != proposal.text:
        raise LatticeError(
            "span violates original_text[start:end] == text at "
            f"[{proposal.start}, {proposal.end})")
    if not proposal.sources:
        raise LatticeError(
            f"span [{proposal.start}, {proposal.end}) carries no provenance")
    for source in proposal.sources:
        if not source.expert_id:
            raise LatticeError(
                f"span [{proposal.start}, {proposal.end}) has a source without an expert id")
        if not source.proposal_id:
            raise LatticeError(
                f"span [{proposal.start}, {proposal.end}) has a source without a proposal id")


def validate_no_text_only_deduplication(proposals: Sequence[SpanProposal]) -> None:
    """Assert repeated surface forms at different offsets survived as distinct nodes.

    This is the positive form of spec §5 case C7: it is not enough to *intend* not
    to deduplicate by text — the lattice must be able to prove two nodes with the
    same text and different coordinates both exist when an expert proposed both.
    """
    seen: set[tuple[int, int]] = set()
    for proposal in proposals:
        if proposal.coordinates in seen:
            raise LatticeError(
                f"duplicate lattice coordinates {proposal.coordinates}; exact "
                "coordinate duplicates must be merged into one node, not repeated")
        seen.add(proposal.coordinates)


def validate_lattice(lattice: SpanLattice) -> None:
    """Validate the whole lattice: every node, plus global identity rules."""
    for proposal in lattice.proposals:
        validate_proposal(proposal, lattice.original_text)
        if proposal.document_id != lattice.document_id:
            raise LatticeError(
                f"node {proposal.coordinates} belongs to document "
                f"{proposal.document_id!r}, not {lattice.document_id!r}")
    validate_no_text_only_deduplication(lattice.proposals)
    ordered = sorted(lattice.proposals, key=lambda p: (p.start, p.end))
    if [p.coordinates for p in ordered] != [p.coordinates for p in lattice.proposals]:
        raise LatticeError("lattice proposals are not in canonical (start, end) order")


__all__ = [
    "validate_lattice",
    "validate_no_text_only_deduplication",
    "validate_proposal",
]
