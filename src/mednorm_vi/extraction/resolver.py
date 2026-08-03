"""Deterministic span and type resolution (0081).

After the verifier has voted, the final entity set is decided here by fixed rules, not by the
model. Two proposals that overlap cannot both be emitted, and the choice between them is made
the same way every time:

1. the complete concept beats a fragment of it - a span that contains another wins;
2. more independent sources beat fewer - agreement is evidence;
3. the mention expert beats a single unsupported proposal;
4. longer beats shorter, then leftmost, so the result is total and reproducible.

Repeated mentions are preserved: identity is `(start, end)`, so the same words at two
different places in the note are two entities, as the organizer schema requires.

One relational rule survives here rather than in the model: a test *result* is only an entity
when a test *name* is present nearby. A bare number with no named measurement is the failure
mode that produced most of the junk results, and it is decidable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..validation.organizer import ASSERTION_TYPES, ORGANIZER_TYPES
from .alignment import DocumentView
from .proposals import SOURCE_E3, Proposal

TYPE_TEST_NAME = "TÊN_XÉT_NGHIỆM"
TYPE_TEST_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"

#: How far back a result may look for the test it belongs to. One line is the unit clinicians
#: actually write in; a result on its own line with no named test above it is unanchored.
RESULT_ANCHOR_CHARS = 160

DROP_UNANCHORED_RESULT = "result_without_test"
DROP_OVERLAP = "overlap_resolved"


@dataclass
class Resolution:
    entities: list[Proposal] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str, amount: int = 1) -> None:
        self.counters[reason] = self.counters.get(reason, 0) + amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.as_dict() for e in self.entities],
            "counters": dict(self.counters),
        }


def _rank(proposal: Proposal) -> tuple[int, int, int, int]:
    """Higher is better. Compared only between overlapping proposals."""
    return (
        len(proposal.sources),
        1 if SOURCE_E3 in proposal.sources else 0,
        proposal.end - proposal.start,
        -proposal.start,
    )


def _prefer(left: Proposal, right: Proposal) -> Proposal:
    """The complete concept first, then evidence, then size. Never a coin flip."""
    if left.contains(right):
        return left
    if right.contains(left):
        return right
    return left if _rank(left) >= _rank(right) else right


def resolve_overlaps(accepted: list[Proposal]) -> tuple[list[Proposal], int]:
    """A maximal non-overlapping set, chosen by fixed precedence."""
    ordered = sorted(accepted, key=lambda p: (-_rank(p)[0], -(p.end - p.start), p.start))
    kept: list[Proposal] = []
    dropped = 0
    for proposal in ordered:
        conflict = next((k for k in kept if k.overlaps(proposal)), None)
        if conflict is None:
            kept.append(proposal)
            continue
        winner = _prefer(conflict, proposal)
        dropped += 1
        if winner.span != conflict.span:
            kept[kept.index(conflict)] = winner
    return sorted(kept, key=lambda p: (p.start, p.end)), dropped


def anchored_results(
    document: DocumentView, entities: list[Proposal]
) -> tuple[list[Proposal], int]:
    """Drop test results that no named test nearby can explain."""
    names = [e for e in entities if e.type == TYPE_TEST_NAME]
    kept: list[Proposal] = []
    dropped = 0
    for entity in entities:
        if entity.type != TYPE_TEST_RESULT:
            kept.append(entity)
            continue
        line = document.line_of(entity.start)
        anchored = any(
            name.end <= entity.start
            and entity.start - name.end <= RESULT_ANCHOR_CHARS
            and (line is None or name.start >= line.start)
            for name in names
        )
        if anchored:
            kept.append(entity)
        else:
            dropped += 1
    return kept, dropped


def resolve(
    document: DocumentView, accepted: list[Proposal]
) -> Resolution:
    """Final spans and types. Deterministic, and independent of proposal order."""
    resolution = Resolution()
    unique: dict[tuple[int, int, str], Proposal] = {}
    for proposal in accepted:
        if proposal.type not in ORGANIZER_TYPES:
            resolution.count("unknown_type")
            continue
        if document.slice(proposal.start, proposal.end) != proposal.text:
            # Impossible through the normal path; a proposal is always a source slice. If
            # it ever happens, the span is wrong and silence would hide it.
            resolution.count("text_offset_mismatch")
            continue
        unique[(proposal.start, proposal.end, proposal.type)] = proposal

    kept, dropped = resolve_overlaps(list(unique.values()))
    resolution.count(DROP_OVERLAP, dropped)

    kept, unanchored = anchored_results(document, kept)
    resolution.count(DROP_UNANCHORED_RESULT, unanchored)

    resolution.entities = kept
    return resolution


def organizer_entity(
    proposal: Proposal, assertions: tuple[str, ...] | None
) -> dict[str, Any]:
    """Organizer JSON for one entity, with no field the type does not support.

    `assertions` is legal only for the three assertion-bearing types; the two laboratory
    types carry `position`, `text`, `type` alone. Emitting `assertions: []` on a lab entity
    is an unsupported field, which has blocked packaging before.
    """
    row: dict[str, Any] = {
        "position": [proposal.start, proposal.end],
        "text": proposal.text,
        "type": proposal.type,
    }
    if proposal.type in ASSERTION_TYPES:
        row["assertions"] = list(assertions or ())
    return row


__all__ = [
    "DROP_OVERLAP",
    "DROP_UNANCHORED_RESULT",
    "RESULT_ANCHOR_CHARS",
    "TYPE_TEST_NAME",
    "TYPE_TEST_RESULT",
    "Resolution",
    "anchored_results",
    "organizer_entity",
    "resolve",
    "resolve_overlaps",
]
