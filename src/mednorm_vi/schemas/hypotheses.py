"""L3 Mention Factory and L4 Boundary & Type Resolver contracts
(spec sections 6-7).

``SpanProposal`` is a high-recall mention from a single expert (no expert emits
a final entity). ``TypedHypothesis`` is a resolved, typed span produced by L4.
Both carry the (absolute, local, normalized) coordinate triplet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spans import SpanCoordinates, SpanProvenance


@dataclass(frozen=True, slots=True)
class SpanProposal:
    """A candidate mention span from one Mention Factory expert (E1-E7).

    ``text`` MUST equal ``original_text[coords.absolute.start:end]``; this is not
    enforced here (dataclasses stay dependency-free) but is checked by
    ``validator.offsets``.
    """

    text: str
    coords: SpanCoordinates
    proposed_types: tuple[str, ...]
    provenance: SpanProvenance
    # Optional distribution over proposed types, if the expert provides one.
    type_scores: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> tuple[int, int]:
        return self.coords.output_position()


@dataclass(frozen=True, slots=True)
class TypedHypothesis:
    """A resolved span with a type distribution, produced by L4.

    ``evidence_ids`` reference the evidence bundle / provenance records that
    justify this hypothesis, enabling replay and ablation.
    """

    hypothesis_id: str
    text: str
    coords: SpanCoordinates
    type_distribution: dict[str, float]
    calibrated_score: float = 0.0
    abstained: bool = False
    source_proposal_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    features: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> tuple[int, int]:
        return self.coords.output_position()

    def argmax_type(self) -> str | None:
        if not self.type_distribution:
            return None
        return max(self.type_distribution.items(), key=lambda kv: kv[1])[0]
