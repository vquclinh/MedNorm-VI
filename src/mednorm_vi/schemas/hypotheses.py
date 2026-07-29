"""L3/L4 hypothesis contracts (spec §19.1).

**Ownership, settled by Audit 0052.** Four types in this repository were named
``SpanProposal``. They are not interchangeable, and collapsing them would erase real
distinctions, so each now has one declared owner:

===============================================  ==========================  =========
type                                             role                        runtime?
===============================================  ==========================  =========
``schemas.hypotheses.SpanProposal``              spec §19.1 abstract shape   **NO**
``mention_factory.models.SpanProposal``          E1/E2 deterministic output  yes
``lattice.models.ExpertSpanProposal``            trainable-expert output     yes
``lattice.models.SpanProposal``                  merged lattice node         yes
===============================================  ==========================  =========

``SpanProposal`` in *this* module is **NON-RUNTIME**. It is the spec's five-field
sketch (``text, coords, proposed_types, provenance, type_scores``), kept as
documentation of the contract §19.1 states. No runtime module constructs or consumes
it, and none should: the concrete lineage is

    E1/E2  ->  mention_factory.models.SpanProposal
    E3/E5/E6/E7  ->  lattice.models.ExpertSpanProposal
    both  ->  lattice.builder.build_span_lattice  ->  lattice.models.SpanProposal
          ->  resolution.canonical  ->  resolution.models.EntityHypothesis

``TypedHypothesis`` below **is** runtime: ``resolution.resolver_v1`` emits it, and
``resolution.canonical`` adapts it to ``EntityHypothesis`` for L5-L9.

See ``docs/architecture/ACTIVE_RUNTIME_MANIFEST.md`` for the full lineage table.
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
