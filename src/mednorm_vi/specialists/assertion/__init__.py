"""L5 Assertion Hydra (spec section 8).

Multi-label assertion resolution over ``isNegated``, ``isHistorical``,
``isFamily``. Stages A1-A6: section prior, cue detector, scope resolver,
entity-cue classifier, LLM adjudicator (only on disagreement), set calibration.

Contract:
    assert_entities(hypotheses, graph) -> list[EvidenceBundle]  # assertion_evidence populated

Status: NOT IMPLEMENTED (bootstrap). Interface only.
"""

from __future__ import annotations

from typing import Protocol

from ...schemas import DocumentGraph, EvidenceBundle, TypedHypothesis


class AssertionSpecialist(Protocol):
    def assert_entities(
        self, hypotheses: list[TypedHypothesis], graph: DocumentGraph
    ) -> list[EvidenceBundle]:
        ...


def assert_entities(
    hypotheses: list[TypedHypothesis], graph: DocumentGraph
) -> list[EvidenceBundle]:
    """TODO(L5-assertion): cue/scope/section/experiencer, multi-label decoding."""
    raise NotImplementedError("Assertion Hydra is not implemented yet (bootstrap).")


__all__ = ["AssertionSpecialist", "assert_entities"]
