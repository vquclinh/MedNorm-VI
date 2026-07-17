"""L5 Assertion Hydra package."""

from __future__ import annotations

from typing import Protocol

from ...schemas import DocumentGraph, EvidenceBundle, TypedHypothesis
from .hydra import AssertionDecision, resolve_assertions


class AssertionSpecialist(Protocol):
    def assert_entities(
        self, hypotheses: list[TypedHypothesis], graph: DocumentGraph
    ) -> list[EvidenceBundle]:
        ...


def assert_entities(
    hypotheses: list[TypedHypothesis], graph: DocumentGraph
) -> list[EvidenceBundle]:
    """Compatibility wrapper for the historical bootstrap protocol."""
    del hypotheses, graph
    raise NotImplementedError("Assertion Hydra compatibility wrapper requires a trained head")


__all__ = ["AssertionDecision", "AssertionSpecialist", "assert_entities", "resolve_assertions"]
