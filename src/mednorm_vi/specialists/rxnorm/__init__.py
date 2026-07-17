"""L5 RxNorm Super Linker package."""

from __future__ import annotations

from typing import Protocol

from ...linking.rxnorm import link_rxnorm
from ...schemas import DocumentGraph, EvidenceBundle, TypedHypothesis

ONTOLOGY = "RXNORM"


class RxNormLinker(Protocol):
    def link(
        self, hypotheses: list[TypedHypothesis], graph: DocumentGraph
    ) -> list[EvidenceBundle]:
        ...


def link(
    hypotheses: list[TypedHypothesis], graph: DocumentGraph
) -> list[EvidenceBundle]:
    """Compatibility wrapper for the historical bootstrap protocol."""
    del hypotheses, graph
    return []


__all__ = ["ONTOLOGY", "RxNormLinker", "link", "link_rxnorm"]
