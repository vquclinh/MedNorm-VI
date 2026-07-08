"""L5 RxNorm Super Linker (spec section 10).

Links MEDICATION entities to RxCUIs. Structured drug parsing (ingredient, salt,
strength, unit, concentration, dose form, release, brand, route, frequency, prn)
precedes graph search over ingredient/component/SCD/brand/dose-form. Honors SCD
vs SBD term types and hard negatives (same ingredient / different strength).
KB is frozen. Never emits ICD-10 codes.

Contract:
    link(hypotheses, graph) -> list[EvidenceBundle]  # candidate_evidence populated (RXNORM)

Status: NOT IMPLEMENTED (bootstrap). Interface only. No index is built.
"""

from __future__ import annotations

from typing import Protocol

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
    """TODO(L5-rxnorm): structured parse + graph search + reranking + set decode."""
    raise NotImplementedError("RxNorm Super Linker is not implemented yet (bootstrap).")


__all__ = ["RxNormLinker", "link", "ONTOLOGY"]
