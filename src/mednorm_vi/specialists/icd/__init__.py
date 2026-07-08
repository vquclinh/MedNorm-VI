"""L5 ICD-10 Super Linker (spec section 9).

Links DIAGNOSIS entities to ICD-10 codes using multi-representation indexes,
weighted RRF fusion, hierarchy expansion, cross-encoder rerank, and a constrained
Qwen3-4B judge that may only SELECT from retrieved candidates. KB is frozen to
the organizer-provided version. Never emits RxNorm codes.

Contract:
    link(hypotheses, graph) -> list[EvidenceBundle]  # candidate_evidence populated (ICD10)

Status: NOT IMPLEMENTED (bootstrap). Interface only. No index is built.
"""

from __future__ import annotations

from typing import Protocol

from ...schemas import DocumentGraph, EvidenceBundle, TypedHypothesis

ONTOLOGY = "ICD10"


class IcdLinker(Protocol):
    def link(
        self, hypotheses: list[TypedHypothesis], graph: DocumentGraph
    ) -> list[EvidenceBundle]:
        ...


def link(
    hypotheses: list[TypedHypothesis], graph: DocumentGraph
) -> list[EvidenceBundle]:
    """TODO(L5-icd): multi-index retrieval + rerank + constrained judge + set decode."""
    raise NotImplementedError("ICD-10 Super Linker is not implemented yet (bootstrap).")


__all__ = ["IcdLinker", "link", "ONTOLOGY"]
