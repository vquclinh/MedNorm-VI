"""L5 ICD-10 Super Linker package."""

from __future__ import annotations

from typing import Protocol

from ...linking.icd10 import link_icd10
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
    """Compatibility wrapper for the historical bootstrap protocol."""
    del hypotheses, graph
    return []


__all__ = ["IcdLinker", "ONTOLOGY", "link", "link_icd10"]
