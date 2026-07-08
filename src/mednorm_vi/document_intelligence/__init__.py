"""L1 — Document Intelligence (spec section 4).

Responsibility: segmentation, section/discourse parsing with priors, and
REVERSIBLE normalization. Produces the immutable :class:`DocumentGraph`.

Contract:
    build_document_graph(document_id: str, original_text: str) -> DocumentGraph

Must never mutate ``original_text`` or lose absolute offsets. Normalization runs
on a copy with a preserved :class:`OffsetAlignment`.

Status: NOT IMPLEMENTED (bootstrap). Interface only.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import DocumentGraph


class DocumentIntelligence(Protocol):
    """Interface for the L1 parser."""

    def build_document_graph(self, document_id: str, original_text: str) -> DocumentGraph:
        ...


def build_document_graph(document_id: str, original_text: str) -> DocumentGraph:
    """TODO(L1): implement section/segment parsing + reversible normalization.

    For now this raises to avoid pretending the layer is solved.
    """
    raise NotImplementedError("L1 Document Intelligence is not implemented yet (bootstrap).")


__all__ = ["DocumentIntelligence", "build_document_graph"]
