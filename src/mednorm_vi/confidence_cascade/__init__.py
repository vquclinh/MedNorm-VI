"""L7 — Confidence Cascade & LLM committee (spec section 12).

Responsibility: route only difficult/high-impact hypotheses to heavier models
(critic Qwen3-1.7B, adjudicator Qwen3-4B). Easy cases bypass the LLM. The LLM
operates under a strict prompt contract: it cannot change offsets outside a
locked option set, cannot emit IDs outside retrieved top-K, and chain-of-thought
is never stored.

Contract:
    adjudicate(hypotheses, bundles, graph) -> list[TypedHypothesis]  # calibrated

Status: NOT IMPLEMENTED (bootstrap). Interface only. No LLM is loaded.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import DocumentGraph, EvidenceBundle, TypedHypothesis

CASCADE_LEVELS = ("fast_path", "specialist_path", "critic", "adjudicator", "global_decoder")


class ConfidenceCascade(Protocol):
    def adjudicate(
        self,
        hypotheses: list[TypedHypothesis],
        bundles: list[EvidenceBundle],
        graph: DocumentGraph,
    ) -> list[TypedHypothesis]:
        ...


def adjudicate(
    hypotheses: list[TypedHypothesis],
    bundles: list[EvidenceBundle],
    graph: DocumentGraph,
) -> list[TypedHypothesis]:
    """TODO(L7): entry-condition routing + constrained LLM adjudication + calibration."""
    raise NotImplementedError("L7 Confidence Cascade is not implemented yet (bootstrap).")


__all__ = ["ConfidenceCascade", "adjudicate", "CASCADE_LEVELS"]
