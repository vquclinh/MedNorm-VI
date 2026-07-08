"""L8 — Metric-aware Set Decoder (spec section 13).

Responsibility: select the metric-optimal prediction set. Expected-WER span
selection; expected-Jaccard candidate-set decoding (no fixed top-K, no blind
top-1); per-label assertion thresholds; entity existence/type utility net of
wrong-type risk. Emits final :class:`EntityPrediction` objects for L9.

Contract:
    decode(hypotheses, bundles, graph) -> list[EntityPrediction]

Status: NOT IMPLEMENTED (bootstrap). Interface only.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import DocumentGraph, EntityPrediction, EvidenceBundle, TypedHypothesis


class MetricDecoder(Protocol):
    def decode(
        self,
        hypotheses: list[TypedHypothesis],
        bundles: list[EvidenceBundle],
        graph: DocumentGraph,
    ) -> list[EntityPrediction]:
        ...


def decode(
    hypotheses: list[TypedHypothesis],
    bundles: list[EvidenceBundle],
    graph: DocumentGraph,
) -> list[EntityPrediction]:
    """TODO(L8): expected WER/Jaccard set decoding; wrong-type-risk-aware selection."""
    raise NotImplementedError("L8 Metric Decoder is not implemented yet (bootstrap).")


__all__ = ["MetricDecoder", "decode"]
