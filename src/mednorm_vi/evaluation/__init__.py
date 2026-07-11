"""PROVISIONAL LOCAL EVALUATOR for MedNorm-VI.

Scores team-owned labeled data (GOLD/SILVER/SYNTHETIC/ORGANIZER_PUBLISHED_EXAMPLE
/EXTERNAL_PERMITTED) against predictions. The organizer competition test set has
**no ground truth**; this evaluator must never be pointed at it as labels.

This is NOT an official evaluator clone. Matching, WER tokenization, clipping, and
aggregation details are provisional and configurable — see
``docs/evaluation/METRIC_ASSUMPTIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .aggregation import aggregate_corpus
from .matching import build_matcher
from .models import (
    CorpusScore,
    Diagnostic,
    EvaluationConfig,
    EvaluationDiagnostics,
    EvaluationDocument,
    PerDocumentScore,
    PerEntityScore,
)
from .scoring import DocumentScoring, score_document

BANNER = "PROVISIONAL LOCAL EVALUATOR"


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Everything produced by one evaluation run (before reporting)."""

    config: EvaluationConfig
    corpus: CorpusScore
    per_document: tuple[PerDocumentScore, ...]
    document_scorings: tuple[DocumentScoring, ...]
    diagnostics: EvaluationDiagnostics
    slots: tuple[PerEntityScore, ...]


def _doc_sort_key(doc_id: str) -> tuple[int, str]:
    return (int(doc_id), doc_id) if doc_id.isdigit() else (1 << 60, doc_id)


def evaluate_corpus(
    ground_truth: dict[str, EvaluationDocument],
    predictions: dict[str, EvaluationDocument],
    config: EvaluationConfig,
) -> EvaluationOutcome:
    """Match, score, and aggregate one corpus deterministically."""
    matcher = build_matcher(config)
    doc_ids = sorted(set(ground_truth) | set(predictions), key=_doc_sort_key)

    scorings: list[DocumentScoring] = []
    all_diags: list[Diagnostic] = []
    for doc_id in doc_ids:
        gt_doc = ground_truth.get(doc_id) or EvaluationDocument(document_id=doc_id, entities=())
        pred_doc = predictions.get(doc_id)
        matching = matcher.match(gt_doc.entities, pred_doc.entities if pred_doc else ())
        scoring = score_document(gt_doc, pred_doc, matching, config)
        scorings.append(scoring)
        all_diags.extend(scoring.diagnostics)

    corpus, per_doc = aggregate_corpus(scorings, config)
    slots = tuple(s for d in scorings for s in d.per_entity)
    return EvaluationOutcome(
        config=config,
        corpus=corpus,
        per_document=tuple(per_doc),
        document_scorings=tuple(scorings),
        diagnostics=EvaluationDiagnostics(items=tuple(all_diags)),
        slots=slots,
    )


__all__ = ["BANNER", "EvaluationOutcome", "evaluate_corpus"]
