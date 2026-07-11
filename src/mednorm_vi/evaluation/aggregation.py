"""Score aggregation policies.

Only ``provisional-v1`` attempts to approximate the published competition score
(``0.3·text + 0.3·assertions + 0.4·candidates``). ``macro-document`` and
``micro-entity-diagnostic`` are DIAGNOSTIC views and must not be confused with
the competition score.

Component construction from per-entity slots (matched + missing + spurious):
  * text:       mean of per-slot text score (unmatched slots contribute 0).
  * assertions: mean of assertion Jaccard over assertion-eligible slots.
  * candidates: candidate-weighted mean of candidate Jaccard over candidate-
                eligible slots, weight = len(GT candidates)+1 (published).
When a component has no eligible slots it is treated as 1.0 (nothing to get
wrong) — a provisional convention, documented in METRIC_ASSUMPTIONS.md.
"""

from __future__ import annotations

from .models import (
    CorpusScore,
    EvaluationConfig,
    PerDocumentScore,
    PerEntityScore,
)
from .scoring import DocumentScoring

AGGREGATION_POLICIES = ("provisional-v1", "macro-document", "micro-entity-diagnostic")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _pool_components(
    slots: list[PerEntityScore], *, weighted_candidates: bool
) -> tuple[float, float, float]:
    text = _mean([s.text_score for s in slots]) if slots else 1.0

    a_slots = [s for s in slots if s.assertions_eligible]
    if a_slots:
        assertions = _mean([(s.assertions.jaccard if s.assertions else 0.0) for s in a_slots])
    else:
        assertions = 1.0

    c_slots = [s for s in slots if s.candidates_eligible]
    if not c_slots:
        candidates = 1.0
    elif weighted_candidates:
        num = sum(s.candidate_weight * (s.candidates.jaccard if s.candidates else 0.0)
                  for s in c_slots)
        den = sum(s.candidate_weight for s in c_slots)
        candidates = num / den if den else 1.0
    else:
        candidates = _mean([(s.candidates.jaccard if s.candidates else 0.0) for s in c_slots])
    return text, assertions, candidates


def _final(config: EvaluationConfig, text: float, assertions: float, candidates: float) -> float:
    return (
        config.weight_text * text
        + config.weight_assertions * assertions
        + config.weight_candidates * candidates
    )


def build_per_document(doc: DocumentScoring, config: EvaluationConfig) -> PerDocumentScore:
    """Per-document provisional score (candidate-weighted)."""
    slots = list(doc.per_entity)
    text, assertions, candidates = _pool_components(slots, weighted_candidates=True)
    n_matched = sum(1 for s in slots if s.slot_kind == "matched")
    n_missing = sum(1 for s in slots if s.slot_kind == "missing")
    n_spurious = sum(1 for s in slots if s.slot_kind == "spurious")
    return PerDocumentScore(
        document_id=doc.document_id,
        provenance=doc.provenance,
        text_score=text,
        assertions_score=assertions,
        candidates_score=candidates,
        final_score=_final(config, text, assertions, candidates),
        n_gt=doc.n_gt,
        n_pred=doc.n_pred,
        n_matched=n_matched,
        n_missing=n_missing,
        n_spurious=n_spurious,
        per_entity=doc.per_entity,
    )


def _group_scores(
    slots: list[PerEntityScore], key: str, config: EvaluationConfig
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[PerEntityScore]] = {}
    for s in slots:
        value = s.entity_type if key == "type" else s.route_case
        if value is None:
            continue
        groups.setdefault(value, []).append(s)
    out: dict[str, dict[str, float]] = {}
    for name in sorted(groups):
        t, a, c = _pool_components(groups[name], weighted_candidates=True)
        out[name] = {
            "text": t,
            "assertions": a,
            "candidates": c,
            "final": _final(config, t, a, c),
            "n_slots": float(len(groups[name])),
        }
    return out


def aggregate_corpus(
    docs: list[DocumentScoring], config: EvaluationConfig
) -> tuple[CorpusScore, list[PerDocumentScore]]:
    """Aggregate per-document scorings into the corpus score for the configured policy."""
    per_doc = [build_per_document(d, config) for d in docs]
    all_slots: list[PerEntityScore] = [s for d in docs for s in d.per_entity]
    policy = config.aggregation_policy

    if policy == "macro-document":
        text = _mean([d.text_score for d in per_doc])
        assertions = _mean([d.assertions_score for d in per_doc])
        candidates = _mean([d.candidates_score for d in per_doc])
    elif policy == "micro-entity-diagnostic":
        text, assertions, candidates = _pool_components(all_slots, weighted_candidates=False)
    else:  # provisional-v1
        text, assertions, candidates = _pool_components(all_slots, weighted_candidates=True)

    corpus = CorpusScore(
        text_score=text,
        assertions_score=assertions,
        candidates_score=candidates,
        final_score=_final(config, text, assertions, candidates),
        aggregation_policy=policy,
        n_documents=len(docs),
        n_entities=sum(d.n_gt for d in docs),
        n_matched=sum(1 for s in all_slots if s.slot_kind == "matched"),
        n_missing=sum(1 for s in all_slots if s.slot_kind == "missing"),
        n_spurious=sum(1 for s in all_slots if s.slot_kind == "spurious"),
        per_type=_group_scores(all_slots, "type", config),
        per_case=_group_scores(all_slots, "case", config),
    )
    return corpus, per_doc


__all__ = ["AGGREGATION_POLICIES", "aggregate_corpus", "build_per_document"]
