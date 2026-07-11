"""Lightweight inter-annotator agreement utilities.

Compares two annotators' entity lists for one document: exact-span agreement,
type agreement, and (over span-agreeing pairs) assertion-set and candidate-set
agreement. Deterministic; no external dependencies.
"""

from __future__ import annotations

from .models import AgreementSummary, AnnotationEntity


def _by_span(entities: list[AnnotationEntity]) -> dict[tuple[int, int], list[AnnotationEntity]]:
    out: dict[tuple[int, int], list[AnnotationEntity]] = {}
    for e in entities:
        out.setdefault(e.position, []).append(e)
    return out


def document_agreement(
    document_id: str,
    annotator_a: list[AnnotationEntity],
    annotator_b: list[AnnotationEntity],
) -> AgreementSummary:
    """Compute an :class:`AgreementSummary` between two annotators for one document."""
    a_by_span = _by_span(annotator_a)
    b_by_span = _by_span(annotator_b)

    n_span_agree = 0
    n_type_agree = 0
    n_assertion_agree = 0
    n_candidate_agree = 0

    for span in sorted(set(a_by_span) & set(b_by_span)):
        a_list = a_by_span[span]
        b_list = b_by_span[span]
        pairs = min(len(a_list), len(b_list))
        n_span_agree += pairs
        for a, b in zip(a_list[:pairs], b_list[:pairs], strict=False):
            if a.type == b.type:
                n_type_agree += 1
            if set(a.assertions) == set(b.assertions):
                n_assertion_agree += 1
            if set(a.candidates) == set(b.candidates):
                n_candidate_agree += 1

    return AgreementSummary(
        document_id=document_id,
        n_a=len(annotator_a),
        n_b=len(annotator_b),
        n_span_agree=n_span_agree,
        n_type_agree=n_type_agree,
        n_assertion_agree=n_assertion_agree,
        n_candidate_agree=n_candidate_agree,
    )


def corpus_agreement(summaries: list[AgreementSummary]) -> dict[str, float]:
    """Aggregate document agreement summaries into corpus-level ratios."""
    total_a = sum(s.n_a for s in summaries)
    total_b = sum(s.n_b for s in summaries)
    span = sum(s.n_span_agree for s in summaries)
    typ = sum(s.n_type_agree for s in summaries)
    asrt = sum(s.n_assertion_agree for s in summaries)
    cand = sum(s.n_candidate_agree for s in summaries)
    denom = max(total_a, total_b)
    return {
        "span_agreement": span / denom if denom else 1.0,
        "type_agreement": typ / denom if denom else 1.0,
        "assertion_agreement": asrt / span if span else 1.0,
        "candidate_agreement": cand / span if span else 1.0,
        "n_documents": float(len(summaries)),
    }


__all__ = ["document_agreement", "corpus_agreement"]
