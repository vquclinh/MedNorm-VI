"""Per-pair and per-slot scoring (matching is done separately in ``matching/``).

Produces ``PerEntityScore`` slots for every matched pair, every unmatched ground
truth (missing), and every unmatched prediction (spurious), plus structured
diagnostics. Aggregation into document/corpus scores lives in ``aggregation.py``.

Assertion/candidate eligibility follows the confirmed per-type field policy:
assertions apply to THUỐC/CHẨN_ĐOÁN/TRIỆU_CHỨNG; candidates apply to
THUỐC/CHẨN_ĐOÁN only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.constants import (
    CANDIDATE_ONTOLOGY_BY_TYPE,
    ORGANIZER_FIELDS_BY_TYPE,
    ORGANIZER_LABEL_BY_TYPE,
)
from . import diagnostics as dx
from .jaccard import jaccard_breakdown
from .models import (
    Diagnostic,
    EntityPair,
    EvaluationConfig,
    EvaluationDocument,
    EvaluationEntity,
    MatchingResult,
    PerEntityScore,
    Provenance,
)
from .wer import compute_wer

# Organizer labels that may carry assertions / candidates (derived from policy).
ASSERTION_LABELS_TYPES: frozenset[str] = frozenset(
    ORGANIZER_LABEL_BY_TYPE[t]
    for t, fields in ORGANIZER_FIELDS_BY_TYPE.items()
    if "assertions" in fields
)
CANDIDATE_LABELS_TYPES: frozenset[str] = frozenset(
    ORGANIZER_LABEL_BY_TYPE[t]
    for t, fields in ORGANIZER_FIELDS_BY_TYPE.items()
    if "candidates" in fields
)
ONTOLOGY_BY_ORGANIZER_LABEL: dict[str, str | None] = {
    ORGANIZER_LABEL_BY_TYPE[t]: CANDIDATE_ONTOLOGY_BY_TYPE[t]
    for t in ORGANIZER_LABEL_BY_TYPE
}


@dataclass(frozen=True, slots=True)
class DocumentScoring:
    """All per-entity slots and diagnostics for one scored document."""

    document_id: str
    provenance: Provenance | None
    n_gt: int
    n_pred: int
    per_entity: tuple[PerEntityScore, ...]
    diagnostics: tuple[Diagnostic, ...]


def allows_assertions(organizer_type: str) -> bool:
    return organizer_type in ASSERTION_LABELS_TYPES


def allows_candidates(organizer_type: str) -> bool:
    return organizer_type in CANDIDATE_LABELS_TYPES


def _candidate_ontology_ok(organizer_type: str, code: str) -> bool:
    ontology = ONTOLOGY_BY_ORGANIZER_LABEL.get(organizer_type)
    if ontology == "RXNORM":
        return code.isdigit()
    return True  # ICD-10 kept as opaque strings (KB membership deferred)


def _score_matched_pair(
    document_id: str,
    gi: int,
    pi: int,
    g: EvaluationEntity,
    p: EvaluationEntity,
    cost: float,
    strategy: str,
    config: EvaluationConfig,
    provenance: Provenance | None,
) -> tuple[PerEntityScore, list[Diagnostic]]:
    diags: list[Diagnostic] = []
    etype = g.type
    wer = compute_wer(
        g.text, p.text, tokenization=config.tokenization, clipping_enabled=config.clipping_enabled
    )
    if wer.substitutions:
        diags.append(Diagnostic(dx.TEXT_TOKEN_SUBSTITUTION, document_id, etype, gi, pi,
                                g.route_case, g.section, provenance))
    if wer.insertions:
        diags.append(Diagnostic(dx.TEXT_TOKEN_INSERTION, document_id, etype, gi, pi,
                                g.route_case, g.section, provenance))
    if wer.deletions:
        diags.append(Diagnostic(dx.TEXT_TOKEN_DELETION, document_id, etype, gi, pi,
                                g.route_case, g.section, provenance))
    if g.position != p.position:
        diags.append(Diagnostic(dx.TEXT_BOUNDARY_ERROR, document_id, etype, gi, pi,
                                g.route_case, g.section, provenance))

    assertions_bd = None
    if allows_assertions(etype):
        assertions_bd = jaccard_breakdown("assertions", g.assertions, p.assertions)
        for _ in assertions_bd.missing:
            diags.append(Diagnostic(dx.ASSERTION_MISSING, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))
        for _ in assertions_bd.extra:
            diags.append(Diagnostic(dx.ASSERTION_EXTRA, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))
        for _ in assertions_bd.gt_duplicates + assertions_bd.pred_duplicates:
            diags.append(Diagnostic(dx.DUPLICATE_ASSERTION, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))

    candidates_bd = None
    candidate_weight = 1.0
    if allows_candidates(etype):
        candidates_bd = jaccard_breakdown("candidates", g.candidates, p.candidates)
        candidate_weight = candidates_bd.weight
        for _ in candidates_bd.missing:
            diags.append(Diagnostic(dx.CANDIDATE_MISSING, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))
        for _ in candidates_bd.extra:
            diags.append(Diagnostic(dx.CANDIDATE_EXTRA, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))
        for _ in candidates_bd.gt_duplicates + candidates_bd.pred_duplicates:
            diags.append(Diagnostic(dx.DUPLICATE_CANDIDATE, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))
        for code in p.candidates:
            if not _candidate_ontology_ok(etype, code):
                diags.append(Diagnostic(dx.CANDIDATE_WRONG_ONTOLOGY, document_id, etype, gi, pi,
                                        g.route_case, g.section, provenance, detail=code))
        if g.candidates and not p.candidates:
            diags.append(Diagnostic(dx.CANDIDATE_EMPTY_WHEN_REQUIRED, document_id, etype, gi, pi,
                                    g.route_case, g.section, provenance))

    pair = EntityPair(
        document_id=document_id,
        gt_index=gi,
        pred_index=pi,
        entity_type=etype,
        gt_text=g.text,
        pred_text=p.text,
        gt_position=g.position,
        pred_position=p.position,
        strategy=strategy,
        cost=cost,
    )
    score = PerEntityScore(
        document_id=document_id,
        slot_kind="matched",
        entity_type=etype,
        text_score=wer.text_score,
        candidate_weight=candidate_weight,
        diagnostics=tuple(d.category for d in diags),
        pair=pair,
        wer=wer,
        assertions=assertions_bd,
        candidates=candidates_bd,
        route_case=g.route_case,
        section=g.section,
        provenance=provenance,
        assertions_eligible=allows_assertions(etype),
        candidates_eligible=allows_candidates(etype),
    )
    return score, diags


def _unmatched_slot(
    document_id: str,
    idx: int,
    ent: EvaluationEntity,
    slot_kind: str,
    provenance: Provenance | None,
) -> tuple[PerEntityScore, list[Diagnostic]]:
    etype = ent.type
    category = dx.MISSING_ENTITY if slot_kind == "missing" else dx.SPURIOUS_ENTITY
    gi = idx if slot_kind == "missing" else None
    pi = idx if slot_kind == "spurious" else None
    diags = [Diagnostic(category, document_id, etype, gi, pi, ent.route_case, ent.section,
                        provenance)]
    weight = float(len(set(ent.candidates)) + 1) if allows_candidates(etype) else 1.0
    score = PerEntityScore(
        document_id=document_id,
        slot_kind=slot_kind,
        entity_type=etype,
        text_score=0.0,
        candidate_weight=weight,
        diagnostics=(category,),
        route_case=ent.route_case,
        section=ent.section,
        provenance=provenance,
        assertions_eligible=allows_assertions(etype),
        candidates_eligible=allows_candidates(etype),
    )
    return score, diags


def _wrong_type_diagnostics(
    document_id: str,
    gt: tuple[EvaluationEntity, ...],
    pred: tuple[EvaluationEntity, ...],
    unmatched_gt: tuple[int, ...],
    unmatched_pred: tuple[int, ...],
    provenance: Provenance | None,
) -> list[Diagnostic]:
    """Flag likely wrong-type predictions: same text, overlapping span, diff type."""
    diags: list[Diagnostic] = []
    for pi in unmatched_pred:
        p = pred[pi]
        for gi in unmatched_gt:
            g = gt[gi]
            if g.type == p.type:
                continue
            if g.text == p.text and (g.start < p.end and p.start < g.end):
                diags.append(Diagnostic(dx.WRONG_TYPE, document_id, p.type, gi, pi,
                                        p.route_case, p.section, provenance,
                                        detail=f"gt_type={g.type}"))
                break
    return diags


def score_document(
    gt_doc: EvaluationDocument,
    pred_doc: EvaluationDocument | None,
    matching: MatchingResult,
    config: EvaluationConfig,
) -> DocumentScoring:
    """Score one document given its matching result."""
    gt = gt_doc.entities
    pred = pred_doc.entities if pred_doc is not None else ()
    provenance = gt_doc.provenance
    per_entity: list[PerEntityScore] = []
    diagnostics: list[Diagnostic] = []

    if provenance is Provenance.SILVER:
        diagnostics.append(Diagnostic(dx.SILVER_LABEL_WARNING, gt_doc.document_id,
                                      provenance=provenance,
                                      detail="scored against SILVER (weak) labels"))

    for dec in matching.pairs:
        g = gt[dec.gt_index]
        p = pred[dec.pred_index]
        score, diags = _score_matched_pair(
            gt_doc.document_id, dec.gt_index, dec.pred_index, g, p, dec.cost,
            matching.strategy, config, provenance,
        )
        per_entity.append(score)
        diagnostics.extend(diags)

    for gi in matching.unmatched_gt:
        score, diags = _unmatched_slot(gt_doc.document_id, gi, gt[gi], "missing", provenance)
        per_entity.append(score)
        diagnostics.extend(diags)

    for pi in matching.unmatched_pred:
        score, diags = _unmatched_slot(gt_doc.document_id, pi, pred[pi], "spurious", provenance)
        per_entity.append(score)
        diagnostics.extend(diags)

    diagnostics.extend(
        _wrong_type_diagnostics(gt_doc.document_id, gt, pred, matching.unmatched_gt,
                                matching.unmatched_pred, provenance)
    )

    return DocumentScoring(
        document_id=gt_doc.document_id,
        provenance=provenance,
        n_gt=len(gt),
        n_pred=len(pred),
        per_entity=tuple(per_entity),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "DocumentScoring",
    "score_document",
    "allows_assertions",
    "allows_candidates",
    "ASSERTION_LABELS_TYPES",
    "CANDIDATE_LABELS_TYPES",
]
