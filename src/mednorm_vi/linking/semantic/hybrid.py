"""Hybrid candidate generation and complete-system assembly (Audit 0072 §3-§6).

The order of operations is the design, and each step exists because a measured failure
demanded it:

    v3 lexical Top-K        precision anchor - the 11.9188 system's retrieval, never replaced
    v4.1 lexical Top-K      coverage - Audit 0069 raised probe recall 1/16 -> 9/16
    dense Top-K             semantic coverage - Vietnamese register the lexicon misses
      -> union, deduplicated by governed identity
      -> REAL cross-encoder reranking          (Audit 0072 §4)
      -> H2 within-family arbitration          (Audit 0072 §5, applied AFTER reranking)
      -> null gate in shadow or conservative   (Audit 0072 §6)

Two invariants hold no matter what the models say:

* **Nothing may invent a code.** Every candidate is looked up in the governed frozen KB
  before it can survive. A generative model reorders and rejects; it never emits.
* **v3 is an anchor, not a veto.** Audit 0070 proved that globally replacing v3 with v4.1
  costs J_candidates, so v3 always contributes. But nothing encodes "v3 wins": the reranker
  may overturn it, and the evaluator counts how often it does.

H2 runs *after* reranking because it is a tie-break over semantics, not a substitute for
them. Running it first would let a lexical family rule reorder candidates the reranker had
not yet judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...kb.indexing import evidence as ev
from ...kb.indexing.retrieval import LocalIndex, search_index
from ..icd10_specificity import (
    POLICY_HIERARCHY_AWARE,
    arbitrate_order,
    compute_features,
)
from ..rxnorm_graph import is_closure_only
from . import null_gate as ng
from .reranker import RerankerBackend, rerank

HYBRID_VERSION = "hybrid-zero-shot-v1"

SOURCE_V3 = "from_v3"
SOURCE_V41 = "from_v41"
SOURCE_DENSE = "from_dense"

#: Bounded pool sizes (Audit 0071 §7). Deliberately modest: the reranker cost is linear in
#: pool size, and Audit 0070 showed an unbounded pool is a liability rather than an asset.
DEFAULT_V3_TOPK = 20
DEFAULT_V41_TOPK = 20
DEFAULT_DENSE_TOPK = 50

NULL_MODE_SHADOW = "shadow"
NULL_MODE_CONSERVATIVE = "conservative"
NULL_MODES = (NULL_MODE_SHADOW, NULL_MODE_CONSERVATIVE)


@dataclass(frozen=True, slots=True)
class PooledCandidate:
    """One candidate before reranking, with where it came from and at what rank."""

    concept_id: str
    sources: tuple[str, ...]
    v3_rank: int | None = None
    v41_rank: int | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float = 0.0
    evidence_tier: str = ""
    channels: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "sources": list(self.sources),
            "v3_rank": self.v3_rank,
            "v41_rank": self.v41_rank,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "lexical_score": round(self.lexical_score, 4),
            "evidence_tier": self.evidence_tier,
        }


def build_pool(
    mention: str,
    *,
    v3_index: LocalIndex | None = None,
    v41_index: LocalIndex | None = None,
    dense_hits: list[tuple[str, float]] | None = None,
    governed_index: LocalIndex | None = None,
    v3_topk: int = DEFAULT_V3_TOPK,
    v41_topk: int = DEFAULT_V41_TOPK,
    dense_topk: int = DEFAULT_DENSE_TOPK,
) -> list[PooledCandidate]:
    """Union the three sources, deduplicated by governed identity.

    ``governed_index`` is the authority on existence. A dense hit naming a concept that is not
    in the frozen KB is dropped here, which is what makes "no model may generate arbitrary
    codes" a structural property rather than a promise.
    """
    merged: dict[str, dict[str, Any]] = {}

    def note(concept_id: str, source: str, **fields: Any) -> None:
        row = merged.setdefault(concept_id, {"sources": [], "channels": (), "lexical_score": 0.0})
        if source not in row["sources"]:
            row["sources"].append(source)
        for key, value in fields.items():
            if value is not None:
                row[key] = value

    if v3_index is not None:
        for rank, hit in enumerate(search_index(v3_index, mention, limit=v3_topk), start=1):
            note(
                hit.concept_id,
                SOURCE_V3,
                v3_rank=rank,
                lexical_score=hit.score,
                channels=hit.channels,
            )
    if v41_index is not None:
        for rank, hit in enumerate(search_index(v41_index, mention, limit=v41_topk), start=1):
            note(
                hit.concept_id,
                SOURCE_V41,
                v41_rank=rank,
                lexical_score=hit.score,
                channels=hit.channels,
                evidence_tier=hit.tier or "",
            )
    for rank, (concept_id, score) in enumerate((dense_hits or [])[:dense_topk], start=1):
        note(concept_id, SOURCE_DENSE, dense_rank=rank, dense_score=float(score))

    authority = governed_index or v3_index or v41_index
    out: list[PooledCandidate] = []
    for concept_id, row in merged.items():
        if authority is not None and not authority.exists(concept_id):
            continue  # a code the governed KB does not contain can never be emitted
        channels = tuple(row.get("channels") or ())
        out.append(
            PooledCandidate(
                concept_id=concept_id,
                sources=tuple(row["sources"]),
                v3_rank=row.get("v3_rank"),
                v41_rank=row.get("v41_rank"),
                dense_rank=row.get("dense_rank"),
                dense_score=row.get("dense_score"),
                lexical_score=float(row.get("lexical_score") or 0.0),
                evidence_tier=str(row.get("evidence_tier") or (ev.tier_of(channels) or "")),
                channels=channels,
            )
        )
    # Deterministic pre-rerank order: lexical anchor first, then dense rank, then identity.
    out.sort(
        key=lambda c: (
            c.v3_rank if c.v3_rank is not None else 10_000,
            c.v41_rank if c.v41_rank is not None else 10_000,
            c.dense_rank if c.dense_rank is not None else 10_000,
            c.concept_id,
        )
    )
    return out


def final_candidate_eligible(index: LocalIndex | None, concept_id: str) -> bool:
    """May the runtime EMIT this concept, not merely reach it (Audit 0075 hotfix)?

    `LocalIndex.exists` answers "is this in the snapshot", which is a weaker question. The
    governed RxNorm KB holds 82,429 searchable concepts and 129,520 `closure_only` ones that
    exist purely so the ingredient/product walk has somewhere to land. Lexical retrieval never
    surfaces them - they are absent from the postings - but **dense retrieval scores every
    document**, so the semantic path could rank one and `exists()` waved it through. L9 then
    refused the whole submission with `kb.candidate_not_final_eligible`.

    This reuses `linking.rxnorm_graph.is_closure_only`, the same predicate the lexical linker
    applies as `DROP_CLOSURE_ONLY`, rather than defining a second notion of eligibility.
    """
    if index is None or not index.exists(concept_id):
        return False
    if index.index_type == "rxnorm" and is_closure_only(index, concept_id):
        return False
    return True


def sanitize_final_candidates(
    codes: tuple[str, ...] | list[str], index: LocalIndex | None
) -> tuple[str, ...]:
    """The single boundary every emitted candidate list passes through.

    Order preserved, deduplicated deterministically, non-emittable ids dropped; an empty
    result is a valid answer. This runs after reranking and hierarchy arbitration so nothing
    downstream can reintroduce an ineligible id.
    """
    out: list[str] = []
    for code in codes:
        if code in out:
            continue
        if final_candidate_eligible(index, code):
            out.append(code)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class HybridResult:
    """A complete-system decision with every stage recoverable."""

    codes: tuple[str, ...]
    pool_size: int
    reranked: tuple[dict[str, Any], ...]
    null_decision: dict[str, Any]
    null_mode: str
    hierarchy_applied: bool
    v3_top1: str = ""
    final_top1: str = ""

    @property
    def v3_top1_kept(self) -> bool:
        return bool(self.v3_top1) and self.v3_top1 == self.final_top1

    @property
    def v3_top1_overturned(self) -> bool:
        return bool(self.v3_top1) and bool(self.final_top1) and self.v3_top1 != self.final_top1


def run_hybrid(
    mention: str,
    query_text: str,
    pool: list[PooledCandidate],
    documents: dict[str, str],
    backend: RerankerBackend,
    *,
    ontology: str,
    icd_index: LocalIndex | None = None,
    governed_index: LocalIndex | None = None,
    context_text: str = "",
    null_mode: str = NULL_MODE_SHADOW,
    apply_hierarchy: bool = True,
    limit: int = 20,
) -> HybridResult:
    """Rerank, then arbitrate hierarchy, then consult the null gate."""
    if null_mode not in NULL_MODES:
        raise ValueError(f"unknown null_mode {null_mode!r}; expected one of {NULL_MODES}")

    v3_top1 = next((c.concept_id for c in pool if c.v3_rank == 1), "")
    if not pool:
        decision = ng.evaluate(
            ng.NullGateEvidence(candidate_count=0, ontology=ontology, mention_text=mention)
        )
        return HybridResult((), 0, (), decision.as_dict(), null_mode, False, v3_top1, "")

    payload = [
        {
            "concept_id": c.concept_id,
            "document": documents.get(c.concept_id, c.concept_id),
            "sources": c.sources,
            "dense_score": c.dense_score,
            "lexical_score": c.lexical_score,
            "evidence_tier": c.evidence_tier,
        }
        for c in pool
    ]
    reranked = rerank(backend, query_text, payload)
    order = [c.concept_id for c in reranked]

    # H2 runs AFTER reranking: it is a within-family tie-break over semantic judgement, not a
    # replacement for it, and it can never move one family past another.
    hierarchy_applied = False
    if apply_hierarchy and ontology == "ICD10" and icd_index is not None and len(order) > 1:
        by_id = {c.concept_id: c for c in reranked}
        features = {
            code: compute_features(
                icd_index,
                code,
                channels=next((c.channels for c in pool if c.concept_id == code), ()),
                lexical_score=by_id[code].reranker_score,
                mention_text=mention,
                context_text=context_text,
            )
            for code in order
        }
        arbitrated = arbitrate_order(order, features, policy=POLICY_HIERARCHY_AWARE)
        hierarchy_applied = arbitrated != order
        order = arbitrated

    top = next((c for c in reranked if c.concept_id == order[0]), reranked[0])
    evidence = ng.NullGateEvidence(
        top_tier=top.evidence_tier,
        top_score=top.reranker_score,
        second_score=reranked[1].reranker_score if len(reranked) > 1 else 0.0,
        dense_score=top.dense_score,
        reranker_score=top.reranker_score,
        source_count=len(top.sources),
        has_exact_evidence=top.evidence_tier in ng.PROTECTED_TIERS,
        mention_text=mention,
        ontology=ontology,
        candidate_count=len(order),
    )
    decision = ng.evaluate(evidence)

    # SHADOW records the decision and changes nothing. That is the whole point: the first
    # zero-shot run collects reranker-aware NULL evidence so calibration can happen later,
    # locally, against the private engineering pack - without a public-tuned threshold.
    governed = icd_index if ontology == "ICD10" else governed_index
    codes = sanitize_final_candidates(order[:limit], governed)
    if null_mode == NULL_MODE_CONSERVATIVE and not decision.emits:
        codes = ()

    return HybridResult(
        codes=codes,
        pool_size=len(pool),
        reranked=tuple(c.as_dict() for c in reranked),
        null_decision=decision.as_dict(),
        null_mode=null_mode,
        hierarchy_applied=hierarchy_applied,
        v3_top1=v3_top1,
        final_top1=order[0] if order else "",
    )


__all__ = [
    "DEFAULT_DENSE_TOPK",
    "final_candidate_eligible",
    "sanitize_final_candidates",
    "DEFAULT_V3_TOPK",
    "DEFAULT_V41_TOPK",
    "HYBRID_VERSION",
    "NULL_MODES",
    "NULL_MODE_CONSERVATIVE",
    "NULL_MODE_SHADOW",
    "SOURCE_DENSE",
    "SOURCE_V3",
    "SOURCE_V41",
    "HybridResult",
    "PooledCandidate",
    "build_pool",
    "run_hybrid",
]
