"""ICD-10 Super Linker: lexical retrieval + hierarchy + specificity (spec §9, §9.3).

The old implementation was three lines of substance: ``search_index(text)``, keep what
exists in the snapshot, return. No hierarchy, no specificity control, no competition
between a broader and a narrower code — even though the hierarchy graph was loaded into
memory on every run. See ``icd10_hierarchy`` for what the graph actually contains and
why direction has to be reconstructed from code length.

The decision procedure, in order:

1. retrieve lexically (exact, accent-insensitive, n-gram, sparse);
2. expand each anchor upward (ancestors), downward (descendants) and — only when the
   anchor's own added detail is unsupported — sideways (siblings);
3. for every reached code, ask the one question spec §9.3 cares about: **does the
   mention's text express the detail this code adds over its parent?**
4. suppress a descendant whose added detail is absent from the text, recording the
   exact missing tokens;
5. prefer the ancestor in that case — the conservative broader-code fallback;
6. keep a specific code when the text does support it;
7. enforce snapshot membership, order deterministically, record everything.

**Depth is never treated as correctness.** A five-character code does not outrank its
four-character parent because it is longer; it outranks it when, and only when, the
mention says the extra thing. ``metadata.specificity`` equals ``len(code) - 3`` for
every record in the locked snapshot, so it is reported for provenance and excluded from
ranking.

Candidate *accuracy* is not reported, here or anywhere: the governed corpus contains
zero ICD-10 codes, so Recall@K and candidate Jaccard are
UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD (see ``evaluation.code_linking``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing import evidence as ev
from ..kb.indexing.evidence import EXACT_CHANNELS
from ..kb.indexing.retrieval import LocalIndex, search_index
from ..resolution.models import EntityHypothesis
from .icd10_hierarchy import (
    ICD10_HIERARCHY_VERSION,
    REL_ANCESTOR,
    REL_DESCENDANT,
    REL_SELF,
    REL_SIBLING,
    HierarchyContext,
    canonical_name,
    content_tokens,
    dotted_code,
    expand_hierarchy,
    hierarchy_context,
)
from .models import LinkedCandidate, LinkerResult

ICD10_LINKER_VERSION = "icd10-hierarchy-linker-v1"

# Retention reasons.
KEEP_EXACT_NAME = "KEEP_EXACT_NAME"
KEEP_LEXICAL = "KEEP_LEXICAL"
KEEP_SPECIFIC_SUPPORTED = "KEEP_SPECIFIC_SUPPORTED"
KEEP_BROADER_FALLBACK = "KEEP_BROADER_FALLBACK"
KEEP_SIBLING_COMPETITION = "KEEP_SIBLING_COMPETITION"

# Suppression reasons.
DROP_UNSUPPORTED_SPECIFICITY = "DROP_UNSUPPORTED_SPECIFICITY"
DROP_NOT_IN_SNAPSHOT = "DROP_NOT_IN_SNAPSHOT"
DROP_NO_LEXICAL_SUPPORT = "DROP_NO_LEXICAL_SUPPORT"
DROP_BUDGET = "DROP_BUDGET"

# Evidence tiers, ordered. Not calibrated probabilities.
TIER_EXACT_NAME = "exact_name"
TIER_SUPPORTED_SPECIFIC = "supported_specific"
TIER_BROADER = "broader"
TIER_LEXICAL = "lexical"
TIER_ORDER: tuple[str, ...] = (
    TIER_EXACT_NAME, TIER_SUPPORTED_SPECIFIC, TIER_BROADER, TIER_LEXICAL)

CANDIDATE_BOUND = 20
RETRIEVAL_LIMIT = 24
# A hierarchy-expanded code with no lexical support of its own needs at least this
# fraction of its content tokens present in the mention to stay in play. Expansion
# must widen recall, not import the whole chapter.
MIN_TOKEN_OVERLAP = 0.5


@dataclass(frozen=True, slots=True)
class Icd10CandidateDecision:
    """Everything spec §9.3 asks to be recorded about one ICD candidate."""

    code: str
    dotted_code: str
    retained: bool
    reason: str
    tier: str
    score: float
    lexical_sources: tuple[str, ...] = field(default_factory=tuple)
    lexical_score: float = 0.0
    hierarchy: HierarchyContext | None = None
    token_overlap: float = 0.0
    snapshot_id: str = ""

    @property
    def missing_detail(self) -> tuple[str, ...]:
        if self.hierarchy is None or self.hierarchy.specificity is None:
            return ()
        return self.hierarchy.specificity.missing_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "dotted_code": self.dotted_code,
            "retained": self.retained, "reason": self.reason, "tier": self.tier,
            "score": round(self.score, 4),
            "lexical_sources": list(self.lexical_sources),
            "lexical_score": round(self.lexical_score, 4),
            "token_overlap": round(self.token_overlap, 4),
            "missing_required_detail": list(self.missing_detail),
            "hierarchy": self.hierarchy.as_dict() if self.hierarchy else None,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class Icd10LinkReport:
    """Per-mention ICD report. Consumed by L6/L7/L8, summarised in manifests."""

    hypothesis_id: str
    decisions: tuple[Icd10CandidateDecision, ...] = field(default_factory=tuple)
    expansion_notes: tuple[str, ...] = field(default_factory=tuple)
    snapshot_id: str = ""
    linker_version: str = ICD10_LINKER_VERSION
    hierarchy_version: str = ICD10_HIERARCHY_VERSION

    @property
    def retained(self) -> tuple[Icd10CandidateDecision, ...]:
        return tuple(d for d in self.decisions if d.retained)

    @property
    def suppressed(self) -> tuple[Icd10CandidateDecision, ...]:
        return tuple(d for d in self.decisions if not d.retained)

    @property
    def best_tier(self) -> str:
        for tier in TIER_ORDER:
            if any(d.tier == tier for d in self.retained):
                return tier
        return TIER_LEXICAL

    @property
    def has_sibling_competition(self) -> bool:
        return any(
            note.startswith("sibling_competition_opened")
            for note in self.expansion_notes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "linker_version": self.linker_version,
            "hierarchy_version": self.hierarchy_version,
            "hypothesis_id": self.hypothesis_id,
            "snapshot_id": self.snapshot_id,
            "best_tier": self.best_tier,
            "retained": len(self.retained),
            "suppressed": len(self.suppressed),
            "sibling_competition": self.has_sibling_competition,
            "expansion_notes": list(self.expansion_notes),
            "decisions": [d.as_dict() for d in self.decisions],
        }


def _token_overlap(index: LocalIndex, code: str, mention_text: str) -> float:
    """Fraction of the code's content tokens present in the mention."""
    code_tokens = content_tokens(canonical_name(index, code))
    if not code_tokens:
        return 0.0
    return len(code_tokens & content_tokens(mention_text)) / len(code_tokens)


def _score(tier: str, lexical: float, overlap: float, depth: int) -> float:
    """Deterministic ordering score. Not a probability.

    Depth contributes only a small positive term, and only *inside* a tier — a
    supported specific code has already earned its tier, so depth breaks ties among
    equally supported codes rather than promoting an unsupported one.
    """
    tier_weight = float(len(TIER_ORDER) - TIER_ORDER.index(tier)) * 1000.0
    return (tier_weight + overlap * 100.0 + float(depth) * 5.0
            + min(lexical, 200.0) / 1000.0)


def link_icd10_hierarchical(
    hypothesis: EntityHypothesis,
    index: LocalIndex,
    *,
    limit: int = CANDIDATE_BOUND,
) -> Icd10LinkReport:
    """Hierarchy-aware ICD-10 linking for one diagnosis hypothesis."""
    if index.index_type != "icd10_vi":
        return Icd10LinkReport(
            hypothesis.hypothesis_id, expansion_notes=("wrong_index_type",),
            snapshot_id=index.source_snapshot_id)

    mention = hypothesis.text
    hits = {h.concept_id: h for h in search_index(index, mention, limit=RETRIEVAL_LIMIT)}
    # Order by the evidence rank tuple, not by the additive score. On an untiered index
    # (competition-v3) the tuple reduces to `(-score, code)`, so this is the previous order.
    anchors = tuple(sorted(hits, key=lambda c: hits[c].rank))
    reached, notes = expand_hierarchy(index, anchors, mention)
    anchor = anchors[0] if anchors else ""

    decisions: list[Icd10CandidateDecision] = []
    for code, origin in reached:
        hit = hits.get(code)
        context = hierarchy_context(index, code, anchor or code, mention)
        overlap = _token_overlap(index, code, mention)
        base: dict[str, Any] = dict(
            code=code, dotted_code=dotted_code(index, code),
            lexical_sources=tuple(hit.channels) if hit else (),
            lexical_score=hit.score if hit else 0.0,
            hierarchy=context, token_overlap=overlap,
            snapshot_id=index.source_snapshot_id)

        if not index.exists(code):
            decisions.append(Icd10CandidateDecision(
                retained=False, reason=DROP_NOT_IN_SNAPSHOT, tier=TIER_LEXICAL,
                score=0.0, **base))
            continue

        channels: tuple[str, ...] = tuple(hit.channels) if hit else ()
        exact = any(c in EXACT_CHANNELS for c in channels)

        # A code reached only by expansion needs its own textual support. Without
        # this, expanding one anchor would drag in every descendant of its block.
        if hit is None and overlap < MIN_TOKEN_OVERLAP:
            decisions.append(Icd10CandidateDecision(
                retained=False, reason=DROP_NO_LEXICAL_SUPPORT, tier=TIER_LEXICAL,
                score=0.0, **base))
            continue

        # Zero content-token overlap means the candidate's name shares no diagnostic
        # word with the mention — it arrived on trigram coincidence alone. An exact
        # name match is exempt, because it *is* the name.
        if not exact and overlap <= 0.0:
            decisions.append(Icd10CandidateDecision(
                retained=False, reason=DROP_NO_LEXICAL_SUPPORT, tier=TIER_LEXICAL,
                score=0.0, **base))
            continue

        # The specificity gate (spec §9.3). A descendant must earn its extra detail.
        assessment = context.specificity
        if (assessment is not None and not assessment.justified
                and origin in {REL_DESCENDANT, REL_SELF, REL_SIBLING}
                and context.depth > 0):
            decisions.append(Icd10CandidateDecision(
                retained=False, reason=DROP_UNSUPPORTED_SPECIFICITY, tier=TIER_LEXICAL,
                score=0.0, **base))
            continue

        if exact:
            tier, reason = TIER_EXACT_NAME, KEEP_EXACT_NAME
        elif assessment is not None and assessment.supported_tokens:
            tier, reason = TIER_SUPPORTED_SPECIFIC, KEEP_SPECIFIC_SUPPORTED
        elif origin == REL_ANCESTOR:
            tier, reason = TIER_BROADER, KEEP_BROADER_FALLBACK
        elif origin == REL_SIBLING:
            tier, reason = TIER_LEXICAL, KEEP_SIBLING_COMPETITION
        else:
            tier, reason = TIER_LEXICAL, KEEP_LEXICAL
        decisions.append(Icd10CandidateDecision(
            retained=True, reason=reason, tier=tier,
            score=_score(tier, base["lexical_score"], overlap, context.depth), **base))

    def evidence_order(decision: Icd10CandidateDecision) -> int:
        """Evidence-tier position within a linker tier (Audit 0069 §4).

        Constant on an untiered index, so competition-v3 keeps its exact previous ordering;
        on a tiered index it is what stops accumulated fuzzy score from promoting a
        one-diacritic false friend above an accent-exact match.
        """
        if not index.tiered:
            return 0
        tier = ev.tier_of(decision.lexical_sources)
        return ev.TIER_ORDER.index(tier) if tier else ev.UNTIERED_RANK

    retained = sorted(
        (d for d in decisions if d.retained),
        key=lambda d: (evidence_order(d), TIER_ORDER.index(d.tier), -d.score, d.code))
    final: list[Icd10CandidateDecision] = list(retained[:limit])
    final.extend(
        Icd10CandidateDecision(**{
            **_decision_fields(d), "retained": False, "reason": DROP_BUDGET})
        for d in retained[limit:])
    final.extend(d for d in decisions if not d.retained)
    return Icd10LinkReport(
        hypothesis.hypothesis_id, tuple(final), notes, index.source_snapshot_id)


def _decision_fields(decision: Icd10CandidateDecision) -> dict[str, Any]:
    return {
        "code": decision.code, "dotted_code": decision.dotted_code,
        "retained": decision.retained, "reason": decision.reason,
        "tier": decision.tier, "score": decision.score,
        "lexical_sources": decision.lexical_sources,
        "lexical_score": decision.lexical_score, "hierarchy": decision.hierarchy,
        "token_overlap": decision.token_overlap, "snapshot_id": decision.snapshot_id,
    }


def link_icd10(
    hypothesis: EntityHypothesis,
    index: LocalIndex,
    *,
    limit: int = CANDIDATE_BOUND,
    dotted_output: bool = False,
) -> LinkerResult:
    """Public L5 entry point. Same signature as before; hierarchy-aware underneath."""
    report = link_icd10_hierarchical(hypothesis, index, limit=limit)
    if "wrong_index_type" in report.expansion_notes:
        return LinkerResult(hypothesis.hypothesis_id, (), ("wrong_index_type",))
    candidates = tuple(
        LinkedCandidate(
            code=decision.dotted_code if dotted_output else decision.code,
            score=decision.score,
            channels=decision.lexical_sources,
            snapshot_id=decision.snapshot_id,
            evidence=_evidence_for(decision),
        )
        for decision in report.retained
    )
    warnings = tuple(report.expansion_notes) + tuple(
        f"suppressed:{d.reason}:{d.code}" for d in report.suppressed
        if d.reason == DROP_UNSUPPORTED_SPECIFICITY)[:8]
    return LinkerResult(hypothesis.hypothesis_id, candidates, warnings)


def _evidence_for(decision: Icd10CandidateDecision) -> tuple[str, ...]:
    evidence = [f"tier:{decision.tier}", f"reason:{decision.reason}"]
    context = decision.hierarchy
    if context is not None:
        evidence.append(f"relationship:{context.relationship}")
        if context.ancestor_path:
            evidence.append(f"ancestors:{'>'.join(context.ancestor_path)}")
        if context.specificity is not None and context.specificity.supported_tokens:
            evidence.append(
                "supported_detail:"
                + ",".join(context.specificity.supported_tokens))
    return tuple(evidence)


__all__ = [
    "CANDIDATE_BOUND",
    "DROP_BUDGET",
    "DROP_NOT_IN_SNAPSHOT",
    "DROP_NO_LEXICAL_SUPPORT",
    "DROP_UNSUPPORTED_SPECIFICITY",
    "ICD10_LINKER_VERSION",
    "KEEP_BROADER_FALLBACK",
    "KEEP_EXACT_NAME",
    "KEEP_LEXICAL",
    "KEEP_SIBLING_COMPETITION",
    "KEEP_SPECIFIC_SUPPORTED",
    "MIN_TOKEN_OVERLAP",
    "TIER_BROADER",
    "TIER_EXACT_NAME",
    "TIER_LEXICAL",
    "TIER_ORDER",
    "TIER_SUPPORTED_SPECIFIC",
    "Icd10CandidateDecision",
    "Icd10LinkReport",
    "link_icd10",
    "link_icd10_hierarchical",
]
