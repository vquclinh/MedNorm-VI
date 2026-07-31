"""L8 deterministic evidence-ranked set selection (spec §13).

**This module does not perform expected-Jaccard decoding, and no longer claims to.**

Spec §13 requires a metric-aware decoder: expected-Jaccard candidate-set search over
calibrated membership probabilities (§13.2), an expected-WER boundary utility (§13.1),
per-label assertion thresholds (§13.3), and entity retention that subtracts wrong-type
risk (§13.4). All four need **calibrated probabilities**, and this project has none:
nothing downstream of L4 is calibrated, no out-of-fold predictions exist, and S6 has
not run. Audit 0051 §16 records that as the ranked-first gap.

The function that used to live here was called ``decode_expected_jaccard`` and did
none of that — it took ``result.candidates[:10]``, an unconditional fixed top-K, which
is precisely what §13.2 rules out ("Top-1 is not always optimal, and fixed top-K is
not used"). The name asserted a capability the body did not have, at every call site.

What this module does instead, honestly:

* **entity retention** by cascade decision plus deterministic evidence, not a
  probability;
* **candidate-set selection** by *evidence tier* — exact ontology evidence is kept,
  weaker evidence is admitted only while it remains comparable to the best evidence
  found, and a configurable maximum acts as a **safety bound, never the rule**;
* **assertion pass-through** that never forces a label while uncertainty remains.

Every retained or removed candidate carries a reason, so the selection is auditable
rather than merely deterministic. :func:`decode_expected_jaccard_calibrated` is the
fail-closed slot for the real §13 decoder; it raises until calibration exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..confidence_cascade.cascade import CascadeDecision
from ..confidence_cascade.escalation import REJECT, CascadeReport
from ..evidence_graph.consistency import GraphConsistencyReport
from ..kb.competition.topk import (
    COMPETITION_ADAPTIVE,
    COMPETITION_TOPK_VERSION,
    MAX_CANDIDATES,
    RankedCandidate,
    TopKPolicy,
    apply_competition_topk,
)
from ..linking.models import LinkedCandidate, LinkerResult
from ..resolution.models import EntityHypothesis
from ..specialists.assertion import AssertionDecision

DECODER_VERSION = "l8-deterministic-evidence-ranked-v1"

# Absolute safety bound on an emitted candidate list. This is NOT the selection
# rule — selection stops when evidence stops supporting a candidate (see
# `_select_candidates`). The bound exists so a pathological index cannot emit a
# thousand codes into one entity's `candidates` field.
CANDIDATE_SAFETY_BOUND = 25

# Evidence tiers, strongest first. A tier is a statement about *where the candidate
# came from*, which is the only ranking signal available without calibration.
TIER_EXACT = "exact_alias"
TIER_STRONG = "strong_lexical"
TIER_WEAK = "weak_lexical"
TIER_ORDER: tuple[str, ...] = (TIER_EXACT, TIER_STRONG, TIER_WEAK)

# A candidate is admitted while its score stays within this ratio of the best score
# in its own tier. Deliberately generous: with no calibration, aggressively pruning
# would trade recall for a confidence this layer does not have. Jaccard rewards a
# correct set, so dropping a plausible code costs as much as adding a wrong one.
RELATIVE_SCORE_FLOOR = 0.60

# Reason codes. Reported per candidate; never accompanied by clinical text.
KEEP_EXACT = "kept_exact_ontology_evidence"
KEEP_WITHIN_TIER = "kept_within_tier_score_band"
DROP_BELOW_BAND = "dropped_below_tier_score_band"
DROP_WEAKER_TIER = "dropped_weaker_tier_than_best_evidence"
DROP_SAFETY_BOUND = "dropped_at_candidate_safety_bound"
DROP_DUPLICATE = "dropped_duplicate_code"
DROP_TYPE_MISMATCH = "dropped_candidate_ontology_mismatch"
# Consistency-driven codes (Audit 0054 §7). L6's typed report is now an input, so a
# contradiction it found is visible in the decoder's own reason codes rather than being
# discovered later by a reader comparing two artifacts.
DROP_GRAPH_CONTRADICTION = "dropped_l6_fatal_contradiction"
DROP_UNSUPPORTED_CANDIDATE = "dropped_candidate_without_graph_support"
KEEP_CONSISTENCY_SUPPORTED = "l6_consistency_no_issues"
WITHHELD_FATAL_CONTRADICTION = "withheld_l6_fatal_contradiction"
# Competition top-k narrowing (Audit 0058 §7). Prefixed so a reader can tell the
# Jaccard-driven cut apart from the evidence-band cut above it.
DROP_COMPETITION_TOPK = "dropped_by_competition_topk_policy"


class CalibratedDecoderUnavailable(RuntimeError):
    """Raised by the future §13 decoder while no calibrated model exists.

    Fail-closed on purpose: a caller asking for expected-Jaccard decoding must get
    an error, not a deterministic approximation wearing the same name.
    """


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """Why one candidate code was retained or removed."""

    code: str
    tier: str
    score: float
    kept: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "tier": self.tier, "score": round(self.score, 6),
                "kept": self.kept, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class DecodedEntity:
    hypothesis: EntityHypothesis
    assertions: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)
    # Audit trail (Audit 0053). Empty for an entity whose type carries no candidates.
    candidate_decisions: tuple[CandidateDecision, ...] = field(default_factory=tuple)
    retention_reason: str = ""
    assertion_uncertain: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "decoder_version": DECODER_VERSION,
            "start": self.hypothesis.start,
            "end": self.hypothesis.end,
            "entity_type": self.hypothesis.entity_type,
            "assertions": list(self.assertions),
            "assertion_uncertain": self.assertion_uncertain,
            "candidates": list(self.candidates),
            "candidate_decisions": [d.as_dict() for d in self.candidate_decisions],
            "retention_reason": self.retention_reason,
            "performs_expected_jaccard_decoding": False,
            "uses_calibrated_probabilities": False,
        }


def _tier_of(candidate: LinkedCandidate) -> str:
    """Which evidence tier a candidate's provenance places it in."""
    channels = {channel.lower() for channel in candidate.channels}
    if channels & {"exact", "exact_alias", "alias", "alias_exact"}:
        return TIER_EXACT
    if channels & {"sparse", "sparse_terms", "bm25", "lexical"}:
        return TIER_STRONG
    return TIER_WEAK


def _select_candidates(
    candidates: Sequence[LinkedCandidate],
    *,
    safety_bound: int,
    blocked_codes: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], tuple[CandidateDecision, ...]]:
    """Select a candidate set by evidence tier and score band.

    Not a fixed top-K: the cut is made where the evidence stops supporting a
    candidate. An exact alias hit ends the search — the ontology already answered —
    and otherwise admission is relative to the best score found in the best tier
    present, so a mention with one good candidate emits one and a mention with five
    comparable candidates emits five.

    ``blocked_codes`` comes from L6's consistency report: a code a fatal issue names
    may not be emitted, whatever its score. Blocked codes are removed *before* the tier
    and band are computed, so one contradicted code cannot drag the band with it.
    """
    decisions: list[CandidateDecision] = []
    if not candidates:
        return (), ()

    if blocked_codes:
        permitted: list[LinkedCandidate] = []
        for candidate in candidates:
            if candidate.code in blocked_codes:
                decisions.append(CandidateDecision(
                    candidate.code, _tier_of(candidate), float(candidate.score), False,
                    DROP_GRAPH_CONTRADICTION))
            else:
                permitted.append(candidate)
        candidates = permitted
        if not candidates:
            return (), tuple(decisions)

    ranked = sorted(
        candidates,
        key=lambda c: (TIER_ORDER.index(_tier_of(c)), -float(c.score), c.code))
    best_tier = _tier_of(ranked[0])

    # An exact ontology hit is emitted alone with its ties: anything weaker is
    # noise beside it, and spec §9.2 gives the exact alias precedence outright.
    if best_tier == TIER_EXACT:
        kept: list[str] = []
        seen: set[str] = set()
        for candidate in ranked:
            tier = _tier_of(candidate)
            if tier != TIER_EXACT:
                decisions.append(CandidateDecision(
                    candidate.code, tier, float(candidate.score), False,
                    DROP_WEAKER_TIER))
                continue
            if candidate.code in seen:
                decisions.append(CandidateDecision(
                    candidate.code, tier, float(candidate.score), False,
                    DROP_DUPLICATE))
                continue
            seen.add(candidate.code)
            kept.append(candidate.code)
            decisions.append(CandidateDecision(
                candidate.code, tier, float(candidate.score), True, KEEP_EXACT))
        return tuple(kept), tuple(decisions)

    best_score = float(ranked[0].score)
    floor = best_score * RELATIVE_SCORE_FLOOR if best_score > 0 else 0.0
    kept = []
    seen = set()
    for candidate in ranked:
        tier = _tier_of(candidate)
        score = float(candidate.score)
        if candidate.code in seen:
            decisions.append(CandidateDecision(
                candidate.code, tier, score, False, DROP_DUPLICATE))
            continue
        if len(kept) >= safety_bound:
            decisions.append(CandidateDecision(
                candidate.code, tier, score, False, DROP_SAFETY_BOUND))
            continue
        if tier != best_tier:
            decisions.append(CandidateDecision(
                candidate.code, tier, score, False, DROP_WEAKER_TIER))
            continue
        if score < floor:
            decisions.append(CandidateDecision(
                candidate.code, tier, score, False, DROP_BELOW_BAND))
            continue
        seen.add(candidate.code)
        kept.append(candidate.code)
        decisions.append(CandidateDecision(
            candidate.code, tier, score, True, KEEP_WITHIN_TIER))
    return tuple(kept), tuple(decisions)


def decode_entities(
    hypotheses: Sequence[EntityHypothesis],
    cascade: Sequence[CascadeDecision],
    assertions: Sequence[AssertionDecision],
    links: Sequence[LinkerResult],
    *,
    candidate_safety_bound: int = CANDIDATE_SAFETY_BOUND,
    consistency: GraphConsistencyReport | None = None,
    escalation: CascadeReport | None = None,
    competition_topk: bool = True,
    topk_policy: TopKPolicy = COMPETITION_ADAPTIVE,
) -> tuple[DecodedEntity, ...]:
    """**The L8 entry point.** Deterministic, evidence-ranked final set selection.

    Truthfully named: it ranks by available evidence and records why. It does not
    estimate expected Jaccard or WER, and it uses no calibrated probability.

    ``competition_topk`` (Audit 0058 §7) applies the Jaccard-aware narrowing on top of
    the evidence-band selection: top-1 per entity, plus a second candidate only on a
    deterministic close tie or a supported ICD specific/unspecified pair. It runs
    **after** :func:`_select_candidates` and can only remove, so it cannot introduce a
    code the L5 offered set does not contain. Passing ``False`` reproduces the
    Audit-0054 behaviour exactly.

    ``consistency`` and ``escalation`` are L6's and L7's typed public contracts
    (Audit 0054). When supplied:

    * a **fatal** contradiction withholds the entity — a contradiction can no longer be
      silently accepted, which was possible while L6 was write-only;
    * codes a fatal issue names are dropped as ``DROP_GRAPH_CONTRADICTION``;
    * an L7 ``REJECT`` disposition withholds the entity;
    * every consistency-driven action appears in the reason codes.

    Omitting them reproduces the Audit-0053 behaviour exactly, so the wiring is
    additive rather than a hidden change of policy.
    """
    accepted = {d.hypothesis_id: d for d in cascade if d.accepted}
    assertion_map = {a.hypothesis_id: a for a in assertions}
    link_map = {result.mention_id: result for result in links}

    out: list[DecodedEntity] = []
    for hypothesis in hypotheses:
        decision = accepted.get(hypothesis.hypothesis_id)
        if decision is None:
            continue

        # L7 disposition, when a cascade report was supplied.
        l7 = (escalation.decision_for(hypothesis.hypothesis_id)
              if escalation is not None else None)
        if l7 is not None and l7.disposition == REJECT:
            continue

        # L6 fatal contradiction. Two different things must not be conflated:
        #   * a fatal issue that names CANDIDATE CODES condemns those codes — a
        #     duplicate RxCUI is a reason to drop the duplicate, not to delete a
        #     correctly-found medication;
        #   * a fatal issue with no codes condemns the ENTITY (an assertion on a type
        #     that cannot carry one, two same-type spans surviving on one interval).
        # Uncertainty (UNRESOLVED) never withholds; it lowers confidence and escalates.
        entity_fatal = any(
            issue.fatal and not issue.candidate_codes
            for issue in (consistency.issues_for(hypothesis.hypothesis_id)
                          if consistency is not None else ()))
        if entity_fatal:
            continue

        blocked = (consistency.blocked_candidate_codes(hypothesis.hypothesis_id)
                   if consistency is not None else frozenset())
        assertion = assertion_map.get(hypothesis.hypothesis_id)
        link = link_map.get(hypothesis.hypothesis_id)
        codes, candidate_decisions = (
            _select_candidates(
                link.candidates, safety_bound=candidate_safety_bound,
                blocked_codes=blocked)
            if link is not None else ((), ()))

        if competition_topk and codes:
            # Narrow, never widen. The input is the list _select_candidates already
            # returned, so every surviving code was offered by L5 and the L8/L9
            # offered-set invariant holds structurally rather than by later check.
            kept_by_code = {d.code: d for d in candidate_decisions if d.kept}
            selection = apply_competition_topk(
                hypothesis.entity_type,
                tuple(
                    RankedCandidate(
                        code, kept_by_code[code].score, kept_by_code[code].tier)
                    for code in codes if code in kept_by_code),
                topk_policy,
            )
            cut = set(codes) - set(selection.codes)
            candidate_decisions = tuple(
                CandidateDecision(d.code, d.tier, d.score, False, DROP_COMPETITION_TOPK)
                if (d.kept and d.code in cut) else d
                for d in candidate_decisions)
            codes = selection.codes

        reasons = list(decision.reasons) or ["cascade_accepted"]
        if consistency is not None:
            issues = consistency.issues_for(hypothesis.hypothesis_id)
            if not issues:
                reasons.append(KEEP_CONSISTENCY_SUPPORTED)
            else:
                reasons.extend(sorted({
                    f"consistency:{issue.rule}:{issue.verdict}" for issue in issues}))
        if l7 is not None:
            reasons.append(f"l7:{l7.disposition}")
        out.append(DecodedEntity(
            hypothesis=hypothesis,
            # An empty label set is preserved as empty: spec §13.3 and the absence
            # of assertion supervision both make a guessed label the expensive error.
            assertions=assertion.labels if assertion is not None else (),
            candidates=codes,
            candidate_decisions=candidate_decisions,
            retention_reason=";".join(reasons),
            assertion_uncertain=bool(
                assertion is not None and getattr(assertion, "uncertain", False)),
        ))
    out.sort(key=lambda e: (e.hypothesis.start, e.hypothesis.end,
                            e.hypothesis.entity_type))
    return tuple(out)


def decode_expected_jaccard_calibrated(
    *_args: Any, **_kwargs: Any
) -> tuple[DecodedEntity, ...]:
    """The real spec §13 decoder. **Disabled: fails closed.**

    Enabling this requires calibrated membership probabilities per candidate, an
    expected-WER boundary utility, per-label assertion thresholds and a wrong-type
    risk term — i.e. S6 calibration over out-of-fold predictions. None exists, so
    this raises rather than silently falling back to :func:`decode_entities`.
    """
    raise CalibratedDecoderUnavailable(
        "expected-Jaccard/WER decoding (spec §13) requires calibrated probabilities; "
        "no calibration model exists (S6 has not run and no out-of-fold predictions "
        "are available). Use decode_entities() for the deterministic evidence-ranked "
        "selection, and do not present its output as metric-optimal.")


def decoder_status() -> Mapping[str, Any]:
    """What L8 actually is, for the run manifest and the runtime manifest."""
    return {
        "decoder_version": DECODER_VERSION,
        "method": "deterministic evidence-tier ranking with a relative score band",
        "performs_expected_jaccard_decoding": False,
        "performs_expected_wer_boundary_utility": False,
        "uses_calibrated_probabilities": False,
        "candidate_selection_rule": (
            "evidence tier + relative score band, then the competition top-k narrowing"
        ),
        "candidate_safety_bound": CANDIDATE_SAFETY_BOUND,
        "fixed_top_k_used": False,
        "competition_topk_policy": COMPETITION_TOPK_VERSION,
        "competition_topk_max_candidates": MAX_CANDIDATES,
        "calibrated_decoder_available": False,
        "spec_section": "13",
    }


__all__ = [
    "CANDIDATE_SAFETY_BOUND",
    "DECODER_VERSION",
    "DROP_BELOW_BAND",
    "DROP_COMPETITION_TOPK",
    "DROP_DUPLICATE",
    "DROP_SAFETY_BOUND",
    "DROP_TYPE_MISMATCH",
    "WITHHELD_FATAL_CONTRADICTION",
    "KEEP_CONSISTENCY_SUPPORTED",
    "DROP_UNSUPPORTED_CANDIDATE",
    "DROP_GRAPH_CONTRADICTION",
    "DROP_WEAKER_TIER",
    "KEEP_EXACT",
    "KEEP_WITHIN_TIER",
    "RELATIVE_SCORE_FLOOR",
    "TIER_EXACT",
    "TIER_ORDER",
    "TIER_STRONG",
    "TIER_WEAK",
    "CalibratedDecoderUnavailable",
    "CandidateDecision",
    "DecodedEntity",
    "decode_entities",
    "decode_expected_jaccard_calibrated",
    "decoder_status",
]
