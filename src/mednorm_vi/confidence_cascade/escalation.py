"""L7 locked-option escalation contract (spec §12, §12.1, principle P7).

**No model is loaded here, and none can be.** This module defines the contract a future
critic/adjudicator backend plugs into, and the deterministic no-model fallback that runs
until one exists. There is no import of ``llm``, no import of ``transformers``, no
network path — a test asserts this by inspecting the module's imports.

The reason the contract comes before the model is spec principle P7: a language model
may **select from a retrieved set, never introduce a value**. That guarantee has to live
in the *plumbing*, not in a prompt, because a prompt is a request and plumbing is an
invariant. So:

* ``LockedOptionSet`` freezes every alternative an escalation may return — boundaries,
  types, assertion label sets, candidate codes;
* ``EscalationDecision`` names its choices by **option id**, not by value, so a decision
  literally cannot express a code, label or coordinate that was not offered;
* ``validate_escalation_decision`` re-checks every id against the locked set and
  refuses anything else, including a well-formed answer to a different question.

Entry conditions are explicit and deterministic (§12): L4 confidence, expert
disagreement, graph consistency, assertion uncertainty, candidate ambiguity, boundary
competition, wrong-type risk, missing structured evidence, repeated-mention conflict.
Every one is recorded whether it fired or not, so "why did this not escalate?" is as
answerable as "why did it?".

When no decision source is attached, ``resolve_escalation`` returns the deterministic
fallback: keep the current best option, disposition UNRESOLVED, reason
``no_decision_source``. It never guesses, and it never silently accepts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..evidence_graph.consistency import (
    CONTRADICTED,
    REC_ESCALATE,
    RULE_ASSERTION_CONFLICT,
    RULE_ICD_HIERARCHY,
    RULE_LAB_PAIR,
    RULE_MED_CONFLICT,
    RULE_OVERLAP,
    RULE_REPEATED_MENTION,
    RULE_RXNORM_STRUCTURED,
    RULE_SECTION_COMPAT,
    UNRESOLVED,
    GraphConsistencyReport,
)
from ..linking.models import LinkerResult
from ..resolution.models import EntityHypothesis
from ..schemas.constants import ASSERTION_LABELS, ORGANIZER_LABEL_BY_TYPE
from ..specialists.assertion import AssertionDecision
from .cascade import CascadeDecision

ESCALATION_CONTRACT_VERSION = "l7-locked-option-escalation-v1"

# --- cascade tiers (spec §12) ----------------------------------------------------
TIER_DETERMINISTIC = "deterministic"
TIER_CRITIC = "critic"
TIER_ADJUDICATOR = "adjudicator"
CASCADE_TIERS: tuple[str, ...] = (TIER_DETERMINISTIC, TIER_CRITIC, TIER_ADJUDICATOR)

# --- dispositions ----------------------------------------------------------------
ACCEPT = "ACCEPT"
REJECT = "REJECT"
UNRESOLVED_DISPOSITION = "UNRESOLVED"
ESCALATE = "ESCALATE"
DISPOSITIONS: tuple[str, ...] = (ACCEPT, REJECT, UNRESOLVED_DISPOSITION, ESCALATE)

# --- entry conditions (spec §12) -------------------------------------------------
COND_LOW_L4_CONFIDENCE = "low_l4_confidence"
COND_EXPERT_DISAGREEMENT = "expert_disagreement"
COND_GRAPH_CONTRADICTION = "graph_contradiction"
COND_GRAPH_UNRESOLVED = "graph_unresolved"
COND_ASSERTION_UNCERTAIN = "assertion_uncertainty"
COND_CANDIDATE_AMBIGUITY = "candidate_ambiguity"
COND_BOUNDARY_COMPETITION = "boundary_competition"
COND_WRONG_TYPE_RISK = "wrong_type_risk"
COND_MISSING_STRUCTURED = "missing_structured_evidence"
COND_REPEATED_MENTION_CONFLICT = "repeated_mention_conflict"

ENTRY_CONDITIONS: tuple[str, ...] = (
    COND_LOW_L4_CONFIDENCE, COND_EXPERT_DISAGREEMENT, COND_GRAPH_CONTRADICTION,
    COND_GRAPH_UNRESOLVED, COND_ASSERTION_UNCERTAIN, COND_CANDIDATE_AMBIGUITY,
    COND_BOUNDARY_COMPETITION, COND_WRONG_TYPE_RISK, COND_MISSING_STRUCTURED,
    COND_REPEATED_MENTION_CONFLICT,
)

# Thresholds for the entry conditions. Fixed and documented rather than searched: this
# milestone forbids broad threshold optimization, and an unsearched threshold that is
# written down is more honest than a tuned one that is not.
LOW_CONFIDENCE_BELOW = 0.35
AMBIGUOUS_CANDIDATE_COUNT = 8
# Two candidates are "tied" when their scores differ by less than this.
CANDIDATE_TIE_EPSILON = 1e-6

# Refusal codes from decision validation.
REFUSE_UNKNOWN_OPTION = "REFUSE_UNKNOWN_OPTION"
REFUSE_WRONG_SUBJECT = "REFUSE_WRONG_SUBJECT"
REFUSE_EMPTY_DECISION = "REFUSE_EMPTY_DECISION"
REFUSE_UNKNOWN_TIER = "REFUSE_UNKNOWN_TIER"
REFUSE_INVENTED_VALUE = "REFUSE_INVENTED_VALUE"


# --- locked options --------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LockedBoundaryOption:
    """One boundary a decision may choose. Coordinates come from L3/L4, never a model."""

    option_id: str
    start: int
    end: int
    kind: str
    proposal_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id, "start": self.start, "end": self.end,
            "kind": self.kind, "proposal_id": self.proposal_id,
        }


@dataclass(frozen=True, slots=True)
class LockedTypeOption:
    """One entity type a decision may choose, from the fixed organizer vocabulary."""

    option_id: str
    entity_type: str
    organizer_label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id, "entity_type": self.entity_type,
            "organizer_label": self.organizer_label,
        }


@dataclass(frozen=True, slots=True)
class LockedAssertionOption:
    """One assertion label **set** a decision may choose.

    Label *sets*, not labels: assertions are multi-label, so offering individual labels
    would let a decision assemble a combination nobody vetted — which is how the
    Audit-0052 all-three-labels defect looked from the outside.
    """

    option_id: str
    labels: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"option_id": self.option_id, "labels": list(self.labels)}


@dataclass(frozen=True, slots=True)
class LockedCandidateOption:
    """One ontology code a decision may keep. Retrieved upstream; never minted here."""

    option_id: str
    code: str
    ontology: str
    snapshot_id: str
    score: float = 0.0
    tier: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id, "code": self.code,
            "ontology": self.ontology, "snapshot_id": self.snapshot_id,
            "score": round(self.score, 4), "tier": self.tier,
        }


@dataclass(frozen=True, slots=True)
class LockedOptionSet:
    """Everything an escalation for one hypothesis is permitted to answer with."""

    subject_id: str
    boundaries: tuple[LockedBoundaryOption, ...] = field(default_factory=tuple)
    types: tuple[LockedTypeOption, ...] = field(default_factory=tuple)
    assertions: tuple[LockedAssertionOption, ...] = field(default_factory=tuple)
    candidates: tuple[LockedCandidateOption, ...] = field(default_factory=tuple)

    def boundary_ids(self) -> frozenset[str]:
        return frozenset(o.option_id for o in self.boundaries)

    def type_ids(self) -> frozenset[str]:
        return frozenset(o.option_id for o in self.types)

    def assertion_ids(self) -> frozenset[str]:
        return frozenset(o.option_id for o in self.assertions)

    def candidate_ids(self) -> frozenset[str]:
        return frozenset(o.option_id for o in self.candidates)

    def offered_codes(self) -> frozenset[str]:
        """The exact code set L9's P7 check should be given for this mention."""
        return frozenset(o.code for o in self.candidates)

    @property
    def option_set_hash(self) -> str:
        return hashlib.sha256(json.dumps({
            "subject_id": self.subject_id,
            "boundaries": [o.as_dict() for o in self.boundaries],
            "types": [o.as_dict() for o in self.types],
            "assertions": [o.as_dict() for o in self.assertions],
            "candidates": [o.as_dict() for o in self.candidates],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "option_set_hash": self.option_set_hash,
            "boundaries": [o.as_dict() for o in self.boundaries],
            "types": [o.as_dict() for o in self.types],
            "assertions": [o.as_dict() for o in self.assertions],
            "candidates": [o.as_dict() for o in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """The evidence an escalation is allowed to see. Ids and scalars, no free text.

    ``mention_text`` is deliberately absent. A future critic will need the text, and it
    will receive it through a separate, explicitly-audited prompt-construction step —
    keeping it out of the bundle means the bundle itself is safe to log, hash and put in
    a run manifest.
    """

    subject_id: str
    entity_type: str
    l4_score: float
    l4_status: str
    expert_ids: tuple[str, ...] = field(default_factory=tuple)
    boundary_alternative_count: int = 0
    assertion_labels: tuple[str, ...] = field(default_factory=tuple)
    assertion_uncertain: bool = False
    candidate_count: int = 0
    best_candidate_tier: str = ""
    top_candidate_gap: float = 0.0
    structured_unresolved_fields: tuple[str, ...] = field(default_factory=tuple)
    consistency_verdicts: Mapping[str, str] = field(default_factory=dict)
    section_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id, "entity_type": self.entity_type,
            "l4_score": round(self.l4_score, 4), "l4_status": self.l4_status,
            "expert_ids": list(self.expert_ids),
            "boundary_alternative_count": self.boundary_alternative_count,
            "assertion_labels": list(self.assertion_labels),
            "assertion_uncertain": self.assertion_uncertain,
            "candidate_count": self.candidate_count,
            "best_candidate_tier": self.best_candidate_tier,
            "top_candidate_gap": round(self.top_candidate_gap, 4),
            "structured_unresolved_fields": list(self.structured_unresolved_fields),
            "consistency_verdicts": dict(sorted(self.consistency_verdicts.items())),
            "section_id": self.section_id,
            "contains_clinical_text": False,
        }


@dataclass(frozen=True, slots=True)
class EscalationRequest:
    """One hypothesis put to the cascade, with its locked options and its reasons."""

    subject_id: str
    tier: str
    bundle: EvidenceBundle
    options: LockedOptionSet
    triggered_conditions: tuple[str, ...] = field(default_factory=tuple)
    skipped_conditions: tuple[str, ...] = field(default_factory=tuple)
    version: str = ESCALATION_CONTRACT_VERSION

    @property
    def should_escalate(self) -> bool:
        return bool(self.triggered_conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalation_contract_version": self.version,
            "subject_id": self.subject_id, "tier": self.tier,
            "should_escalate": self.should_escalate,
            "triggered_conditions": list(self.triggered_conditions),
            "skipped_conditions": list(self.skipped_conditions),
            "bundle": self.bundle.as_dict(),
            "options": self.options.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """A decision expressed **only** as option ids. It cannot carry a new value."""

    subject_id: str
    disposition: str
    tier: str = TIER_DETERMINISTIC
    boundary_option_id: str | None = None
    type_option_id: str | None = None
    assertion_option_id: str | None = None
    candidate_option_ids: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    source: str = "deterministic_fallback"
    version: str = ESCALATION_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalation_contract_version": self.version,
            "subject_id": self.subject_id, "disposition": self.disposition,
            "tier": self.tier, "boundary_option_id": self.boundary_option_id,
            "type_option_id": self.type_option_id,
            "assertion_option_id": self.assertion_option_id,
            "candidate_option_ids": list(self.candidate_option_ids),
            "reasons": list(self.reasons), "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class EscalationValidationResult:
    """Whether a decision stayed inside its locked option set."""

    subject_id: str
    accepted: bool
    refusals: tuple[str, ...] = field(default_factory=tuple)
    detail: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id, "accepted": self.accepted,
            "refusals": list(self.refusals), "detail": list(self.detail),
        }


class EscalationDecisionSource(Protocol):
    """What a future critic/adjudicator backend must implement.

    A model backend satisfies this by returning ids from the request's option set.
    Nothing in this contract lets it return anything else, which is the point.
    """

    name: str

    def decide(self, request: EscalationRequest) -> EscalationDecision: ...


# --- building the locked option set ----------------------------------------------
def build_locked_options(
    hypothesis: EntityHypothesis,
    *,
    assertion: AssertionDecision | None = None,
    link_result: LinkerResult | None = None,
    ontology: str = "",
) -> LockedOptionSet:
    """Freeze every alternative that already exists upstream. Nothing is created."""
    boundaries = [LockedBoundaryOption(
        option_id=f"b:{hypothesis.chosen_proposal_id}",
        start=hypothesis.start, end=hypothesis.end,
        kind=hypothesis.boundary_evidence.chosen_kind,
        proposal_id=hypothesis.chosen_proposal_id)]
    boundaries.extend(
        LockedBoundaryOption(
            option_id=f"b:{alternative.proposal_id}:{alternative.start}:{alternative.end}",
            start=alternative.start, end=alternative.end, kind=alternative.kind,
            proposal_id=alternative.proposal_id)
        for alternative in hypothesis.retained_alternatives)

    # Type options: the resolved type plus any type an expert proposed. Never the whole
    # organizer vocabulary — a type nobody proposed is not an alternative, it is a guess.
    proposed = dict.fromkeys(
        (hypothesis.entity_type, *hypothesis.type_evidence.proposed_types))
    types = [
        LockedTypeOption(
            option_id=f"t:{entity_type}", entity_type=entity_type,
            # `entity_type` is already the organizer-facing label on the canonical
            # path; the lookup covers a caller that passes an internal name.
            organizer_label=ORGANIZER_LABEL_BY_TYPE.get(entity_type, entity_type))
        for entity_type in proposed if entity_type
    ]

    # Assertion options: the current set, the empty set, and each single label. Any
    # combination beyond these must be added deliberately, never assembled by a model.
    current = tuple(assertion.labels) if assertion else ()
    label_sets: list[tuple[str, ...]] = [current, ()]
    label_sets.extend((label,) for label in sorted(ASSERTION_LABELS))
    seen_sets: dict[tuple[str, ...], None] = {}
    for labels in label_sets:
        seen_sets.setdefault(tuple(sorted(labels)), None)
    assertions = [
        LockedAssertionOption(
            option_id=f"a:{'+'.join(labels) if labels else 'none'}", labels=labels)
        for labels in seen_sets
    ]

    candidates = tuple(
        LockedCandidateOption(
            option_id=f"c:{candidate.code}", code=candidate.code, ontology=ontology,
            snapshot_id=candidate.snapshot_id, score=candidate.score,
            tier=next((e.removeprefix("tier:") for e in candidate.evidence
                       if e.startswith("tier:")), ""))
        for candidate in (link_result.candidates if link_result else ())
    )
    return LockedOptionSet(
        subject_id=hypothesis.hypothesis_id,
        boundaries=tuple(dict.fromkeys(boundaries)),
        types=tuple(types), assertions=tuple(assertions), candidates=candidates)


def build_evidence_bundle(
    hypothesis: EntityHypothesis,
    *,
    assertion: AssertionDecision | None = None,
    link_result: LinkerResult | None = None,
    consistency: GraphConsistencyReport | None = None,
    section_id: str = "",
    structured_unresolved: Sequence[str] = (),
) -> EvidenceBundle:
    """Collect the scalar evidence an escalation may reason over."""
    candidates = link_result.candidates if link_result else ()
    gap = 0.0
    if len(candidates) >= 2:
        ordered = sorted((c.score for c in candidates), reverse=True)
        gap = ordered[0] - ordered[1]
    verdicts: dict[str, str] = {}
    if consistency is not None:
        for rule in (RULE_SECTION_COMPAT, RULE_LAB_PAIR, RULE_ASSERTION_CONFLICT,
                     RULE_ICD_HIERARCHY, RULE_RXNORM_STRUCTURED, RULE_MED_CONFLICT):
            verdicts[rule] = consistency.verdict_for(rule, hypothesis.hypothesis_id)
    return EvidenceBundle(
        subject_id=hypothesis.hypothesis_id, entity_type=hypothesis.entity_type,
        l4_score=hypothesis.score, l4_status=hypothesis.status,
        expert_ids=tuple(hypothesis.expert_ids),
        boundary_alternative_count=len(hypothesis.retained_alternatives),
        assertion_labels=tuple(assertion.labels) if assertion else (),
        assertion_uncertain=bool(assertion.uncertain) if assertion else False,
        candidate_count=len(candidates),
        best_candidate_tier=next(
            (e.removeprefix("tier:") for c in candidates for e in c.evidence
             if e.startswith("tier:")), ""),
        top_candidate_gap=gap,
        structured_unresolved_fields=tuple(structured_unresolved),
        consistency_verdicts=verdicts, section_id=section_id)


def evaluate_entry_conditions(
    hypothesis: EntityHypothesis,
    bundle: EvidenceBundle,
    consistency: GraphConsistencyReport | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which escalation conditions fired, and which were checked and did not.

    Both halves are returned because "this did not escalate, and here is everything
    that was checked" is the answer an auditor needs; a bare empty list is not.
    """
    triggered: list[str] = []
    skipped: list[str] = []

    def check(condition: str, fired: bool) -> None:
        (triggered if fired else skipped).append(condition)

    check(COND_LOW_L4_CONFIDENCE, bundle.l4_score < LOW_CONFIDENCE_BELOW)
    check(COND_EXPERT_DISAGREEMENT, len(bundle.expert_ids) > 1)
    # More than ONE retained alternative, not merely one: L4 retains the runner-up
    # for almost every mention, so `> 0` made this condition fire on nearly everything
    # and escalation stopped discriminating. Measured on the medication fixture: 6/6
    # subjects escalated before this change.
    check(COND_BOUNDARY_COMPETITION, bundle.boundary_alternative_count > 1)
    check(COND_ASSERTION_UNCERTAIN, bundle.assertion_uncertain)
    check(COND_CANDIDATE_AMBIGUITY, (
        bundle.candidate_count >= AMBIGUOUS_CANDIDATE_COUNT
        and bundle.top_candidate_gap <= CANDIDATE_TIE_EPSILON))
    check(COND_WRONG_TYPE_RISK, len(hypothesis.type_evidence.proposed_types) > 1)
    check(COND_MISSING_STRUCTURED, bool(bundle.structured_unresolved_fields))

    issues = (consistency.issues_for(hypothesis.hypothesis_id)
              if consistency is not None else ())
    check(COND_GRAPH_CONTRADICTION, any(i.verdict == CONTRADICTED for i in issues))
    # Only an UNRESOLVED issue whose own recommendation is ESCALATE. The consistency
    # layer distinguishes "emit with caution" from "someone must adjudicate this", and
    # ignoring that distinction escalates everything.
    check(COND_GRAPH_UNRESOLVED, any(
        i.verdict == UNRESOLVED and i.recommendation == REC_ESCALATE for i in issues))
    check(COND_REPEATED_MENTION_CONFLICT, any(
        i.rule == RULE_REPEATED_MENTION and i.verdict in {CONTRADICTED, UNRESOLVED}
        for i in issues))
    # Overlap competition is a document-level relation, so it is read from the issue
    # list rather than from the single-hypothesis bundle.
    if any(i.rule == RULE_OVERLAP and i.verdict == CONTRADICTED for i in issues):
        if COND_GRAPH_CONTRADICTION not in triggered:
            triggered.append(COND_GRAPH_CONTRADICTION)
            skipped = [c for c in skipped if c != COND_GRAPH_CONTRADICTION]
    return tuple(sorted(triggered)), tuple(sorted(skipped))


def build_escalation_request(
    hypothesis: EntityHypothesis,
    *,
    assertion: AssertionDecision | None = None,
    link_result: LinkerResult | None = None,
    consistency: GraphConsistencyReport | None = None,
    ontology: str = "",
    section_id: str = "",
    structured_unresolved: Sequence[str] = (),
    tier: str = TIER_CRITIC,
) -> EscalationRequest:
    """Assemble one escalation request. Always safe to build; escalating is separate."""
    bundle = build_evidence_bundle(
        hypothesis, assertion=assertion, link_result=link_result,
        consistency=consistency, section_id=section_id,
        structured_unresolved=structured_unresolved)
    triggered, skipped = evaluate_entry_conditions(hypothesis, bundle, consistency)
    return EscalationRequest(
        subject_id=hypothesis.hypothesis_id, tier=tier, bundle=bundle,
        options=build_locked_options(
            hypothesis, assertion=assertion, link_result=link_result,
            ontology=ontology),
        triggered_conditions=triggered, skipped_conditions=skipped)


def validate_escalation_decision(
    request: EscalationRequest, decision: EscalationDecision
) -> EscalationValidationResult:
    """Refuse any decision that steps outside the locked option set (spec P7)."""
    refusals: list[str] = []
    detail: list[str] = []

    if decision.subject_id != request.subject_id:
        refusals.append(REFUSE_WRONG_SUBJECT)
        detail.append(f"decision for {decision.subject_id!r}, "
                      f"request for {request.subject_id!r}")
    if decision.disposition not in DISPOSITIONS:
        refusals.append(REFUSE_EMPTY_DECISION)
        detail.append(f"unknown disposition {decision.disposition!r}")
    if decision.tier not in CASCADE_TIERS:
        refusals.append(REFUSE_UNKNOWN_TIER)
        detail.append(f"unknown tier {decision.tier!r}")

    options = request.options
    for value, allowed, label in (
        (decision.boundary_option_id, options.boundary_ids(), "boundary"),
        (decision.type_option_id, options.type_ids(), "type"),
        (decision.assertion_option_id, options.assertion_ids(), "assertion"),
    ):
        if value is not None and value not in allowed:
            refusals.append(REFUSE_UNKNOWN_OPTION)
            detail.append(f"{label} option {value!r} was not offered")
    for option_id in decision.candidate_option_ids:
        if option_id not in options.candidate_ids():
            refusals.append(REFUSE_INVENTED_VALUE)
            detail.append(f"candidate option {option_id!r} was not offered")

    if (decision.disposition == ACCEPT
            and decision.boundary_option_id is None
            and decision.type_option_id is None
            and decision.assertion_option_id is None
            and not decision.candidate_option_ids):
        refusals.append(REFUSE_EMPTY_DECISION)
        detail.append("ACCEPT with no chosen option decides nothing")

    return EscalationValidationResult(
        subject_id=request.subject_id, accepted=not refusals,
        refusals=tuple(dict.fromkeys(refusals)), detail=tuple(detail))


def deterministic_fallback(request: EscalationRequest) -> EscalationDecision:
    """The no-model answer. Explicit, conservative, and never a silent accept.

    It keeps whatever L4 already chose and marks the subject UNRESOLVED when an entry
    condition fired. UNRESOLVED is not rejection: it means "a stage that does not exist
    yet was needed here", which is the truthful state of L7 today.
    """
    boundary = request.options.boundaries[0].option_id if request.options.boundaries else None
    type_option = request.options.types[0].option_id if request.options.types else None
    if not request.should_escalate:
        return EscalationDecision(
            subject_id=request.subject_id, disposition=ACCEPT,
            tier=TIER_DETERMINISTIC, boundary_option_id=boundary,
            type_option_id=type_option,
            candidate_option_ids=tuple(
                o.option_id for o in request.options.candidates),
            reasons=("no_entry_condition_fired",))
    return EscalationDecision(
        subject_id=request.subject_id, disposition=UNRESOLVED_DISPOSITION,
        tier=TIER_DETERMINISTIC, boundary_option_id=boundary,
        type_option_id=type_option,
        candidate_option_ids=tuple(o.option_id for o in request.options.candidates),
        reasons=("no_decision_source", *request.triggered_conditions))


def resolve_escalation(
    request: EscalationRequest,
    source: EscalationDecisionSource | None = None,
) -> tuple[EscalationDecision, EscalationValidationResult]:
    """Run one escalation. Without a source, the deterministic fallback answers.

    A decision from a source is validated before it is returned; a refused decision is
    replaced by the fallback, so an out-of-contract answer degrades to conservative
    behaviour instead of propagating.
    """
    if source is None or not request.should_escalate:
        decision = deterministic_fallback(request)
        return decision, validate_escalation_decision(request, decision)
    proposed = source.decide(request)
    validation = validate_escalation_decision(request, proposed)
    if validation.accepted:
        return proposed, validation
    fallback = deterministic_fallback(request)
    return EscalationDecision(
        subject_id=fallback.subject_id, disposition=fallback.disposition,
        tier=fallback.tier, boundary_option_id=fallback.boundary_option_id,
        type_option_id=fallback.type_option_id,
        assertion_option_id=fallback.assertion_option_id,
        candidate_option_ids=fallback.candidate_option_ids,
        reasons=(*fallback.reasons, f"source_refused:{source.name}", *validation.refusals),
        source=f"deterministic_fallback_after_refusal:{source.name}"), validation


@dataclass(frozen=True, slots=True)
class CascadeReport:
    """Document-level L7 outcome. Summarised in the run manifest."""

    document_id: str
    requests: tuple[EscalationRequest, ...] = field(default_factory=tuple)
    decisions: tuple[EscalationDecision, ...] = field(default_factory=tuple)
    validations: tuple[EscalationValidationResult, ...] = field(default_factory=tuple)
    version: str = ESCALATION_CONTRACT_VERSION

    def disposition_counts(self) -> dict[str, int]:
        counts = {disposition: 0 for disposition in DISPOSITIONS}
        for decision in self.decisions:
            counts[decision.disposition] = counts.get(decision.disposition, 0) + 1
        return counts

    def condition_counts(self) -> dict[str, int]:
        counts = {condition: 0 for condition in ENTRY_CONDITIONS}
        for request in self.requests:
            for condition in request.triggered_conditions:
                counts[condition] = counts.get(condition, 0) + 1
        return counts

    @property
    def escalated(self) -> int:
        return sum(1 for r in self.requests if r.should_escalate)

    @property
    def refused(self) -> int:
        return sum(1 for v in self.validations if not v.accepted)

    def decision_for(self, subject_id: str) -> EscalationDecision | None:
        return next((d for d in self.decisions if d.subject_id == subject_id), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalation_contract_version": self.version,
            "document_id": self.document_id,
            "subjects": len(self.requests),
            "escalated": self.escalated,
            "refused_decisions": self.refused,
            "disposition_counts": self.disposition_counts(),
            "condition_counts": self.condition_counts(),
            "contains_clinical_text": False,
        }


def run_cascade_escalation(
    document_id: str,
    hypotheses: Sequence[EntityHypothesis],
    *,
    cascade: Sequence[CascadeDecision] = (),
    assertions: Sequence[AssertionDecision] = (),
    link_results: Sequence[LinkerResult] = (),
    consistency: GraphConsistencyReport | None = None,
    source: EscalationDecisionSource | None = None,
    ontology_by_type: Mapping[str, str] | None = None,
) -> CascadeReport:
    """Build and resolve an escalation for every hypothesis, in document order."""
    assertion_by_id = {a.hypothesis_id: a for a in assertions}
    links_by_id = {r.mention_id: r for r in link_results}
    cascade_by_id = {c.hypothesis_id: c for c in cascade}
    ontologies = ontology_by_type or {}

    requests: list[EscalationRequest] = []
    decisions: list[EscalationDecision] = []
    validations: list[EscalationValidationResult] = []
    for hypothesis in sorted(hypotheses, key=lambda h: (h.start, h.hypothesis_id)):
        request = build_escalation_request(
            hypothesis,
            assertion=assertion_by_id.get(hypothesis.hypothesis_id),
            link_result=links_by_id.get(hypothesis.hypothesis_id),
            consistency=consistency,
            ontology=ontologies.get(hypothesis.entity_type, ""))
        decision, validation = resolve_escalation(request, source)
        # A cascade rejection is authoritative: L7 must not resurrect what L7's own
        # threshold stage rejected.
        cascade_decision = cascade_by_id.get(hypothesis.hypothesis_id)
        if cascade_decision is not None and not cascade_decision.accepted:
            decision = EscalationDecision(
                subject_id=decision.subject_id, disposition=REJECT,
                tier=TIER_DETERMINISTIC,
                boundary_option_id=decision.boundary_option_id,
                type_option_id=decision.type_option_id,
                reasons=("cascade_rejected", *cascade_decision.reasons),
                source=decision.source)
        requests.append(request)
        decisions.append(decision)
        validations.append(validation)
    return CascadeReport(
        document_id, tuple(requests), tuple(decisions), tuple(validations))


__all__ = [
    "ACCEPT",
    "AMBIGUOUS_CANDIDATE_COUNT",
    "CASCADE_TIERS",
    "COND_ASSERTION_UNCERTAIN",
    "COND_BOUNDARY_COMPETITION",
    "COND_CANDIDATE_AMBIGUITY",
    "COND_EXPERT_DISAGREEMENT",
    "COND_GRAPH_CONTRADICTION",
    "COND_GRAPH_UNRESOLVED",
    "COND_LOW_L4_CONFIDENCE",
    "COND_MISSING_STRUCTURED",
    "COND_REPEATED_MENTION_CONFLICT",
    "COND_WRONG_TYPE_RISK",
    "DISPOSITIONS",
    "ENTRY_CONDITIONS",
    "ESCALATE",
    "ESCALATION_CONTRACT_VERSION",
    "LOW_CONFIDENCE_BELOW",
    "REFUSE_EMPTY_DECISION",
    "REFUSE_INVENTED_VALUE",
    "REFUSE_UNKNOWN_OPTION",
    "REFUSE_UNKNOWN_TIER",
    "REFUSE_WRONG_SUBJECT",
    "REJECT",
    "TIER_ADJUDICATOR",
    "TIER_CRITIC",
    "TIER_DETERMINISTIC",
    "UNRESOLVED_DISPOSITION",
    "CascadeReport",
    "EscalationDecision",
    "EscalationDecisionSource",
    "EscalationRequest",
    "EscalationValidationResult",
    "EvidenceBundle",
    "LockedAssertionOption",
    "LockedBoundaryOption",
    "LockedCandidateOption",
    "LockedOptionSet",
    "LockedTypeOption",
    "build_escalation_request",
    "build_evidence_bundle",
    "build_locked_options",
    "deterministic_fallback",
    "evaluate_entry_conditions",
    "resolve_escalation",
    "run_cascade_escalation",
    "validate_escalation_decision",
]
