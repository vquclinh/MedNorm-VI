"""Typed contracts for the organizer-policy registry (Phase 1C-A).

These record the organizer's *confirmed facts* and *intentionally unresolved
policies* as data — never as executable assumptions. A hypothesis is never
presented as a confirmed rule. Nothing here loads a model or hits the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Allowed status values for a policy hypothesis (widen only with review).
POLICY_STATUSES: frozenset[str] = frozenset(
    {"unresolved", "investigating", "hypothesis", "hypothesis_leading", "resolved", "rejected"}
)
CONFIDENCE_LEVELS: frozenset[str] = frozenset({"none", "low", "medium", "high"})


@dataclass(frozen=True, slots=True)
class ConfirmedFact:
    """A fact explicitly confirmed by the organizer (with its confirming source)."""

    fact_id: str
    statement: str
    source: str
    note: str | None = None
    since: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyOption:
    """One candidate answer to an unresolved policy question."""

    id: str
    description: str


@dataclass(frozen=True, slots=True)
class PolicyHypothesis:
    """An unresolved / hypothesised organizer policy — NOT a confirmed rule.

    ``internal_default`` is what MedNorm-VI does *until* the policy resolves; it
    carries no claim about the organizer's actual choice.
    """

    policy_id: str
    title: str
    status: str
    question: str = ""
    description: str = ""
    options: tuple[PolicyOption, ...] = field(default_factory=tuple)
    supporting_observation: str = ""
    contradicting_evidence: str = ""
    confidence: str = "low"
    test_method: str = ""
    leaderboard_experiment_id: str | None = None
    internal_default: str | None = None
    linked_unresolved: str | None = None
    registry: str = ""

    @property
    def is_confirmed(self) -> bool:
        return False  # a hypothesis is never a confirmed organizer rule

    @property
    def is_open(self) -> bool:
        return self.status not in ("resolved", "rejected")


@dataclass(frozen=True, slots=True)
class OrganizerPolicyRegistry:
    """All loaded organizer-policy registries, plus a deterministic hash."""

    confirmed_facts: tuple[ConfirmedFact, ...]
    unresolved_policies: tuple[PolicyHypothesis, ...]
    position_policy_ids: tuple[str, ...]
    default_position_policy: str
    rxnorm_decoding_hypotheses: tuple[PolicyHypothesis, ...]
    icd_format_hypotheses: tuple[PolicyHypothesis, ...]
    historical_hypotheses: tuple[PolicyHypothesis, ...]
    config_hash: str

    def policy(self, policy_id: str) -> PolicyHypothesis | None:
        for group in (self.unresolved_policies, self.rxnorm_decoding_hypotheses,
                      self.icd_format_hypotheses, self.historical_hypotheses):
            for p in group:
                if p.policy_id == policy_id:
                    return p
        return None

    def fact(self, fact_id: str) -> ConfirmedFact | None:
        for f in self.confirmed_facts:
            if f.fact_id == fact_id:
                return f
        return None

    @property
    def open_policy_count(self) -> int:
        return sum(1 for p in self.unresolved_policies if p.is_open)

    @property
    def hypothesis_count(self) -> int:
        return (len(self.unresolved_policies) + len(self.rxnorm_decoding_hypotheses)
                + len(self.icd_format_hypotheses) + len(self.historical_hypotheses))


__all__ = [
    "POLICY_STATUSES",
    "CONFIDENCE_LEVELS",
    "ConfirmedFact",
    "PolicyOption",
    "PolicyHypothesis",
    "OrganizerPolicyRegistry",
]
