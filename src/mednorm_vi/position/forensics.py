"""Position-policy forensics (Phase 1C-A).

Given a sample input text and a sample output (entity ``text`` + observed
``position``) produced under the organizer's *undisclosed* policy, compare every
registered policy hypothesis and report which best explains the observed offsets:
exact/near matches, systematic deltas, per-line cumulative shift, max offset, and
CRLF/LF, inclusive/exclusive, and byte/code-point evidence.

This module contains NO organizer sample text — callers pass file paths; tests
use synthetic fixtures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .encoders import PositionEncodingError
from .registry import PositionPolicyRegistry


@dataclass(frozen=True, slots=True)
class Observation:
    """One observed output entity: its text and the organizer-reported position."""

    entity_text: str
    org_start: int
    org_end: int


@dataclass(frozen=True, slots=True)
class PolicyStat:
    policy_id: str
    matched: int  # observations with at least one locatable raw candidate
    exact: int  # encoded position equals the observed position exactly
    near: int  # within near_tolerance on start
    systematic_start_delta: int | None  # modal (observed_start - encoded_start)
    max_abs_start_delta: int


@dataclass(frozen=True, slots=True)
class ForensicsReport:
    n_observations: int
    n_locatable: int
    policy_stats: tuple[PolicyStat, ...]
    best_policy_id: str | None
    byte_vs_codepoint: str  # "byte" | "codepoint" | "inconclusive"
    line_ending_evidence: str  # "raw" | "lf_canonical" | "crlf_canonical" | "inconclusive"
    interval_evidence: str  # "half_open" | "closed" | "inconclusive"
    cumulative_line_shift: tuple[tuple[int, int, int], ...]  # (obs_index, org_start, delta)
    max_offset: int
    notes: tuple[str, ...] = field(default_factory=tuple)


def _candidate_spans(text: str, entity_text: str) -> list[tuple[int, int]]:
    if not entity_text:
        return []
    out: list[tuple[int, int]] = []
    i = text.find(entity_text)
    while i != -1:
        out.append((i, i + len(entity_text)))
        i = text.find(entity_text, i + 1)
    return out


def _best_candidate_delta(
    registry: PositionPolicyRegistry, policy_id: str, text: str,
    spans: list[tuple[int, int]], obs: Observation,
) -> tuple[int, int, bool] | None:
    """Return (start_delta, end_delta, exact) for the candidate closest on start."""
    best: tuple[int, int, bool] | None = None
    best_key = None
    for s, e in spans:
        try:
            enc = registry.encode(policy_id, text, s, e)
        except PositionEncodingError:
            continue
        sd = obs.org_start - enc.start
        ed = obs.org_end - enc.end
        exact = enc.start == obs.org_start and enc.end == obs.org_end
        key = (abs(sd), abs(ed), s)
        if best_key is None or key < best_key:
            best_key = key
            best = (sd, ed, exact)
    return best


def analyze(
    registry: PositionPolicyRegistry, text: str, observations: list[Observation],
    *, near_tolerance: int = 2,
) -> ForensicsReport:
    """Compare every registered policy against the observed positions."""
    per_policy: list[PolicyStat] = []
    exact_by_policy: dict[str, int] = {}
    locatable = 0
    for obs in observations:
        if _candidate_spans(text, obs.entity_text):
            locatable += 1

    for policy_id in registry.policy_ids:
        matched = exact = near = 0
        deltas: list[int] = []
        max_abs = 0
        for obs in observations:
            spans = _candidate_spans(text, obs.entity_text)
            if not spans:
                continue
            res = _best_candidate_delta(registry, policy_id, text, spans, obs)
            if res is None:
                continue
            sd, _ed, is_exact = res
            matched += 1
            deltas.append(sd)
            max_abs = max(max_abs, abs(sd))
            if is_exact:
                exact += 1
            elif abs(sd) <= near_tolerance:
                near += 1
        modal = Counter(deltas).most_common(1)[0][0] if deltas else None
        per_policy.append(PolicyStat(policy_id, matched, exact, near, modal, max_abs))
        exact_by_policy[policy_id] = exact

    best_policy_id = None
    if per_policy:
        best = max(per_policy, key=lambda p: (p.exact, p.near, -p.max_abs_start_delta, p.policy_id))
        best_policy_id = best.policy_id if best.exact > 0 or best.near > 0 else None

    # byte vs code-point evidence
    byte_exact = exact_by_policy.get("utf8-byte-half-open", 0)
    cp_exact = exact_by_policy.get("raw-codepoint-half-open", 0)
    if byte_exact > cp_exact:
        byte_vs_codepoint = "byte"
    elif cp_exact > byte_exact:
        byte_vs_codepoint = "codepoint"
    else:
        byte_vs_codepoint = "inconclusive"

    # line-ending evidence
    le_scores = {
        "raw": exact_by_policy.get("raw-codepoint-half-open", 0),
        "lf_canonical": exact_by_policy.get("canonical-lf-codepoint", 0),
        "crlf_canonical": exact_by_policy.get("canonical-crlf-codepoint", 0),
    }
    le_best = max(le_scores.items(), key=lambda kv: (kv[1], kv[0]))
    line_ending_evidence = le_best[0] if le_best[1] > 0 and \
        list(le_scores.values()).count(le_best[1]) == 1 else "inconclusive"

    # interval evidence (under raw-codepoint): does org_end match exclusive or inclusive?
    excl = incl = 0
    for obs in observations:
        spans = _candidate_spans(text, obs.entity_text)
        if not spans:
            continue
        s, e = spans[0]
        if obs.org_end == e:
            excl += 1
        elif obs.org_end == e - 1:
            incl += 1
    if excl > incl:
        interval_evidence = "half_open"
    elif incl > excl:
        interval_evidence = "closed"
    else:
        interval_evidence = "inconclusive"

    # per-line cumulative shift under raw-codepoint
    cumulative: list[tuple[int, int, int]] = []
    max_offset = 0
    for idx, obs in enumerate(observations):
        spans = _candidate_spans(text, obs.entity_text)
        if not spans:
            continue
        res = _best_candidate_delta(registry, "raw-codepoint-half-open", text, spans, obs)
        if res is None:
            continue
        sd = res[0]
        cumulative.append((idx, obs.org_start, sd))
        max_offset = max(max_offset, abs(sd))

    notes: list[str] = []
    if best_policy_id is None:
        notes.append("no registered policy explained any observed position")
    if byte_vs_codepoint == "byte":
        notes.append("multi-byte deltas suggest UTF-8 byte offsets")
    if line_ending_evidence == "crlf_canonical":
        notes.append("per-line growing deltas suggest CRLF reconstruction")

    return ForensicsReport(
        n_observations=len(observations), n_locatable=locatable,
        policy_stats=tuple(per_policy), best_policy_id=best_policy_id,
        byte_vs_codepoint=byte_vs_codepoint, line_ending_evidence=line_ending_evidence,
        interval_evidence=interval_evidence, cumulative_line_shift=tuple(cumulative),
        max_offset=max_offset, notes=tuple(notes))


__all__ = ["Observation", "PolicyStat", "ForensicsReport", "analyze"]
