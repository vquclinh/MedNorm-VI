"""Milestone 3B competition KB policy, candidate v3 builders and runtime top-k.

Milestone 3A froze a clinically governed candidate KB in which every uncertain record
was ``REQUIRES_REVIEW`` and nothing reached the runtime without a human. This package
adds the competition-oriented reading of the same frozen evidence: what the runtime
may retrieve, what it may offer to the organizer, and how many candidates it should
emit under a Jaccard-scored metric.

See :mod:`.policy` for the governance vocabulary that keeps the two apart — in
particular, why no builder here can ever record ``CLINICALLY_APPROVED``.
"""

from __future__ import annotations

from .policy import (
    CLINICALLY_APPROVED,
    COMPETITION_POLICY_VERSION,
    COMPETITION_RUNTIME_ELIGIBLE,
    EXCLUDED,
    RETRIEVAL_ONLY,
    RUNTIME_ROLES,
    UNMAPPED,
    ClinicalApprovalForbidden,
    assert_automated_decision,
    may_emit_final_candidate,
)
from .topk import (
    COMPETITION_TOPK_VERSION,
    MAX_CANDIDATES,
    RankedCandidate,
    TopKSelection,
    apply_competition_topk,
)

__all__ = [
    "CLINICALLY_APPROVED",
    "COMPETITION_POLICY_VERSION",
    "COMPETITION_RUNTIME_ELIGIBLE",
    "COMPETITION_TOPK_VERSION",
    "EXCLUDED",
    "MAX_CANDIDATES",
    "RETRIEVAL_ONLY",
    "RUNTIME_ROLES",
    "UNMAPPED",
    "ClinicalApprovalForbidden",
    "RankedCandidate",
    "TopKSelection",
    "apply_competition_topk",
    "assert_automated_decision",
    "may_emit_final_candidate",
]
