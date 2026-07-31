"""Governance vocabulary for the Milestone-3B competition KB (Audit 0058).

Milestone 3A froze a *clinically governed* candidate KB: every uncertain record was
marked ``REQUIRES_REVIEW`` and nothing could reach the runtime without a human
adjudicating it. That is the correct posture for a clinical deployment and the wrong
posture for a scored competition, because it blocks the runtime on thousands of
manual decisions that the organizer's metric never asks about.

Milestone 3B separates the two questions that 3A had fused into one:

    "may a human rely on this mapping clinically?"      -> CLINICALLY_APPROVED
    "may the runtime offer this code to the scorer?"    -> COMPETITION_RUNTIME_ELIGIBLE

An automated builder may answer the second. It may **never** answer the first: no
code path in this package is permitted to emit :data:`CLINICALLY_APPROVED`, and
:func:`assert_automated_decision` enforces that at the one place decisions are made.
Representing a machine's lexical inference as a clinician's approval would be a
governance lie, and it is the single failure mode this module exists to prevent.

The remaining roles are deliberately narrow. ``RETRIEVAL_ONLY`` widens *how the
runtime looks things up* without widening *what the runtime may emit* — a
normalization or query-expansion aid whose target code must still earn candidacy
through the searchable concept index and mention evidence. ``EXCLUDED`` is a
positive statement that the evidence contradicts the mapping. ``UNMAPPED`` is the
honest absence of a target.
"""

from __future__ import annotations

COMPETITION_POLICY_VERSION = "competition-kb-policy-v1"

# --- runtime roles ---------------------------------------------------------------

#: A human clinician adjudicated this record. **No automated path may emit this.**
CLINICALLY_APPROVED = "CLINICALLY_APPROVED"
#: The runtime may offer this record's code to the organizer as a final candidate.
COMPETITION_RUNTIME_ELIGIBLE = "COMPETITION_RUNTIME_ELIGIBLE"
#: The record may normalize a surface or expand retrieval. It may **not** emit a code.
RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
#: Source evidence contradicts the record; it is inert everywhere.
EXCLUDED = "EXCLUDED"
#: No target could be established from local source evidence.
UNMAPPED = "UNMAPPED"

RUNTIME_ROLES: tuple[str, ...] = (
    CLINICALLY_APPROVED,
    COMPETITION_RUNTIME_ELIGIBLE,
    RETRIEVAL_ONLY,
    EXCLUDED,
    UNMAPPED,
)

#: The only decisions a builder may reach without a human in the loop.
AUTOMATED_DECISIONS: frozenset[str] = frozenset({RETRIEVAL_ONLY, EXCLUDED, UNMAPPED})

#: Roles whose records may appear in a final organizer candidate list.
FINAL_CANDIDATE_ROLES: frozenset[str] = frozenset(
    {CLINICALLY_APPROVED, COMPETITION_RUNTIME_ELIGIBLE}
)

# --- confidence bands ------------------------------------------------------------
#
# Evidence bands, not calibrated probabilities. Nothing in this repository produces a
# calibrated probability (Audit 0053 §19), so a numeric score here would be a
# fabricated confidence. These are ordinal labels over *what kind of evidence exists*.

CONFIDENCE_SOURCE_EXACT = "source_exact_match"
CONFIDENCE_SOURCE_SYNONYM = "source_synonym_match"
CONFIDENCE_LEXICAL_EQUIVALENCE = "lexical_equivalence_unadjudicated"
CONFIDENCE_NONE = "no_supporting_evidence"

CONFIDENCE_BANDS: tuple[str, ...] = (
    CONFIDENCE_SOURCE_EXACT,
    CONFIDENCE_SOURCE_SYNONYM,
    CONFIDENCE_LEXICAL_EQUIVALENCE,
    CONFIDENCE_NONE,
)


class ClinicalApprovalForbidden(RuntimeError):
    """Raised when an automated path tries to claim clinical approval.

    Fail loudly rather than downgrade silently: a builder that *tried* to write
    ``CLINICALLY_APPROVED`` has a policy bug, and quietly rewriting the value to
    ``RETRIEVAL_ONLY`` would hide it behind an output that looks correct.
    """


def assert_automated_decision(decision: str) -> str:
    """Return ``decision`` if a builder may reach it without human adjudication.

    Raises :class:`ClinicalApprovalForbidden` for :data:`CLINICALLY_APPROVED` and
    :class:`ValueError` for anything outside the governed vocabulary.
    """
    if decision == CLINICALLY_APPROVED:
        raise ClinicalApprovalForbidden(
            "an automated builder may never record CLINICALLY_APPROVED; "
            f"allowed automated decisions are {sorted(AUTOMATED_DECISIONS)}"
        )
    if decision not in AUTOMATED_DECISIONS:
        raise ValueError(f"unknown automated crosswalk decision: {decision!r}")
    return decision


def may_emit_final_candidate(runtime_role: str) -> bool:
    """Whether a record with this role may appear in a final organizer candidate list."""
    return runtime_role in FINAL_CANDIDATE_ROLES


__all__ = [
    "AUTOMATED_DECISIONS",
    "CLINICALLY_APPROVED",
    "COMPETITION_POLICY_VERSION",
    "COMPETITION_RUNTIME_ELIGIBLE",
    "CONFIDENCE_BANDS",
    "CONFIDENCE_LEXICAL_EQUIVALENCE",
    "CONFIDENCE_NONE",
    "CONFIDENCE_SOURCE_EXACT",
    "CONFIDENCE_SOURCE_SYNONYM",
    "EXCLUDED",
    "FINAL_CANDIDATE_ROLES",
    "RETRIEVAL_ONLY",
    "RUNTIME_ROLES",
    "UNMAPPED",
    "ClinicalApprovalForbidden",
    "assert_automated_decision",
    "may_emit_final_candidate",
]
