"""ZS0 — the pure zero-shot / pretrained submission baseline (Audit 0048).

No weights this project fitted, no head this project initialized, no external
API, no internet at runtime, and no `internal_test` anywhere. What it does use is
deterministic grammar, pinned pretrained checkpoints and locked ontologies.

    profile      the three arms and the zero-shot contract they must satisfy
    proposals    GLiNER and constrained-Qwen mention proposals
    resolver     the conservative deterministic resolver
    assertions   cue-and-scope logic, with Qwen only where it is uncertain
    linking      ontology-grounded candidates; codes are retrieved, not generated
    ledger       the exact parameter ledger and the 9B gate
    submission   arm selection, the organizer gate and package validation
"""

from __future__ import annotations

from .ledger import (
    MAX_ACTIVE_PARAMETERS,
    BudgetExceeded,
    LedgerEntry,
    LedgerReport,
    UnverifiedComponent,
    build_ledger,
    render_ledger,
)
from .linking import (
    Candidate,
    LinkingError,
    LinkingResult,
    OntologySnapshot,
    assert_codes_in_snapshot,
    link_mention,
)
from .profile import (
    ORGANIZER_AUTHORIZATION,
    ZS0_A,
    ZS0_ARMS,
    ZS0_B,
    ZS0_C,
    ZS0Arm,
    ZS0ProfileError,
    all_arms,
    assert_zs0_components_allowed,
    assert_zs0_feature_flags,
    build_arm,
)
from .proposals import (
    ProposalRejected,
    RejectionLedger,
    convert_gliner_spans,
    qwen_proposals,
    resolve_occurrence,
)
from .resolver import ResolvedMention, ResolverThresholds, emitted, resolve
from .submission import (
    ArmResult,
    OrganizerPreconditions,
    SubmissionError,
    assert_organizer_inference_allowed,
    select_arm,
    validate_package,
)

__all__ = [
    "MAX_ACTIVE_PARAMETERS",
    "ORGANIZER_AUTHORIZATION",
    "ZS0_A",
    "ZS0_ARMS",
    "ZS0_B",
    "ZS0_C",
    "ArmResult",
    "BudgetExceeded",
    "Candidate",
    "LedgerEntry",
    "LedgerReport",
    "LinkingError",
    "LinkingResult",
    "OntologySnapshot",
    "OrganizerPreconditions",
    "ProposalRejected",
    "RejectionLedger",
    "ResolvedMention",
    "ResolverThresholds",
    "SubmissionError",
    "UnverifiedComponent",
    "ZS0Arm",
    "ZS0ProfileError",
    "all_arms",
    "assert_codes_in_snapshot",
    "assert_organizer_inference_allowed",
    "assert_zs0_components_allowed",
    "assert_zs0_feature_flags",
    "build_arm",
    "build_ledger",
    "convert_gliner_spans",
    "emitted",
    "link_mention",
    "qwen_proposals",
    "render_ledger",
    "resolve",
    "resolve_occurrence",
    "select_arm",
    "validate_package",
]
