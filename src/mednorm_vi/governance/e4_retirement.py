"""E4 PhoBERT-W2NER: retired from the active architecture (Audits 0048, 0051).

E4 was researched to completion and then withdrawn by the human owner. Audit 0048
recorded the decision while keeping the implementation in the tree; Audit 0051
removed the implementation, its notebook, its configs and its tests, because dead
code preserved only because it once existed is a reactivation risk, not evidence.

This module is the **whole** surviving runtime record. It is deliberately small: a
status, the final measured result, the exact measured parameter count, the
checkpoint history, and four assertions that keep E4 unreachable. Everything
narrative lives in the audits, which are append-only and were not modified.

    status                RETIRED_FROM_ACTIVE_ARCHITECTURE
    evidence              a valid, completed Stage-2 tiny-recipe ablation
    best measured result  exact F1 0.7333 (group_balanced_ce), gate 0.95, no pass
    checkpoints           produced (training, diagnostic, reproduction); none
                          passed the acceptance gate; none retained — see
                          CHECKPOINT_HISTORY
    parameters            371,289,161 total (369,163,264 encoder +
                          2,125,897 head), PROGRAMMATICALLY_VERIFIED,
                          reconciled in Audit 0044

The architecture PDF describes a candidate *super*-architecture and lists E4 among
seven possible mention experts. The active runtime stack is a validated subset of
it. Retiring E4 changes the subset; the document is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

E4_RETIREMENT_VERSION = "e4-retirement-v3"

RETIRED_FROM_ACTIVE_ARCHITECTURE = "RETIRED_FROM_ACTIVE_ARCHITECTURE"

# The identifiers that must never reappear in an active path. Kept as data so the
# checks below and the tests that drive them cannot drift from each other.
E4_EXPERT_ID = "E4_phobert_w2ner"
E4_EXPERT_ID_LOWER = "e4_phobert_w2ner"
E4_FEATURE_FLAG = "enable_e4_phobert_w2ner"
E4_CHECKPOINT_KEY = "mention/phobert_w2ner"
E4_MODEL_ID = "vinai/phobert-large"

E4_FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    E4_EXPERT_ID, E4_EXPERT_ID_LOWER, E4_FEATURE_FLAG, E4_CHECKPOINT_KEY)


class E4RetirementError(RuntimeError):
    """Raised when something tries to use retired E4 in an active path."""


# ---------------------------------------------------------------------------
# The completed Stage-2 evidence the decision rests on
# ---------------------------------------------------------------------------
#
# A VALID run, not a runtime failure: full epoch bound, full warmup, peak learning
# rates reached, and a real same-state save/reload reproduction that passed for
# every recipe. `group_balanced_ce` more than doubled the reference baseline —
# confirming Audit 0047's diagnosis that background dominance was the binding
# constraint — and still fell well short of memorizing twelve examples. No recipe
# reached the unchanged exact-F1 >= 0.95 gate, so no recipe was selected and the
# chain stopped there.
STAGE2_FINAL_RESULT: Mapping[str, Any] = {
    "stage": "tiny_recipe_ablation",
    "run_validity": "VALID_COMPLETED_RUN",
    "examples": 12,
    "gold_mentions": 22,
    "all_required_types_present": True,
    "epochs_per_recipe": 200,
    "optimizer_steps_per_recipe": 600,
    "warmup_fully_served": True,
    "peak_learning_rate_reached": True,
    "save_reload_reproduction_passed": True,
    "recipes": {
        "reference_ce": {"best_exact_f1": 0.3448, "save_reload_reproduced": True},
        "group_balanced_ce": {"best_exact_f1": 0.7333, "save_reload_reproduced": True},
        "hard_negative_ce": {"best_exact_f1": 0.3704, "save_reload_reproduced": True},
    },
    "target_exact_f1": 0.95,
    "any_recipe_met_the_gate": False,
    "selected_recipe": None,
    "later_stages_reached": False,
}

E4_BEST_ACHIEVED_EXACT_F1 = 0.7333
E4_BEST_ACHIEVED_RECIPE = "group_balanced_ce"

# ---------------------------------------------------------------------------
# The exact historical parameter count
# ---------------------------------------------------------------------------
#
# These are MEASURED figures, not planning estimates. Audit 0044 restored both
# `best.pt` and `latest.pt`, instantiated the model twice, and reconciled the
# instantiated parameter total against each checkpoint's own declared count with
# zero missing and zero unexpected keys on both the encoder and the head. The four
# numbers agreed exactly.
#
# Do not conflate this with `configs/parameter_budget.yaml`'s "approximately
# 0.370B" row: that is spec §17's PLANNING figure for PhoBERT-large, is labelled as
# such, and is no longer summed into the active plan.
E4_ENCODER_PARAMETERS = 369_163_264
E4_W2NER_HEAD_PARAMETERS = 2_125_897
E4_TOTAL_PARAMETERS = 371_289_161
PARAMETER_COUNT_STATUS = "PROGRAMMATICALLY_VERIFIED"
PARAMETER_COUNT_EVIDENCE = "docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md"

# ---------------------------------------------------------------------------
# Checkpoint history
# ---------------------------------------------------------------------------
#
# E4 DID produce checkpoints. Saying otherwise would understate how far the
# experiment got and would make Audits 0043/0044 unreadable.
CHECKPOINT_HISTORY = (
    "E4 produced training, diagnostic and reproduction checkpoints. "
    "No E4 checkpoint passed the acceptance gate. "
    "No E4 checkpoint is retained in the current tree or active deployment. "
    "Obsolete E4 artifacts were deleted after evidence was recorded in the "
    "append-only audits."
)

# Digests recorded IN an audit, and therefore independently checkable.
E4_AUDITED_CHECKPOINT_SHA256: Mapping[str, str] = {
    "full_training/checkpoints/best.pt":
        "4cc934eb5d072bcf827e46745bbcc308beda3552b4156c4c4504f571aa0bd16f",
    "full_training/checkpoints/latest.pt":
        "22b9017da3e5a56b7086c7f03dda1aed7e78b5e7b9f844d6171ee9333355bf07",
}

# Operator-reported from the later tiny-overfit execution. Kept SEPARATE from the
# audited digests above because it appears in no audit: the tiny-overfit stage
# serialized and reloaded checkpoints, and this is the digest the operator
# reported for the hard-negative recipe. It is preserved as provenance, and
# labelled so nobody later cites it as audit-verified.
E4_OPERATOR_REPORTED_CHECKPOINT_SHA256: Mapping[str, str] = {
    "tiny_overfit/hard_negative_ce":
        "93495b355a52d9742e89247f4afd07ddf876b24a5338e4359c30172f60e8f558",
}

# Where the full narrative lives. These audits are append-only and unmodified.
EVIDENCE_AUDITS: tuple[str, ...] = (
    "docs/audits/0043-e4-post-training-collapse-diagnosis.md",
    "docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md",
    "docs/audits/0045-clean-slate-e4-replacement-and-gated-training.md",
    "docs/audits/0046-e4-tiny-overfit-execution-contract-repair.md",
    "docs/audits/0047-e4-positive-cell-objective-repair.md",
    "docs/audits/0048-e4-retirement-and-zs0-zero-shot-baseline.md",
    "docs/audits/0051-repository-wide-architecture-review-and-cleanup.md",
)


@dataclass(frozen=True, slots=True)
class E4RetirementRecord:
    """The retirement decision and its evidence, as data rather than prose."""

    status: str = RETIRED_FROM_ACTIVE_ARCHITECTURE
    decided_by: str = "human_owner"
    version: str = E4_RETIREMENT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "retirement_version": self.version,
            "expert_id": E4_EXPERT_ID,
            "model_id": E4_MODEL_ID,
            "status": self.status,
            "decided_by": self.decided_by,
            "evidence": dict(STAGE2_FINAL_RESULT),
            "best_achieved_exact_f1": E4_BEST_ACHIEVED_EXACT_F1,
            "best_achieved_recipe": E4_BEST_ACHIEVED_RECIPE,
            "encoder_parameters": E4_ENCODER_PARAMETERS,
            "w2ner_head_parameters": E4_W2NER_HEAD_PARAMETERS,
            "total_parameters": E4_TOTAL_PARAMETERS,
            "parameter_count_status": PARAMETER_COUNT_STATUS,
            "parameter_count_evidence": PARAMETER_COUNT_EVIDENCE,
            "checkpoints_ever_produced": True,
            "checkpoint_history": CHECKPOINT_HISTORY,
            "audited_checkpoint_sha256": dict(E4_AUDITED_CHECKPOINT_SHA256),
            "operator_reported_checkpoint_sha256":
                dict(E4_OPERATOR_REPORTED_CHECKPOINT_SHA256),
            "any_checkpoint_passed_the_acceptance_gate": False,
            "checkpoint_retained_in_tree": False,
            "source_deleted": True,
            "source_deleted_by_audit": "0051",
            "evidence_audits": list(EVIDENCE_AUDITS),
            "audits_modified": False,
            "architecture_pdf_modified": False,
            "counted_in_any_deployment_ledger": False,
        }


def _reason(subject: str) -> str:
    return (
        f"{subject}; E4 is {RETIRED_FROM_ACTIVE_ARCHITECTURE}. Its completed "
        f"Stage-2 ablation selected no recipe (best exact F1 "
        f"{E4_BEST_ACHIEVED_EXACT_F1} by {E4_BEST_ACHIEVED_RECIPE} against a 0.95 "
        "gate), no checkpoint was ever produced, and its implementation was "
        "removed in Audit 0051. The audits remain as the record.")


def assert_e4_absent_from_flags(
    feature_flags: Mapping[str, Any], *, profile: str
) -> None:
    """No profile may carry the retired flag — set either way."""
    if E4_FEATURE_FLAG in feature_flags:
        raise E4RetirementError(_reason(
            f"profile {profile!r} still declares {E4_FEATURE_FLAG!r}"))


def assert_no_e4_checkpoint_required(
    required: Sequence[str], *, profile: str
) -> None:
    """No active profile may require or expect an E4 checkpoint."""
    if E4_CHECKPOINT_KEY in set(required):
        raise E4RetirementError(_reason(
            f"profile {profile!r} requires {E4_CHECKPOINT_KEY!r}"))


def assert_e4_absent_from_registry(component_ids: Sequence[str]) -> None:
    """No active expert or model registry may list E4."""
    listed = sorted(
        set(component_ids) & {E4_EXPERT_ID, E4_EXPERT_ID_LOWER})
    if listed:
        raise E4RetirementError(_reason(f"an active registry lists {listed}"))


def assert_e4_absent_from_ledger(component_ids: Sequence[str]) -> None:
    """E4 carries no parameters in any deployment ledger."""
    listed = sorted(set(component_ids) & {E4_EXPERT_ID, E4_EXPERT_ID_LOWER})
    if listed:
        raise E4RetirementError(_reason(
            f"a deployment ledger lists {listed}, which loads no weights"))


__all__ = [
    "E4_BEST_ACHIEVED_EXACT_F1",
    "E4_BEST_ACHIEVED_RECIPE",
    "E4_CHECKPOINT_KEY",
    "E4_EXPERT_ID",
    "E4_EXPERT_ID_LOWER",
    "E4_FEATURE_FLAG",
    "E4_FORBIDDEN_IDENTIFIERS",
    "E4_AUDITED_CHECKPOINT_SHA256",
    "E4_ENCODER_PARAMETERS",
    "E4_MODEL_ID",
    "E4_OPERATOR_REPORTED_CHECKPOINT_SHA256",
    "E4_RETIREMENT_VERSION",
    "E4_TOTAL_PARAMETERS",
    "E4_W2NER_HEAD_PARAMETERS",
    "EVIDENCE_AUDITS",
    "CHECKPOINT_HISTORY",
    "PARAMETER_COUNT_EVIDENCE",
    "PARAMETER_COUNT_STATUS",
    "RETIRED_FROM_ACTIVE_ARCHITECTURE",
    "STAGE2_FINAL_RESULT",
    "E4RetirementError",
    "E4RetirementRecord",
    "assert_e4_absent_from_flags",
    "assert_e4_absent_from_ledger",
    "assert_e4_absent_from_registry",
    "assert_no_e4_checkpoint_required",
]
