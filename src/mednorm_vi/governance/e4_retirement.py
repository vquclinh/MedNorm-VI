"""E4 retirement from the active stack (Audit 0048).

The human owner retired E4 PhoBERT-W2NER after a completed, valid Stage-2
ablation. This module records that decision as data rather than prose, so the
retirement is checkable by tests and by the runtime rather than remembered.

**Retired, not deleted.** The E4 source, its tests and Audits 0043-0047 remain in
the repository as historical research evidence. What changes is that E4 is off in
every active profile, absent from every active registry and ledger, and
mechanically unable to reach Stage 3 or Stage 4.

The architecture PDF describes a candidate *super*-architecture. The active
runtime stack is a validated subset of it, and E4 is no longer in that subset.
Nothing here modifies the PDF.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

E4_RETIREMENT_VERSION = "e4-retirement-v1"

RETIRED_FROM_ACTIVE_STACK = "RETIRED_FROM_ACTIVE_STACK"

E4_EXPERT_ID = "E4_phobert_w2ner"
E4_FEATURE_FLAG = "enable_e4_phobert_w2ner"
E4_CHECKPOINT_KEY = "mention/phobert_w2ner"

# Stages that must never run now. Both are blocked at the gate and in config.
E4_FORBIDDEN_STAGES: tuple[str, ...] = ("subset_smoke", "full_training")


class E4RetirementError(RuntimeError):
    """Raised when something tries to use retired E4 in an active path."""


# ---------------------------------------------------------------------------
# The completed Stage-2 evidence this decision rests on
# ---------------------------------------------------------------------------
#
# A VALID run, recorded exactly: full epoch bound, full warmup, peak learning
# rates, and — after the Audit-0047 repair — a real same-state save/reload
# reproduction that passed for every recipe. No recipe reached the unchanged
# exact-F1 >= 0.95 gate, so `selected_recipe` is null and the chain stops.
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
    "stage3_blocked": True,
    "stage4_blocked": True,
}

# The best result any recipe achieved, for the record. group_balanced_ce more
# than doubled the baseline — the Audit-0047 diagnosis of background dominance
# was correct — and still fell well short of memorizing 12 examples.
E4_BEST_ACHIEVED_EXACT_F1 = 0.7333
E4_BEST_ACHIEVED_RECIPE = "group_balanced_ce"


@dataclass(frozen=True, slots=True)
class E4RetirementRecord:
    """The retirement decision, its evidence, and what it does and does not do."""

    status: str = RETIRED_FROM_ACTIVE_STACK
    decided_by: str = "human_owner"
    version: str = E4_RETIREMENT_VERSION

    @property
    def preserved(self) -> tuple[str, ...]:
        return (
            "src/mednorm_vi/training/phase2/e4/",
            "tests/unit/test_e4_clean_training.py",
            "notebooks/MedNorm_E4_Clean_Training.ipynb",
            "configs/training/phase2_e4.yaml",
            "docs/audits/0043-e4-post-training-collapse-diagnosis.md",
            "docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md",
            "docs/audits/0045-clean-slate-e4-replacement-and-gated-training.md",
            "docs/audits/0046-e4-tiny-overfit-execution-contract-repair.md",
            "docs/audits/0047-e4-positive-cell-objective-repair.md",
        )

    @property
    def withdrawn(self) -> tuple[str, ...]:
        return (
            "active and default pipeline feature flags",
            "active inference expert registries",
            "the deployment parameter ledger",
            "any expectation that an E4 checkpoint exists or loads",
            "Stage 3 (subset smoke) and Stage 4 (full training)",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "retirement_version": self.version,
            "expert_id": E4_EXPERT_ID,
            "status": self.status,
            "decided_by": self.decided_by,
            "evidence": dict(STAGE2_FINAL_RESULT),
            "best_achieved_exact_f1": E4_BEST_ACHIEVED_EXACT_F1,
            "best_achieved_recipe": E4_BEST_ACHIEVED_RECIPE,
            "preserved_as_historical_evidence": list(self.preserved),
            "withdrawn_from": list(self.withdrawn),
            "source_deleted": False,
            "audits_modified": False,
            "architecture_pdf_modified": False,
            "forbidden_stages": list(E4_FORBIDDEN_STAGES),
        }


def assert_e4_disabled(feature_flags: Mapping[str, Any], *, profile: str) -> None:
    """Refuse a profile that enables retired E4."""
    if bool(feature_flags.get(E4_FEATURE_FLAG, False)):
        raise E4RetirementError(
            f"profile {profile!r} enables {E4_FEATURE_FLAG}; E4 is "
            f"{RETIRED_FROM_ACTIVE_STACK} and must be false in every active "
            "profile. Its source and audits are preserved; only its activation "
            "is withdrawn.")


def assert_no_e4_checkpoint_required(required: Sequence[str], *, profile: str) -> None:
    """No active profile may require or expect an E4 checkpoint."""
    if E4_CHECKPOINT_KEY in set(required):
        raise E4RetirementError(
            f"profile {profile!r} requires {E4_CHECKPOINT_KEY!r}; retired E4 has "
            "no checkpoint and none is expected")


def assert_stage_not_forbidden(stage: str) -> None:
    """Stage 3 and Stage 4 must never run for E4."""
    if stage in E4_FORBIDDEN_STAGES:
        raise E4RetirementError(
            f"E4 stage {stage!r} must never run: E4 is "
            f"{RETIRED_FROM_ACTIVE_STACK}. The completed Stage-2 ablation "
            f"selected no recipe (best exact F1 {E4_BEST_ACHIEVED_EXACT_F1} by "
            f"{E4_BEST_ACHIEVED_RECIPE}, gate 0.95), so there is nothing to "
            "promote.")


def assert_e4_absent_from_ledger(component_ids: Sequence[str]) -> None:
    """E4 carries no parameters in the active deployment ledger."""
    if E4_EXPERT_ID in set(component_ids) or "e4_phobert_w2ner" in set(component_ids):
        raise E4RetirementError(
            "the active deployment ledger lists E4; a retired expert loads no "
            "weights and contributes no parameters")


__all__ = [
    "E4_BEST_ACHIEVED_EXACT_F1",
    "E4_BEST_ACHIEVED_RECIPE",
    "E4_CHECKPOINT_KEY",
    "E4_EXPERT_ID",
    "E4_FEATURE_FLAG",
    "E4_FORBIDDEN_STAGES",
    "E4_RETIREMENT_VERSION",
    "RETIRED_FROM_ACTIVE_STACK",
    "STAGE2_FINAL_RESULT",
    "E4RetirementError",
    "E4RetirementRecord",
    "assert_e4_absent_from_ledger",
    "assert_e4_disabled",
    "assert_no_e4_checkpoint_required",
    "assert_stage_not_forbidden",
]
