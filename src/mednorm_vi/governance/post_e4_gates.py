"""Post-E4 workflow gating (Audit 0042).

A real E4 full-training run is executing on Colab while this milestone is built.
Nine downstream tasks genuinely cannot be done until that run produces a
validated artifact, and the honest way to represent them is as explicitly blocked
records rather than as work quietly left undone.

The gate is deliberately strict and fails closed: a task is unblocked only when a
validated E4 full artifact exists, its validator passed, both checkpoints are
verified, validation metrics are readable, and internal_test was never touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

BLOCKED_ON_E4_FULL_ARTIFACT = "BLOCKED_ON_E4_FULL_ARTIFACT"
UNBLOCKED = "UNBLOCKED"

POST_E4_GATE_VERSION = "post-e4-gate-v1"

# The exact conditions that release the gate.
E4_GATE_CONDITIONS: tuple[str, ...] = (
    "full_training_complete",
    "artifact_validator_ok",
    "best_checkpoint_verified",
    "latest_checkpoint_verified",
    "validation_metrics_available",
    "internal_test_not_accessed",
)


@dataclass(frozen=True, slots=True)
class PostE4Task:
    """One downstream task that must wait for the E4 full artifact."""

    task_id: str
    description: str
    status: str = BLOCKED_ON_E4_FULL_ARTIFACT
    requires_internal_test: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "requires_internal_test": self.requires_internal_test,
        }


# The nine tasks named by the milestone, in order. None of them is executed here.
POST_E4_TASKS: tuple[PostE4Task, ...] = (
    PostE4Task("validate_e4_full_artifact",
               "Run the read-only validator against the real E4 full artifact"),
    PostE4Task("capture_e4_checkpoint_hashes",
               "Record best.pt and latest.pt SHA-256 from the persisted artifact"),
    PostE4Task("read_e4_validation_metrics",
               "Read final governed-validation metrics from the artifact"),
    PostE4Task("compare_e3_versus_e4",
               "Compare E3 ViHealthBERT against E4 PhoBERT-W2NER on governed validation"),
    PostE4Task("evaluate_e3_plus_e4_union",
               "Evaluate the E3 + E4 proposal union for complementarity"),
    PostE4Task("generate_frozen_e4_proposals",
               "Generate frozen E4 proposals for learned L4 v2 training"),
    PostE4Task("train_learned_l4_v2",
               "Train the learned L4 v2 resolver on frozen proposals"),
    PostE4Task("access_internal_test",
               "Any internal_test evaluation", requires_internal_test=True),
    PostE4Task("submit_leaderboard_predictions",
               "Produce and submit leaderboard predictions"),
)


@dataclass(frozen=True, slots=True)
class E4GateStatus:
    """Whether the post-E4 gate is open, and precisely what is missing."""

    open: bool
    satisfied: tuple[str, ...] = field(default_factory=tuple)
    missing: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""
    version: str = POST_E4_GATE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "post_e4_gate_version": self.version,
            "gate_open": self.open,
            "satisfied_conditions": list(self.satisfied),
            "missing_conditions": list(self.missing),
            "detail": self.detail,
        }


def evaluate_post_e4_gate(evidence: Mapping[str, Any] | None) -> E4GateStatus:
    """Open the gate only on complete, positive evidence. Fails closed.

    ``evidence`` is what a future milestone will read from the real artifact. A
    missing mapping, a missing key, or any falsy condition keeps the gate shut —
    absence of evidence is never treated as success.
    """
    if not evidence:
        return E4GateStatus(
            open=False, satisfied=(), missing=E4_GATE_CONDITIONS,
            detail="no E4 full artifact evidence supplied")

    satisfied: list[str] = []
    missing: list[str] = []
    for condition in E4_GATE_CONDITIONS:
        if condition == "internal_test_not_accessed":
            # Inverted: the artifact must report internal_test_accessed == False.
            accessed = evidence.get("internal_test_accessed", True)
            (satisfied if accessed is False else missing).append(condition)
            continue
        (satisfied if bool(evidence.get(condition, False)) else missing).append(condition)

    return E4GateStatus(
        open=not missing, satisfied=tuple(satisfied), missing=tuple(missing),
        detail=("E4 full artifact validated" if not missing
                else "E4 full artifact evidence incomplete"))


def post_e4_task_table(
    gate: E4GateStatus | None = None,
    tasks: Sequence[PostE4Task] = POST_E4_TASKS,
) -> dict[str, Any]:
    """Report every downstream task with its current blocked/unblocked status."""
    status = gate or evaluate_post_e4_gate(None)
    resolved = []
    for task in tasks:
        task_status = BLOCKED_ON_E4_FULL_ARTIFACT if not status.open else UNBLOCKED
        resolved.append({**task.as_dict(), "status": task_status})
    return {
        "post_e4_gate_version": POST_E4_GATE_VERSION,
        "gate": status.as_dict(),
        "tasks": resolved,
        "blocked_task_count": sum(
            1 for task in resolved if task["status"] == BLOCKED_ON_E4_FULL_ARTIFACT),
        "frozen_proposal_generation_allowed": status.open,
        "internal_test_accessed": False,
    }


def assert_frozen_proposals_allowed(gate: E4GateStatus) -> None:
    """Frozen L4-v2 proposals may only be generated after the gate opens."""
    if not gate.open:
        raise RuntimeError(
            "frozen E4 proposal generation is "
            f"{BLOCKED_ON_E4_FULL_ARTIFACT}; missing: {', '.join(gate.missing)}")


__all__ = [
    "BLOCKED_ON_E4_FULL_ARTIFACT",
    "E4_GATE_CONDITIONS",
    "POST_E4_GATE_VERSION",
    "POST_E4_TASKS",
    "UNBLOCKED",
    "E4GateStatus",
    "PostE4Task",
    "assert_frozen_proposals_allowed",
    "evaluate_post_e4_gate",
    "post_e4_task_table",
]
