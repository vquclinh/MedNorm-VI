"""Explicit freeze gate for future one-shot internal_test evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .artifacts import ArtifactValidationReport
from .common import Phase2ReadinessError, canonical_json_sha256, is_immutable_revision

INTERNAL_TEST_AUTHORIZATION = "I_AUTHORIZE_ONE_SHOT_INTERNAL_TEST_PHASE2"
INTERNAL_TEST_GATE_SCHEMA_VERSION = "phase2-internal-test-freeze-gate-v1"


class InternalTestGateError(Phase2ReadinessError):
    """Raised when a held-out evaluation gate is incomplete."""


@dataclass(frozen=True, slots=True)
class InternalTestFreezeGateReport:
    schema_version: str
    authorized: bool
    ready: bool
    failures: tuple[str, ...]
    frozen_profile_hash: str
    internal_test_accessed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authorized": self.authorized,
            "ready": self.ready,
            "failures": list(self.failures),
            "frozen_profile_hash": self.frozen_profile_hash,
            "internal_test_accessed": self.internal_test_accessed,
        }


def evaluate_internal_test_freeze_gate(
    *,
    artifact_reports: Sequence[ArtifactValidationReport],
    frozen_feature_flags: Mapping[str, bool],
    frozen_thresholds: Mapping[str, float],
    config_hashes: Mapping[str, str],
    checkpoint_hashes: Mapping[str, str],
    validation_ablation_complete: bool,
    validation_ablation_hash: str,
    model_revisions: Mapping[str, str],
    authorization: str,
) -> InternalTestFreezeGateReport:
    failures: list[str] = []
    for report in artifact_reports:
        if not report.ok:
            failures.append(f"artifact_not_valid:{report.expected_expert_id}")
    for name, enabled in frozen_feature_flags.items():
        if enabled and not checkpoint_hashes:
            failures.append(f"enabled_flag_without_checkpoint_hash:{name}")
    if not frozen_feature_flags:
        failures.append("frozen_feature_flags_missing")
    if not frozen_thresholds:
        failures.append("frozen_thresholds_missing")
    if not config_hashes:
        failures.append("config_hashes_missing")
    if not checkpoint_hashes:
        failures.append("checkpoint_hashes_missing")
    if not validation_ablation_complete:
        failures.append("validation_ablation_incomplete")
    if not validation_ablation_hash:
        failures.append("validation_ablation_hash_missing")
    for model_id, revision in model_revisions.items():
        if not is_immutable_revision(revision, allow_local_architecture=True):
            failures.append(f"model_revision_not_immutable:{model_id}")
    authorized = authorization == INTERNAL_TEST_AUTHORIZATION
    if not authorized:
        failures.append("operator_authorization_missing_or_invalid")
    profile_hash = canonical_json_sha256(
        {
            "feature_flags": dict(frozen_feature_flags),
            "thresholds": dict(frozen_thresholds),
            "config_hashes": dict(config_hashes),
            "checkpoint_hashes": dict(checkpoint_hashes),
            "validation_ablation_hash": validation_ablation_hash,
            "model_revisions": dict(model_revisions),
        }
    )
    return InternalTestFreezeGateReport(
        schema_version=INTERNAL_TEST_GATE_SCHEMA_VERSION,
        authorized=authorized,
        ready=not failures,
        failures=tuple(failures),
        frozen_profile_hash=profile_hash,
        internal_test_accessed=False,
    )


def require_internal_test_gate_ready(report: InternalTestFreezeGateReport) -> None:
    if not report.ready:
        raise InternalTestGateError("; ".join(report.failures))


__all__ = [
    "INTERNAL_TEST_AUTHORIZATION",
    "INTERNAL_TEST_GATE_SCHEMA_VERSION",
    "InternalTestFreezeGateReport",
    "InternalTestGateError",
    "evaluate_internal_test_freeze_gate",
    "require_internal_test_gate_ready",
]
