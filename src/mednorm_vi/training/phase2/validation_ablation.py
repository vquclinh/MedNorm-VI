"""Governed-validation-only ablation helpers for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...evaluation.exact_mention import (
    BOTH_BOUNDARY,
    LEFT_BOUNDARY,
    RIGHT_BOUNDARY,
    WRONG_TYPE,
    Mention,
    evaluate_examples,
)
from ...evaluation.l3_l4_ablation_v2 import (
    ARM_ALL_L3,
    ARM_E3_E5,
    ARM_E3_E5_E6,
    ARM_E3_E6,
    ARM_E3_ONLY,
    ARM_L4_V1,
    ARM_L4_V2,
    STATUS_EVALUABLE,
    STATUS_UNAVAILABLE_UNTRAINED,
    AblationArmStatus,
)
from .common import Phase2ReadinessError, canonical_json_sha256, write_json

VALIDATION_ABLATION_SCHEMA_VERSION = "phase2-validation-ablation-v1"
VALIDATION_ABLATION_AUTHORIZATION = "I_AUTHORIZE_PHASE2_VALIDATION_ABLATION"
SUPPORTED_VALIDATION_ARMS: tuple[str, ...] = (
    ARM_E3_ONLY,
    ARM_E3_E5,
    ARM_E3_E6,
    ARM_E3_E5_E6,
    ARM_ALL_L3,
    ARM_L4_V1,
    ARM_L4_V2,
)


class ValidationAblationError(Phase2ReadinessError):
    """Raised when validation-only ablation would violate the protocol."""


@dataclass(frozen=True, slots=True)
class ValidationAblationArmResult:
    arm: str
    status: str
    reason: str
    report: Mapping[str, Any]
    proposal_count: int
    final_count: int
    config_hashes: Mapping[str, str]
    checkpoint_hashes: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        if self.status != STATUS_EVALUABLE:
            return {
                "arm": self.arm,
                "status": self.status,
                "reason": self.reason,
                "config_hashes": dict(self.config_hashes),
                "checkpoint_hashes": dict(self.checkpoint_hashes),
            }
        micro = dict(self.report["micro"])
        categories = dict(self.report["error_categories"])
        return {
            "arm": self.arm,
            "status": self.status,
            "reason": self.reason,
            "exact_precision": micro["precision"],
            "exact_recall": micro["recall"],
            "exact_f1": micro["f1"],
            "per_type": self.report["by_type"],
            "boundary_errors": (
                int(categories.get(LEFT_BOUNDARY, 0))
                + int(categories.get(RIGHT_BOUNDARY, 0))
                + int(categories.get(BOTH_BOUNDARY, 0))
            ),
            "wrong_type_errors": int(categories.get(WRONG_TYPE, 0)),
            "proposal_count": self.proposal_count,
            "final_count": self.final_count,
            "config_hashes": dict(self.config_hashes),
            "checkpoint_hashes": dict(self.checkpoint_hashes),
        }


@dataclass(frozen=True, slots=True)
class ValidationAblationReport:
    schema_version: str
    split: str
    results: tuple[ValidationAblationArmResult, ...]
    internal_test_accessed: bool
    config_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "results": [result.as_dict() for result in self.results],
            "internal_test_accessed": self.internal_test_accessed,
            "config_sha256": self.config_sha256,
        }


def _validate_examples_are_validation(examples: Sequence[Mapping[str, Any]]) -> None:
    for index, example in enumerate(examples, start=1):
        split = str(example.get("split", "validation"))
        if split == "internal_test":
            raise ValidationAblationError("validation ablation may not access internal_test")
        if split != "validation":
            raise ValidationAblationError(f"example {index} is not from validation split")


def _mentions_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mention, ...]:
    mentions: list[Mention] = []
    for row in rows:
        mentions.append(
            Mention(
                start=int(row["start"]),
                end=int(row["end"]),
                entity_type=str(row["entity_type"]),
                text=str(row["text"]),
                route=str(row.get("route", "")),
                section=str(row.get("section", "")),
                provenance=str(row.get("provenance", "unknown")),
            )
        )
    return tuple(mentions)


def run_validation_ablation(
    examples: Sequence[Mapping[str, Any]],
    predictions_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    arm_statuses: Sequence[AblationArmStatus],
    config_hashes_by_arm: Mapping[str, Mapping[str, str]],
    checkpoint_hashes_by_arm: Mapping[str, Mapping[str, str]],
) -> ValidationAblationReport:
    """Evaluate only governed-validation arms with validated checkpoints."""
    _validate_examples_are_validation(examples)
    status_by_arm = {status.arm: status for status in arm_statuses}
    results: list[ValidationAblationArmResult] = []
    for arm in SUPPORTED_VALIDATION_ARMS:
        status = status_by_arm.get(arm)
        if status is None:
            results.append(
                ValidationAblationArmResult(
                    arm=arm,
                    status=STATUS_UNAVAILABLE_UNTRAINED,
                    reason="arm not present in availability plan",
                    report={},
                    proposal_count=0,
                    final_count=0,
                    config_hashes=config_hashes_by_arm.get(arm, {}),
                    checkpoint_hashes=checkpoint_hashes_by_arm.get(arm, {}),
                )
            )
            continue
        if status.status != STATUS_EVALUABLE:
            results.append(
                ValidationAblationArmResult(
                    arm=arm,
                    status=status.status,
                    reason=status.reason,
                    report={},
                    proposal_count=0,
                    final_count=0,
                    config_hashes=config_hashes_by_arm.get(arm, {}),
                    checkpoint_hashes=checkpoint_hashes_by_arm.get(arm, {}),
                )
            )
            continue
        predictions = {
            str(example_id): _mentions_from_rows(rows)
            for example_id, rows in predictions_by_arm.get(arm, {}).items()
        }
        report = evaluate_examples(
            examples,
            predictions=predictions,
            label=f"{arm}_governed_validation",
            config={
                "arm": arm,
                "config_hashes": dict(config_hashes_by_arm.get(arm, {})),
                "checkpoint_hashes": dict(checkpoint_hashes_by_arm.get(arm, {})),
            },
        )
        final_count = int(report["predicted_mentions"])
        proposal_count = sum(len(rows) for rows in predictions_by_arm.get(arm, {}).values())
        results.append(
            ValidationAblationArmResult(
                arm=arm,
                status=STATUS_EVALUABLE,
                reason=status.reason,
                report=report,
                proposal_count=proposal_count,
                final_count=final_count,
                config_hashes=config_hashes_by_arm.get(arm, {}),
                checkpoint_hashes=checkpoint_hashes_by_arm.get(arm, {}),
            )
        )
    payload_hash = canonical_json_sha256([result.as_dict() for result in results])
    return ValidationAblationReport(
        schema_version=VALIDATION_ABLATION_SCHEMA_VERSION,
        split="validation",
        results=tuple(results),
        internal_test_accessed=False,
        config_sha256=payload_hash,
    )


def write_validation_ablation_report(
    path: str | Path,
    report: ValidationAblationReport,
) -> str:
    write_json(path, report.as_dict())
    return canonical_json_sha256(report.as_dict())


__all__ = [
    "SUPPORTED_VALIDATION_ARMS",
    "VALIDATION_ABLATION_AUTHORIZATION",
    "VALIDATION_ABLATION_SCHEMA_VERSION",
    "ValidationAblationArmResult",
    "ValidationAblationError",
    "ValidationAblationReport",
    "run_validation_ablation",
    "write_validation_ablation_report",
]
