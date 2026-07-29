"""Six-file artifact contract and read-only validators for Phase 2."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...lattice.models import EXPERT_XLMR_MRC
from ...mention_factory.mrc import MRC_QUERY_VERSION, TYPE_QUERY_ORDER
from ...resolution.learned_v2 import (
    RESOLVER_V2_VERSION,
    SUPPORTED_BOUNDARY_ACTIONS,
    TYPE_DROP,
    TYPE_ORDER,
)
from .common import (
    Phase2ReadinessError,
    canonical_json_sha256,
    is_immutable_revision,
    read_json,
    sha256_file,
    validate_hex_digest,
    write_json,
)

ARTIFACT_SCHEMA_VERSION = "phase2-artifact-v1"
CHECKPOINT_SCHEMA_VERSION = "phase2-checkpoint-v1"
STATUS_FULLY_TRAINED = "FULLY_TRAINED"
STATUS_SMOKE_EXECUTED = "SMOKE_EXECUTED"
STATUS_ARTIFACT_VALIDATED = "ARTIFACT_VALIDATED"
MODE_SMOKE = "smoke"
MODE_FULL = "full"

REQUIRED_ARTIFACT_FILES: tuple[str, ...] = (
    "checkpoints/best.pt",
    "checkpoints/latest.pt",
    "logs/training_history.jsonl",
    "resolved_config.json",
    "validation_metrics.json",
    "training_manifest.json",
)

MODEL_ARTIFACT_KINDS: dict[str, str] = {
    "e5": EXPERT_XLMR_MRC,
    "l4": RESOLVER_V2_VERSION,
}


class ArtifactValidationError(Phase2ReadinessError):
    """Raised only by fail-fast wrappers; validators otherwise report failures."""


@dataclass(frozen=True, slots=True)
class Phase2CheckpointHashSet:
    best: str
    latest: str

    def as_dict(self) -> dict[str, str]:
        return {"best": self.best, "latest": self.latest}


@dataclass(frozen=True, slots=True)
class Phase2TrainingManifest:
    stage_id: str
    expert_id: str
    mode: str
    status: str
    run_completed: bool
    interrupted_reason: str
    safe_to_resume: bool
    repository_commit: str
    corpus_hashes: Mapping[str, str]
    data_hashes: Mapping[str, str]
    config_sha256: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    query_revision: str
    query_hash: str
    seed: int
    completed_epochs: int
    optimizer_steps: int
    effective_batch_size: int
    parameter_count: int
    checkpoint_hashes: Mapping[str, str]
    best_metric: float
    best_metric_name: str
    best_criterion: str
    train_split_id: str
    validation_split_id: str
    internal_test_accessed: bool
    initialization_source: str
    label_space: Sequence[str]
    boundary_action_space: Sequence[str] = field(default_factory=tuple)
    type_action_space: Sequence[str] = field(default_factory=tuple)
    threshold_config: Mapping[str, float] = field(default_factory=dict)
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    best_latest_identical_allowed: bool = False
    best_latest_identical_reason: str = ""
    # Optional per-expert accounting (Audit 0039). An expert that records real
    # optimizer-step, gradient-accumulation, precision and weight-format
    # accounting here cannot merely relabel a batch size. Experts that do not use
    # it leave it empty, so E5 and L4 manifests are unaffected.
    training_accounting: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "stage_id": self.stage_id,
            "expert_id": self.expert_id,
            "mode": self.mode,
            "status": self.status,
            "repository_commit": self.repository_commit,
            "config_sha256": self.config_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "train_split_id": self.train_split_id,
            "validation_split_id": self.validation_split_id,
            "initialization_source": self.initialization_source,
        }
        missing = tuple(sorted(key for key, value in required.items() if not value))
        if missing:
            raise ArtifactValidationError(
                "Phase-2 training manifest missing fields: " + ",".join(missing)
            )
        if self.mode not in {MODE_SMOKE, MODE_FULL}:
            raise ArtifactValidationError("manifest mode must be smoke or full")
        if self.internal_test_accessed:
            raise ArtifactValidationError("Phase-2 training manifest accessed internal_test")
        if self.completed_epochs < 0 or self.optimizer_steps < 0:
            raise ArtifactValidationError("epochs and optimizer steps must be non-negative")
        if self.effective_batch_size < 0 or self.parameter_count < 0:
            raise ArtifactValidationError("batch size and parameter count must be non-negative")
        validate_hex_digest(self.config_sha256, field_name="config_sha256")
        for key, digest in self.checkpoint_hashes.items():
            validate_hex_digest(str(digest), field_name=f"checkpoint_hashes.{key}")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["corpus_hashes"] = dict(self.corpus_hashes)
        payload["data_hashes"] = dict(self.data_hashes)
        payload["checkpoint_hashes"] = dict(self.checkpoint_hashes)
        payload["label_space"] = list(self.label_space)
        payload["boundary_action_space"] = list(self.boundary_action_space)
        payload["type_action_space"] = list(self.type_action_space)
        payload["threshold_config"] = dict(self.threshold_config)
        payload["training_accounting"] = dict(self.training_accounting)
        return payload

    def write(self, path: str | Path) -> None:
        self.validate()
        write_json(path, self.as_dict())


@dataclass(frozen=True, slots=True)
class ArtifactValidationReport:
    artifact_dir: str
    expected_expert_id: str
    expected_mode: str
    ok: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    checkpoint_hashes: Mapping[str, str]
    manifest_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "expected_expert_id": self.expected_expert_id,
            "expected_mode": self.expected_mode,
            "ok": self.ok,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "checkpoint_hashes": dict(self.checkpoint_hashes),
            "manifest_sha256": self.manifest_sha256,
        }


def _load_manifest(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, Phase2ReadinessError) as exc:
        failures.append(f"manifest_unreadable:{exc}")
        return {}


def _read_json_or_report(path: Path, label: str, failures: list[str]) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, Phase2ReadinessError) as exc:
        failures.append(f"{label}_unreadable:{exc}")
        return {}


def _history_rows(path: Path, failures: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    failures.append(f"history_row_not_object:{line_number}")
                    continue
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"history_unreadable:{exc}")
    return rows


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any] | None:
    """Read a Phase-2 checkpoint payload.

    Colab notebooks save real checkpoints with ``torch.save``. Unit tests and
    read-only validators may also use canonical JSON bytes with the same schema,
    so the validator can run without Torch being installed.
    """
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping):
            return payload
    except Exception:  # pragma: no cover - fallback is covered without torch dependence
        pass
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def write_checkpoint_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a schema-valid checkpoint payload.

    Operators in Colab should use the same payload shape with ``torch.save``.
    This helper prefers Torch when present and otherwise writes JSON bytes; both
    forms are accepted by the read-only validator.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(dict(payload), target)
        return
    except Exception:  # pragma: no cover - JSON fallback depends on environment
        target.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )


def checkpoint_payload(
    *,
    expert_id: str,
    mode: str,
    config_sha256: str,
    model_revision: str,
    parameter_count: int,
    label_space: Sequence[str],
    tokenizer_revision: str = "",
    query_revision: str = "",
    boundary_action_space: Sequence[str] = (),
    type_action_space: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "expert_id": expert_id,
        "mode": mode,
        "config_sha256": config_sha256,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "query_revision": query_revision,
        "parameter_count": parameter_count,
        "label_space": list(label_space),
        "boundary_action_space": list(boundary_action_space),
        "type_action_space": list(type_action_space),
        "internal_test_accessed": False,
        "model_state": {},
    }


def _expected_label_space(expert_id: str) -> tuple[str, ...]:
    if expert_id == EXPERT_XLMR_MRC:
        return TYPE_QUERY_ORDER
    if expert_id == RESOLVER_V2_VERSION:
        return TYPE_ORDER + (TYPE_DROP,)
    return ()


def _check_revision_fields(
    manifest: Mapping[str, Any],
    expected_expert_id: str,
    failures: list[str],
) -> None:
    model_revision = str(manifest.get("model_revision", ""))
    tokenizer_revision = str(manifest.get("tokenizer_revision", ""))
    if expected_expert_id == EXPERT_XLMR_MRC:
        if not is_immutable_revision(model_revision):
            failures.append("model_revision_not_immutable")
        if not is_immutable_revision(tokenizer_revision):
            failures.append("tokenizer_revision_not_immutable")
    elif expected_expert_id == RESOLVER_V2_VERSION and not is_immutable_revision(
        model_revision,
        allow_local_architecture=True,
    ):
        failures.append("learned_l4_model_revision_not_stable")


def _check_history(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    failures: list[str],
) -> None:
    completed_epochs = int(manifest.get("completed_epochs", -1))
    if completed_epochs > 0 and len(rows) < completed_epochs:
        failures.append("training_history_shorter_than_completed_epochs")
    required = {"epoch", "mode", "train_loss", "validation_exact_f1", "optimizer_steps"}
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            failures.append(f"training_history_schema_missing:{index}:{','.join(missing)}")


def _check_checkpoint_payload(
    path: Path,
    manifest: Mapping[str, Any],
    expected_expert_id: str,
    expected_mode: str,
    failures: list[str],
) -> None:
    payload = _load_checkpoint_payload(path)
    if payload is None:
        failures.append(f"checkpoint_payload_unreadable:{path.name}")
        return
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "expert_id": expected_expert_id,
        "mode": expected_mode,
        "config_sha256": manifest.get("config_sha256"),
        "model_revision": manifest.get("model_revision"),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"checkpoint_payload_mismatch:{path.name}:{key}")
    if bool(payload.get("internal_test_accessed", False)):
        failures.append(f"checkpoint_payload_internal_test_accessed:{path.name}")
    if int(payload.get("parameter_count", -1)) != int(manifest.get("parameter_count", -2)):
        failures.append(f"checkpoint_payload_parameter_count_mismatch:{path.name}")
    label_space = tuple(str(v) for v in payload.get("label_space", ()))
    expected_label_space = _expected_label_space(expected_expert_id)
    if expected_label_space and label_space != expected_label_space:
        failures.append(f"checkpoint_payload_label_space_mismatch:{path.name}")
    if expected_expert_id == RESOLVER_V2_VERSION:
        actions = tuple(str(v) for v in payload.get("boundary_action_space", ()))
        types = tuple(str(v) for v in payload.get("type_action_space", ()))
        if actions != SUPPORTED_BOUNDARY_ACTIONS:
            failures.append(f"checkpoint_payload_boundary_actions_mismatch:{path.name}")
        if types != TYPE_ORDER + (TYPE_DROP,):
            failures.append(f"checkpoint_payload_type_actions_mismatch:{path.name}")


def _check_no_base_cache(root: Path, failures: list[str]) -> None:
    forbidden_dirs = {"huggingface", "model_cache", "transformers", "base_model"}
    forbidden_suffixes = {".safetensors", ".bin", ".ckpt", ".pth"}
    for child in root.rglob("*"):
        parts = set(child.relative_to(root).parts)
        if child.is_dir() and child.name in forbidden_dirs:
            failures.append(f"base_model_cache_embedded:{child.relative_to(root)}")
        if child.is_file() and child.suffix in forbidden_suffixes:
            failures.append(f"base_model_weight_embedded:{child.relative_to(root)}")
        if "checkpoints" not in parts and child.is_file() and child.suffix == ".pt":
            failures.append(f"checkpoint_outside_checkpoint_dir:{child.relative_to(root)}")


def validate_phase2_artifact(
    artifact_dir: str | Path,
    *,
    expected_expert_id: str,
    expected_mode: str = MODE_FULL,
    require_completed: bool = True,
) -> ArtifactValidationReport:
    root = Path(artifact_dir)
    failures: list[str] = []
    warnings: list[str] = []
    if expected_mode not in {MODE_SMOKE, MODE_FULL}:
        failures.append("expected_mode_must_be_smoke_or_full")
    for rel in REQUIRED_ARTIFACT_FILES:
        if not (root / rel).is_file():
            failures.append(f"missing_required_file:{rel}")
    manifest_path = root / "training_manifest.json"
    manifest = _load_manifest(manifest_path, failures) if manifest_path.exists() else {}
    metrics = _read_json_or_report(root / "validation_metrics.json", "metrics", failures)
    resolved_config = _read_json_or_report(root / "resolved_config.json", "config", failures)
    history = _history_rows(root / "logs" / "training_history.jsonl", failures)

    checkpoint_hashes: dict[str, str] = {}
    for name in ("best", "latest"):
        checkpoint = root / "checkpoints" / f"{name}.pt"
        if checkpoint.exists():
            checkpoint_hashes[name] = sha256_file(checkpoint)

    if manifest:
        if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            failures.append("artifact_schema_version_mismatch")
        if manifest.get("expert_id") != expected_expert_id:
            failures.append("expert_id_mismatch")
        if manifest.get("mode") != expected_mode:
            failures.append("mode_mismatch")
        if require_completed and not bool(manifest.get("run_completed", False)):
            failures.append("run_not_completed")
        if require_completed and expected_mode == MODE_FULL:
            if str(manifest.get("status", "")) != STATUS_FULLY_TRAINED:
                failures.append("full_artifact_status_not_fully_trained")
            if int(manifest.get("completed_epochs", 0)) <= 0:
                failures.append("full_artifact_has_no_completed_epochs")
            if int(manifest.get("optimizer_steps", 0)) <= 0:
                failures.append("full_artifact_has_no_optimizer_steps")
        if bool(manifest.get("internal_test_accessed", False)):
            failures.append("manifest_internal_test_accessed")
        if str(manifest.get("train_split_id")) == str(manifest.get("validation_split_id")):
            failures.append("train_validation_split_identity_equal")
        if expected_mode == MODE_FULL and "smoke" in str(
            manifest.get("initialization_source", "")
        ).lower():
            failures.append("full_artifact_initialized_from_smoke_checkpoint")
        if int(manifest.get("parameter_count", -1)) <= 0:
            failures.append("parameter_count_not_positive")
        try:
            validate_hex_digest(str(manifest.get("config_sha256", "")), field_name="config_sha256")
        except Phase2ReadinessError:
            failures.append("config_sha256_not_64_hex")
        _check_revision_fields(manifest, expected_expert_id, failures)
        for key, digest in checkpoint_hashes.items():
            recorded = str(dict(manifest.get("checkpoint_hashes", {})).get(key, ""))
            if recorded != digest:
                failures.append(f"checkpoint_hash_mismatch:{key}")
        if metrics:
            metric_name = str(manifest.get("best_metric_name", ""))
            if metric_name and metric_name in metrics:
                if float(metrics[metric_name]) != float(manifest.get("best_metric", "nan")):
                    failures.append("best_metric_manifest_metrics_mismatch")
            if bool(metrics.get("internal_test_accessed", False)):
                failures.append("metrics_internal_test_accessed")
        if resolved_config:
            config_hash = canonical_json_sha256(resolved_config)
            if config_hash != manifest.get("config_sha256"):
                failures.append("resolved_config_hash_mismatch")
            if bool(resolved_config.get("internal_test_accessed", False)):
                failures.append("resolved_config_internal_test_accessed")
        _check_history(history, manifest, failures)
        for name in ("best", "latest"):
            checkpoint = root / "checkpoints" / f"{name}.pt"
            if checkpoint.exists():
                _check_checkpoint_payload(
                    checkpoint,
                    manifest,
                    expected_expert_id,
                    expected_mode,
                    failures,
                )
        if (
            "best" in checkpoint_hashes
            and "latest" in checkpoint_hashes
            and checkpoint_hashes["best"] == checkpoint_hashes["latest"]
            and not bool(manifest.get("best_latest_identical_allowed", False))
        ):
            failures.append("best_latest_checkpoints_byte_identical_without_justification")
        if (
            bool(manifest.get("best_latest_identical_allowed", False))
            and not str(manifest.get("best_latest_identical_reason", "")).strip()
        ):
            failures.append("best_latest_identical_reason_missing")
        for digest_name in ("corpus_hashes", "data_hashes"):
            digest_map = manifest.get(digest_name, {})
            if not isinstance(digest_map, Mapping) or not digest_map:
                failures.append(f"{digest_name}_missing")
            elif any(len(str(value)) != 64 for value in digest_map.values()):
                failures.append(f"{digest_name}_contains_non_sha256")
        if expected_expert_id == EXPERT_XLMR_MRC:
            if manifest.get("query_revision") != MRC_QUERY_VERSION:
                failures.append("query_revision_mismatch")
            if not str(manifest.get("query_hash", "")):
                failures.append("query_hash_missing")
    _check_no_base_cache(root, failures)

    manifest_sha256 = sha256_file(manifest_path) if manifest_path.exists() else ""
    return ArtifactValidationReport(
        artifact_dir=str(root),
        expected_expert_id=expected_expert_id,
        expected_mode=expected_mode,
        ok=not failures,
        failures=tuple(failures),
        warnings=tuple(warnings),
        checkpoint_hashes=checkpoint_hashes,
        manifest_sha256=manifest_sha256,
    )


def validate_e5_artifact(
    artifact_dir: str | Path,
    *,
    mode: str = MODE_FULL,
) -> ArtifactValidationReport:
    return validate_phase2_artifact(
        artifact_dir,
        expected_expert_id=EXPERT_XLMR_MRC,
        expected_mode=mode,
    )


def validate_l4_artifact(
    artifact_dir: str | Path,
    *,
    mode: str = MODE_FULL,
) -> ArtifactValidationReport:
    return validate_phase2_artifact(
        artifact_dir,
        expected_expert_id=RESOLVER_V2_VERSION,
        expected_mode=mode,
    )


def require_valid_artifact(report: ArtifactValidationReport) -> None:
    if not report.ok:
        raise ArtifactValidationError("; ".join(report.failures))


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "MODE_FULL",
    "MODE_SMOKE",
    "MODEL_ARTIFACT_KINDS",
    "REQUIRED_ARTIFACT_FILES",
    "STATUS_ARTIFACT_VALIDATED",
    "STATUS_FULLY_TRAINED",
    "STATUS_SMOKE_EXECUTED",
    "ArtifactValidationError",
    "ArtifactValidationReport",
    "Phase2CheckpointHashSet",
    "Phase2TrainingManifest",
    "checkpoint_payload",
    "require_valid_artifact",
    "validate_e5_artifact",
    "validate_l4_artifact",
    "validate_phase2_artifact",
    "write_checkpoint_payload",
]
