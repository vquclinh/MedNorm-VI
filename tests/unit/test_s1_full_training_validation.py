"""Read-only full-training artifact validation (Audit 0031).

Artifacts are synthesized in tmp_path: no Drive access, no Torch, no GPU, no
training, and no real checkpoint tensors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mednorm_vi.training.s1_full_training import (
    BEST_METRIC_KEY,
    BEST_METRIC_MODE,
    CHECKPOINT_REQUIRED_KEYS,
    COLAB_BEST_CHECKPOINT,
    FULL_TRAINING_MODE,
    LOCAL_BEST_CHECKPOINT_RELATIVE,
    S1_BEST_CHECKPOINT_ENV,
    SMOKE_MODE,
    full_training_output_paths,
    resolve_s1_best_checkpoint,
    validate_best_checkpoint_only,
    validate_full_training_artifact,
)
from mednorm_vi.training.s1_mention_smoke import ENTITY_TYPE_ORDER

PINNED = "f89e80b461e86f9cfc1c84019bd819830c24b6c5"
# The digests reported for the completed run; test data only, never a source default.
BEST_BYTES = b"best-checkpoint-payload"
LATEST_BYTES = b"latest-checkpoint-payload"
BEST_SHA = hashlib.sha256(BEST_BYTES).hexdigest()
LATEST_SHA = hashlib.sha256(LATEST_BYTES).hexdigest()
BEST_METRIC = 0.7194053623573136
EPOCHS = 4
STEPS = 2976


def _resolved_config() -> dict:
    return {"pinned_revision": PINNED, "num_epochs": EPOCHS, "seed": 20260723,
            "best_metric_key": BEST_METRIC_KEY, "best_metric_mode": BEST_METRIC_MODE}


def _config_sha(resolved: dict) -> str:
    return hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest(resolved: dict) -> dict:
    return {
        "manifest_version": 1, "stage_id": "S1", "status": FULL_TRAINING_MODE,
        "smoke_only_not_full_training": False, "architecture_spec_version": "1.1",
        "repository": {"resolved_commit": "c" * 40},
        "corpus": {"corpus_manifest_sha256": "a" * 64},
        "model": {
            "registry_model_id": "vihealthbert_span_type",
            "hf_model_id": "demdecuong/vihealthbert-base-word",
            "requested_revision": "main", "pinned_model_revision": PINNED,
            "tokenizer_revision": PINNED, "initialize_from": "pretrained_base",
            "initialized_from_smoke_checkpoint": False,
        },
        "hyperparameters": resolved,
        "effective_batch_size": 32,
        "schedule": {"total_optimizer_steps": STEPS, "steps_per_epoch": 744},
        "completed_epochs": EPOCHS, "completed_optimizer_steps": STEPS,
        "validation_metrics": {BEST_METRIC_KEY: BEST_METRIC, "token_micro_f1": 0.81},
        "best_checkpoint_criterion": {"key": BEST_METRIC_KEY, "mode": BEST_METRIC_MODE},
        "artifacts": {
            "checkpoint_sha256": {"best_checkpoint": BEST_SHA,
                                  "latest_checkpoint": LATEST_SHA},
            "smoke_artifact_dir": "/drive/artifacts/s1_mention_first_run_smoke_v5",
        },
        "config_sha256": _config_sha(resolved),
        "run_completed": True, "interrupted_reason": "", "safe_to_resume": True,
    }


def _artifact(tmp_path: Path, *, manifest=None, epochs=EPOCHS) -> Path:
    base = tmp_path / "s1_mention_full_training_v1"
    resolved = _resolved_config()
    paths = full_training_output_paths(base)
    for key in paths.values():
        Path(key).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["best_checkpoint"]).write_bytes(BEST_BYTES)
    Path(paths["latest_checkpoint"]).write_bytes(LATEST_BYTES)
    Path(paths["resolved_config"]).write_text(json.dumps(resolved), encoding="utf-8")
    Path(paths["validation_metrics"]).write_text(
        json.dumps({BEST_METRIC_KEY: BEST_METRIC}), encoding="utf-8")
    Path(paths["training_history"]).write_text(
        "\n".join(json.dumps({"event": "validation", "epoch": e + 1,
                              BEST_METRIC_KEY: BEST_METRIC}) for e in range(epochs)),
        encoding="utf-8")
    Path(paths["training_manifest"]).write_text(
        json.dumps(manifest if manifest is not None else _manifest(resolved)),
        encoding="utf-8")
    return base


def _payload(mode=FULL_TRAINING_MODE) -> dict:
    resolved = _resolved_config()
    return {key: True for key in CHECKPOINT_REQUIRED_KEYS} | {
        "mode": mode, "epoch": EPOCHS, "global_step": STEPS, "best_metric": BEST_METRIC,
        "entity_type_order": list(ENTITY_TYPE_ORDER), "seed": 20260723,
        "pinned_model_revision": PINNED, "config_sha256": _config_sha(resolved),
    }


def _validate(base: Path, **kwargs):
    options = {
        "expected_checkpoint_sha256": {"best_checkpoint": BEST_SHA,
                                       "latest_checkpoint": LATEST_SHA},
        "expected_pinned_revision": PINNED,
    }
    options.update(kwargs)
    return validate_full_training_artifact(base, **options)


# --- happy path ---------------------------------------------------------------

def test_a_complete_run_validates(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path))
    assert outcome.validated is True
    assert outcome.failures == ()
    assert outcome.best_metric == pytest.approx(BEST_METRIC)
    assert outcome.completed_epochs == EPOCHS
    assert outcome.completed_optimizer_steps == STEPS
    assert outcome.pinned_model_revision == PINNED
    assert outcome.checkpoint_sha256 == {"best_checkpoint": BEST_SHA,
                                         "latest_checkpoint": LATEST_SHA}


def test_checkpoint_schema_is_verified_when_payloads_are_supplied(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path), checkpoint_payloads={
        "best_checkpoint": _payload(), "latest_checkpoint": _payload()})
    assert outcome.validated is True
    assert outcome.diagnostics["checkpoint_schema_checked"] is True


def test_schema_check_is_reported_as_not_performed_without_payloads(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path))
    assert outcome.diagnostics["checkpoint_schema_checked"] is False


# --- required files -----------------------------------------------------------

@pytest.mark.parametrize("artifact", [
    "best_checkpoint", "latest_checkpoint", "training_history",
    "resolved_config", "validation_metrics", "training_manifest",
])
def test_every_required_artifact_is_required(tmp_path: Path, artifact) -> None:
    base = _artifact(tmp_path)
    Path(full_training_output_paths(base)[artifact]).unlink()
    outcome = _validate(base)
    assert outcome.validated is False
    assert any(f"required artifact missing: {artifact}" in f for f in outcome.failures)
    assert outcome.diagnostics["files_present"][artifact] is False


def test_a_missing_manifest_stops_immediately(tmp_path: Path) -> None:
    base = _artifact(tmp_path)
    Path(full_training_output_paths(base)["training_manifest"]).unlink()
    outcome = _validate(base)
    assert outcome.validated is False
    assert outcome.checkpoint_sha256 == {}


# --- three-way hash agreement -------------------------------------------------

def test_checkpoint_hash_must_match_the_manifest(tmp_path: Path) -> None:
    resolved = _resolved_config()
    manifest = _manifest(resolved)
    manifest["artifacts"]["checkpoint_sha256"]["best_checkpoint"] = "0" * 64
    outcome = _validate(_artifact(tmp_path, manifest=manifest))
    assert outcome.validated is False
    assert any("best_checkpoint SHA-256 does not match the manifest" in f
               for f in outcome.failures)


def test_checkpoint_hash_must_match_the_operator_supplied_hash(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path), expected_checkpoint_sha256={
        "best_checkpoint": "b" * 64, "latest_checkpoint": LATEST_SHA})
    assert outcome.validated is False
    assert any("operator-supplied hash" in f for f in outcome.failures)


@pytest.mark.parametrize("supplied", ["", "   ", "not-a-hash", "0" * 63])
def test_a_missing_or_malformed_expected_hash_blocks(tmp_path: Path, supplied) -> None:
    outcome = _validate(_artifact(tmp_path), expected_checkpoint_sha256={
        "best_checkpoint": supplied, "latest_checkpoint": LATEST_SHA})
    assert outcome.validated is False
    assert any("was not supplied" in f for f in outcome.failures)
    # the recomputed digest is reported so the operator can confirm it
    assert any(BEST_SHA in f for f in outcome.failures)


def test_no_run_specific_digest_is_hardcoded_in_source() -> None:
    import re
    source = (Path(__file__).resolve().parents[2] / "src" / "mednorm_vi" / "training"
              / "s1_full_training.py").read_text(encoding="utf-8")
    assert not re.search(r"[0-9a-f]{64}", source)


def test_identical_best_and_latest_checkpoints_are_rejected(tmp_path: Path) -> None:
    base = _artifact(tmp_path)
    paths = full_training_output_paths(base)
    Path(paths["latest_checkpoint"]).write_bytes(BEST_BYTES)
    manifest = _manifest(_resolved_config())
    manifest["artifacts"]["checkpoint_sha256"]["latest_checkpoint"] = BEST_SHA
    Path(paths["training_manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    outcome = _validate(base, expected_checkpoint_sha256={
        "best_checkpoint": BEST_SHA, "latest_checkpoint": BEST_SHA})
    assert outcome.validated is False
    assert any("byte-identical" in f for f in outcome.failures)


# --- run state, accounting, revision ------------------------------------------

@pytest.mark.parametrize("path,value,fragment", [
    (("status",), SMOKE_MODE, "status is FULL_TRAINING"),
    (("smoke_only_not_full_training",), True, "smoke_only_not_full_training is false"),
    (("run_completed",), False, "run_completed is true"),
    (("interrupted_reason",), "CUDA out of memory", "interrupted_reason is empty"),
    (("safe_to_resume",), False, "safe_to_resume is true"),
    (("completed_epochs",), 2, "completed_epochs does not match the plan"),
    (("completed_optimizer_steps",), 1488, "completed_optimizer_steps does not match"),
])
def test_run_state_and_accounting_are_enforced(tmp_path, path, value, fragment) -> None:
    manifest = _manifest(_resolved_config())
    node = manifest
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    outcome = _validate(_artifact(tmp_path, manifest=manifest))
    assert outcome.validated is False
    assert any(fragment in f for f in outcome.failures), outcome.failures


@pytest.mark.parametrize("revision,fragment", [
    ("main", "not immutable"), ("", "not immutable"), ("a" * 40, "expected revision"),
])
def test_pinned_revision_is_enforced(tmp_path, revision, fragment) -> None:
    manifest = _manifest(_resolved_config())
    manifest["model"]["pinned_model_revision"] = revision
    outcome = _validate(_artifact(tmp_path, manifest=manifest))
    assert outcome.validated is False
    assert any(fragment in f for f in outcome.failures)


# --- the smoke checkpoint was not the initializer ------------------------------

def test_smoke_checkpoint_initialization_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(_resolved_config())
    manifest["model"]["initialize_from"] = "smoke_checkpoint"
    manifest["model"]["initialized_from_smoke_checkpoint"] = True
    outcome = _validate(_artifact(tmp_path, manifest=manifest))
    assert outcome.validated is False
    assert any("did not initialize from the pretrained base" in f for f in outcome.failures)
    assert any("was NOT the initializer" in f for f in outcome.failures)


def test_a_smoke_mode_checkpoint_payload_is_rejected(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path), checkpoint_payloads={
        "best_checkpoint": _payload(mode=SMOKE_MODE)})
    assert outcome.validated is False
    assert any("carries the SMOKE_ONLY mode" in f for f in outcome.failures)


@pytest.mark.parametrize("missing", CHECKPOINT_REQUIRED_KEYS)
def test_incomplete_resume_metadata_is_rejected(tmp_path: Path, missing) -> None:
    payload = _payload()
    payload.pop(missing)
    outcome = _validate(_artifact(tmp_path), checkpoint_payloads={"best_checkpoint": payload})
    assert outcome.validated is False
    assert any(f"missing the resume field {missing!r}" in f for f in outcome.failures)


def test_a_checkpoint_from_another_revision_or_label_space_is_rejected(tmp_path: Path) -> None:
    drifted = _payload() | {"pinned_model_revision": "d" * 40}
    assert any("different pinned revision" in f
               for f in _validate(_artifact(tmp_path),
                                  checkpoint_payloads={"best_checkpoint": drifted}).failures)
    relabelled = _payload() | {"entity_type_order": ["DIAGNOSIS"]}
    assert any("different label space" in f
               for f in _validate(_artifact(tmp_path),
                                  checkpoint_payloads={"best_checkpoint": relabelled}).failures)


# --- config hash, metric, history ---------------------------------------------

def test_resolved_config_must_hash_to_the_manifest_config_sha(tmp_path: Path) -> None:
    base = _artifact(tmp_path)
    Path(full_training_output_paths(base)["resolved_config"]).write_text(
        json.dumps(_resolved_config() | {"num_epochs": 99}), encoding="utf-8")
    outcome = _validate(base)
    assert outcome.validated is False
    assert any("does not hash to the manifest config_sha256" in f for f in outcome.failures)


def test_validation_metrics_file_must_agree_with_the_manifest(tmp_path: Path) -> None:
    base = _artifact(tmp_path)
    Path(full_training_output_paths(base)["validation_metrics"]).write_text(
        json.dumps({BEST_METRIC_KEY: 0.1}), encoding="utf-8")
    outcome = _validate(base)
    assert outcome.validated is False
    assert any("disagrees with the manifest on the best metric" in f for f in outcome.failures)


@pytest.mark.parametrize("metric", [-0.1, 1.5, "high", None])
def test_an_implausible_best_metric_is_rejected(tmp_path: Path, metric) -> None:
    manifest = _manifest(_resolved_config())
    manifest["validation_metrics"][BEST_METRIC_KEY] = metric
    outcome = _validate(_artifact(tmp_path, manifest=manifest))
    assert outcome.validated is False
    assert any("not a value in [0, 1]" in f for f in outcome.failures)


def test_history_must_have_one_validation_record_per_completed_epoch(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path, epochs=2))
    assert outcome.validated is False
    assert any("validation record(s) but the manifest reports" in f for f in outcome.failures)


def test_base_model_cache_inside_the_artifact_is_rejected(tmp_path: Path) -> None:
    base = _artifact(tmp_path)
    (base / "hf_cache").mkdir()
    (base / "hf_cache" / "model.safetensors").write_bytes(b"weights")
    outcome = _validate(base)
    assert outcome.validated is False
    assert any("base-model cache files" in f for f in outcome.failures)


def test_every_failure_is_reported_not_just_the_first(tmp_path: Path) -> None:
    manifest = _manifest(_resolved_config())
    manifest["run_completed"] = False
    manifest["completed_epochs"] = 1
    manifest["model"]["pinned_model_revision"] = "main"
    outcome = _validate(_artifact(tmp_path, manifest=manifest))
    assert len(outcome.failures) >= 4
    assert outcome.as_dict()["failed_condition_count"] == len(outcome.failures)


# --- checkpoint location contract (Audit 0032) --------------------------------

def test_env_override_wins_over_everything(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "best.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(BEST_BYTES)
    location = resolve_s1_best_checkpoint(
        repository_root=tmp_path, environ={S1_BEST_CHECKPOINT_ENV: str(target)},
        in_colab=True)
    assert location.environment == "env_override"
    assert location.path == target and location.exists is True


def test_colab_uses_the_drive_artifact_path() -> None:
    location = resolve_s1_best_checkpoint(environ={}, in_colab=True)
    assert location.environment == "colab"
    assert location.path == COLAB_BEST_CHECKPOINT
    assert str(location.path).startswith("/content/drive/")


def test_local_uses_the_ignored_repository_checkpoint_directory(tmp_path: Path) -> None:
    location = resolve_s1_best_checkpoint(repository_root=tmp_path, environ={}, in_colab=False)
    assert location.environment == "local"
    assert location.path == tmp_path / LOCAL_BEST_CHECKPOINT_RELATIVE
    assert location.path.parts[-3:] == ("checkpoint", "s1_mention_full_training_v1", "best.pt")
    assert location.exists is False


def test_an_empty_env_override_is_ignored(tmp_path: Path) -> None:
    location = resolve_s1_best_checkpoint(
        repository_root=tmp_path, environ={S1_BEST_CHECKPOINT_ENV: "   "}, in_colab=False)
    assert location.environment == "local"


def test_require_names_the_missing_path_and_how_to_fix_it(tmp_path: Path) -> None:
    location = resolve_s1_best_checkpoint(repository_root=tmp_path, environ={}, in_colab=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        location.require()
    message = str(excinfo.value)
    assert str(location.path) in message
    assert S1_BEST_CHECKPOINT_ENV in message


# --- best-checkpoint-only validation ------------------------------------------

def _best_only(tmp_path: Path, **kwargs):
    path = tmp_path / "best.pt"
    path.write_bytes(BEST_BYTES)
    options = {
        "expected_sha256": BEST_SHA, "expected_pinned_revision": PINNED,
        "expected_epoch": EPOCHS, "expected_global_step": STEPS,
        "payload": _payload() | {"model_state_dict": True},
    }
    options.update(kwargs)
    return validate_best_checkpoint_only(path, **options)


def test_best_checkpoint_only_validates_and_never_claims_the_full_artifact(tmp_path) -> None:
    outcome = _best_only(tmp_path)
    payload = outcome.as_dict()
    assert outcome.best_checkpoint_validated is True
    # The whole point: a lone best.pt can never establish the full artifact.
    assert payload["full_artifact_validated"] is False
    assert payload["best_checkpoint_validated"] is True
    assert payload["epoch"] == EPOCHS and payload["global_step"] == STEPS
    assert payload["checkpoint_sha256"] == BEST_SHA
    assert outcome.diagnostics["scope"] == "best_checkpoint_only"
    assert outcome.diagnostics["full_artifact_files_checked"] == []


def test_full_artifact_flag_is_false_even_when_everything_else_fails(tmp_path) -> None:
    outcome = _best_only(tmp_path, expected_epoch=99)
    assert outcome.as_dict()["full_artifact_validated"] is False
    assert outcome.best_checkpoint_validated is False


@pytest.mark.parametrize("override,fragment", [
    ({"expected_epoch": 3}, "epoch does not match"),
    ({"expected_global_step": 1488}, "global_step does not match"),
    ({"expected_pinned_revision": "d" * 40}, "pinned model revision does not match"),
    ({"expected_sha256": ""}, "was not supplied"),
    ({"expected_sha256": "0" * 64}, "does not match the operator-supplied hash"),
])
def test_best_checkpoint_mismatches_are_rejected(tmp_path, override, fragment) -> None:
    outcome = _best_only(tmp_path, **override)
    assert outcome.best_checkpoint_validated is False
    assert any(fragment in f for f in outcome.failures), outcome.failures


@pytest.mark.parametrize("payload_override,fragment", [
    ({"mode": SMOKE_MODE}, "SMOKE_ONLY"),
    ({"mode": "OTHER"}, "mode is not FULL_TRAINING"),
    ({"entity_type_order": ["DIAGNOSIS"]}, "label space"),
    ({"model_state_dict": False}, "no model_state_dict"),
    ({"pinned_model_revision": "main"}, "not immutable"),
])
def test_best_checkpoint_payload_defects_are_rejected(tmp_path, payload_override, fragment):
    payload = _payload() | {"model_state_dict": True} | payload_override
    outcome = _best_only(tmp_path, payload=payload)
    assert outcome.best_checkpoint_validated is False
    assert any(fragment in f for f in outcome.failures), outcome.failures


def test_omitting_the_payload_never_silently_passes(tmp_path: Path) -> None:
    outcome = _best_only(tmp_path, payload=None)
    assert outcome.best_checkpoint_validated is False
    assert any("payload was not supplied" in f for f in outcome.failures)
    assert outcome.diagnostics["payload_inspected"] is False


def test_a_missing_checkpoint_file_is_reported(tmp_path: Path) -> None:
    outcome = validate_best_checkpoint_only(
        tmp_path / "absent.pt", expected_sha256=BEST_SHA, expected_pinned_revision=PINNED,
        expected_epoch=EPOCHS, expected_global_step=STEPS)
    assert outcome.best_checkpoint_validated is False
    assert any("does not exist" in f for f in outcome.failures)


def test_validation_does_not_modify_the_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(BEST_BYTES)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    validate_best_checkpoint_only(
        path, expected_sha256=BEST_SHA, expected_pinned_revision=PINNED,
        expected_epoch=EPOCHS, expected_global_step=STEPS,
        payload=_payload() | {"model_state_dict": True})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
