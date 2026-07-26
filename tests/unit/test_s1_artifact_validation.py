"""S1 smoke-artifact validation tests (Audit 0025).

Deterministic and offline: manifests and checkpoints are synthesized in tmp_path.
No Drive access, no downloads, no training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mednorm_vi.training.s1_artifact_validation import (
    CHECKPOINT_RELATIVE_PATH,
    EXPECTED_SMOKE_CHECKPOINT_SHA256,
    ArtifactValidationError,
    ValidationOutcome,
    find_base_model_cache_files,
    is_immutable_revision,
    load_smoke_manifest,
    pinned_revision_from_outcome,
    validate_smoke_artifact,
)
from mednorm_vi.training.s1_mention_smoke import load_smoke_config

REPO = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = load_smoke_config(REPO / "configs" / "training" / "s1_mention_first_run_smoke.yaml")
EXPECTED_CORPUS = SMOKE_CONFIG["corpus"]
RESOLVED_REVISION = "b" * 40
REPO_COMMIT = "c" * 40


def _manifest() -> dict:
    """A manifest matching the confirmed successful Colab smoke run."""
    return {
        "status": "SMOKE_ONLY",
        "smoke_only_not_full_training": True,
        "full_training_readiness": True,
        "repository": {"resolved_commit": REPO_COMMIT},
        "corpus": {
            "corpus_manifest_sha256": EXPECTED_CORPUS["expected_corpus_manifest_sha256"],
            "train_sha256": EXPECTED_CORPUS["expected_train_sha256"],
            "total_examples": EXPECTED_CORPUS["expected_total_examples"],
            "train_examples": EXPECTED_CORPUS["expected_train_examples"],
            "validation_examples": EXPECTED_CORPUS["expected_validation_examples"],
            "internal_test_examples": EXPECTED_CORPUS["expected_internal_test_examples"],
            "vietmed_status": EXPECTED_CORPUS["expected_vietmed_status"],
            "cross_split_family_leakage": 0,
            "eval_map_approximate_entities": 0,
        },
        "model": {
            "registry_model_id": "vihealthbert_span_type",
            "hf_model_id": "demdecuong/vihealthbert-base-word",
            "requested_revision": "main",
            "resolved_model_revision": RESOLVED_REVISION,
        },
        "tokenizer": {
            "tokenizer_class": "PhobertTokenizer",
            "tokenizer_is_fast": False,
            "tokenizer_revision": "main",
        },
        "alignment": {
            "alignment_backend": "character_offset_reconstruction",
            "tokenizer_equivalence_checked": True,
            "tokenizer_equivalence_examples": 12,
            "tokenizer_equivalence_failures": 0,
            "unalignable_example_count": 0,
        },
        "word_segmentation": {
            "segmenter_mode": "vncorenlp",
            "word_segmenter": "VnCoreNLP RDRSegmenter",
            "degraded_fallback": False,
            "word_segmenter_version": "py_vncorenlp==0.1.4",
            "word_segmenter_resource_hashes": {"VnCoreNLP-1.2.jar": "d" * 64},
        },
        "environment": {
            "s1_dependency_closure_verified": True,
            "blocking_dependency_conflicts": [],
            "non_blocking_dependency_conflicts": [
                "gradio 5.49.1 requires huggingface-hub<1.0,>=0.28.1, but you have "
                "huggingface-hub 1.0.1 which is incompatible.",
            ],
            "numpy_abi_preflight_passed": True,
            "dependency_restart_completed": True,
            "pip_check_passed": False,          # unrelated Colab image conflicts
            "pip_check_output": "ipython 7.34.0 requires jedi>=0.16, which is not installed.",
            "dependency_contract_version": "s1-colab-deps-v2",
        },
        "loss_values": {
            "train_loss_finite": True,
            "backward_completed": True,
            "optimizer_step_confirmed": True,
        },
        "validation_metrics": {"validation_completed": True},
        "artifacts": {"checkpoint_sha256": ""},
    }


def _artifact(tmp_path: Path, manifest: dict, checkpoint_bytes: bytes = b"smoke") -> Path:
    """Write a manifest + checkpoint whose recorded hash matches the bytes."""
    base = tmp_path / "s1_mention_first_run_smoke"
    checkpoint = base / CHECKPOINT_RELATIVE_PATH
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(checkpoint_bytes)
    if not manifest["artifacts"]["checkpoint_sha256"]:
        manifest["artifacts"]["checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    (base / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return base


def _validate(base: Path, expected_sha: str | None = None):
    if expected_sha is None:
        expected_sha = hashlib.sha256(
            (base / CHECKPOINT_RELATIVE_PATH).read_bytes()).hexdigest()
    return validate_smoke_artifact(
        base, expected_corpus=EXPECTED_CORPUS, expected_checkpoint_sha256=expected_sha)


# --- happy path ---------------------------------------------------------------

def test_valid_artifact_passes_every_condition(tmp_path: Path) -> None:
    outcome = _validate(_artifact(tmp_path, _manifest()))
    assert outcome.smoke_validated is True
    assert outcome.failures == ()
    assert outcome.resolved_model_revision == RESOLVED_REVISION


def test_failed_global_pip_check_does_not_invalidate_the_artifact(tmp_path: Path) -> None:
    """Audit 0024 scoping: the S1 closure gate is what matters, not the whole image."""
    outcome = _validate(_artifact(tmp_path, _manifest()))
    assert outcome.smoke_validated is True
    assert outcome.diagnostics["pip_check_passed"] is False        # preserved, not fatal
    assert outcome.diagnostics["non_blocking_dependency_conflicts"]
    assert "jedi" in outcome.diagnostics["pip_check_output"]


def test_blocking_closure_conflicts_do_invalidate_the_artifact(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["environment"]["blocking_dependency_conflicts"] = [
        "transformers 4.44.2 requires tokenizers<0.20, but you have tokenizers 0.21.0"]
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any("blocking dependency conflicts" in f for f in outcome.failures)


# --- checkpoint hashing -------------------------------------------------------

def test_checkpoint_hash_must_match_the_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["artifacts"]["checkpoint_sha256"] = "0" * 64
    base = _artifact(tmp_path, manifest)
    outcome = _validate(base)
    assert outcome.smoke_validated is False
    assert any("does not match the manifest" in f for f in outcome.failures)


def test_checkpoint_hash_must_match_the_expected_colab_hash(tmp_path: Path) -> None:
    base = _artifact(tmp_path, _manifest())
    outcome = _validate(base, expected_sha=EXPECTED_SMOKE_CHECKPOINT_SHA256)
    assert outcome.smoke_validated is False
    assert any("expected Colab hash" in f for f in outcome.failures)


def test_hash_is_recomputed_from_the_bytes_not_read_from_the_manifest(tmp_path: Path) -> None:
    """Tampering with the checkpoint after the manifest was written is detected."""
    base = _artifact(tmp_path, _manifest())
    (base / CHECKPOINT_RELATIVE_PATH).write_bytes(b"tampered")
    outcome = validate_smoke_artifact(
        base, expected_corpus=EXPECTED_CORPUS,
        expected_checkpoint_sha256=hashlib.sha256(b"tampered").hexdigest())
    assert outcome.smoke_validated is False
    assert any("does not match the manifest" in f for f in outcome.failures)


def test_missing_checkpoint_file_is_reported(tmp_path: Path) -> None:
    base = _artifact(tmp_path, _manifest())
    (base / CHECKPOINT_RELATIVE_PATH).unlink()
    outcome = validate_smoke_artifact(
        base, expected_corpus=EXPECTED_CORPUS,
        expected_checkpoint_sha256=EXPECTED_SMOKE_CHECKPOINT_SHA256)
    assert outcome.smoke_validated is False
    assert any("checkpoint file does not exist" in f for f in outcome.failures)


# --- readiness evidence -------------------------------------------------------

@pytest.mark.parametrize("path,value,fragment", [
    (("status",), "FULL_TRAINING", "status is SMOKE_ONLY"),
    (("smoke_only_not_full_training",), False, "smoke_only_not_full_training"),
    (("full_training_readiness",), False, "full_training_readiness is true"),
    (("repository", "resolved_commit"), "", "repository commit SHA recorded"),
    (("tokenizer", "tokenizer_is_fast"), True, "tokenizer_is_fast is false"),
    (("tokenizer", "tokenizer_class"), "RobertaTokenizerFast", "PhobertTokenizer"),
    (("alignment", "tokenizer_equivalence_examples"), 0, "at least one example"),
    (("alignment", "tokenizer_equivalence_failures"), 3, "equivalence failures are zero"),
    (("alignment", "unalignable_example_count"), 2, "zero unalignable examples"),
    (("word_segmentation", "degraded_fallback"), True, "degraded segmentation fallback"),
    (("word_segmentation", "segmenter_mode"), "whitespace_fallback", "production VnCoreNLP"),
    (("word_segmentation", "word_segmenter_resource_hashes"), {}, "resource hashes"),
    (("environment", "s1_dependency_closure_verified"), False, "dependency closure verified"),
    (("environment", "numpy_abi_preflight_passed"), False, "NumPy ABI preflight"),
    (("loss_values", "train_loss_finite"), False, "train loss is finite"),
    (("loss_values", "backward_completed"), False, "backward pass completed"),
    (("loss_values", "optimizer_step_confirmed"), False, "AdamW optimizer step"),
    (("validation_metrics", "validation_completed"), False, "tiny validation completed"),
    (("model", "registry_model_id"), "other_model", "model registry id"),
    (("model", "hf_model_id"), "bert-base-uncased", "hugging face model id"),
])
def test_missing_or_false_evidence_fails_validation(tmp_path, path, value, fragment) -> None:
    manifest = _manifest()
    node = manifest
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any(fragment in f for f in outcome.failures), outcome.failures


@pytest.mark.parametrize("key,value", [
    ("corpus_manifest_sha256", "0" * 64),
    ("train_sha256", "0" * 64),
    ("train_examples", 1),
    ("vietmed_status", "excluded"),
])
def test_governed_corpus_mismatch_fails_validation(tmp_path, key, value) -> None:
    manifest = _manifest()
    manifest["corpus"][key] = value
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any(key in f for f in outcome.failures)


@pytest.mark.parametrize("key,label", [
    ("cross_split_family_leakage", "cross-split family leakage is zero"),
    ("eval_map_approximate_entities", "approximate mappings are zero"),
])
def test_leakage_and_approximate_mappings_must_be_zero(tmp_path, key, label) -> None:
    manifest = _manifest()
    manifest["corpus"][key] = 4
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any(label in f for f in outcome.failures)


def test_every_failure_is_reported_not_just_the_first(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["status"] = "FULL_TRAINING"
    manifest["full_training_readiness"] = False
    manifest["alignment"]["tokenizer_equivalence_failures"] = 1
    outcome = _validate(_artifact(tmp_path, manifest))
    assert len(outcome.failures) >= 3
    assert outcome.as_dict()["failed_condition_count"] == len(outcome.failures)


# --- model revision -----------------------------------------------------------

@pytest.mark.parametrize("revision,immutable", [
    ("a" * 40, True),
    ("main", False),
    ("master", False),
    ("", False),
    ("v1.0", False),
    ("a" * 39, False),
    ("A" * 40, False),               # uppercase is not a git object id
])
def test_only_a_40_hex_commit_is_immutable(revision, immutable) -> None:
    assert is_immutable_revision(revision) is immutable


@pytest.mark.parametrize("revision", ["main", "", "v1.0"])
def test_mutable_resolved_revision_fails_validation(tmp_path, revision) -> None:
    manifest = _manifest()
    manifest["model"]["resolved_model_revision"] = revision
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any("resolved model revision is immutable" in f for f in outcome.failures)


def test_pinned_revision_comes_only_from_a_validated_artifact(tmp_path: Path) -> None:
    good = _validate(_artifact(tmp_path / "ok", _manifest()))
    assert pinned_revision_from_outcome(good) == RESOLVED_REVISION

    manifest = _manifest()
    manifest["full_training_readiness"] = False
    failed = _validate(_artifact(tmp_path / "bad", manifest))
    with pytest.raises(ArtifactValidationError, match="failed validation"):
        pinned_revision_from_outcome(failed)


def test_pinned_revision_is_never_invented() -> None:
    """No immutable revision is a BLOCKER to report, never a value to guess."""
    outcome = ValidationOutcome(
        passed=True, failures=(), checkpoint_sha256="e" * 64,
        manifest_checkpoint_sha256="e" * 64, resolved_model_revision="main")
    with pytest.raises(ArtifactValidationError, match="BLOCKER"):
        pinned_revision_from_outcome(outcome)


# --- artifact hygiene ---------------------------------------------------------

def test_base_model_cache_inside_the_artifact_fails_validation(tmp_path: Path) -> None:
    base = _artifact(tmp_path, _manifest())
    (base / "hf_cache").mkdir()
    (base / "hf_cache" / "model.safetensors").write_bytes(b"weights")
    outcome = _validate(base)
    assert outcome.smoke_validated is False
    assert any("base-model cache files" in f for f in outcome.failures)
    assert find_base_model_cache_files(base) == ["hf_cache/model.safetensors"]


def test_the_smoke_checkpoint_itself_is_not_flagged_as_cache(tmp_path: Path) -> None:
    assert find_base_model_cache_files(_artifact(tmp_path, _manifest())) == []


# --- unreadable artifacts -----------------------------------------------------

def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="missing smoke manifest"):
        load_smoke_manifest(tmp_path)


def test_malformed_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "training_manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="not valid JSON"):
        load_smoke_manifest(tmp_path)
