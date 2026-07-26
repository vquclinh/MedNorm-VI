"""S1 smoke-artifact validation tests (Audit 0025).

Deterministic and offline: manifests and checkpoints are synthesized in tmp_path.
No Drive access, no downloads, no training.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from mednorm_vi.training.s1_artifact_validation import (
    CHECKPOINT_RELATIVE_PATH,
    ArtifactValidationError,
    ValidationOutcome,
    find_base_model_cache_files,
    is_immutable_revision,
    is_valid_sha256,
    load_smoke_manifest,
    pinned_revision_from_outcome,
    validate_smoke_artifact,
)
from mednorm_vi.training.s1_mention_smoke import (
    load_smoke_config,
    smoke_artifact_paths_from_config,
)

REPO = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = load_smoke_config(REPO / "configs" / "training" / "s1_mention_first_run_smoke.yaml")
EXPECTED_CORPUS = SMOKE_CONFIG["corpus"]
RESOLVED_REVISION = "b" * 40
REPO_COMMIT = "c" * 40
# The digest recorded by the historical v1 run, kept only as test data. It no
# longer lives in the reusable validator.
HISTORICAL_V1_CHECKPOINT_SHA256 = (
    "7310f69acfae278f36753c4b356979737f17388c74f6703a362bb22788892213")


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
            "alignment_backend": "phobert-slow-char-alignment-v2",
            "tokenizer_equivalence_checked": True,
            "tokenizer_equivalence_examples": 12,
            "tokenizer_equivalence_failures": 0,
            "tokenizer_equivalence_considered": 12,
            "tokenizer_equivalence_skipped_unmappable": 0,
            "examples_considered": 12,
            "aligned_example_count": 12,
            "unalignable_example_count": 0,
            "governed_exclusion_count": 0,
            "unalignable_examples": [],
            "governed_exclusions": [],
            "reason_code_counts": {},
            "counters_reconciled": True,
            "boundary_merge_masked_word_count": 1,
            "boundary_merge_affected_entity_count": 2,
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


def test_checkpoint_hash_must_match_the_operator_supplied_hash(tmp_path: Path) -> None:
    base = _artifact(tmp_path, _manifest())
    outcome = _validate(base, expected_sha=HISTORICAL_V1_CHECKPOINT_SHA256)
    assert outcome.smoke_validated is False
    assert any("operator-supplied" in f for f in outcome.failures)


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
        expected_checkpoint_sha256=HISTORICAL_V1_CHECKPOINT_SHA256)
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


# --- alignment counter reconciliation and diagnostic privacy (Audit 0026) -----

def test_counters_must_reconcile_with_the_examples_considered(tmp_path: Path) -> None:
    """aligned + unalignable + governed exclusions must equal examples considered."""
    manifest = _manifest()
    manifest["alignment"]["aligned_example_count"] = 7      # the pre-fix manifest shape
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any("counters reconcile" in f for f in outcome.failures)


def test_equivalence_and_alignment_must_cover_the_same_examples(tmp_path: Path) -> None:
    """The old manifest scored equivalence over 12 but alignment over 8."""
    manifest = _manifest()
    manifest["alignment"]["tokenizer_equivalence_considered"] = 8
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any("cover the same examples" in f for f in outcome.failures)


def test_governed_exclusions_reconcile_without_blocking(tmp_path: Path) -> None:
    """A tracked exclusion keeps the books balanced and does NOT fail validation."""
    manifest = _manifest()
    manifest["alignment"].update({
        "aligned_example_count": 11, "governed_exclusion_count": 1,
        "governed_exclusions": [{
            "source": "vimq", "split": "train",
            "privacy_safe_example_id": "0123456789abcdef",
            "stage": "governed_exclusion", "reason_code": "GOVERNED_EXCLUSION",
            "exception_type": ""}],
    })
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is True
    assert outcome.diagnostics["governed_exclusions"]


def test_unexpected_unalignable_examples_still_fail_validation(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["alignment"].update({
        "aligned_example_count": 11, "unalignable_example_count": 1,
        "unalignable_examples": [{
            "source": "vimq", "split": "validation",
            "privacy_safe_example_id": "0123456789abcdef",
            "stage": "word_mapping",
            "reason_code": "NON_SEPARATOR_GAP_INSIDE_WORD",
            "exception_type": "AlignmentError"}],
    })
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any("zero unalignable examples" in f for f in outcome.failures)
    # the privacy-safe diagnostic is surfaced so the operator can act on it
    assert outcome.diagnostics["unalignable_examples"][0]["reason_code"] == (
        "NON_SEPARATOR_GAP_INSIDE_WORD")


@pytest.mark.parametrize("entry", [
    {"source": "vimq", "split": "train", "privacy_safe_example_id": "0123456789abcdef",
     "stage": "word_mapping", "reason_code": "X", "exception_type": "AlignmentError",
     "text": "benh nhan dau bung"},                       # raw clinical text
    {"source": "vimq", "split": "train", "privacy_safe_example_id": "vimq:dev:000958",
     "stage": "word_mapping", "reason_code": "X", "exception_type": "AlignmentError"},
    {"source": "vimq", "split": "train", "privacy_safe_example_id": "",
     "stage": "word_mapping", "reason_code": "X", "exception_type": "AlignmentError"},
])
def test_diagnostics_carrying_content_or_verbatim_ids_fail_validation(tmp_path, entry) -> None:
    """A manifest must never smuggle clinical text through a diagnostic field."""
    manifest = _manifest()
    manifest["alignment"].update({
        "aligned_example_count": 11, "governed_exclusion_count": 1,
        "governed_exclusions": [entry]})
    outcome = _validate(_artifact(tmp_path, manifest))
    assert outcome.smoke_validated is False
    assert any("privacy-safe" in f for f in outcome.failures)


# --- artifact lifecycle: v1 historical vs v2 corrected (Audit 0026) -----------

def test_v1_and_v2_smoke_output_directories_are_distinct() -> None:
    paths = smoke_artifact_paths_from_config(SMOKE_CONFIG)
    assert paths.artifact_version == "v2"
    assert paths.artifact_dir.endswith("s1_mention_first_run_smoke_v2")
    assert paths.previous_artifact_dir.endswith("s1_mention_first_run_smoke")
    assert paths.artifact_dir != paths.previous_artifact_dir
    assert "FULL_TRAINING_READINESS_FALSE" in paths.previous_artifact_status


def test_corrected_smoke_config_cannot_target_the_historical_artifact(tmp_path: Path) -> None:
    """The rerun must never overwrite the v1 evidence."""
    import yaml
    doc = yaml.safe_load(
        (REPO / "configs" / "training" / "s1_mention_first_run_smoke.yaml").read_text(
            encoding="utf-8"))
    doc["output"]["artifact_dir"] = doc["output"]["previous_artifact_dir"]
    with pytest.raises(ValueError, match="must differ from the historical"):
        smoke_artifact_paths_from_config(doc)


def test_smoke_config_output_section_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing output section"):
        smoke_artifact_paths_from_config({"limits": {}})


# --- the expected hash is supplied at runtime, never hardcoded ---------------

def test_validator_source_contains_no_run_specific_checkpoint_hash() -> None:
    """Accepting a new run must never require editing Python source."""
    source = (REPO / "src" / "mednorm_vi" / "training"
              / "s1_artifact_validation.py").read_text(encoding="utf-8")
    assert HISTORICAL_V1_CHECKPOINT_SHA256 not in source
    assert "EXPECTED_SMOKE_CHECKPOINT_SHA256 = " not in source
    assert not re.search(r"[0-9a-f]{64}", source), "a 64-hex digest is hardcoded"


# ("A" * 64 is NOT here: the validator lower-cases first, so an upper-case digest
# is a valid hash that simply has to match - see the case-tolerance test below.)
@pytest.mark.parametrize("expected", ["", "   ", None, "not-a-hash", "abc123",
                                      "0" * 63, "0" * 65, "g" * 64])
def test_missing_or_malformed_expected_hash_blocks_readiness(tmp_path, expected) -> None:
    base = _artifact(tmp_path, _manifest())
    outcome = validate_smoke_artifact(
        base, expected_corpus=EXPECTED_CORPUS, expected_checkpoint_sha256=expected)
    assert outcome.smoke_validated is False
    assert any("expected checkpoint SHA-256 was not supplied" in f for f in outcome.failures)
    # The recomputed digest is reported so the operator can confirm it.
    assert outcome.checkpoint_sha256
    assert any(outcome.checkpoint_sha256 in f for f in outcome.failures)
    assert outcome.diagnostics["expected_checkpoint_sha256_supplied"] is False


def test_a_new_run_is_accepted_by_supplying_its_hash_only(tmp_path: Path) -> None:
    """Two different runs both validate, with no source change between them."""
    for payload in (b"rerun-v2-alpha", b"rerun-v2-beta"):
        base = _artifact(tmp_path / payload.decode(), _manifest(), checkpoint_bytes=payload)
        digest = hashlib.sha256(payload).hexdigest()
        outcome = validate_smoke_artifact(
            base, expected_corpus=EXPECTED_CORPUS, expected_checkpoint_sha256=digest)
        assert outcome.smoke_validated is True
        assert outcome.checkpoint_sha256 == digest
        assert outcome.diagnostics["expected_checkpoint_sha256_supplied"] is True


def test_expected_hash_is_case_and_whitespace_tolerant(tmp_path: Path) -> None:
    base = _artifact(tmp_path, _manifest(), checkpoint_bytes=b"rerun")
    digest = hashlib.sha256(b"rerun").hexdigest()
    outcome = validate_smoke_artifact(
        base, expected_corpus=EXPECTED_CORPUS,
        expected_checkpoint_sha256=f"  {digest.upper()}  ")
    assert outcome.smoke_validated is True


def test_all_three_hashes_must_agree(tmp_path: Path) -> None:
    """recomputed == manifest == operator-supplied, or nothing is granted."""
    manifest = _manifest()
    manifest["artifacts"]["checkpoint_sha256"] = "0" * 64      # manifest disagrees
    base = _artifact(tmp_path, manifest, checkpoint_bytes=b"rerun")
    outcome = validate_smoke_artifact(
        base, expected_corpus=EXPECTED_CORPUS,
        expected_checkpoint_sha256=hashlib.sha256(b"rerun").hexdigest())
    assert outcome.smoke_validated is False
    assert any("does not match the manifest" in f for f in outcome.failures)


@pytest.mark.parametrize("value,valid", [
    ("a" * 64, True), ("0123456789abcdef" * 4, True),
    ("A" * 64, False), ("a" * 63, False), ("", False), ("zz", False),
])
def test_sha256_validity_rule(value, valid) -> None:
    assert is_valid_sha256(value) is valid


def test_outcome_reports_the_artifact_directory_it_validated(tmp_path: Path) -> None:
    base = _artifact(tmp_path, _manifest())
    outcome = _validate(base)
    assert outcome.diagnostics["artifact_dir"] == str(base)
