"""Validation of the REAL S1 smoke artifact produced on Colab (Audit 0025).

The S1 first-run smoke notebook writes two files to Drive::

    artifacts/s1_mention_first_run_smoke/training_manifest.json
    artifacts/s1_mention_first_run_smoke/checkpoint/s1_mention_smoke_model.pt

A screenshot is not evidence. This module reads the actual manifest, recomputes
the checkpoint SHA-256 from the bytes on disk, and checks every condition the
smoke was supposed to prove. It reports **all** failures rather than the first
one, so a Colab operator sees the complete picture in a single run.

Pure Python: no Torch, no Transformers, no network. The notebook supplies the
Drive path; everything here is file reading and comparison.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .s1_mention_smoke import sha256_file

# Relative layout of the smoke artifact directory.
MANIFEST_NAME = "training_manifest.json"
CHECKPOINT_RELATIVE_PATH = "checkpoint/s1_mention_smoke_model.pt"

# NO run-specific checkpoint hash lives here. This module is reused across smoke
# reruns, so the expected hash is supplied by the operator at validation time
# (a notebook config variable or MEDNORM_EXPECTED_SMOKE_CHECKPOINT_SHA256) and is
# never baked into source. Accepting a new run must never require a code edit.
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# A Hugging Face commit hash: the only revision form that is truly immutable.
IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MUTABLE_REVISIONS = frozenset({"", "main", "master", "head", "HEAD", "latest"})

# Base-model cache files must never be copied into the reviewed artifact.
FORBIDDEN_ARTIFACT_SUFFIXES = (".safetensors", ".bin", ".h5", ".onnx", ".msgpack")

# The only keys a privacy-safe alignment diagnostic may carry. Anything else could
# smuggle clinical text into a manifest that is reviewed and shared.
ALLOWED_DIAGNOSTIC_KEYS = frozenset({
    "source", "split", "privacy_safe_example_id", "stage", "reason_code", "exception_type",
})
# A hashed example handle: 16 lowercase hex characters.
PRIVACY_SAFE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


class ArtifactValidationError(ValueError):
    """Raised when the artifact cannot be read at all (not a failed check)."""


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """The result of validating one real smoke artifact directory."""

    passed: bool
    failures: tuple[str, ...]
    checkpoint_sha256: str
    manifest_checkpoint_sha256: str
    resolved_model_revision: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def smoke_validated(self) -> bool:
        """Only a completely clean run may be called a validated smoke."""
        return self.passed and not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "smoke_artifact_validated": self.smoke_validated,
            "failed_conditions": list(self.failures),
            "failed_condition_count": len(self.failures),
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_checkpoint_sha256": self.manifest_checkpoint_sha256,
            "resolved_model_revision": self.resolved_model_revision,
            "diagnostics": dict(self.diagnostics),
        }


def is_valid_sha256(value: str) -> bool:
    """True only for a full lowercase 64-hex digest."""
    return bool(SHA256_PATTERN.match(str(value or "").strip()))


def is_immutable_revision(revision: str) -> bool:
    """True only for a full 40-hex commit hash.

    ``main`` moves. Pinning production training to a moving reference makes the
    run unreproducible, so it is rejected even though it resolves today.
    """
    candidate = str(revision or "").strip()
    if candidate in MUTABLE_REVISIONS:
        return False
    return bool(IMMUTABLE_REVISION_PATTERN.match(candidate))


def _get(payload: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from nested manifest mappings."""
    node: Any = payload
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def load_smoke_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Read the real manifest; a missing or malformed file is fatal."""
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ArtifactValidationError(f"missing smoke manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"smoke manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("smoke manifest must be a JSON object")
    return manifest


def find_base_model_cache_files(artifact_dir: str | Path) -> list[str]:
    """Base-model weights copied into the artifact directory (must be none)."""
    base = Path(artifact_dir)
    if not base.is_dir():
        return []
    checkpoint = (base / CHECKPOINT_RELATIVE_PATH).resolve()
    return sorted(
        str(path.relative_to(base))
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix in FORBIDDEN_ARTIFACT_SUFFIXES
        and path.resolve() != checkpoint
    )


def _boolean_conditions(manifest: Mapping[str, Any]) -> list[tuple[str, bool, Any]]:
    """(label, holds, observed) for every evidence condition in the manifest.

    Each entry checks a real recorded value, not merely that a key exists.
    """
    tokenizer_class = str(_get(manifest, "tokenizer.tokenizer_class", ""))
    equivalence_examples = _get(manifest, "alignment.tokenizer_equivalence_examples", 0)
    equivalence_failures = _get(manifest, "alignment.tokenizer_equivalence_failures", 1)
    unalignable = _get(manifest, "alignment.unalignable_example_count", 1)
    blocking = _get(manifest, "environment.blocking_dependency_conflicts", ["<absent>"])
    resource_hashes = _get(manifest, "word_segmentation.word_segmenter_resource_hashes", {})
    return [
        ("status is SMOKE_ONLY",
         manifest.get("status") == "SMOKE_ONLY", manifest.get("status")),
        ("smoke_only_not_full_training is true",
         manifest.get("smoke_only_not_full_training") is True,
         manifest.get("smoke_only_not_full_training")),
        ("repository commit SHA recorded",
         bool(re.fullmatch(r"[0-9a-f]{40}", str(_get(manifest, "repository.resolved_commit", "")))),
         _get(manifest, "repository.resolved_commit")),
        ("model registry id is vihealthbert_span_type",
         _get(manifest, "model.registry_model_id") == "vihealthbert_span_type",
         _get(manifest, "model.registry_model_id")),
        ("hugging face model id is demdecuong/vihealthbert-base-word",
         _get(manifest, "model.hf_model_id") == "demdecuong/vihealthbert-base-word",
         _get(manifest, "model.hf_model_id")),
        ("requested model revision recorded",
         bool(str(_get(manifest, "model.requested_revision", "")).strip()),
         _get(manifest, "model.requested_revision")),
        ("resolved model revision is immutable and non-empty",
         is_immutable_revision(str(_get(manifest, "model.resolved_model_revision", ""))),
         _get(manifest, "model.resolved_model_revision")),
        ("tokenizer class is PhobertTokenizer",
         "phoberttokenizer" in tokenizer_class.lower(), tokenizer_class),
        ("tokenizer_is_fast is false",
         _get(manifest, "tokenizer.tokenizer_is_fast") is False,
         _get(manifest, "tokenizer.tokenizer_is_fast")),
        ("slow-tokenizer alignment backend recorded",
         bool(str(_get(manifest, "alignment.alignment_backend", "")).strip()),
         _get(manifest, "alignment.alignment_backend")),
        ("tokenizer equivalence checked",
         _get(manifest, "alignment.tokenizer_equivalence_checked") is True,
         _get(manifest, "alignment.tokenizer_equivalence_checked")),
        ("tokenizer equivalence covered at least one example",
         isinstance(equivalence_examples, int) and equivalence_examples >= 1,
         equivalence_examples),
        ("tokenizer equivalence failures are zero",
         equivalence_failures == 0, equivalence_failures),
        ("zero unalignable examples", unalignable == 0, unalignable),
        ("production VnCoreNLP segmentation used",
         _get(manifest, "word_segmentation.segmenter_mode") == "vncorenlp"
         and "VnCoreNLP" in str(_get(manifest, "word_segmentation.word_segmenter", "")),
         _get(manifest, "word_segmentation.word_segmenter")),
        ("degraded segmentation fallback is false",
         _get(manifest, "word_segmentation.degraded_fallback") is False,
         _get(manifest, "word_segmentation.degraded_fallback")),
        ("segmenter resource hashes are non-empty",
         isinstance(resource_hashes, Mapping) and len(resource_hashes) > 0,
         len(resource_hashes) if isinstance(resource_hashes, Mapping) else resource_hashes),
        ("S1 dependency closure verified",
         _get(manifest, "environment.s1_dependency_closure_verified") is True,
         _get(manifest, "environment.s1_dependency_closure_verified")),
        ("blocking dependency conflicts are empty",
         isinstance(blocking, Sequence) and not isinstance(blocking, str) and len(blocking) == 0,
         blocking),
        ("NumPy ABI preflight passed",
         _get(manifest, "environment.numpy_abi_preflight_passed") is True,
         _get(manifest, "environment.numpy_abi_preflight_passed")),
        ("dependency restart completed",
         _get(manifest, "environment.dependency_restart_completed") is True,
         _get(manifest, "environment.dependency_restart_completed")),
        ("train loss is finite",
         _get(manifest, "loss_values.train_loss_finite") is True,
         _get(manifest, "loss_values.train_loss_finite")),
        ("backward pass completed",
         _get(manifest, "loss_values.backward_completed") is True,
         _get(manifest, "loss_values.backward_completed")),
        ("AdamW optimizer step completed",
         _get(manifest, "loss_values.optimizer_step_confirmed") is True,
         _get(manifest, "loss_values.optimizer_step_confirmed")),
        ("tiny validation completed",
         _get(manifest, "validation_metrics.validation_completed") is True,
         _get(manifest, "validation_metrics.validation_completed")),
        ("full_training_readiness is true",
         manifest.get("full_training_readiness") is True,
         manifest.get("full_training_readiness")),
    ]


def _diagnostic_conditions(manifest: Mapping[str, Any]) -> list[tuple[str, bool, Any]]:
    """Counter reconciliation and privacy of the alignment diagnostics.

    Unexpected failures must block; tracked governed exclusions must not. Both are
    only trustworthy if every considered example is accounted for exactly once and
    no diagnostic carries free-form content.
    """
    alignment = manifest.get("alignment")
    if not isinstance(alignment, Mapping):
        return [("alignment block recorded", False, alignment)]
    checks: list[tuple[str, bool, Any]] = []
    considered = alignment.get("examples_considered")
    aligned = alignment.get("aligned_example_count")
    unalignable = alignment.get("unalignable_example_count")
    excluded = alignment.get("governed_exclusion_count", 0)
    if isinstance(considered, int):
        total = sum(v for v in (aligned, unalignable, excluded) if isinstance(v, int))
        checks.append((
            "alignment counters reconcile (aligned + unalignable + governed exclusions "
            "== examples considered)",
            total == considered,
            {"aligned": aligned, "unalignable": unalignable,
             "governed_exclusions": excluded, "considered": considered}))
        checks.append((
            "tokenizer equivalence and alignment cover the same examples",
            alignment.get("tokenizer_equivalence_considered") == considered,
            {"equivalence_considered": alignment.get("tokenizer_equivalence_considered"),
             "alignment_considered": considered}))
    diagnostics: list[Any] = []
    for key in ("unalignable_examples", "governed_exclusions"):
        entries = alignment.get(key) or []
        if not isinstance(entries, Sequence) or isinstance(entries, str):
            checks.append((f"{key} is a list", False, entries))
            continue
        diagnostics.extend(entries)
    offending = [
        entry for entry in diagnostics
        if not isinstance(entry, Mapping) or set(entry) - ALLOWED_DIAGNOSTIC_KEYS
        or not PRIVACY_SAFE_ID_PATTERN.match(str(entry.get("privacy_safe_example_id", "")))
    ]
    checks.append((
        "alignment diagnostics are privacy-safe (hashed ids, no free-form fields)",
        not offending, offending[:3]))
    return checks


def _corpus_conditions(
    manifest: Mapping[str, Any], expected_corpus: Mapping[str, Any],
) -> list[tuple[str, bool, Any]]:
    """Governed-corpus identity: the run must have used the approved corpus."""
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        return [("governed corpus block recorded", False, corpus)]
    checks: list[tuple[str, bool, Any]] = []
    for key, expected in (
        ("corpus_manifest_sha256", expected_corpus.get("expected_corpus_manifest_sha256")),
        ("train_sha256", expected_corpus.get("expected_train_sha256")),
        ("total_examples", expected_corpus.get("expected_total_examples")),
        ("train_examples", expected_corpus.get("expected_train_examples")),
        ("validation_examples", expected_corpus.get("expected_validation_examples")),
        ("internal_test_examples", expected_corpus.get("expected_internal_test_examples")),
        ("vietmed_status", expected_corpus.get("expected_vietmed_status")),
    ):
        observed = corpus.get(key)
        checks.append((f"governed corpus {key} matches the approved corpus",
                       observed == expected, {"got": observed, "expected": expected}))
    for key, label in (
        ("cross_split_family_leakage", "cross-split family leakage is zero"),
        ("eval_map_approximate_entities",
         "validation/internal-test approximate mappings are zero"),
    ):
        observed = corpus.get(key)
        checks.append((label, observed == 0, observed))
    return checks


def validate_smoke_artifact(
    artifact_dir: str | Path, *,
    expected_corpus: Mapping[str, Any],
    expected_checkpoint_sha256: str,
    hasher: Callable[[str | Path], str] = sha256_file,
) -> ValidationOutcome:
    """Validate the real S1 smoke artifact and report every failed condition.

    The checkpoint SHA-256 is recomputed from the bytes on disk and must equal
    **both** the hash recorded inside the manifest **and**
    ``expected_checkpoint_sha256``, which the operator supplies explicitly for the
    run being accepted. A missing or malformed expected hash never grants
    readiness: the recomputed digest is reported so it can be confirmed, without
    editing any source file.

    A failed global ``pip check`` does not invalidate the artifact: environment
    health is scoped to the S1 dependency closure (Audit 0024), so unrelated
    Colab conflicts are carried through as diagnostics only.
    """
    base = Path(artifact_dir)
    manifest = load_smoke_manifest(base)
    failures: list[str] = []

    for label, holds, observed in (
        *_boolean_conditions(manifest),
        *_corpus_conditions(manifest, expected_corpus),
        *_diagnostic_conditions(manifest),
    ):
        if not holds:
            failures.append(f"{label} (observed: {observed!r})")

    checkpoint_path = base / CHECKPOINT_RELATIVE_PATH
    manifest_sha = str(_get(manifest, "artifacts.checkpoint_sha256", ""))
    expected_sha = str(expected_checkpoint_sha256 or "").strip().lower()
    computed_sha = ""
    if not checkpoint_path.is_file():
        failures.append(f"checkpoint file does not exist (observed: {str(checkpoint_path)!r})")
    else:
        computed_sha = hasher(checkpoint_path)
        if computed_sha != manifest_sha:
            failures.append(
                "recomputed checkpoint SHA-256 does not match the manifest "
                f"(observed: {computed_sha!r}, manifest: {manifest_sha!r})")
        if not is_valid_sha256(expected_sha):
            # Never silently accept whatever is on disk: ask for confirmation and
            # show the digest the operator would be confirming.
            failures.append(
                "expected checkpoint SHA-256 was not supplied (or is malformed): set "
                "EXPECTED_SMOKE_CHECKPOINT_SHA256 to the 64-hex digest of the run you "
                f"are accepting. The recomputed digest is {computed_sha!r} "
                f"(observed: {expected_checkpoint_sha256!r})")
        elif computed_sha != expected_sha:
            failures.append(
                "recomputed checkpoint SHA-256 does not match the operator-supplied "
                f"expected hash (observed: {computed_sha!r}, expected: {expected_sha!r})")

    cache_files = find_base_model_cache_files(base)
    if cache_files:
        failures.append(f"base-model cache files inside the artifact (observed: {cache_files!r})")

    resolved_revision = str(_get(manifest, "model.resolved_model_revision", ""))
    return ValidationOutcome(
        passed=not failures,
        failures=tuple(failures),
        checkpoint_sha256=computed_sha,
        manifest_checkpoint_sha256=manifest_sha,
        resolved_model_revision=resolved_revision,
        diagnostics={
            # Preserved, never treated as failures (Audit 0024 scoping).
            "pip_check_passed": _get(manifest, "environment.pip_check_passed"),
            "pip_check_output": _get(manifest, "environment.pip_check_output", ""),
            "non_blocking_dependency_conflicts": _get(
                manifest, "environment.non_blocking_dependency_conflicts", []),
            "dependency_contract_version": _get(
                manifest, "environment.dependency_contract_version", ""),
            "requested_model_revision": _get(manifest, "model.requested_revision", ""),
            "tokenizer_revision": _get(manifest, "tokenizer.tokenizer_revision", ""),
            "word_segmenter_version": _get(
                manifest, "word_segmentation.word_segmenter_version", ""),
            "unalignable_examples": _get(manifest, "alignment.unalignable_examples", []),
            "governed_exclusions": _get(manifest, "alignment.governed_exclusions", []),
            "reason_code_counts": _get(manifest, "alignment.reason_code_counts", {}),
            "boundary_merge_masked_word_count": _get(
                manifest, "alignment.boundary_merge_masked_word_count", 0),
            "resolved_commit": _get(manifest, "repository.resolved_commit", ""),
            "artifact_dir": str(base),
            "expected_checkpoint_sha256_supplied": is_valid_sha256(expected_sha),
        },
    )


def pinned_revision_from_outcome(outcome: ValidationOutcome) -> str:
    """The immutable revision full training must pin, or raise.

    Refuses to invent a revision: if the validated manifest does not carry a
    trustworthy immutable commit hash, that is a blocker to report, not a value
    to guess.
    """
    if not outcome.smoke_validated:
        raise ArtifactValidationError(
            "cannot pin a model revision from an artifact that failed validation")
    if not is_immutable_revision(outcome.resolved_model_revision):
        raise ArtifactValidationError(
            "validated manifest has no immutable resolved model revision "
            f"(observed: {outcome.resolved_model_revision!r}); this is a BLOCKER for full "
            "training - do not invent a revision")
    return outcome.resolved_model_revision


__all__ = [
    "ALLOWED_DIAGNOSTIC_KEYS",
    "CHECKPOINT_RELATIVE_PATH",
    "MANIFEST_NAME",
    "MUTABLE_REVISIONS",
    "ArtifactValidationError",
    "ValidationOutcome",
    "find_base_model_cache_files",
    "is_immutable_revision",
    "is_valid_sha256",
    "load_smoke_manifest",
    "pinned_revision_from_outcome",
    "validate_smoke_artifact",
]
