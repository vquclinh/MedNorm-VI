"""S1 mention first-run smoke contracts.

Pure helpers for the Colab smoke notebook. They validate the governed corpus,
prepare tiny deterministic train/validation subsets, and build token-level
span/type labels plus loss masks without importing Torch or Transformers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..data_engine.annotation_coverage import SourceCoverage, example_loss_mask
from ..schemas.constants import ENTITY_TYPES
from .phobert_alignment import (
    EXPECTED_REASON_CODES,
    REASON_ENTITY_OFFSET_INVARIANT_VIOLATED,
    REASON_GOVERNED_EXCLUSION,
    STAGE_GOVERNED_EXCLUSION,
    STAGE_SUBTOKEN_ENCODING,
    STAGE_TOKENIZER_EQUIVALENCE,
    UNEXPECTED_REASON_CODES,
    AlignmentError,
    align_subtokens,
    classify_truncated_entities,
    entities_touched_by_boundary_merge,
    map_segmented_words,
    segmented_text_to_words,
)

ENTITY_TYPE_ORDER: tuple[str, ...] = tuple(sorted(ENTITY_TYPES))
ENTITY_TYPE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ENTITY_TYPE_ORDER)}


@dataclass(frozen=True, slots=True)
class ExpectedCorpus:
    total_examples: int
    train_examples: int
    validation_examples: int
    internal_test_examples: int
    corpus_manifest_sha256: str
    train_sha256: str
    vietmed_status: str
    cross_split_family_leakage: int
    eval_map_approximate_entities: int


@dataclass(frozen=True, slots=True)
class SmokeLimits:
    max_train_examples: int
    max_validation_examples: int
    max_train_batches: int
    max_validation_batches: int
    max_optimizer_steps: int
    batch_size: int
    max_sequence_length: int
    learning_rate: float

    def validate(self) -> None:
        if self.max_optimizer_steps < 1 or self.max_optimizer_steps > 2:
            raise ValueError("S1 smoke max_optimizer_steps must be 1 or 2")
        if self.max_train_batches < 1 or self.max_train_batches > 3:
            raise ValueError("S1 smoke max_train_batches must be between 1 and 3")
        if self.max_validation_batches < 1 or self.max_validation_batches > 2:
            raise ValueError("S1 smoke max_validation_batches must be 1 or 2")
        if self.max_train_examples < self.batch_size:
            raise ValueError("S1 smoke max_train_examples must cover at least one batch")
        if self.batch_size < 1 or self.batch_size > 4:
            raise ValueError("S1 smoke batch_size must be between 1 and 4")
        if self.max_sequence_length < 16 or self.max_sequence_length > 256:
            raise ValueError("S1 smoke max_sequence_length must be between 16 and 256")


def privacy_safe_example_id(example_id: str) -> str:
    """Stable, non-reversible handle for one example.

    Governed example ids look like ``vimq:dev:000958`` — structural, not clinical.
    They are hashed anyway so nothing derived from the corpus is copied verbatim
    into a manifest or audit, while a reviewer can still recompute the digest
    locally to identify the row.
    """
    return hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AlignmentDiagnostic:
    """Why one example produced no supervision. Never contains text.

    Only these fields are recorded: dataset/split identity, a hashed example id,
    the pipeline stage, a stable reason code, and the exception class name.
    """

    source_dataset: str
    split: str
    privacy_safe_example_id: str
    stage: str
    reason_code: str
    exception_type: str

    @property
    def expected(self) -> bool:
        """True for tracked governance decisions, False for real failures."""
        return self.reason_code in EXPECTED_REASON_CODES

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source_dataset,
            "split": self.split,
            "privacy_safe_example_id": self.privacy_safe_example_id,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "exception_type": self.exception_type,
        }


def alignment_diagnostic(
    example: Mapping[str, Any], *, split: str, stage: str, error: Exception,
) -> AlignmentDiagnostic:
    """Build a text-free diagnostic for an example that could not be encoded."""
    reason = str(getattr(error, "reason_code", "") or "")
    if reason not in (*UNEXPECTED_REASON_CODES, *EXPECTED_REASON_CODES):
        reason = REASON_ENTITY_OFFSET_INVARIANT_VIOLATED if isinstance(
            error, ValueError) and not isinstance(error, AlignmentError) else reason
    return AlignmentDiagnostic(
        source_dataset=str(example.get("source_dataset", "")),
        split=str(split),
        privacy_safe_example_id=privacy_safe_example_id(example.get("example_id", "")),
        stage=str(stage),
        reason_code=reason or "UNCLASSIFIED",
        exception_type=type(error).__name__,
    )


def summarize_alignment_diagnostics(
    diagnostics: Sequence[AlignmentDiagnostic],
) -> dict[str, Any]:
    """Split diagnostics into blocking failures and tracked governed exclusions."""
    unexpected = [d for d in diagnostics if not d.expected]
    expected = [d for d in diagnostics if d.expected]
    by_reason: dict[str, int] = {}
    for diagnostic in diagnostics:
        by_reason[diagnostic.reason_code] = by_reason.get(diagnostic.reason_code, 0) + 1
    return {
        # Blocks full-training readiness (the validator requires this to be 0).
        "unalignable_example_count": len(unexpected),
        # Deterministic, tracked governance decisions; never blocks.
        "governed_exclusion_count": len(expected),
        "unalignable_examples": [d.as_dict() for d in unexpected],
        "governed_exclusions": [d.as_dict() for d in expected],
        "reason_code_counts": dict(sorted(by_reason.items())),
    }


@dataclass(frozen=True, slots=True)
class AlignmentPreflightReport:
    """Whole-corpus alignment result, computed WITHOUT any model.

    The S1 smoke only ever exercised a dozen examples, so alignment defects kept
    surfacing example by example. This report is the acceptance gate instead: it
    covers every supervised example of every split and must be clean before a
    long GPU run is allowed to start.
    """

    examples_considered: int
    aligned_example_count: int
    unalignable_example_count: int
    governed_exclusion_count: int
    equivalence_checked_count: int
    equivalence_failure_count: int
    by_split: dict[str, dict[str, int]]
    by_source: dict[str, dict[str, int]]
    reason_code_counts: dict[str, int]
    stage_counts: dict[str, int]
    unalignable_examples: list[dict[str, str]]
    governed_exclusions: list[dict[str, str]]

    @property
    def counters_reconciled(self) -> bool:
        return (
            self.aligned_example_count
            + self.unalignable_example_count
            + self.governed_exclusion_count
        ) == self.examples_considered

    @property
    def passed(self) -> bool:
        """Zero UNEXPECTED failures, and the books balance.

        Governed exclusions are tracked decisions and do not block; an unexpected
        alignment or equivalence failure always does.
        """
        return (
            self.unalignable_example_count == 0
            and self.equivalence_failure_count == 0
            and self.counters_reconciled
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "alignment_preflight_passed": self.passed,
            "counters_reconciled": self.counters_reconciled,
            "examples_considered": self.examples_considered,
            "aligned_example_count": self.aligned_example_count,
            "unalignable_example_count": self.unalignable_example_count,
            "governed_exclusion_count": self.governed_exclusion_count,
            "equivalence_checked_count": self.equivalence_checked_count,
            "equivalence_failure_count": self.equivalence_failure_count,
            "by_split": {k: dict(v) for k, v in sorted(self.by_split.items())},
            "by_source": {k: dict(v) for k, v in sorted(self.by_source.items())},
            "reason_code_counts": dict(sorted(self.reason_code_counts.items())),
            "stage_counts": dict(sorted(self.stage_counts.items())),
            "unalignable_examples": list(self.unalignable_examples),
            "governed_exclusions": list(self.governed_exclusions),
        }


def run_alignment_preflight(
    splits: Mapping[str, Iterable[Mapping[str, Any]]], *,
    encode: Callable[[Mapping[str, Any]], Any],
    verify_equivalence: Callable[[Mapping[str, Any]], None] | None = None,
    governed_exclusions: Mapping[str, Any] | None = None,
    supervised: Callable[[Mapping[str, Any]], bool] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[AlignmentPreflightReport, dict[str, list[Any]]]:
    """Align every example of every split, with no model forward or backward.

    ``encode`` performs segmentation, character alignment and mention encoding and
    raises on failure; ``verify_equivalence`` re-checks per-word tokenization
    against whole-sentence tokenization. Both are supplied by the caller so this
    stays free of Torch and Transformers.

    Returns the report together with the encoded features per split, so the
    caller does not have to encode the corpus a second time.
    """
    exclusions = dict(governed_exclusions or {})
    diagnostics: list[AlignmentDiagnostic] = []
    features: dict[str, list[Any]] = {}
    by_split: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, int]] = {}
    considered = aligned = equivalence_checked = equivalence_failures = 0

    def bump(table: dict[str, dict[str, int]], key: str, field: str) -> None:
        table.setdefault(key, {"considered": 0, "aligned": 0,
                               "unalignable": 0, "governed_exclusion": 0})[field] += 1

    for split, rows in splits.items():
        features[split] = []
        for index, row in enumerate(rows):
            if supervised is not None and not supervised(row):
                continue
            source = str(row.get("source_dataset", ""))
            considered += 1
            bump(by_split, split, "considered")
            bump(by_source, source, "considered")
            if progress is not None and considered % 2000 == 0:
                progress(split, index, considered)

            if privacy_safe_example_id(row.get("example_id", "")) in exclusions:
                diagnostics.append(governed_exclusion_diagnostic(row, split=split))
                bump(by_split, split, "governed_exclusion")
                bump(by_source, source, "governed_exclusion")
                continue
            try:
                feature = encode(row)
            except (AlignmentError, ValueError) as exc:
                diagnostics.append(alignment_diagnostic(
                    row, split=split, stage=STAGE_SUBTOKEN_ENCODING, error=exc))
                bump(by_split, split, "unalignable")
                bump(by_source, source, "unalignable")
                continue
            if verify_equivalence is not None:
                try:
                    verify_equivalence(row)
                except (AlignmentError, ValueError) as exc:
                    equivalence_failures += 1
                    diagnostics.append(alignment_diagnostic(
                        row, split=split, stage=STAGE_TOKENIZER_EQUIVALENCE, error=exc))
                    bump(by_split, split, "unalignable")
                    bump(by_source, source, "unalignable")
                    continue
                equivalence_checked += 1
            aligned += 1
            bump(by_split, split, "aligned")
            bump(by_source, source, "aligned")
            features[split].append(feature)

    summary = summarize_alignment_diagnostics(diagnostics)
    stage_counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        stage_counts[diagnostic.stage] = stage_counts.get(diagnostic.stage, 0) + 1
    report = AlignmentPreflightReport(
        examples_considered=considered,
        aligned_example_count=aligned,
        unalignable_example_count=summary["unalignable_example_count"],
        governed_exclusion_count=summary["governed_exclusion_count"],
        equivalence_checked_count=equivalence_checked,
        equivalence_failure_count=equivalence_failures,
        by_split=by_split,
        by_source=by_source,
        reason_code_counts=summary["reason_code_counts"],
        stage_counts=stage_counts,
        unalignable_examples=summary["unalignable_examples"],
        governed_exclusions=summary["governed_exclusions"],
    )
    return report, features


def load_governed_exclusions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the tracked, deterministic governed-exclusion policy.

    Keyed by privacy-safe example id. An example may only be skipped without
    blocking readiness when it is listed here with a recorded justification —
    never silently. An empty policy (the current state) means every failure is
    unexpected and blocks.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError("governed exclusion policy must be a mapping")
    entries: dict[str, dict[str, Any]] = {}
    for entry in doc.get("exclusions") or []:
        handle = str(entry.get("privacy_safe_example_id", "")).strip()
        justification = str(entry.get("justification", "")).strip()
        if not handle or not justification:
            raise ValueError(
                f"governed exclusion needs privacy_safe_example_id and justification: {entry}")
        entries[handle] = dict(entry)
    return entries


def governed_exclusion_diagnostic(
    example: Mapping[str, Any], *, split: str,
) -> AlignmentDiagnostic:
    """Diagnostic for an example skipped by the tracked exclusion policy."""
    return AlignmentDiagnostic(
        source_dataset=str(example.get("source_dataset", "")),
        split=str(split),
        privacy_safe_example_id=privacy_safe_example_id(example.get("example_id", "")),
        stage=STAGE_GOVERNED_EXCLUSION,
        reason_code=REASON_GOVERNED_EXCLUSION,
        exception_type="",
    )


@dataclass(frozen=True, slots=True)
class SmokeArtifactPaths:
    """Where the smoke rerun writes, and what it must never touch.

    Every earlier artifact is immutable evidence of what that run actually
    produced, so each rerun claims its own versioned directory.
    """

    artifact_version: str
    artifact_dir: str
    previous_artifacts: tuple[tuple[str, str, str], ...] = ()   # (version, path, status)

    @property
    def previous_artifact_dirs(self) -> tuple[str, ...]:
        return tuple(path for _, path, _ in self.previous_artifacts)

    def validate(self) -> None:
        if not self.artifact_version:
            raise ValueError("smoke output needs an explicit artifact_version")
        if not self.artifact_dir:
            raise ValueError("smoke output needs an artifact_dir")
        if not self.previous_artifacts:
            raise ValueError("smoke output must record the historical artifacts")
        current = Path(self.artifact_dir).resolve()
        for version, path, _ in self.previous_artifacts:
            if Path(path).resolve() == current:
                raise ValueError(
                    "the corrected smoke artifact_dir must differ from the historical "
                    f"{version} artifact: {self.artifact_dir}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "artifact_dir": self.artifact_dir,
            "previous_artifacts": [
                {"version": v, "path": p, "status": s} for v, p, s in self.previous_artifacts
            ],
        }


def smoke_artifact_paths_from_config(config: Mapping[str, Any]) -> SmokeArtifactPaths:
    """Read the tracked, versioned smoke output contract."""
    output = config.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("S1 smoke config missing output section")
    previous = tuple(
        (str(e.get("version", "")), str(e.get("path", "")), str(e.get("status", "")))
        for e in (output.get("previous_artifacts") or [])
    )
    paths = SmokeArtifactPaths(
        artifact_version=str(output.get("artifact_version", "")),
        artifact_dir=str(output.get("artifact_dir", "")),
        previous_artifacts=previous,
    )
    paths.validate()
    return paths


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: str | Path) -> int:
    with Path(path).open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def load_smoke_config(path: str | Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError("S1 smoke config must be a mapping")
    return dict(doc)


def expected_corpus_from_config(config: Mapping[str, Any]) -> ExpectedCorpus:
    corpus = config.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("S1 smoke config missing corpus section")
    return ExpectedCorpus(
        total_examples=int(corpus["expected_total_examples"]),
        train_examples=int(corpus["expected_train_examples"]),
        validation_examples=int(corpus["expected_validation_examples"]),
        internal_test_examples=int(corpus["expected_internal_test_examples"]),
        corpus_manifest_sha256=str(corpus["expected_corpus_manifest_sha256"]),
        train_sha256=str(corpus["expected_train_sha256"]),
        vietmed_status=str(corpus["expected_vietmed_status"]),
        cross_split_family_leakage=int(corpus["expected_cross_split_family_leakage"]),
        eval_map_approximate_entities=int(corpus["expected_eval_map_approximate_entities"]),
    )


def smoke_limits_from_config(config: Mapping[str, Any]) -> SmokeLimits:
    limits = config.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("S1 smoke config missing limits section")
    smoke_limits = SmokeLimits(
        max_train_examples=int(limits["max_train_examples"]),
        max_validation_examples=int(limits["max_validation_examples"]),
        max_train_batches=int(limits["max_train_batches"]),
        max_validation_batches=int(limits["max_validation_batches"]),
        max_optimizer_steps=int(limits["max_optimizer_steps"]),
        batch_size=int(limits["batch_size"]),
        max_sequence_length=int(limits["max_sequence_length"]),
        learning_rate=float(limits["learning_rate"]),
    )
    smoke_limits.validate()
    return smoke_limits


def verify_governed_corpus(
    corpus_dir: str | Path, expected: ExpectedCorpus,
) -> dict[str, Any]:
    """Fail fast unless the governed corpus matches the Audit 0020 data gate."""
    base = Path(corpus_dir)
    required = [
        base / "canonical_examples" / "public_ner_examples.jsonl",
        base / "manifests" / "corpus_manifest.json",
        base / "manifests" / "split_manifest.json",
        base / "manifests" / "annotation_coverage.json",
        base / "splits" / "train.jsonl",
        base / "splits" / "validation.jsonl",
        base / "splits" / "internal_test.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing governed corpus file(s): {missing}")

    corpus_manifest_path = base / "manifests" / "corpus_manifest.json"
    split_manifest_path = base / "manifests" / "split_manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    corpus_manifest_sha = sha256_file(corpus_manifest_path)
    train_sha = sha256_file(base / "splits" / "train.jsonl")
    canonical_lines = count_jsonl(base / "canonical_examples" / "public_ner_examples.jsonl")
    train_lines = count_jsonl(base / "splits" / "train.jsonl")
    validation_lines = count_jsonl(base / "splits" / "validation.jsonl")
    internal_test_lines = count_jsonl(base / "splits" / "internal_test.jsonl")
    if corpus_manifest_sha != expected.corpus_manifest_sha256:
        raise ValueError(
            "corpus manifest hash mismatch: "
            f"{corpus_manifest_sha} != {expected.corpus_manifest_sha256}"
        )
    if train_sha != expected.train_sha256:
        raise ValueError(f"train split hash mismatch: {train_sha} != {expected.train_sha256}")
    counts = split_manifest.get("counts", {})
    checks = {
        "total_examples": int(corpus_manifest.get("total_examples", -1)),
        "train_examples": int(counts.get("train", -1)),
        "validation_examples": int(counts.get("validation", -1)),
        "internal_test_examples": int(counts.get("internal_test", -1)),
        "vietmed_status": str(corpus_manifest.get("vietmed_status", "")),
        "cross_split_family_leakage": int(
            split_manifest.get("cross_split_family_leakage", -1)
        ),
        "eval_map_approximate_entities": int(
            split_manifest.get("eval_map_approximate_entities", -1)
        ),
    }
    expected_checks: dict[str, int | str] = {
        "total_examples": expected.total_examples,
        "train_examples": expected.train_examples,
        "validation_examples": expected.validation_examples,
        "internal_test_examples": expected.internal_test_examples,
        "vietmed_status": expected.vietmed_status,
        "cross_split_family_leakage": expected.cross_split_family_leakage,
        "eval_map_approximate_entities": expected.eval_map_approximate_entities,
    }
    mismatches = {
        key: {"got": checks[key], "expected": expected_checks[key]}
        for key in expected_checks if checks[key] != expected_checks[key]
    }
    if mismatches:
        raise ValueError(f"governed corpus mismatch: {mismatches}")
    line_checks = {
        "canonical_jsonl_lines": canonical_lines,
        "train_jsonl_lines": train_lines,
        "validation_jsonl_lines": validation_lines,
        "internal_test_jsonl_lines": internal_test_lines,
    }
    expected_lines = {
        "canonical_jsonl_lines": expected.total_examples,
        "train_jsonl_lines": expected.train_examples,
        "validation_jsonl_lines": expected.validation_examples,
        "internal_test_jsonl_lines": expected.internal_test_examples,
    }
    line_mismatches = {
        key: {"got": line_checks[key], "expected": expected_lines[key]}
        for key in expected_lines if line_checks[key] != expected_lines[key]
    }
    if line_mismatches:
        raise ValueError(f"governed corpus JSONL count mismatch: {line_mismatches}")
    return {
        **checks,
        **line_checks,
        "total_entities": int(corpus_manifest.get("total_entities", -1)),
        "corpus_manifest_sha256": corpus_manifest_sha,
        "train_sha256": train_sha,
        "validation_sha256": split_manifest["sha256"]["validation"],
        "internal_test_sha256": split_manifest["sha256"]["internal_test"],
        "split_policy_version": split_manifest.get("split_policy_version", ""),
    }


def load_coverage(corpus_dir: str | Path) -> dict[str, SourceCoverage]:
    path = Path(corpus_dir) / "manifests" / "annotation_coverage.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    coverage: dict[str, SourceCoverage] = {}
    for source, fields in raw.items():
        coverage[source] = SourceCoverage(
            source_dataset=source,
            span=bool(fields.get("boundary")),
            entity_type=bool(fields.get("entity_type")),
            assertions=bool(fields.get("assertions")),
            icd_candidates=bool(fields.get("icd_candidates")),
            rxnorm_candidates=bool(fields.get("rxnorm_candidates")),
            test_result_pairing=bool(fields.get("test_result_pairing")),
            patient_context=bool(fields.get("patient_context")),
            relations=bool(fields.get("relations")),
        )
    return coverage


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("canonical JSONL row must be an object")
                yield row


def select_deterministic_examples(
    split_path: str | Path, *, limit: int, seed: int, require_entities: bool,
    source_dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Return a fixed tiny subset without depending on file order alone."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    for row in iter_jsonl(split_path):
        if source_dataset is not None and row.get("source_dataset") != source_dataset:
            continue
        if require_entities and not row.get("entities"):
            continue
        key = f"{seed}:{row.get('example_id', '')}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        candidates.append((digest, row))
    candidates.sort(key=lambda item: item[0])
    chosen = [row for _, row in candidates[:limit]]
    if len(chosen) < limit:
        raise ValueError(f"only selected {len(chosen)} example(s), expected {limit}")
    return chosen


def loss_mask_for_example(
    example: Mapping[str, Any], coverage_by_source: Mapping[str, SourceCoverage],
) -> dict[str, bool]:
    source = str(example.get("source_dataset", ""))
    coverage = coverage_by_source.get(source)
    if coverage is None:
        raise ValueError(f"missing annotation coverage for source {source!r}")
    return example_loss_mask(dict(example), coverage)


def verify_vietmed_train_only_masks(
    examples: Sequence[Mapping[str, Any]], coverage_by_source: Mapping[str, SourceCoverage],
) -> dict[str, int]:
    checked_examples = 0
    checked_entities = 0
    for example in examples:
        if example.get("source_dataset") != "vietmed_ner":
            continue
        checked_examples += 1
        mask = loss_mask_for_example(example, coverage_by_source)
        if mask != {
            "span": True,
            "entity_type": True,
            "assertions": False,
            "icd_candidates": False,
            "rxnorm_candidates": False,
        }:
            raise ValueError(f"unexpected VietMed loss mask: {mask}")
        for entity in example.get("entities", []):
            if entity.get("target_type") != "MEDICATION":
                raise ValueError("VietMed entity target_type must be MEDICATION")
            if entity.get("mapping_status") != "MAP_APPROXIMATE":
                raise ValueError("VietMed entity mapping_status must be MAP_APPROXIMATE")
            checked_entities += 1
    return {"vietmed_examples": checked_examples, "vietmed_entities": checked_entities}


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def encode_mention_example_slow(
    example: Mapping[str, Any], tokenizer: Any, *,
    coverage_by_source: Mapping[str, SourceCoverage],
    max_length: int,
    segmented_text: str | None = None,
) -> dict[str, Any]:
    """Encode one example with a SLOW tokenizer (PhoBERT/ViHealthBERT-Word).

    ``PhobertTokenizer`` has no fast implementation, so ``offset_mapping`` is
    unavailable. Character spans are reconstructed by
    :mod:`mednorm_vi.training.phobert_alignment`: segmented words are mapped back
    to original half-open spans, then every BPE piece inherits its word's span.
    Labels are then assigned by the SAME character-overlap rule used by the fast
    path, so supervision semantics are unchanged.

    ``segmented_text`` is the RDRSegmenter output (underscore-joined syllables).
    When omitted, the original text is treated as already whitespace-segmented.
    When word segmentation merges across a gold entity boundary, the straddling
    word and every subtoken of the entities it touches are masked out and counted
    (:data:`BOUNDARY_MERGE_POLICY`); the rest of the example keeps its
    supervision. No token is ever given an incompatible label.
    """
    text = str(example.get("text", ""))
    entities = [
        (int(e["start"]), int(e["end"])) for e in example.get("entities", []) or []
    ]
    words = map_segmented_words(text, segmented_text_to_words(segmented_text or text))
    straddling_words, boundary_merged_entities = entities_touched_by_boundary_merge(
        words, entities)
    aligned = align_subtokens(
        words, tokenizer, max_length=max_length,
        cls_token_id=getattr(tokenizer, "cls_token_id", None),
        sep_token_id=getattr(tokenizer, "sep_token_id", None))
    offsets: list[tuple[int, int]] = [
        (0, 0) if piece.is_special else (piece.original_start, piece.original_end)
        for piece in aligned.subtokens
    ]
    input_ids = [piece.token_id for piece in aligned.subtokens]
    encoded = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "offset_mapping": offsets,
    }
    feature = _labels_from_offsets(example, encoded, text=text,
                                   coverage_by_source=coverage_by_source)

    # Partial-truncation policy: an entity whose supervised subtokens only
    # partially survived must NOT keep partial labels. Mask every retained
    # subtoken of that entity (labels zeroed, label_mask 0) and count it.
    report = classify_truncated_entities(words, aligned, entities)
    for entity_index in report.partially_truncated:
        ent_start, ent_end = entities[entity_index]
        for position, piece in enumerate(aligned.subtokens):
            if piece.is_special:
                continue
            if piece.original_start < ent_end and ent_start < piece.original_end:
                feature["labels"][position] = [0 for _ in ENTITY_TYPE_ORDER]
                feature["label_mask"][position] = 0

    # Boundary-merge policy: mask the straddling words and every subtoken of the
    # entities they made ambiguous, rather than discarding the whole example.
    straddling = set(straddling_words)
    for position, piece in enumerate(aligned.subtokens):
        if piece.is_special:
            continue
        ambiguous = piece.source_word_index in straddling
        if not ambiguous:
            for entity_index in boundary_merged_entities:
                ent_start, ent_end = entities[entity_index]
                if piece.original_start < ent_end and ent_start < piece.original_end:
                    ambiguous = True
                    break
        if ambiguous:
            feature["labels"][position] = [0 for _ in ENTITY_TYPE_ORDER]
            feature["label_mask"][position] = 0

    feature["boundary_merge_masked_word_count"] = len(straddling_words)
    feature["boundary_merge_affected_entity_count"] = len(boundary_merged_entities)
    feature["truncated"] = aligned.truncated
    feature["fully_dropped_entity_count"] = report.fully_dropped_count
    feature["partially_truncated_entity_count"] = report.partially_truncated_count
    feature["truncated_entity_count"] = report.truncated_entity_count
    return feature


def encode_mention_example(
    example: Mapping[str, Any], tokenizer: Any, *,
    coverage_by_source: Mapping[str, SourceCoverage],
    max_length: int,
) -> dict[str, Any]:
    """Tokenize one example and build token-level multi-type labels.

    Requires a tokenizer that returns ``offset_mapping`` (a fast tokenizer or a
    test double). For the slow PhoBERT/ViHealthBERT-Word path used by S1 on
    Colab, call :func:`encode_mention_example_slow` instead.
    """
    text = str(example.get("text", ""))
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return _labels_from_offsets(example, encoded, text=text,
                                coverage_by_source=coverage_by_source)


def _labels_from_offsets(
    example: Mapping[str, Any], encoded: Mapping[str, Any], *, text: str,
    coverage_by_source: Mapping[str, SourceCoverage],
) -> dict[str, Any]:
    """Shared label construction from (start, end) offsets — one rule for both paths."""
    offsets = [tuple(v) for v in encoded["offset_mapping"]]
    labels = [[0 for _ in ENTITY_TYPE_ORDER] for _ in offsets]
    label_mask = [0 for _ in offsets]
    mask = loss_mask_for_example(example, coverage_by_source)
    if mask["span"] and mask["entity_type"]:
        for i, pair in enumerate(offsets):
            start, end = int(pair[0]), int(pair[1])
            if end > start:
                label_mask[i] = 1
        for entity in example.get("entities", []):
            target_type = str(entity.get("target_type", ""))
            if target_type not in ENTITY_TYPE_TO_ID:
                raise ValueError(f"unsupported target_type {target_type!r}")
            ent_start = int(entity["start"])
            ent_end = int(entity["end"])
            if text[ent_start:ent_end] != entity.get("text"):
                raise ValueError("entity offset/text invariant failed")
            type_id = ENTITY_TYPE_TO_ID[target_type]
            for i, pair in enumerate(offsets):
                start, end = int(pair[0]), int(pair[1])
                if end > start and _overlaps(start, end, ent_start, ent_end):
                    labels[i][type_id] = 1
    return {
        "example_id": str(example.get("example_id", "")),
        "source_dataset": str(example.get("source_dataset", "")),
        "input_ids": [int(v) for v in encoded["input_ids"]],
        "attention_mask": [int(v) for v in encoded["attention_mask"]],
        "labels": labels,
        "label_mask": label_mask,
        "loss_mask": mask,
    }


def pad_encoded_features(
    features: Sequence[Mapping[str, Any]], *, pad_token_id: int,
) -> dict[str, Any]:
    if not features:
        raise ValueError("cannot collate an empty batch")
    max_len = max(len(f["input_ids"]) for f in features)
    batch_ids: list[list[int]] = []
    batch_attention: list[list[int]] = []
    batch_labels: list[list[list[int]]] = []
    batch_label_mask: list[list[int]] = []
    example_ids: list[str] = []
    for feature in features:
        length = len(feature["input_ids"])
        pad = max_len - length
        batch_ids.append(list(feature["input_ids"]) + [pad_token_id] * pad)
        batch_attention.append(list(feature["attention_mask"]) + [0] * pad)
        zero_label = [0 for _ in ENTITY_TYPE_ORDER]
        batch_labels.append(list(feature["labels"]) + [zero_label.copy() for _ in range(pad)])
        batch_label_mask.append(list(feature["label_mask"]) + [0] * pad)
        example_ids.append(str(feature["example_id"]))
    return {
        "example_ids": example_ids,
        "input_ids": batch_ids,
        "attention_mask": batch_attention,
        "labels": batch_labels,
        "label_mask": batch_label_mask,
    }


def preparation_digest(features: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(features, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ENTITY_TYPE_ORDER",
    "ENTITY_TYPE_TO_ID",
    "AlignmentDiagnostic",
    "AlignmentPreflightReport",
    "run_alignment_preflight",
    "alignment_diagnostic",
    "governed_exclusion_diagnostic",
    "load_governed_exclusions",
    "privacy_safe_example_id",
    "summarize_alignment_diagnostics",
    "ExpectedCorpus",
    "SmokeArtifactPaths",
    "SmokeLimits",
    "smoke_artifact_paths_from_config",
    "count_jsonl",
    "encode_mention_example",
    "expected_corpus_from_config",
    "iter_jsonl",
    "load_coverage",
    "load_smoke_config",
    "loss_mask_for_example",
    "pad_encoded_features",
    "preparation_digest",
    "select_deterministic_examples",
    "sha256_file",
    "smoke_limits_from_config",
    "verify_governed_corpus",
    "verify_vietmed_train_only_masks",
]
