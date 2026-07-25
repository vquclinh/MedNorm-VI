"""S1 mention first-run smoke contracts.

Pure helpers for the Colab smoke notebook. They validate the governed corpus,
prepare tiny deterministic train/validation subsets, and build token-level
span/type labels plus loss masks without importing Torch or Transformers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..data_engine.annotation_coverage import SourceCoverage, example_loss_mask
from ..schemas.constants import ENTITY_TYPES

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


def encode_mention_example(
    example: Mapping[str, Any], tokenizer: Any, *,
    coverage_by_source: Mapping[str, SourceCoverage],
    max_length: int,
) -> dict[str, Any]:
    """Tokenize one example and build token-level multi-type labels.

    The tokenizer must be a fast tokenizer or test double returning
    ``input_ids``, ``attention_mask``, and ``offset_mapping``.
    """
    text = str(example.get("text", ""))
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
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
    "ExpectedCorpus",
    "SmokeLimits",
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
