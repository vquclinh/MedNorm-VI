"""S1 mention first-run smoke helpers.

These tests use synthetic fixtures and pure Python helpers only. They do not
import Torch/Transformers, download models, or run training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mednorm_vi.training.s1_mention_smoke import (
    ENTITY_TYPE_ORDER,
    ExpectedCorpus,
    encode_mention_example,
    expected_corpus_from_config,
    load_coverage,
    load_smoke_config,
    pad_encoded_features,
    preparation_digest,
    select_deterministic_examples,
    sha256_file,
    smoke_limits_from_config,
    verify_governed_corpus,
    verify_vietmed_train_only_masks,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "training" / "s1_mention_first_run_smoke.yaml"
REAL_CORPUS = REPO / "data" / "derived" / "training_corpora" / "mednorm_vi_training_v1"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _example(example_id: str, *, source: str = "vimedner") -> dict[str, Any]:
    text = "benh a"
    return {
        "example_id": example_id,
        "document_id": example_id,
        "text": text,
        "source_dataset": source,
        "entities": [{
            "text": "benh",
            "start": 0,
            "end": 4,
            "target_type": "DIAGNOSIS" if source != "vietmed_ner" else "MEDICATION",
            "source_label": "ten_benh" if source != "vietmed_ner" else "DRUGCHEMICAL",
            "mapping_status": "MAP_EXACT" if source != "vietmed_ner" else "MAP_APPROXIMATE",
        }],
    }


def _tiny_corpus(root: Path) -> ExpectedCorpus:
    train_rows = [_example("train:1"), _example("train:2", source="vietmed_ner")]
    validation_rows = [_example("validation:1")]
    internal_rows = [_example("internal_test:1")]
    canonical = [*train_rows, *validation_rows, *internal_rows]
    _write_jsonl(root / "canonical_examples" / "public_ner_examples.jsonl", canonical)
    _write_jsonl(root / "splits" / "train.jsonl", train_rows)
    _write_jsonl(root / "splits" / "validation.jsonl", validation_rows)
    _write_jsonl(root / "splits" / "internal_test.jsonl", internal_rows)
    corpus_manifest = {
        "schema_version": "canonical_example_v1",
        "total_examples": 4,
        "total_entities": 4,
        "vietmed_status": "included_from_artifacts",
    }
    _write_json(root / "manifests" / "corpus_manifest.json", corpus_manifest)
    _write_json(root / "manifests" / "split_manifest.json", {
        "counts": {"train": 2, "validation": 1, "internal_test": 1},
        "cross_split_family_leakage": 0,
        "eval_map_approximate_entities": 0,
        "sha256": {
            "train": sha256_file(root / "splits" / "train.jsonl"),
            "validation": sha256_file(root / "splits" / "validation.jsonl"),
            "internal_test": sha256_file(root / "splits" / "internal_test.jsonl"),
        },
        "split_policy_version": "split-policy-v2-duplicate-family",
    })
    _write_json(root / "manifests" / "annotation_coverage.json", {
        "vimedner": {
            "boundary": True,
            "entity_type": True,
            "assertions": False,
            "icd_candidates": False,
            "rxnorm_candidates": False,
        },
        "vietmed_ner": {
            "boundary": True,
            "entity_type": True,
            "assertions": False,
            "icd_candidates": False,
            "rxnorm_candidates": False,
        },
    })
    return ExpectedCorpus(
        total_examples=4,
        train_examples=2,
        validation_examples=1,
        internal_test_examples=1,
        corpus_manifest_sha256=sha256_file(root / "manifests" / "corpus_manifest.json"),
        train_sha256=sha256_file(root / "splits" / "train.jsonl"),
        vietmed_status="included_from_artifacts",
        cross_split_family_leakage=0,
        eval_map_approximate_entities=0,
    )


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int] | list[tuple[int, int]]]:
        parts = text.split()
        offsets: list[tuple[int, int]] = [(0, 0)]
        pos = 0
        ids = [101]
        for i, part in enumerate(parts, 1):
            start = text.index(part, pos)
            end = start + len(part)
            offsets.append((start, end))
            ids.append(100 + i)
            pos = end
        offsets.append((0, 0))
        ids.append(102)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def test_s1_smoke_config_is_strictly_bounded() -> None:
    cfg = load_smoke_config(CONFIG)
    expected = expected_corpus_from_config(cfg)
    limits = smoke_limits_from_config(cfg)
    assert expected.total_examples == 35916
    assert expected.train_examples == 33826
    assert expected.validation_examples == 1045
    assert expected.internal_test_examples == 1045
    assert expected.corpus_manifest_sha256 == (
        "a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3"
    )
    assert expected.train_sha256 == (
        "892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a"
    )
    assert limits.max_optimizer_steps == 1
    assert limits.max_train_batches <= 2
    assert limits.max_train_examples == 8


@pytest.mark.skipif(not REAL_CORPUS.is_dir(), reason="ignored governed corpus not present")
def test_real_governed_corpus_matches_s1_gate_when_present() -> None:
    cfg = load_smoke_config(CONFIG)
    report = verify_governed_corpus(REAL_CORPUS, expected_corpus_from_config(cfg))
    assert report["total_examples"] == 35916
    assert report["train_examples"] == 33826
    assert report["validation_examples"] == 1045
    assert report["internal_test_examples"] == 1045


def test_governed_corpus_verifier_rejects_count_mismatch(tmp_path: Path) -> None:
    expected = _tiny_corpus(tmp_path)
    bad = ExpectedCorpus(
        total_examples=expected.total_examples,
        train_examples=3,
        validation_examples=expected.validation_examples,
        internal_test_examples=expected.internal_test_examples,
        corpus_manifest_sha256=expected.corpus_manifest_sha256,
        train_sha256=expected.train_sha256,
        vietmed_status=expected.vietmed_status,
        cross_split_family_leakage=expected.cross_split_family_leakage,
        eval_map_approximate_entities=expected.eval_map_approximate_entities,
    )
    with pytest.raises(ValueError, match="governed corpus mismatch"):
        verify_governed_corpus(tmp_path, bad)


def test_governed_corpus_verifier_rejects_train_hash_mismatch(tmp_path: Path) -> None:
    expected = _tiny_corpus(tmp_path)
    train_path = tmp_path / "splits" / "train.jsonl"
    rows = train_path.read_text(encoding="utf-8").splitlines()
    train_path.write_text(rows[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train split hash mismatch"):
        verify_governed_corpus(tmp_path, expected)


def test_governed_corpus_verifier_rejects_actual_jsonl_count_mismatch(tmp_path: Path) -> None:
    expected = _tiny_corpus(tmp_path)
    train_path = tmp_path / "splits" / "train.jsonl"
    rows = train_path.read_text(encoding="utf-8").splitlines()
    train_path.write_text(rows[0] + "\n", encoding="utf-8")
    expected = ExpectedCorpus(
        total_examples=expected.total_examples,
        train_examples=expected.train_examples,
        validation_examples=expected.validation_examples,
        internal_test_examples=expected.internal_test_examples,
        corpus_manifest_sha256=expected.corpus_manifest_sha256,
        train_sha256=sha256_file(train_path),
        vietmed_status=expected.vietmed_status,
        cross_split_family_leakage=expected.cross_split_family_leakage,
        eval_map_approximate_entities=expected.eval_map_approximate_entities,
    )
    with pytest.raises(ValueError, match="JSONL count mismatch"):
        verify_governed_corpus(tmp_path, expected)


def test_deterministic_selection_and_digest(tmp_path: Path) -> None:
    expected = _tiny_corpus(tmp_path)
    verify_governed_corpus(tmp_path, expected)
    a = select_deterministic_examples(
        tmp_path / "splits" / "train.jsonl",
        limit=2,
        seed=20260723,
        require_entities=True,
    )
    b = select_deterministic_examples(
        tmp_path / "splits" / "train.jsonl",
        limit=2,
        seed=20260723,
        require_entities=True,
    )
    assert a == b
    assert preparation_digest(a) == preparation_digest(b)
    only_vietmed = select_deterministic_examples(
        tmp_path / "splits" / "train.jsonl",
        limit=1,
        seed=20260723,
        require_entities=True,
        source_dataset="vietmed_ner",
    )
    assert only_vietmed[0]["source_dataset"] == "vietmed_ner"


def test_vietmed_approximate_entities_have_span_type_only_masks(tmp_path: Path) -> None:
    _tiny_corpus(tmp_path)
    coverage = load_coverage(tmp_path)
    train = list(select_deterministic_examples(
        tmp_path / "splits" / "train.jsonl",
        limit=2,
        seed=20260723,
        require_entities=True,
    ))
    report = verify_vietmed_train_only_masks(train, coverage)
    assert report == {"vietmed_examples": 1, "vietmed_entities": 1}


def test_encode_and_pad_mention_features(tmp_path: Path) -> None:
    _tiny_corpus(tmp_path)
    coverage = load_coverage(tmp_path)
    example = _example("x")
    encoded = encode_mention_example(
        example,
        FakeTokenizer(),
        coverage_by_source=coverage,
        max_length=32,
    )
    diagnosis_id = ENTITY_TYPE_ORDER.index("DIAGNOSIS")
    assert encoded["label_mask"] == [0, 1, 1, 0]
    assert encoded["labels"][1][diagnosis_id] == 1
    batch = pad_encoded_features([encoded, encoded], pad_token_id=0)
    assert len(batch["input_ids"]) == 2
    assert len(batch["input_ids"][0]) == len(batch["labels"][0]) == len(batch["label_mask"][0])
