"""Synthetic tests for Audit 0012 public NER intake governance."""

from __future__ import annotations

import json
from pathlib import Path

from mednorm_vi.data_engine.models import CanonicalDocument
from mednorm_vi.data_engine.public_ner import (
    BioExample,
    PublicDatasetGovernanceError,
    bio_examples_to_documents,
    duplicate_groups_by_hash,
    file_tree_summary,
    load_json_records,
    parse_git_revision,
    phoner_json_matches_conll,
    read_conll,
    read_parquet_metadata,
    require_training_allowed,
    validate_bio_examples,
    vimq_records_to_documents,
    vimq_span_stats,
)
from mednorm_vi.resources.models import LicenseRecord, RedistributionPolicy, SourceRecord
from mednorm_vi.resources.ner import (
    LabelMapping,
    NerDatasetManifest,
    load_ner_manifest,
    validate_ner_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def _write_minimal_parquet(path: Path) -> None:
    def varint(value: int) -> bytes:
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                return bytes(out)

    def zigzag(value: int) -> bytes:
        return varint((value << 1) ^ (value >> 63))

    def binary(value: str) -> bytes:
        raw = value.encode("utf-8")
        return varint(len(raw)) + raw

    schema_element = bytes([0x48]) + binary("schema") + b"\x00"
    row_group = bytes([0x36]) + zigzag(3) + b"\x00"
    footer = (
        bytes([0x15])
        + zigzag(2)
        + bytes([0x19, 0x1C])
        + schema_element
        + bytes([0x16])
        + zigzag(3)
        + bytes([0x19, 0x1C])
        + row_group
        + bytes([0x28])
        + binary("synthetic")
        + b"\x00"
    )
    path.write_bytes(b"PAR1" + footer + len(footer).to_bytes(4, "little") + b"PAR1")


def test_four_public_ner_manifests_validate() -> None:
    paths = [
        "phoner-covid19-public-ner.yaml",
        "vimedner-public-ner.yaml",
        "vimq-public-ner.yaml",
        "vietmed-ner-public-ner.yaml",
    ]
    for name in paths:
        manifest = load_ner_manifest(REPO / "data" / "manifests" / name)
        result = validate_ner_manifest(manifest)
        assert result.ok, (name, result.errors)
        # Governance separation (Audit 0017): the SOURCE license fact is preserved
        # (REVIEW_REQUIRED), and training permission lives in a SEPARATE training_use
        # block — the source license is never rewritten to reflect user permission.
        assert manifest.license.status == "REVIEW_REQUIRED"
        assert manifest.training_use.get("status") == "USER_ATTESTED_PERMISSION"
        assert manifest.training_use.get("internal_training_allowed") is True
        assert manifest.training_use.get("raw_redistribution_allowed") is False
        assert "ner_training" in manifest.permitted_use


def test_source_revision_parsing_and_tree_hash_determinism(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".git" / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "a.txt").write_text("alpha\n", encoding="utf-8")
    assert parse_git_revision(root) == "a" * 40
    assert file_tree_summary(root) == file_tree_summary(root)


def test_conll_json_and_bio_validation(tmp_path: Path) -> None:
    conll = tmp_path / "train.conll"
    conll.write_text("thuốc B-DRUG\nabc I-DRUG\n\nx I-DISEASE\n", encoding="utf-8")
    examples = read_conll(conll, split="train")
    result = validate_bio_examples(examples)
    assert result.entity_counts == {"DRUG": 1}
    assert result.malformed_count == 1
    jsonl = tmp_path / "train.json"
    jsonl.write_text(
        json.dumps({"words": ["thuốc", "abc"], "tags": ["B-DRUG", "I-DRUG"]})
        + "\n"
        + json.dumps({"words": ["x"], "tags": ["I-DISEASE"]})
        + "\n",
        encoding="utf-8",
    )
    loaded = load_json_records(jsonl)
    assert loaded.format == "jsonl"
    assert phoner_json_matches_conll(loaded.records, examples)


def test_malformed_json_is_counted(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"ok": true}\n{bad}\n', encoding="utf-8")
    result = load_json_records(path)
    assert result.format == "jsonl"
    assert result.malformed_count == 1
    assert len(result.records) == 1


def test_vimq_span_reader_and_canonical_offsets() -> None:
    records = (
        {
            "sentence": "uống thuốc a",
            "seq_label": [[1, 2, "drug"]],
            "sent_label": "treatment",
        },
    )
    stats = vimq_span_stats(records)
    assert stats["entity_counts"] == {"drug": 1}
    docs = vimq_records_to_documents("vimq", "train", records, {"drug": "MEDICATION"})
    assert docs[0].annotations[0].text == "thuốc a"
    assert docs[0].annotations[0].span.start == 5


def test_bio_adapter_preserves_offsets() -> None:
    examples = (BioExample("train-000001", ("a", "b"), ("B-DRUG", "I-DRUG")),)
    docs = bio_examples_to_documents("d", "train", examples, {"DRUG": "MEDICATION"})
    annotation = docs[0].annotations[0]
    assert docs[0].text[annotation.span.start : annotation.span.end] == "a b"


def test_parquet_footer_metadata_and_malformed_schema(tmp_path: Path) -> None:
    path = tmp_path / "tiny.parquet"
    _write_minimal_parquet(path)
    meta = read_parquet_metadata(path)
    assert meta.num_rows == 3
    assert meta.row_group_rows == 3
    assert meta.schema_names == ("schema",)
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    try:
        read_parquet_metadata(bad)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("malformed parquet must fail")


def test_label_mapping_completeness_and_governance_gate() -> None:
    manifest = NerDatasetManifest(
        dataset_id="d",
        title="t",
        source=SourceRecord(organization="x"),
        license=LicenseRecord(status="REVIEW_REQUIRED"),
        redistribution=RedistributionPolicy(permission="unknown"),
        original_labels=("A", "B"),
        label_mappings=(LabelMapping("A", "a", "REVIEW", mapping_status="ambiguous"),),
    )
    result = validate_ner_manifest(manifest)
    assert any(issue.code == "ner.unmapped_labels" for issue in result.errors)
    try:
        require_training_allowed(manifest)
    except PublicDatasetGovernanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("REVIEW_REQUIRED datasets must not be training-eligible")


def test_duplicate_leakage_and_ignore_policy() -> None:
    docs = {
        "train": (CanonicalDocument("a", "same text"),),
        "dev": (CanonicalDocument("b", "same   text"),),
    }
    groups = duplicate_groups_by_hash(docs)
    assert groups["cross_split"]
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/external/public_ner/**" in gitignore
