"""VietMed-NER adapter + governed-corpus inclusion (Audit 0018).

Pure conversion logic is tested WITHOUT pyarrow (synthetic decoded rows). The
pyarrow Parquet reader is skip-guarded. No restricted raw VietMed text is used.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mednorm_vi.data_engine.corpus_build import build_governed_corpus
from mednorm_vi.data_engine.vietmed_ner import (
    DETERMINISTIC_REPAIR,
    EXCLUDE,
    HUMAN_REVIEW_REQUIRED,
    VietMedMapping,
    VietMedSchemaError,
    classify_row,
    convert_rows,
    load_vietmed_artifacts,
    load_vietmed_mapping,
    resolve_bio_columns,
    write_artifacts,
)

REPO = Path(__file__).resolve().parents[2]
_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None

# Synthetic mapping mirroring vietmed_ner_v1.yaml's concrete map (no raw dataset text).
MAP = VietMedMapping(
    type_mapping={"DRUGCHEMICAL": "MEDICATION"},
    status_by_label={
        "DRUGCHEMICAL": {"classification": "MAP_APPROXIMATE", "exclude_from_validation": True},
        "DISEASESYMTOM": {"classification": "REVIEW_REQUIRED", "exclude_from_validation": True},
    },
    version=1, config_hash="x")


def _row(words, labels, split="train", rid="000001"):
    return {"words": words, "labels": labels, "split": split, "record_id": rid}


# --- pure conversion ----------------------------------------------------------

def test_valid_bio_conversion_and_half_open_offsets() -> None:
    rows = [_row(["uống", "paracetamol", "500mg", "mỗi", "ngày"],
                 ["O", "B-DRUGCHEMICAL", "I-DRUGCHEMICAL", "O", "O"])]
    examples, summary = convert_rows(rows, mapping=MAP)
    assert summary["offset_invalid"] == 0 and summary["accepted"] == 1
    e = examples[0]
    assert len(e["entities"]) == 1
    ent = e["entities"][0]
    assert ent["target_type"] == "MEDICATION" and ent["source_label"] == "DRUGCHEMICAL"
    assert ent["mapping_status"] == "MAP_APPROXIMATE"
    assert e["text"][ent["start"]:ent["end"]] == ent["text"] == "paracetamol 500mg"


def test_empty_row_excluded() -> None:
    assert classify_row([], []) == EXCLUDE
    _, summary = convert_rows([_row([], [])], mapping=MAP)
    assert summary["excluded"] == 1 and summary["examples_emitted"] == 0


def test_words_tags_mismatch_is_human_review_and_fails_write(tmp_path: Path) -> None:
    # Authoritative policy: length mismatch -> HUMAN_REVIEW_REQUIRED; run must fail
    # and NO artifact may be written.
    assert classify_row(["a", "b"], ["O"]) == HUMAN_REVIEW_REQUIRED
    examples, summary = convert_rows([_row(["a", "b"], ["O"])], mapping=MAP)
    assert summary["human_review_required"] == 1 and summary["examples_emitted"] == 0
    with pytest.raises(ValueError):
        write_artifacts(tmp_path, examples, summary, mapping=MAP, source_hashes={})
    assert not (tmp_path / "canonical_examples" / "vietmed_ner_examples.jsonl").exists()


# --- column resolution (words/tokens ; labels/ner_tags/tags-string) ----------

def test_resolve_columns_real_vietmed_schema_picks_string_labels() -> None:
    # words(str) + tags(int) + labels(str) -> word=words, tag=labels (int tags skipped)
    types = {"words": "list<element: string>", "tags": "list<element: int64>",
             "labels": "list<element: string>", "text": "string"}
    assert resolve_bio_columns(types) == ("words", "labels")


@pytest.mark.parametrize("types,expected", [
    ({"tokens": "list<string>", "tags": "list<string>"}, ("tokens", "tags")),
    ({"words": "list<string>", "ner_tags": "list<string>"}, ("words", "ner_tags")),
    ({"words": "list<string>", "labels": "list<string>"}, ("words", "labels")),
])
def test_resolve_columns_variants(types, expected) -> None:
    assert resolve_bio_columns(types) == expected


@pytest.mark.parametrize("types", [
    {"words": "list<string>"},                                       # no tag column
    {"labels": "list<string>"},                                      # no word column
    {"words": "list<string>", "tags": "list<int64>"},                # tags is int, not BIO-string
    {"words": "list<string>", "labels": "list<string>", "ner_tags": "list<string>"},  # ambiguous
    {"words": "list<string>", "tokens": "list<string>", "labels": "list<string>"},  # ambiguous word
])
def test_resolve_columns_fail_on_missing_or_ambiguous(types) -> None:
    with pytest.raises(VietMedSchemaError):
        resolve_bio_columns(types)


def test_orphan_i_tag_is_deterministic_repair() -> None:
    # I- with no preceding B- -> orphan -> ignored (repair); no entity emitted.
    rows = [_row(["sốt", "cao"], ["I-DRUGCHEMICAL", "O"])]
    assert classify_row(rows[0]["words"], rows[0]["labels"]) == DETERMINISTIC_REPAIR
    examples, summary = convert_rows(rows, mapping=MAP)
    assert summary["deterministic_repair"] == 1
    assert examples[0]["entities"] == []


def test_non_concrete_labels_dropped() -> None:
    rows = [_row(["ho", "khan"], ["B-DISEASESYMTOM", "I-DISEASESYMTOM"])]
    examples, _ = convert_rows(rows, mapping=MAP)
    assert examples[0]["entities"] == []            # REVIEW_REQUIRED label -> no entity


def test_split_and_record_provenance_preserved() -> None:
    examples, _ = convert_rows(
        [_row(["x"], ["O"], split="validation", rid="000042")], mapping=MAP)
    e = examples[0]
    assert e["source_split"] == "validation" and e["source_record_id"] == "000042"
    assert e["document_id"] == "vietmed_ner:validation:000042"


def test_convert_is_deterministic() -> None:
    rows = [_row(["a", "b"], ["B-DRUGCHEMICAL", "O"]),
            _row(["c"], ["O"], rid="000002")]
    a, _ = convert_rows(rows, mapping=MAP)
    b, _ = convert_rows(rows, mapping=MAP)
    assert a == b


def test_real_mapping_file_loads_concrete_drugchemical() -> None:
    m = load_vietmed_mapping(
        REPO / "configs" / "resources" / "label_mappings" / "public_ner" / "vietmed_ner_v1.yaml")
    assert m.type_mapping == {"DRUGCHEMICAL": "MEDICATION"}
    assert len(m.config_hash) == 64


# --- artifacts (audio excluded; hashes; fail-fast) ---------------------------

def test_write_and_load_artifacts_roundtrip(tmp_path: Path) -> None:
    rows = [_row(["uống", "aspirin"], ["O", "B-DRUGCHEMICAL"])]
    examples, summary = convert_rows(rows, mapping=MAP)
    manifest = write_artifacts(tmp_path, examples, summary, mapping=MAP,
                               source_hashes={"train.parquet": "abc"})
    assert manifest["audio_excluded"] is True
    # exact artifact filenames exist
    assert (tmp_path / "canonical_examples" / "vietmed_ner_examples.jsonl").is_file()
    assert (tmp_path / "manifests" / "vietmed_ner_preprocessing_manifest.json").is_file()
    for q in ("label_inventory", "offset_validation", "repair_exclusion_summary", "split_summary"):
        assert (tmp_path / "quality" / f"vietmed_ner_{q}.json").is_file()
    loaded = load_vietmed_artifacts(tmp_path)
    assert loaded == examples                        # roundtrip
    # no audio anywhere in the emitted JSONL
    jsonl = (tmp_path / "canonical_examples" / "vietmed_ner_examples.jsonl").read_text()
    assert "audio" not in jsonl


def test_load_absent_returns_none(tmp_path: Path) -> None:
    assert load_vietmed_artifacts(tmp_path) is None


def test_load_corrupt_artifact_fails_fast(tmp_path: Path) -> None:
    rows = [_row(["a"], ["B-DRUGCHEMICAL"])]
    examples, summary = convert_rows(rows, mapping=MAP)
    write_artifacts(tmp_path, examples, summary, mapping=MAP, source_hashes={})
    # tamper with the JSONL so its sha256 no longer matches the manifest
    jp = tmp_path / "canonical_examples" / "vietmed_ner_examples.jsonl"
    jp.write_text(jp.read_text() + '{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_vietmed_artifacts(tmp_path)


def test_write_fails_when_offset_invalid_nonzero(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_artifacts(tmp_path, [], {"offset_invalid": 1, "rows_in": 0, "accepted": 0,
                                       "deterministic_repair": 0, "excluded": 0,
                                       "human_review_required": 0, "excluded_records": [],
                                       "entities_emitted": 0}, mapping=MAP, source_hashes={})


# --- governed-corpus inclusion / pending -------------------------------------

def _tiny_artifact(dir_: Path) -> None:
    rows = [_row(["uống", "aspirin"], ["O", "B-DRUGCHEMICAL"], rid="000001"),
            _row(["dùng", "ibuprofen"], ["O", "B-DRUGCHEMICAL"], rid="000002")]
    examples, summary = convert_rows(rows, mapping=MAP)
    write_artifacts(dir_, examples, summary, mapping=MAP, source_hashes={"t": "h"})


_VIMEDNER_TRAIN = REPO / "data" / "external" / "public_ner" / "vimedner" / "data" / "train.txt"
_HAS_DATA = _VIMEDNER_TRAIN.is_file()


def test_builder_include_true_fails_fast_when_absent(tmp_path: Path) -> None:
    # Fails fast BEFORE reading the public datasets (absent artifact dir).
    with pytest.raises(FileNotFoundError):
        build_governed_corpus(
            out_dir=tmp_path / "o", include_vietmed=True,
            vietmed_artifact_dir=tmp_path / "missing")


@pytest.mark.skipif(not _HAS_DATA, reason="public NER data not present locally")
def test_builder_includes_vietmed_when_artifacts_present(tmp_path: Path) -> None:
    art = tmp_path / "vietmed_v1"
    _tiny_artifact(art)
    r = build_governed_corpus(out_dir=tmp_path / "o", include_vietmed=True,
                              vietmed_artifact_dir=art)
    assert r["vietmed_status"] == "included_from_artifacts"
    # VietMed MEDICATION entities are MAP_APPROXIMATE -> train only (eval stays MAP_EXACT).
    assert r["eval_map_approximate_entities"] == 0
    cm = json.loads((tmp_path / "o" / "manifests" / "corpus_manifest.json").read_text())
    assert cm["vietmed_status"] == "included_from_artifacts"


@pytest.mark.skipif(not _HAS_DATA, reason="public NER data not present locally")
def test_builder_pending_when_artifacts_absent(tmp_path: Path) -> None:
    r = build_governed_corpus(out_dir=tmp_path / "o",
                              vietmed_artifact_dir=tmp_path / "none")
    assert r["vietmed_status"] == "not_included_pending"


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed locally")
def test_parquet_reader_available_when_pyarrow_present() -> None:  # pragma: no cover
    from mednorm_vi.data_engine.vietmed_ner import read_vietmed_parquet
    assert callable(read_vietmed_parquet)
