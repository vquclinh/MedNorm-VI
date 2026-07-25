"""Deterministic VietMed-NER Parquet adapter (Colab CPU preprocessing).

VietMed-NER ships as Hugging Face Parquet with columns
``words``/``tags``/``labels``/``text``/``audio``/``duration``. This module converts
the BIO ``labels`` into canonical half-open entity spans, applies the versioned v1
mapping (``DRUGCHEMICAL -> THUỐC`` MAP_APPROXIMATE), classifies malformed rows, and
writes clearly named derived artifacts. **Audio is never read or emitted.**

The Parquet READ step needs ``pyarrow`` (Colab only) and is isolated in
:func:`read_vietmed_parquet`. All CONVERSION logic is pure Python and testable
without pyarrow. Deterministic across runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..schemas.constants import ENTITY_TYPES, TYPE_BY_ORGANIZER_LABEL
from .public_ner import _token_offsets

ADAPTER_VERSION = "vietmed-adapter-v1"
SCHEMA_VERSION = "canonical_example_v1"
DATASET_ID = "vietmed_ner"
SOURCE_REVISION = "e3d0393c733858402a7c04228f45d351d2ce6d8f"
DEFAULT_MAPPING = "configs/resources/label_mappings/public_ner/vietmed_ner_v1.yaml"

# Row repair classification.
ACCEPT_AS_IS = "ACCEPT_AS_IS"
DETERMINISTIC_REPAIR = "DETERMINISTIC_REPAIR"
EXCLUDE = "EXCLUDE"
HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"

# Column resolution: candidate names for the word column and the BIO-tag column.
# The BIO-tag column must hold BIO *strings* (O / B-* / I-*); an int-id column such
# as VietMed's ``tags`` is NOT a BIO-string column and is ignored for tag resolution.
WORD_COLUMN_CANDIDATES = ("words", "tokens")
BIO_TAG_COLUMN_CANDIDATES = ("labels", "ner_tags", "tags")


class VietMedSchemaError(ValueError):
    """Raised when the word or BIO-tag column cannot be resolved unambiguously."""


def _is_string_list_type(type_str: str) -> bool:
    t = type_str.lower()
    return "list" in t and "string" in t


def resolve_bio_columns(column_types: dict[str, str]) -> tuple[str, str]:
    """Resolve exactly one word column and one BIO-tag (string) column by name+type.

    ``column_types`` maps column name -> a type string (e.g. ``"list<element: string>"``,
    ``"list<element: int64>"``). Fails (``VietMedSchemaError``) if either the word
    column or the BIO-string tag column is missing or ambiguous. VietMed's
    ``tags`` (int ids) is correctly skipped in favour of the string ``labels``.
    """
    word_hits = [c for c in WORD_COLUMN_CANDIDATES
                 if c in column_types and _is_string_list_type(column_types[c])]
    if len(word_hits) == 0:
        raise VietMedSchemaError(
            f"no word column among {WORD_COLUMN_CANDIDATES} in {sorted(column_types)}")
    if len(word_hits) > 1:
        raise VietMedSchemaError(f"ambiguous word column: {word_hits}")
    tag_hits = [c for c in BIO_TAG_COLUMN_CANDIDATES
                if c in column_types and _is_string_list_type(column_types[c])]
    if len(tag_hits) == 0:
        raise VietMedSchemaError(
            f"no BIO-string tag column among {BIO_TAG_COLUMN_CANDIDATES} "
            f"in {sorted(column_types)} (int-id 'tags' is not a BIO-string column)")
    if len(tag_hits) > 1:
        raise VietMedSchemaError(f"ambiguous BIO-string tag column: {tag_hits}")
    return word_hits[0], tag_hits[0]


@dataclass(frozen=True, slots=True)
class VietMedMapping:
    type_mapping: dict[str, str]                 # source label -> internal ENTITY_TYPE
    status_by_label: dict[str, dict[str, Any]]   # source label -> {classification, ...}
    version: int
    config_hash: str


def _internal_type(target: str) -> str:
    if target in ENTITY_TYPES:
        return target
    return TYPE_BY_ORGANIZER_LABEL.get(target, "")


def load_vietmed_mapping(path: str | Path = DEFAULT_MAPPING) -> VietMedMapping:
    raw = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(raw) or {}
    type_mapping: dict[str, str] = {}
    status: dict[str, dict[str, Any]] = {}
    for m in doc.get("mappings", []) or []:
        src = str(m["source_label"])
        internal = _internal_type(str(m.get("target", "")))
        cls = str(m.get("classification", ""))
        status[src] = {"classification": cls,
                       "exclude_from_validation": bool(m.get("exclude_from_validation", False))}
        if internal in ENTITY_TYPES and cls in {"MAP_EXACT", "MAP_APPROXIMATE"} \
                and bool(m.get("allowed_for_type_training", False)):
            type_mapping[src] = internal
    return VietMedMapping(
        type_mapping=type_mapping, status_by_label=status,
        version=int(doc.get("version", 1)),
        config_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest())


def _bio_spans(words: list[str], labels: list[str]) -> tuple[list[tuple[int, int, str]], bool]:
    """Return (start_char, end_char, source_label) spans from BIO ``labels`` over
    joined-token text, plus a flag if an orphan ``I-`` tag was repaired (ignored).

    Mirrors the repository's BIO->span convention: a run opens on ``B-`` and closes
    on the next ``O``/``B-``; ``I-`` tags without an open run are orphans and are
    deterministically ignored (a repair). Half-open ``[start, end)`` offsets.
    """
    offsets = _token_offsets(words)
    spans: list[tuple[int, int, str]] = []
    active: str | None = None
    active_start = 0
    repaired = False
    for idx, tag in enumerate((*labels, "O")):
        is_b = tag.startswith("B-")
        is_i = tag.startswith("I-")
        if active is not None and (tag == "O" or is_b):
            spans.append((offsets[active_start][0], offsets[idx - 1][1], active))
            active = None
        if is_b:
            active = tag.split("-", 1)[1]
            active_start = idx
        elif is_i and active is None:
            repaired = True                       # orphan I- -> ignored
    return spans, repaired


def _row_example(words: list[str], labels: list[str], *, split: str, record_id: str,
                 mapping: VietMedMapping) -> dict[str, Any]:
    text = " ".join(words)
    spans, _ = _bio_spans(words, labels)
    doc_id = f"{DATASET_ID}:{split}:{record_id}"
    entities = []
    for start, end, src_label in spans:
        internal = mapping.type_mapping.get(src_label)
        if internal is None:
            continue                              # non-concrete labels dropped
        surface = text[start:end]                 # half-open; validated in convert_rows
        cls = mapping.status_by_label.get(src_label, {}).get("classification", "")
        entities.append({
            "text": surface, "start": start, "end": end, "target_type": internal,
            "source_label": src_label, "mapping_status": cls,
            "mapping_confidence": 0.9 if cls == "MAP_EXACT" else 0.7,
            "provenance": f"{DATASET_ID}:{split}", "quality_flags": [],
        })
    return {
        "example_id": doc_id, "document_id": doc_id, "text": text, "entities": entities,
        "source_dataset": DATASET_ID, "source_revision": SOURCE_REVISION,
        "source_split": split, "source_record_id": record_id,
        "provenance": ADAPTER_VERSION, "quality_flags": [],
        "transformation_version": SCHEMA_VERSION,
    }


def classify_row(words: Any, labels: Any) -> str:
    """Deterministic per-row policy (authoritative):

        empty row            -> EXCLUDE
        words/tags mismatch  -> HUMAN_REVIEW_REQUIRED  (fails the run; see convert_rows)
        orphan I-tag         -> DETERMINISTIC_REPAIR
        otherwise            -> ACCEPT_AS_IS
    """
    if not isinstance(words, list) or not isinstance(labels, list):
        return EXCLUDE
    if len(words) == 0 or len(labels) == 0:
        return EXCLUDE
    if len(words) != len(labels):
        return HUMAN_REVIEW_REQUIRED
    _, repaired = _bio_spans([str(w) for w in words], [str(x) for x in labels])
    return DETERMINISTIC_REPAIR if repaired else ACCEPT_AS_IS


def convert_rows(
    rows: list[dict[str, Any]], *, mapping: VietMedMapping,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert decoded VietMed rows -> canonical examples + repair/exclusion summary.

    Each row: ``{"words": [...], "labels": [...], "split": str, "record_id": str}``.
    Audio is never referenced. Deterministic.
    """
    examples: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    excluded: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    for row in rows:
        words, labels = row.get("words"), row.get("labels")
        split = str(row.get("split", ""))
        record_id = str(row.get("record_id", ""))
        decision = classify_row(words, labels)
        decisions[decision] += 1
        if decision == EXCLUDE:
            excluded.append({"record": f"{split}:{record_id}", "issue": "empty_row"})
            continue
        if decision == HUMAN_REVIEW_REQUIRED:
            review.append({"record": f"{split}:{record_id}",
                           "issue": "words_tags_length_mismatch"})
            continue  # not emitted; write_artifacts refuses when review count > 0
        assert isinstance(words, list) and isinstance(labels, list)  # narrowed by classify_row
        examples.append(_row_example([str(w) for w in words], [str(x) for x in labels],
                                     split=split, record_id=record_id, mapping=mapping))
    off_invalid = sum(
        1 for e in examples for en in e["entities"]
        if e["text"][en["start"]:en["end"]] != en["text"]
    )
    summary = {
        "adapter_version": ADAPTER_VERSION,
        "rows_in": len(rows),
        "accepted": decisions[ACCEPT_AS_IS],
        "deterministic_repair": decisions[DETERMINISTIC_REPAIR],
        "excluded": decisions[EXCLUDE],
        "human_review_required": decisions[HUMAN_REVIEW_REQUIRED],
        "excluded_records": excluded,            # aggregate provenance; no raw clinical text
        "human_review_records": review,          # aggregate provenance; no raw clinical text
        "examples_emitted": len(examples),
        "entities_emitted": sum(len(e["entities"]) for e in examples),
        "offset_invalid": off_invalid,
    }
    return examples, summary


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_artifacts(
    out_dir: str | Path, examples: list[dict[str, Any]], summary: dict[str, Any], *,
    mapping: VietMedMapping, source_hashes: dict[str, str], repo_commit: str = "",
    resolved_columns: dict[str, str] | None = None,
    notebook: str = "notebooks/MedNorm_Data_VietMed_Preprocess.ipynb",
) -> dict[str, Any]:
    """Write the clearly-named derived artifact tree.

    Fails fast (no artifact written) when ``offset_invalid != 0`` or when any row was
    classified HUMAN_REVIEW_REQUIRED (words/tags length mismatch) — a valid artifact
    must never be produced while such a mismatch exists.
    """
    if summary["offset_invalid"] != 0:
        raise ValueError(f"offset_invalid must be 0, got {summary['offset_invalid']}")
    if summary.get("human_review_required", 0) != 0:
        raise ValueError(
            f"{summary['human_review_required']} row(s) need HUMAN_REVIEW_REQUIRED "
            "(words/tags length mismatch); refusing to write a partial artifact")
    out = Path(out_dir)
    (out / "canonical_examples").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "quality").mkdir(parents=True, exist_ok=True)

    jsonl = "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in examples)
    (out / "canonical_examples" / "vietmed_ner_examples.jsonl").write_text(jsonl, encoding="utf-8")
    jsonl_sha = _sha256_text(jsonl)

    label_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for e in examples:
        split_counts[e["source_split"]] += 1
        for en in e["entities"]:
            label_counts[en["source_label"]] += 1
            type_counts[en["target_type"]] += 1
            status_counts[en["mapping_status"]] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION, "adapter_version": ADAPTER_VERSION,
        "dataset_id": DATASET_ID, "source_revision": SOURCE_REVISION,
        "mapping_version": mapping.version, "mapping_config_hash": mapping.config_hash,
        "source_files": source_hashes,
        "row_count": summary["rows_in"], "accepted": summary["accepted"],
        "repaired": summary["deterministic_repair"], "excluded": summary["excluded"],
        "examples": len(examples), "entities": summary["entities_emitted"],
        "per_label_counts": dict(sorted(label_counts.items())),
        "per_type_counts": dict(sorted(type_counts.items())),
        "per_mapping_status_counts": dict(sorted(status_counts.items())),
        "offset_invalid": summary["offset_invalid"],
        "examples_jsonl_sha256": jsonl_sha,
        "repo_commit": repo_commit, "creation_notebook": notebook,
        "resolved_columns": resolved_columns or {},
        "audio_excluded": True,
    }

    def _dump(rel: str, payload: Any) -> None:
        (out / rel).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _dump("manifests/vietmed_ner_preprocessing_manifest.json", manifest)
    _dump("manifests/vietmed_ner_source_hashes.json", source_hashes)
    _dump("quality/vietmed_ner_label_inventory.json",
          {"per_label_counts": dict(sorted(label_counts.items())),
           "label_inventory_hash": _sha256_text(
               json.dumps(sorted(label_counts), separators=(",", ":")))})
    _dump("quality/vietmed_ner_offset_validation.json",
          {"checked": summary["entities_emitted"], "invalid": summary["offset_invalid"]})
    _dump("quality/vietmed_ner_repair_exclusion_summary.json",
          {k: summary[k] for k in ("rows_in", "accepted", "deterministic_repair",
                                    "excluded", "human_review_required", "excluded_records")})
    _dump("quality/vietmed_ner_split_summary.json",
          {"per_split_examples": dict(sorted(split_counts.items()))})
    return manifest


def load_vietmed_artifacts(artifact_dir: str | Path) -> list[dict[str, Any]] | None:
    """Load returned VietMed canonical examples if present AND hash-valid; else None.

    Fails fast (ValueError) if the manifest is present but its recorded
    ``examples_jsonl_sha256`` does not match the JSONL on disk (corrupted artifact) —
    there is no silent fallback.
    """
    base = Path(artifact_dir)
    jsonl_path = base / "canonical_examples" / "vietmed_ner_examples.jsonl"
    manifest_path = base / "manifests" / "vietmed_ner_preprocessing_manifest.json"
    if not (jsonl_path.is_file() and manifest_path.is_file()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text = jsonl_path.read_text(encoding="utf-8")
    if _sha256_text(text) != manifest.get("examples_jsonl_sha256"):
        raise ValueError("vietmed artifact corrupted: JSONL sha256 != manifest")
    if manifest.get("offset_invalid", 1) != 0:
        raise ValueError("vietmed artifact manifest reports offset_invalid != 0")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_vietmed_parquet(data_dir: str | Path) -> list[dict[str, Any]]:  # pragma: no cover
    """Read words/labels/split/record_id from the VietMed Parquet splits (Colab).

    Requires ``pyarrow`` (not a repo runtime dependency). **Audio is never read.**
    Fails fast if no Parquet file is found or required columns are absent.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "read_vietmed_parquet requires pyarrow (install in Colab); "
            "conversion logic is available without pyarrow via convert_rows()."
        ) from exc
    base = Path(data_dir)
    splits = {"train": "train-00000-of-00001.parquet",
              "validation": "validation-00000-of-00001.parquet",
              "test": "test-00000-of-00001.parquet"}
    rows: list[dict[str, Any]] = []
    resolved: tuple[str, str] | None = None
    for split, fname in splits.items():
        path = base / fname
        if not path.is_file():
            continue
        schema = pq.read_schema(path)
        column_types = {f.name: str(f.type) for f in schema}
        word_col, tag_col = resolve_bio_columns(column_types)   # by name AND type
        if resolved is None:
            resolved = (word_col, tag_col)
        elif resolved != (word_col, tag_col):
            raise VietMedSchemaError(
                f"inconsistent resolved columns across splits: {resolved} vs {(word_col, tag_col)}")
        table = pq.read_table(path, columns=[word_col, tag_col])   # audio NEVER read
        words_col = table.column(word_col).to_pylist()
        tags_col = table.column(tag_col).to_pylist()
        for i, (w, la) in enumerate(zip(words_col, tags_col, strict=True)):
            rows.append({"words": w, "labels": la, "split": split, "record_id": f"{i:06d}"})
    if resolved is None:
        raise FileNotFoundError(f"no VietMed Parquet files found under {base}")
    return rows


__all__ = [
    "ADAPTER_VERSION", "DATASET_ID", "VietMedMapping", "VietMedSchemaError",
    "load_vietmed_mapping", "resolve_bio_columns",
    "classify_row", "convert_rows", "write_artifacts", "load_vietmed_artifacts",
    "read_vietmed_parquet",
    "WORD_COLUMN_CANDIDATES", "BIO_TAG_COLUMN_CANDIDATES",
    "ACCEPT_AS_IS", "DETERMINISTIC_REPAIR", "EXCLUDE", "HUMAN_REVIEW_REQUIRED",
]
