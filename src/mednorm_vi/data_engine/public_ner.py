"""Public Vietnamese NER dataset intake helpers.

These readers are deliberately deterministic and offline. They inspect and adapt
local public-NER payloads into the Data Engine contracts without downloading
data, changing raw files, or deciding legal usability.
"""

from __future__ import annotations

import hashlib
import json
import struct
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..resources.ner import NerDatasetManifest
from ..schemas.constants import ENTITY_TYPES
from ..schemas.spans import Span
from .models import CanonicalAnnotation, CanonicalDocument


@dataclass(frozen=True, slots=True)
class FileTreeSummary:
    root: str
    file_count: int
    total_bytes: int
    tree_hash_sha256: str
    zero_byte_files: tuple[str, ...] = field(default_factory=tuple)
    symlinks: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BioExample:
    example_id: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BioValidation:
    label_vocab: tuple[str, ...]
    entity_counts: dict[str, int]
    token_count: int
    malformed_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class JsonLoadResult:
    records: tuple[Any, ...]
    format: str
    malformed_count: int = 0


@dataclass(frozen=True, slots=True)
class ParquetMetadata:
    path: str
    bytes: int
    version: int
    num_rows: int
    row_group_count: int
    row_group_rows: int
    schema_names: tuple[str, ...]
    created_by: str
    footer_length: int


class PublicDatasetGovernanceError(RuntimeError):
    """Raised when a public dataset is not governed for the requested use."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_tree_summary(root: str | Path) -> FileTreeSummary:
    """Hash a local tree as sorted ``relative_path<TAB>file_sha256`` entries."""

    base = Path(root)
    entries: list[tuple[str, int, str]] = []
    zero_byte: list[str] = []
    symlinks: list[str] = []
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base).as_posix()
        if path.is_symlink():
            symlinks.append(rel)
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size == 0:
            zero_byte.append(rel)
        entries.append((rel, size, _sha256_file(path)))
    payload = "".join(f"{rel}\t{sha}\n" for rel, _, sha in entries)
    return FileTreeSummary(
        root=base.as_posix(),
        file_count=len(entries),
        total_bytes=sum(size for _, size, _ in entries),
        tree_hash_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        zero_byte_files=tuple(zero_byte),
        symlinks=tuple(symlinks),
    )


def label_inventory_hash(labels: Iterable[str]) -> str:
    payload = json.dumps(sorted(set(labels)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_git_revision(root: str | Path) -> str:
    head = Path(root) / ".git" / "HEAD"
    if not head.is_file():
        return ""
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref:"):
        ref = value.split(" ", 1)[1]
        ref_path = Path(root) / ".git" / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        packed = Path(root) / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                sha, packed_ref = line.split(" ", 1)
                if packed_ref == ref:
                    return sha
        return ""
    return value


def parse_huggingface_snapshot_id(root: str | Path) -> str:
    trees = sorted((Path(root) / ".cache" / "huggingface" / "trees").glob("*.json"))
    if trees:
        return trees[0].stem
    metadata = sorted((Path(root) / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    for path in metadata:
        first = path.read_text(encoding="utf-8").splitlines()[0].strip()
        if first:
            return first
    return ""


def read_conll(path: str | Path, *, split: str) -> tuple[BioExample, ...]:
    examples: list[BioExample] = []
    tokens: list[str] = []
    tags: list[str] = []
    idx = 1
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if tokens:
                examples.append(
                    BioExample(f"{split}-{idx:06d}", tuple(tokens), tuple(tags))
                )
                idx += 1
                tokens = []
                tags = []
            continue
        parts = stripped.split()
        if len(parts) < 2:
            tokens.append(stripped)
            tags.append("__MALFORMED__")
        else:
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        examples.append(BioExample(f"{split}-{idx:06d}", tuple(tokens), tuple(tags)))
    return tuple(examples)


def validate_bio_examples(examples: Sequence[BioExample]) -> BioValidation:
    labels: Counter[str] = Counter()
    entities: Counter[str] = Counter()
    malformed = 0
    errors: list[str] = []
    token_count = 0
    for example in examples:
        previous = "O"
        for token_idx, tag in enumerate(example.tags):
            labels[tag] += 1
            token_count += 1
            if tag == "O":
                previous = tag
                continue
            if "-" not in tag:
                malformed += 1
                errors.append(f"{example.example_id}:{token_idx}:bad_label:{tag}")
                previous = tag
                continue
            prefix, label = tag.split("-", 1)
            if prefix not in {"B", "I", "E", "S"}:
                malformed += 1
                errors.append(f"{example.example_id}:{token_idx}:bad_prefix:{tag}")
            if prefix in {"B", "S"}:
                entities[label] += 1
            if prefix == "I" and (
                previous == "O" or "-" not in previous or previous.split("-", 1)[1] != label
            ):
                malformed += 1
                errors.append(f"{example.example_id}:{token_idx}:bad_transition:{tag}")
            previous = tag
    return BioValidation(
        label_vocab=tuple(sorted(labels)),
        entity_counts=dict(sorted(entities.items())),
        token_count=token_count,
        malformed_count=malformed,
        errors=tuple(errors),
    )


def load_json_records(path: str | Path) -> JsonLoadResult:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[Any] = []
        malformed = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
        return JsonLoadResult(tuple(records), "jsonl", malformed)
    if isinstance(value, list):
        return JsonLoadResult(tuple(value), "json_array")
    return JsonLoadResult((value,), "json_object")


def phoner_json_matches_conll(records: Sequence[Any], examples: Sequence[BioExample]) -> bool:
    if len(records) != len(examples):
        return False
    for record, example in zip(records, examples, strict=True):
        if not isinstance(record, dict):
            return False
        words = tuple(str(x) for x in record.get("words", []) or [])
        tags = tuple(str(x) for x in record.get("tags", []) or [])
        if words != example.tokens or tags != example.tags:
            return False
    return True


def vimq_span_stats(records: Sequence[Any]) -> dict[str, Any]:
    sent_labels: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    malformed = 0
    token_count = 0
    for record in records:
        if not isinstance(record, dict):
            malformed += 1
            continue
        tokens = str(record.get("sentence", "")).split()
        token_count += len(tokens)
        sent_labels[str(record.get("sent_label", ""))] += 1
        for span in record.get("seq_label", []) or []:
            if not (
                isinstance(span, list)
                and len(span) == 3
                and isinstance(span[0], int)
                and isinstance(span[1], int)
            ):
                malformed += 1
                continue
            start, end, label = span
            if start < 0 or end < start or end >= len(tokens):
                malformed += 1
            entity_counts[str(label)] += 1
    return {
        "records": len(records),
        "tokens": token_count,
        "sentence_labels": dict(sorted(sent_labels.items())),
        "entity_counts": dict(sorted(entity_counts.items())),
        "malformed": malformed,
    }


def _token_offsets(tokens: Sequence[str]) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end + 1
    return tuple(offsets)


def bio_examples_to_documents(
    dataset_id: str,
    split: str,
    examples: Sequence[BioExample],
    label_mapping: Mapping[str, str],
) -> tuple[CanonicalDocument, ...]:
    documents: list[CanonicalDocument] = []
    for example in examples:
        text = " ".join(example.tokens)
        offsets = _token_offsets(example.tokens)
        annotations: list[CanonicalAnnotation] = []
        active_label = ""
        active_start = 0
        for idx, tag in enumerate((*example.tags, "O")):
            if tag == "O" or tag.startswith("B-") or tag.startswith("S-"):
                if active_label:
                    target = label_mapping.get(active_label)
                    if target in ENTITY_TYPES:
                        start = offsets[active_start][0]
                        end = offsets[idx - 1][1]
                        annotations.append(
                            CanonicalAnnotation(
                                annotation_id=(
                                    f"{dataset_id}-{example.example_id}-ann-"
                                    f"{len(annotations) + 1:04d}"
                                ),
                                document_id=f"{dataset_id}:{split}:{example.example_id}",
                                span=Span(start, end),
                                text=text[start:end],
                                entity_type=target,
                                source=dataset_id,
                            )
                        )
                active_label = ""
            if tag.startswith("B-"):
                active_label = tag.split("-", 1)[1]
                active_start = idx
            elif tag.startswith("S-"):
                label = tag.split("-", 1)[1]
                target = label_mapping.get(label)
                if target in ENTITY_TYPES:
                    start, end = offsets[idx]
                    annotations.append(
                        CanonicalAnnotation(
                            annotation_id=(
                                f"{dataset_id}-{example.example_id}-ann-"
                                f"{len(annotations) + 1:04d}"
                            ),
                            document_id=f"{dataset_id}:{split}:{example.example_id}",
                            span=Span(start, end),
                            text=text[start:end],
                            entity_type=target,
                            source=dataset_id,
                        )
                    )
        documents.append(
            CanonicalDocument(
                document_id=f"{dataset_id}:{split}:{example.example_id}",
                text=text,
                source_id=dataset_id,
                metadata={"split": split, "offset_convention": "joined_tokens"},
                annotations=tuple(annotations),
            )
        )
    return tuple(documents)


def vimq_records_to_documents(
    dataset_id: str,
    split: str,
    records: Sequence[Any],
    label_mapping: Mapping[str, str],
) -> tuple[CanonicalDocument, ...]:
    documents: list[CanonicalDocument] = []
    for row_idx, record in enumerate(records, 1):
        if not isinstance(record, dict):
            continue
        tokens = str(record.get("sentence", "")).split()
        text = " ".join(tokens)
        offsets = _token_offsets(tokens)
        doc_id = f"{dataset_id}:{split}:{row_idx:06d}"
        annotations: list[CanonicalAnnotation] = []
        for span in record.get("seq_label", []) or []:
            if not (
                isinstance(span, list)
                and len(span) == 3
                and isinstance(span[0], int)
                and isinstance(span[1], int)
            ):
                continue
            start_token, end_token, source_label = span
            target = label_mapping.get(str(source_label))
            if target not in ENTITY_TYPES or start_token < 0 or end_token >= len(offsets):
                continue
            start = offsets[start_token][0]
            end = offsets[end_token][1]
            annotations.append(
                CanonicalAnnotation(
                    annotation_id=f"{doc_id}-ann-{len(annotations) + 1:04d}",
                    document_id=doc_id,
                    span=Span(start, end),
                    text=text[start:end],
                    entity_type=target,
                    source=dataset_id,
                )
            )
        documents.append(
            CanonicalDocument(
                document_id=doc_id,
                text=text,
                source_id=dataset_id,
                metadata={
                    "split": split,
                    "offset_convention": "source_token_span_inclusive",
                    "sentence_label": str(record.get("sent_label", "")),
                },
                annotations=tuple(annotations),
            )
        )
    return tuple(documents)


class _CompactReader:
    stop = 0
    boolean_true = 1
    boolean_false = 2
    byte = 3
    i16 = 4
    i32 = 5
    i64 = 6
    double = 7
    binary = 8
    list_ = 9
    set_ = 10
    map_ = 11
    struct_ = 12
    float_ = 13

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def _read_byte(self) -> int:
        if self.position >= len(self.data):
            raise ValueError("unexpected end of thrift payload")
        value = self.data[self.position]
        self.position += 1
        return value

    def _read_varint(self) -> int:
        shift = 0
        result = 0
        while True:
            byte = self._read_byte()
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    @staticmethod
    def _zigzag(value: int) -> int:
        return (value >> 1) ^ -(value & 1)

    def _read_binary(self) -> str:
        length = self._read_varint()
        value = self.data[self.position : self.position + length]
        self.position += length
        return value.decode("utf-8", errors="replace")

    def read_struct(self) -> dict[int, Any]:
        output: dict[int, Any] = {}
        field_id = 0
        while True:
            header = self._read_byte()
            if header == self.stop:
                break
            compact_type = header & 0x0F
            delta = (header & 0xF0) >> 4
            if delta:
                field_id += delta
            else:
                field_id = self._zigzag(self._read_varint())
            output[field_id] = self._read_value(compact_type)
        return output

    def _read_value(self, compact_type: int) -> Any:
        if compact_type == self.boolean_true:
            return True
        if compact_type == self.boolean_false:
            return False
        if compact_type == self.byte:
            value = self._read_byte()
            return value - 256 if value > 127 else value
        if compact_type in {self.i16, self.i32, self.i64}:
            return self._zigzag(self._read_varint())
        if compact_type == self.double:
            value = struct.unpack("<d", self.data[self.position : self.position + 8])[0]
            self.position += 8
            return value
        if compact_type == self.float_:
            value = struct.unpack("<f", self.data[self.position : self.position + 4])[0]
            self.position += 4
            return value
        if compact_type == self.binary:
            return self._read_binary()
        if compact_type == self.struct_:
            return self.read_struct()
        if compact_type in {self.list_, self.set_}:
            header = self._read_byte()
            size = (header & 0xF0) >> 4
            element_type = header & 0x0F
            if size == 15:
                size = self._read_varint()
            return [self._read_value(element_type) for _ in range(size)]
        if compact_type == self.map_:
            size = self._read_varint()
            if size == 0:
                return []
            type_byte = self._read_byte()
            key_type = (type_byte & 0xF0) >> 4
            value_type = type_byte & 0x0F
            return [
                (self._read_value(key_type), self._read_value(value_type))
                for _ in range(size)
            ]
        raise ValueError(f"unsupported thrift compact type {compact_type}")


def read_parquet_metadata(path: str | Path) -> ParquetMetadata:
    parquet = Path(path)
    size = parquet.stat().st_size
    if size < 12:
        raise ValueError(f"{parquet}: too small to be a parquet file")
    with parquet.open("rb") as fh:
        if fh.read(4) != b"PAR1":
            raise ValueError(f"{parquet}: missing parquet header")
        fh.seek(-8, 2)
        footer_length = int.from_bytes(fh.read(4), "little")
        if fh.read(4) != b"PAR1":
            raise ValueError(f"{parquet}: missing parquet footer")
        if footer_length <= 0 or footer_length > size - 8:
            raise ValueError(f"{parquet}: invalid parquet footer length")
        fh.seek(-(8 + footer_length), 2)
        footer = fh.read(footer_length)
    reader = _CompactReader(footer)
    metadata = reader.read_struct()
    schema = metadata.get(2, [])
    row_groups = metadata.get(4, [])
    if not isinstance(schema, list) or not isinstance(row_groups, list):
        raise ValueError(f"{parquet}: malformed parquet metadata")
    schema_names = tuple(
        str(item.get(4, "")) for item in schema if isinstance(item, dict) and item.get(4)
    )
    row_group_rows = sum(
        int(item.get(3, 0)) for item in row_groups if isinstance(item, dict)
    )
    return ParquetMetadata(
        path=parquet.as_posix(),
        bytes=size,
        version=int(metadata.get(1, 0)),
        num_rows=int(metadata.get(3, 0)),
        row_group_count=len(row_groups),
        row_group_rows=row_group_rows,
        schema_names=schema_names,
        created_by=str(metadata.get(6, "")),
        footer_length=footer_length,
    )


def normalized_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFC", " ".join(text.casefold().split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def duplicate_groups_by_hash(
    documents_by_split: Mapping[str, Sequence[CanonicalDocument]],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for split, documents in documents_by_split.items():
        for document in documents:
            by_hash[normalized_text_hash(document.text)].append(f"{split}:{document.document_id}")
    groups = tuple(tuple(sorted(ids)) for ids in by_hash.values() if len(ids) > 1)
    leakage = tuple(group for group in groups if len({item.split(":", 1)[0] for item in group}) > 1)
    return {"duplicates": tuple(sorted(groups)), "cross_split": tuple(sorted(leakage))}


def require_training_allowed(manifest: NerDatasetManifest) -> None:
    """Gate training use on the SEPARATE user-attested `training_use` block, NOT on
    the source `license` status (which is preserved as the original source fact).

    A dataset may be REVIEW_REQUIRED as a source license yet still be internally
    training-eligible under a user attestation; raw redistribution stays forbidden.
    """
    tu = manifest.training_use
    if str(tu.get("status", "")) != "USER_ATTESTED_PERMISSION" or not tu.get(
        "internal_training_allowed", tu.get("permission_attested_by_user", False)
    ):
        raise PublicDatasetGovernanceError(
            f"{manifest.dataset_id}: training_use does not attest internal training permission"
        )
    if "ner_training" not in manifest.permitted_use:
        raise PublicDatasetGovernanceError(
            f"{manifest.dataset_id}: manifest does not permit ner_training"
        )
    if manifest.readiness_status not in {"training_ready", "analysis_and_training_ready"}:
        raise PublicDatasetGovernanceError(
            f"{manifest.dataset_id}: readiness_status {manifest.readiness_status} "
            "is not training-ready"
        )


__all__ = [
    "BioExample",
    "BioValidation",
    "FileTreeSummary",
    "JsonLoadResult",
    "ParquetMetadata",
    "PublicDatasetGovernanceError",
    "bio_examples_to_documents",
    "duplicate_groups_by_hash",
    "file_tree_summary",
    "label_inventory_hash",
    "load_json_records",
    "parse_git_revision",
    "parse_huggingface_snapshot_id",
    "phoner_json_matches_conll",
    "read_conll",
    "read_parquet_metadata",
    "require_training_allowed",
    "validate_bio_examples",
    "vimq_records_to_documents",
    "vimq_span_stats",
]
