"""Serializable local KB index contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    index_id: str
    index_type: str
    source_snapshot_id: str
    source_hash: str
    config_hash: str
    builder_version: str
    deterministic_index_hash: str
    record_count: int
    concept_count: int
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexRecord:
    concept_id: str
    canonical_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchIndex:
    metadata: IndexMetadata
    records: tuple[IndexRecord, ...]
    exact: dict[str, tuple[str, ...]]
    exact_ascii: dict[str, tuple[str, ...]]
    ngrams: dict[str, tuple[str, ...]]
    sparse_terms: dict[str, tuple[str, ...]]
    graph: dict[str, tuple[str, ...]] = field(default_factory=dict)


__all__ = ["IndexMetadata", "IndexRecord", "SearchIndex"]
