"""Retrieval over generated local JSON indexes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalization import char_ngrams, normalize_text, tokens


@dataclass(frozen=True, slots=True)
class CandidateHit:
    concept_id: str
    score: float
    channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalIndex:
    index_type: str
    source_snapshot_id: str
    records: dict[str, dict[str, Any]]
    exact: dict[str, list[str]]
    exact_ascii: dict[str, list[str]]
    ngrams: dict[str, list[str]]
    sparse_terms: dict[str, list[str]]
    graph: dict[str, list[str]]

    def exists(self, concept_id: str) -> bool:
        return concept_id in self.records


def load_index(path: str | Path) -> LocalIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {
        str(row["concept_id"]): row
        for row in payload.get("records", [])
        if isinstance(row, dict) and row.get("concept_id")
    }
    meta = payload.get("metadata", {})
    return LocalIndex(
        index_type=str(meta.get("index_type", "")),
        source_snapshot_id=str(meta.get("source_snapshot_id", "")),
        records=records,
        exact={str(k): list(v) for k, v in payload.get("exact", {}).items()},
        exact_ascii={str(k): list(v) for k, v in payload.get("exact_ascii", {}).items()},
        ngrams={str(k): list(v) for k, v in payload.get("ngrams", {}).items()},
        sparse_terms={str(k): list(v) for k, v in payload.get("sparse_terms", {}).items()},
        graph={str(k): list(v) for k, v in payload.get("graph", {}).items()},
    )


def search_index(index: LocalIndex, query: str, *, limit: int = 20) -> tuple[CandidateHit, ...]:
    """Retrieve candidates via exact, accent-insensitive, n-gram, and sparse channels."""
    scores: dict[str, float] = {}
    channels: dict[str, set[str]] = {}

    def add(cui: str, score: float, channel: str) -> None:
        scores[cui] = scores.get(cui, 0.0) + score
        channels.setdefault(cui, set()).add(channel)

    for cui in index.exact.get(normalize_text(query), []):
        add(cui, 100.0, "exact")
    for cui in index.exact_ascii.get(normalize_text(query, strip_accents=True), []):
        add(cui, 80.0, "exact_ascii")
    for gram in char_ngrams(query):
        for cui in index.ngrams.get(gram, []):
            add(cui, 1.0, "char_ngram")
    for token in tokens(query):
        for cui in index.sparse_terms.get(token, []):
            add(cui, 2.0, "sparse")

    hits = [
        CandidateHit(concept_id=cui, score=score, channels=tuple(sorted(channels[cui])))
        for cui, score in scores.items()
        if index.exists(cui)
    ]
    hits.sort(key=lambda h: (-h.score, h.concept_id))
    return tuple(hits[:limit])


__all__ = ["CandidateHit", "LocalIndex", "load_index", "search_index"]
