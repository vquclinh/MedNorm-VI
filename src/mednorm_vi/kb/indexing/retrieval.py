"""Retrieval over generated local JSON indexes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evidence import UNTIERED_RANK, rank_key, tier_of
from .normalization import (
    accent_marked_ngrams,
    accent_marked_tokens,
    char_ngrams,
    normalize_text,
    tokens,
)


@dataclass(frozen=True, slots=True)
class CandidateHit:
    concept_id: str
    score: float
    channels: tuple[str, ...]
    #: Lexicographic ranking key, ascending - see `kb.indexing.evidence.rank_key`. Present
    #: on every hit so callers can order by evidence instead of by an additive score.
    rank: tuple[int, int, int, float, int, str] = (UNTIERED_RANK, 0, 0, 0.0, 0, "")
    #: Strongest evidence tier (A-F), or "" for an index that carries no tier postings.
    tier: str = ""


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
    #: Tier postings (Audit 0069). Empty for competition-v3 and RxNorm, which therefore keep
    #: their previous ordering untouched.
    exact_canonical: dict[str, list[str]] = field(default_factory=dict)
    exact_alias: dict[str, list[str]] = field(default_factory=dict)
    exact_synonym: dict[str, list[str]] = field(default_factory=dict)
    ngrams_accent: dict[str, list[str]] = field(default_factory=dict)
    sparse_accent: dict[str, list[str]] = field(default_factory=dict)

    @property
    def tiered(self) -> bool:
        """True when this index can be ranked by evidence tier rather than by score."""
        return bool(self.exact_canonical or self.exact_alias or self.exact_synonym)

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
        exact_canonical={str(k): list(v) for k, v in payload.get("exact_canonical", {}).items()},
        exact_alias={str(k): list(v) for k, v in payload.get("exact_alias", {}).items()},
        exact_synonym={str(k): list(v) for k, v in payload.get("exact_synonym", {}).items()},
        ngrams_accent={str(k): list(v) for k, v in payload.get("ngrams_accent", {}).items()},
        sparse_accent={str(k): list(v) for k, v in payload.get("sparse_accent", {}).items()},
    )


def search_index(index: LocalIndex, query: str, *, limit: int = 20) -> tuple[CandidateHit, ...]:
    """Retrieve candidates via exact, accent-insensitive, n-gram, and sparse channels."""
    scores: dict[str, float] = {}
    channels: dict[str, set[str]] = {}

    def add(cui: str, score: float, channel: str) -> None:
        scores[cui] = scores.get(cui, 0.0) + score
        channels.setdefault(cui, set()).add(channel)

    accented = normalize_text(query)
    stripped = normalize_text(query, strip_accents=True)

    if index.tiered:
        # Tiered index (Audit 0069): the exact channel is split by evidence kind so the
        # ranking can tell an exact canonical hit from an exact alias hit from a synonym.
        for cui in index.exact_canonical.get(accented, []):
            add(cui, 100.0, "exact_canonical")
        for cui in index.exact_alias.get(accented, []):
            add(cui, 100.0, "exact_alias")
        for cui in index.exact_synonym.get(accented, []):
            add(cui, 100.0, "exact_synonym")
        for gram in accent_marked_ngrams(query):
            for cui in index.ngrams_accent.get(gram, []):
                add(cui, 1.0, "char_ngram_accent")
        for token in accent_marked_tokens(query):
            for cui in index.sparse_accent.get(token, []):
                add(cui, 2.0, "sparse_accent")
    else:
        for cui in index.exact.get(accented, []):
            add(cui, 100.0, "exact")

    for cui in index.exact_ascii.get(stripped, []):
        add(cui, 80.0, "exact_ascii")
    for gram in char_ngrams(query):
        for cui in index.ngrams.get(gram, []):
            add(cui, 1.0, "char_ngram")
    for token in tokens(query):
        for cui in index.sparse_terms.get(token, []):
            add(cui, 2.0, "sparse")

    hierarchy = _hierarchy_signal(index, scores) if index.tiered else {}
    hits = []
    for cui, score in scores.items():
        if not index.exists(cui):
            continue
        found = tuple(sorted(channels[cui]))
        hits.append(
            CandidateHit(
                concept_id=cui,
                score=score,
                channels=found,
                rank=rank_key(
                    concept_id=cui,
                    channels=found,
                    lexical_score=score,
                    hierarchy_signal=hierarchy.get(cui, 0),
                    tiered=index.tiered,
                ),
                tier=tier_of(found) if index.tiered else "",
            )
        )
    # One ascending sort over the rank tuple expresses the whole contract. For an untiered
    # index every rank begins (UNTIERED_RANK, 0, 0, -score, 0, id), which is exactly the
    # previous `(-score, concept_id)` ordering - so v3 and RxNorm results do not move.
    hits.sort(key=lambda h: h.rank)
    return tuple(hits[:limit])


def _hierarchy_signal(index: LocalIndex, scores: dict[str, float]) -> dict[str, int]:
    """How many of a concept's ICD graph neighbours also matched this query.

    Used only as a late tie-break: when two candidates are otherwise indistinguishable, the
    one whose hierarchical neighbourhood also matched is the likelier region of the tree.
    """
    matched = set(scores)
    return {
        cui: sum(1 for neighbour in index.graph.get(cui, ()) if neighbour in matched)
        for cui in matched
    }


__all__ = ["CandidateHit", "LocalIndex", "load_index", "search_index"]
