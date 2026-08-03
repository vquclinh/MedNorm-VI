"""BioSyn-style character lexical view (0082), built on the index the repo already has.

Inspection first: `kb.indexing.retrieval.search_index` is **already** a character n-gram
retriever. It carries `ngrams` (accent-stripped 3-grams), `ngrams_accent` (diacritic-bearing
3-grams), token postings and tiered exact postings, all as inverted lists. So this module
does **not** build a second index, and deliberately does not build a global TF-IDF character
matrix - 227,257 governed documents against a character vocabulary is exactly the blind
allocation the brief warns about, and it buys nothing the postings do not already give.

What was missing is BioSyn's *scoring* half: the inverted index ranks by how many n-grams a
concept shares, not by how similar the two strings are. A long label sharing many grams
outranks a short exact-ish one. So the design here is:

    existing inverted index  ->  cheap shortlist  ->  character n-gram similarity rerank

Similarity is Dice over 3-gram sets, computed **only for the shortlist**, so memory is
bounded by the shortlist size rather than by the KB. Surface forms are memoized per concept
so a repeated candidate is not re-tokenized, and the memo is bounded the same way.

Everything is deterministic: same query, same index, same order, every time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kb.indexing.normalization import char_ngrams, normalize_text
from ..kb.indexing.retrieval import LocalIndex, search_index

#: BioSyn uses character n-grams; 3 is what this repository's index is already built with,
#: so the rerank scores the same signal the shortlist was drawn from.
NGRAM_N = 3

#: How many candidates the inverted index contributes before reranking. Wide enough that the
#: right concept is usually inside it, small enough that the rerank stays trivial.
DEFAULT_SHORTLIST = 50

SPARSE_VERSION = "ontofusion-sparse-v1"


class SparseProvenanceMismatch(RuntimeError):
    """Raised when a recorded sparse configuration does not describe this KB."""


@dataclass(frozen=True, slots=True)
class SparseSettings:
    """Everything that changes what a sparse score means. Recorded, and checked on reload."""

    ngram_n: int = NGRAM_N
    strip_accents: bool = True
    shortlist: int = DEFAULT_SHORTLIST
    version: str = SPARSE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "ngram_n": self.ngram_n, "strip_accents": self.strip_accents,
            "shortlist": self.shortlist, "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class SparseProvenance:
    """Which KB a sparse configuration was validated against.

    Separate from the GraphCENT dense caches on purpose: those record a model revision and
    an embedding row order, neither of which exists here. Sharing one manifest would make
    both weaker.
    """

    settings: SparseSettings
    icd_snapshot: str
    rxnorm_snapshot: str
    document_checksum: str
    row_order_checksum: str
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.as_dict(),
            "icd_snapshot": self.icd_snapshot,
            "rxnorm_snapshot": self.rxnorm_snapshot,
            "document_checksum": self.document_checksum,
            "row_order_checksum": self.row_order_checksum,
            "rows": self.rows,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> SparseProvenance:
        settings = row.get("settings") or {}
        return SparseProvenance(
            settings=SparseSettings(
                ngram_n=int(settings.get("ngram_n", NGRAM_N)),
                strip_accents=bool(settings.get("strip_accents", True)),
                shortlist=int(settings.get("shortlist", DEFAULT_SHORTLIST)),
                version=str(settings.get("version", SPARSE_VERSION)),
            ),
            icd_snapshot=str(row.get("icd_snapshot", "")),
            rxnorm_snapshot=str(row.get("rxnorm_snapshot", "")),
            document_checksum=str(row.get("document_checksum", "")),
            row_order_checksum=str(row.get("row_order_checksum", "")),
            rows=int(row.get("rows", 0)),
        )

    def assert_compatible(self, other: SparseProvenance) -> None:
        problems = [
            name
            for name, mine, theirs in (
                ("settings", self.settings.as_dict(), other.settings.as_dict()),
                ("icd_snapshot", self.icd_snapshot, other.icd_snapshot),
                ("rxnorm_snapshot", self.rxnorm_snapshot, other.rxnorm_snapshot),
                ("document_checksum", self.document_checksum, other.document_checksum),
                ("row_order_checksum", self.row_order_checksum, other.row_order_checksum),
                ("rows", self.rows, other.rows),
            )
            if mine != theirs
        ]
        if problems:
            raise SparseProvenanceMismatch(
                "recorded sparse provenance does not describe this knowledge base: "
                f"{problems}. Refusing to reuse it."
            )


def write_provenance(path: Path, provenance: SparseProvenance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(provenance.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def read_provenance(path: Path) -> SparseProvenance | None:
    if not path.is_file():
        return None
    return SparseProvenance.from_dict(json.loads(path.read_text(encoding="utf-8")))


def grams(text: str, settings: SparseSettings) -> frozenset[str]:
    return frozenset(
        char_ngrams(text, n=settings.ngram_n, strip_accents=settings.strip_accents)
    )


def dice(left: frozenset[str], right: frozenset[str]) -> float:
    """Sørensen-Dice over character n-gram sets. 1.0 identical, 0.0 disjoint."""
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


@dataclass
class SurfaceMemo:
    """Concept -> its governed surface forms, computed once per run and bounded by use."""

    index: LocalIndex
    settings: SparseSettings = field(default_factory=SparseSettings)
    _cache: dict[str, tuple[frozenset[str], ...]] = field(default_factory=dict)

    def forms(self, concept_id: str) -> tuple[frozenset[str], ...]:
        cached = self._cache.get(concept_id)
        if cached is not None:
            return cached
        record = self.index.records.get(concept_id) or {}
        texts = [str(record.get("canonical_name") or "")]
        texts.extend(str(a) for a in (record.get("aliases") or ()))
        built = tuple(grams(t, self.settings) for t in texts if t.strip())
        self._cache[concept_id] = built
        return built

    def similarity(self, concept_id: str, query_grams: frozenset[str]) -> float:
        """Best character similarity over the concept's own surface forms."""
        return max(
            (dice(query_grams, form) for form in self.forms(concept_id)), default=0.0
        )

    @property
    def size(self) -> int:
        return len(self._cache)


@dataclass(frozen=True, slots=True)
class SparseHit:
    concept_id: str
    similarity: float
    shortlist_rank: int


def sparse_search(
    index: LocalIndex | None,
    query: str,
    *,
    top_k: int,
    memo: SurfaceMemo | None = None,
    settings: SparseSettings | None = None,
) -> list[SparseHit]:
    """Shortlist from the existing inverted index, then rerank by character similarity.

    Ties are broken by the shortlist position and then by concept id, so the output is a
    total order that does not depend on dictionary iteration.
    """
    if index is None or not query.strip():
        return []
    config = settings or SparseSettings()
    surface = memo if memo is not None else SurfaceMemo(index, config)
    shortlist = search_index(index, query, limit=config.shortlist)
    if not shortlist:
        return []
    query_grams = grams(query, config)
    scored = [
        SparseHit(
            concept_id=hit.concept_id,
            similarity=surface.similarity(hit.concept_id, query_grams),
            shortlist_rank=position,
        )
        for position, hit in enumerate(shortlist, start=1)
    ]
    scored.sort(key=lambda h: (-h.similarity, h.shortlist_rank, h.concept_id))
    return scored[:top_k]


def normalized_query(text: str) -> str:
    return normalize_text(text)


__all__ = [
    "DEFAULT_SHORTLIST",
    "NGRAM_N",
    "SPARSE_VERSION",
    "SparseHit",
    "SparseProvenance",
    "SparseProvenanceMismatch",
    "SparseSettings",
    "SurfaceMemo",
    "dice",
    "grams",
    "normalized_query",
    "read_provenance",
    "sparse_search",
    "write_provenance",
]
