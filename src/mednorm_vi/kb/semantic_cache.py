"""Multi-retriever candidate generation with row-order-safe caches (GraphCENT 0080).

Three things this module refuses to get wrong.

**Pooling is per model.** `pool()` dispatches on the registry's declared contract, so
cross-lingual SapBERT is read at CLS and the BioBERT sentence model is mean-pooled over its
attention mask. Using one recipe for both is a silent quality loss with no error message.

**Row order is a checked contract.** Audit 0073 shipped a scored submission whose dense cache
was indexed in a different order than the documents it described - ICD mentions were scored
against RxNorm vectors and the ids still looked valid. Every cache here carries the document
checksum, the row-order checksum and the shape, and `CachedIndex.validate` refuses to retrieve
until they agree.

**Retrieval is over governed ids only.** The encoders learned from UMLS and SNOMED; the index
contains nothing but our ICD and RxNorm concepts, so a candidate outside the governed KB is
not filtered out later - it never exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..models.registry import POOLING_CLS, POOLING_MEAN, RetrieverSpec, is_hub_revision

CACHE_MANIFEST_VERSION = "graphcent-cache-v1"

#: Small per-retriever depth, following CENT's small-candidate-context principle.
TOP_K_PER_RETRIEVER = 5
#: Total curated context handed to the LLM.
MAX_CANDIDATE_CONTEXT = 10


def checksum(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class Encoder(Protocol):
    """Anything that turns text into unit-norm vectors. Injected, so tests need no weights."""

    key: str

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> Any: ...


def pool(hidden_state: Any, attention_mask: Any, pooling: str) -> Any:
    """Model-specific pooling. Never guess: the registry declares which one applies."""
    import torch

    if pooling == POOLING_CLS:
        pooled = hidden_state[:, 0]  # CLS before pooler, per SapBERT/ClinLinker cards
    elif pooling == POOLING_MEAN:
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        pooled = (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    else:  # pragma: no cover - registry validates this
        raise ValueError(f"unknown pooling {pooling!r}")
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


@dataclass(frozen=True, slots=True)
class KbDocument:
    """One governed concept rendered for semantic retrieval."""

    concept_id: str
    ontology: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"concept_id": self.concept_id, "ontology": self.ontology, "text": self.text}


@dataclass(frozen=True, slots=True)
class CacheManifest:
    """Everything needed to prove a cache describes the documents it is used with."""

    model_repo: str
    revision: str
    pooling: str
    normalized: bool
    document_checksum: str
    row_order_checksum: str
    rows: int
    dim: int
    dtype: str
    version: str = CACHE_MANIFEST_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_repo": self.model_repo,
            "revision": self.revision,
            "pooling": self.pooling,
            "normalized": self.normalized,
            "document_checksum": self.document_checksum,
            "row_order_checksum": self.row_order_checksum,
            "rows": self.rows,
            "dim": self.dim,
            "dtype": self.dtype,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> CacheManifest:
        return CacheManifest(
            model_repo=row["model_repo"],
            revision=row["revision"],
            pooling=row["pooling"],
            normalized=bool(row["normalized"]),
            document_checksum=row["document_checksum"],
            row_order_checksum=row["row_order_checksum"],
            rows=int(row["rows"]),
            dim=int(row["dim"]),
            dtype=row["dtype"],
            version=row.get("version", CACHE_MANIFEST_VERSION),
        )


class CacheMismatch(RuntimeError):
    """Raised when a cache cannot be proved to describe the current documents."""


@dataclass
class CachedIndex:
    """Vectors plus the documents they describe, with the alignment checked before use."""

    spec: RetrieverSpec
    documents: list[KbDocument]
    vectors: Any
    manifest: CacheManifest

    def validate(self) -> None:
        problems: list[str] = []
        if self.manifest.version != CACHE_MANIFEST_VERSION:
            problems.append(f"manifest version {self.manifest.version!r} unsupported")
        if self.manifest.rows != len(self.documents):
            problems.append(
                f"cache has {self.manifest.rows} rows, documents have {len(self.documents)}"
            )
        if int(self.vectors.shape[0]) != len(self.documents):
            problems.append(
                f"tensor has {int(self.vectors.shape[0])} rows, documents have "
                f"{len(self.documents)}"
            )
        if int(self.vectors.shape[1]) != self.manifest.dim:
            problems.append(
                f"tensor has dim {int(self.vectors.shape[1])}, manifest says "
                f"{self.manifest.dim}"
            )
        if str(self.vectors.dtype) != self.manifest.dtype:
            problems.append(
                f"tensor dtype is {self.vectors.dtype}, manifest says {self.manifest.dtype}"
            )
        if not self.manifest.normalized:
            problems.append("cache manifest does not declare normalized vectors")
        if self.manifest.document_checksum != checksum([d.text for d in self.documents]):
            problems.append("document text changed since the cache was built")
        if self.manifest.row_order_checksum != checksum([d.concept_id for d in self.documents]):
            problems.append(
                "row order changed since the cache was built - a known failure mode, and "
                "one that is silent at retrieval time"
            )
        if self.manifest.pooling != self.spec.pooling:
            problems.append(
                f"cache pooled with {self.manifest.pooling!r}, spec declares {self.spec.pooling!r}"
            )
        if self.manifest.model_repo != self.spec.repo_id:
            problems.append(
                f"cache built from {self.manifest.model_repo!r}, spec declares "
                f"{self.spec.repo_id!r}"
            )
        if not is_hub_revision(self.spec.revision):
            problems.append(
                f"spec revision {self.spec.revision or 'empty'} is not an immutable hub commit"
            )
        if not is_hub_revision(self.manifest.revision):
            problems.append(
                f"cache revision {self.manifest.revision or 'empty'} is not an immutable "
                "hub commit"
            )
        elif self.manifest.revision != self.spec.revision:
            # Same repo, different snapshot: the vectors are from other weights than the
            # ones this run deploys, and nothing downstream would notice.
            problems.append(
                f"cache built at revision {self.manifest.revision or 'empty'}, this run "
                f"deploys {self.spec.revision}"
            )
        if problems:
            raise CacheMismatch(
                f"{self.spec.key}: cache is not usable:\n  - " + "\n  - ".join(problems)
            )

    def search(self, query_vector: Any, *, ontology: str, top_k: int) -> list[tuple[str, int]]:
        """`(concept_id, rank)` restricted to one ontology. Scores stay internal."""
        import torch

        rows = [i for i, d in enumerate(self.documents) if d.ontology == ontology]
        if not rows:
            return []
        subset = self.vectors[rows]
        scores = subset @ query_vector
        count = min(top_k, len(rows))
        top = torch.topk(scores, count)
        return [
            (self.documents[rows[i]].concept_id, rank + 1)
            for rank, i in enumerate(top.indices.tolist())
        ]


def write_cache(path: Path, index: CachedIndex) -> None:
    """Atomic: manifest and tensor appear together or not at all."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_tmp = path.with_suffix(".pt.tmp")
    manifest_tmp = path.with_suffix(".manifest.json.tmp")
    torch.save(index.vectors, tensor_tmp)
    manifest_tmp.write_text(
        json.dumps(index.manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    tensor_tmp.replace(path.with_suffix(".pt"))
    manifest_tmp.replace(path.with_suffix(".manifest.json"))


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Provenance for one candidate. Raw similarities are deliberately not carried."""

    concept_id: str
    lexical_rank: int | None = None
    retriever_ranks: dict[str, int] = field(default_factory=dict)
    exact_alias_match: bool = False

    @property
    def supporting_retrievers(self) -> int:
        return len(self.retriever_ranks)

    @property
    def lexically_backed(self) -> bool:
        return self.lexical_rank is not None

    def reciprocal_rank(self) -> float:
        """Internal ordering only; never shown to the LLM."""
        total = 1.0 / (60 + self.lexical_rank) if self.lexical_rank else 0.0
        for rank in self.retriever_ranks.values():
            total += 1.0 / (60 + rank)
        return total

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "lexical_rank": self.lexical_rank,
            "retriever_ranks": dict(sorted(self.retriever_ranks.items())),
            "supporting_retrievers": self.supporting_retrievers,
            "exact_alias_match": self.exact_alias_match,
        }


def fuse(
    lexical: Sequence[tuple[str, int]],
    per_retriever: dict[str, Sequence[tuple[str, int]]],
    *,
    exact_alias_ids: frozenset[str] = frozenset(),
    governed_ids: frozenset[str] | None = None,
    limit: int = MAX_CANDIDATE_CONTEXT,
) -> list[CandidateEvidence]:
    """Union, dedupe by governed id, order by reciprocal rank, cap the context.

    A candidate absent from `governed_ids` is dropped here, so no downstream stage has to
    trust that retrieval stayed inside the KB.
    """
    merged: dict[str, dict[str, Any]] = {}
    for concept_id, rank in lexical:
        if governed_ids is not None and concept_id not in governed_ids:
            continue
        merged.setdefault(concept_id, {"lexical": None, "ranks": {}})["lexical"] = rank
    for key, hits in per_retriever.items():
        for concept_id, rank in hits:
            if governed_ids is not None and concept_id not in governed_ids:
                continue
            merged.setdefault(concept_id, {"lexical": None, "ranks": {}})["ranks"][key] = rank

    evidence = [
        CandidateEvidence(
            concept_id=concept_id,
            lexical_rank=row["lexical"],
            retriever_ranks=dict(row["ranks"]),
            exact_alias_match=concept_id in exact_alias_ids,
        )
        for concept_id, row in merged.items()
    ]
    evidence.sort(key=lambda e: (-e.reciprocal_rank(), e.concept_id))
    return evidence[:limit]


__all__ = [
    "CACHE_MANIFEST_VERSION",
    "MAX_CANDIDATE_CONTEXT",
    "TOP_K_PER_RETRIEVER",
    "CacheManifest",
    "CacheMismatch",
    "CachedIndex",
    "CandidateEvidence",
    "Encoder",
    "KbDocument",
    "checksum",
    "fuse",
    "pool",
    "write_cache",
]
