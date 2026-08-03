"""Building, loading and querying the governed semantic caches.

Each retriever gets its own cache because pooling and tokenisation are not interchangeable.
Every cache carries the document checksum, the row-order checksum, the pooling contract and
the model revision it was built at, and retrieval refuses to start if any of them disagrees -
a cache built by other weights is silently wrong, which is the worst kind of wrong.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import RuntimeConfig
from ..models.encoders import build_encoder
from ..models.qwen import log
from ..models.registry import RetrieverSpec, RevisionNotPinned, is_hub_revision
from .indexing.normalization import normalize_text
from .indexing.retrieval import LocalIndex, search_index
from .ontology import build_kb_documents
from .rxnorm.structured import StructuredDrug
from .semantic_cache import (
    CachedIndex,
    CacheManifest,
    KbDocument,
    checksum,
    write_cache,
)


def cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / f"{key}_kb"


def build_index_for(
    spec: RetrieverSpec,
    documents: list[KbDocument],
    config: RuntimeConfig,
    *,
    encoder_factory: Any = build_encoder,
) -> CachedIndex:
    """Embed the governed corpus with one retriever, then release it."""
    if not is_hub_revision(spec.revision):
        raise RevisionNotPinned(
            f"{spec.key}: refusing to build a cache against revision "
            f"{spec.revision or 'empty'}. An index whose model identity is unknown "
            "cannot be validated on reload; run download-models first so the pinned "
            "revision is recorded."
        )
    encoder = encoder_factory(spec, config.model_root, config.device)
    started = time.time()
    log(f"{spec.key}: embedding {len(documents):,} governed documents")
    vectors = encoder.encode(
        [d.text for d in documents], batch_size=config.embed_batch_size
    )
    revision = spec.revision
    dim = int(vectors.shape[1]) if len(vectors) else 0
    encoder.unload()
    log(f"{spec.key}: done in {time.time() - started:.1f}s, dim {dim}")

    index = CachedIndex(
        spec=spec, documents=documents, vectors=vectors,
        manifest=CacheManifest(
            model_repo=spec.repo_id, revision=revision, pooling=spec.pooling,
            normalized=True,
            document_checksum=checksum([d.text for d in documents]),
            row_order_checksum=checksum([d.concept_id for d in documents]),
            rows=len(documents), dim=dim, dtype=str(vectors.dtype),
        ),
    )
    index.validate()  # alignment proved before anything is written
    write_cache(cache_path(config.cache_root, spec.key), index)
    return index


def load_index(
    spec: RetrieverSpec, documents: list[KbDocument], config: RuntimeConfig
) -> CachedIndex:
    """Restore a cache and refuse it unless it describes these exact documents."""
    import torch

    base = cache_path(config.cache_root, spec.key)
    manifest_path = base.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{spec.key}: no cache manifest at {manifest_path}")
    index = CachedIndex(
        spec=spec, documents=documents,
        vectors=torch.load(base.with_suffix(".pt"), map_location="cpu"),
        manifest=CacheManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        ),
    )
    index.validate()
    return index


@dataclass
class RetrievalEvidence:
    """Per-span retrieval results, keyed by span text. Collected before Qwen loads."""

    per_span: dict[str, dict[str, list[tuple[str, int]]]] = field(default_factory=dict)

    def record(self, span: str, key: str, hits: list[tuple[str, int]]) -> None:
        self.per_span.setdefault(span, {})[key] = hits

    def for_span(self, span: str) -> dict[str, list[tuple[str, int]]]:
        return self.per_span.get(span, {})


def precompute_retrieval(
    spans: Sequence[tuple[str, str, str]],
    specs: Sequence[RetrieverSpec],
    documents: list[KbDocument],
    config: RuntimeConfig,
    *,
    encoder_factory: Any = build_encoder,
) -> RetrievalEvidence:
    """Embed every mention once per retriever, then unload before disambiguation.

    ``spans`` is ``(text, ontology, retriever_role)``. Batching all mentions per model
    keeps the number of load/unload cycles at one per retriever for the entire run, while
    preserving the registry contract that a retriever only contributes to declared roles.
    """
    evidence = RetrievalEvidence()
    unique = sorted({(text, ontology, role) for text, ontology, role in spans})
    if not unique:
        return evidence

    for spec in specs:
        relevant = [(text, ontology, role) for text, ontology, role in unique if role in spec.role]
        if not relevant:
            continue
        texts = [text for text, _, _ in relevant]
        index = load_index(spec, documents, config)
        encoder = encoder_factory(spec, config.model_root, config.device)
        vectors = encoder.encode(texts, batch_size=config.embed_batch_size)
        encoder.unload()
        for position, (text, ontology, _role) in enumerate(relevant):
            hits = index.search(
                vectors[position], ontology=ontology, top_k=config.top_k_per_retriever
            )
            evidence.record(f"{ontology}\x00{text}", spec.key, hits)
        del index
        log(f"{spec.key}: retrieved for {len(relevant):,} unique spans, model unloaded")
    return evidence


def lexical_hits(
    index: LocalIndex | None, mention: str, limit: int
) -> list[tuple[str, int]]:
    if index is None:
        return []
    return [
        (hit.concept_id, rank)
        for rank, hit in enumerate(search_index(index, mention, limit=limit), start=1)
    ]


def exact_alias_ids(index: LocalIndex | None, mention: str) -> frozenset[str]:
    """Concepts whose governed label or alias equals the mention after normalization."""
    if index is None:
        return frozenset()
    key = normalize_text(mention)
    out = set(index.exact.get(key, []))
    for posting in ("exact_canonical", "exact_alias", "exact_synonym"):
        out.update(getattr(index, posting, {}).get(key, []))
    return frozenset(out)


def governed_kb(
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug] | None,
) -> list[KbDocument]:
    return build_kb_documents(icd_index, rxnorm_index, structured)


__all__ = [
    "RetrievalEvidence",
    "build_index_for",
    "cache_path",
    "exact_alias_ids",
    "governed_kb",
    "lexical_hits",
    "load_index",
    "precompute_retrieval",
]
