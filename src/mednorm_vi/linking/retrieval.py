"""Multi-view diversified retrieval and candidate union (0082).

CENT's finding drives the shape here: a small candidate set drawn from *diverse* retrievers
beats a deep top-k from one. So every retriever contributes a handful, the union is
deduplicated by governed concept id, and the cap is small. Widening one retriever to top-20
is exactly what this refuses to do.

Three query views multiply the diversity without multiplying the retrievers: the raw
Vietnamese mention (always present), a canonical Vietnamese term, and a canonical English
term. Which retriever sees which view is decided per retriever, because forcing an English
query through an index of Vietnamese ICD labels retrieves noise.

Retriever identity and per-retriever rank are recorded for deterministic gating and
diagnostics, and are deliberately kept out of everything the final model sees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing.retrieval import LocalIndex
from ..kb.ontology import ONTOLOGY_ICD
from ..models.registry import ROLE_DIAGNOSIS, ROLE_DRUG
from .sparse import SparseSettings, SurfaceMemo, sparse_search

RETRIEVER_SPARSE = "sparse_char_ngram"
RETRIEVER_EXACT = "governed_exact_alias"

VIEW_RAW = "raw"
VIEW_VI = "canonical_vi"
VIEW_EN = "canonical_en"
VIEWS: tuple[str, ...] = (VIEW_RAW, VIEW_VI, VIEW_EN)


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Per-role top-k and which views each retriever is allowed to see.

    ClinLinker is absent from the drug policy, matching the registry role it already
    declares; the two are checked against each other at runtime so they cannot drift.
    """

    sparse_top_k: int = 4
    dense_top_k: int = 4
    candidate_cap: int = 10
    #: The English view is a retrieval view for the multilingual encoders only. The governed
    #: lexical indices are Vietnamese/ICD surface forms, where an English query is noise.
    sparse_views: tuple[str, ...] = (VIEW_RAW, VIEW_VI)
    dense_views: tuple[str, ...] = (VIEW_RAW, VIEW_VI, VIEW_EN)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sparse_top_k": self.sparse_top_k, "dense_top_k": self.dense_top_k,
            "candidate_cap": self.candidate_cap,
            "sparse_views": list(self.sparse_views), "dense_views": list(self.dense_views),
        }


@dataclass
class Candidate:
    """One governed concept and which retrievers found it, with which view.

    `supporting` and `best_rank` exist for deterministic gating and diagnostics. They are
    never rendered into a prompt.
    """

    concept_id: str
    ontology: str
    supporting: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    exact_alias: bool = False

    def record(self, retriever: str, view: str, rank: int) -> None:
        self.supporting.setdefault(retriever, []).append((view, rank))

    @property
    def retriever_count(self) -> int:
        return len(self.supporting)

    @property
    def view_count(self) -> int:
        return len({view for hits in self.supporting.values() for view, _ in hits})

    @property
    def best_rank(self) -> int:
        return min(
            (rank for hits in self.supporting.values() for _, rank in hits), default=999
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "ontology": self.ontology,
            "exact_alias": self.exact_alias,
            "supporting_retrievers": sorted(self.supporting),
            "retriever_count": self.retriever_count,
            "view_count": self.view_count,
            "best_rank": self.best_rank,
        }


@dataclass
class CandidateSet:
    """Union of everything retrieved for one mention, deduplicated by concept id."""

    ontology: str
    candidates: dict[str, Candidate] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, reason: str, amount: int = 1) -> None:
        if amount:
            self.counters[reason] = self.counters.get(reason, 0) + amount

    def add(self, concept_id: str, retriever: str, view: str, rank: int) -> Candidate:
        candidate = self.candidates.get(concept_id)
        if candidate is None:
            candidate = Candidate(concept_id=concept_id, ontology=self.ontology)
            self.candidates[concept_id] = candidate
        candidate.record(retriever, view, rank)
        self.count(f"retrieved_{retriever}")
        self.count(f"retrieved_view_{view}")
        return candidate

    def mark_exact(self, concept_id: str) -> None:
        candidate = self.candidates.get(concept_id)
        if candidate is not None:
            candidate.exact_alias = True

    @property
    def size(self) -> int:
        return len(self.candidates)

    def capped(self, cap: int) -> list[Candidate]:
        """The cap is a deterministic function of evidence, then of concept id.

        Evidence here means *how many independent retrievers and views* found a concept -
        never a similarity score, which would smuggle retriever ranking into the cut.
        """
        ordered = sorted(
            self.candidates.values(),
            key=lambda c: (
                -int(c.exact_alias), -c.retriever_count, -c.view_count, c.best_rank,
                c.concept_id,
            ),
        )
        return ordered[:cap]


def retrieve_sparse(
    candidates: CandidateSet,
    index: LocalIndex | None,
    views: dict[str, str],
    policy: RetrievalPolicy,
    *,
    memo: SurfaceMemo | None = None,
    settings: SparseSettings | None = None,
) -> None:
    """Character-similarity retrieval for every view the policy allows."""
    for view in policy.sparse_views:
        query = views.get(view, "")
        if not query:
            continue
        for rank, hit in enumerate(
            sparse_search(
                index, query, top_k=policy.sparse_top_k, memo=memo, settings=settings
            ),
            start=1,
        ):
            candidates.add(hit.concept_id, RETRIEVER_SPARSE, view, rank)


def retrieve_dense(
    candidates: CandidateSet,
    hits_by_retriever: dict[str, list[tuple[str, str, int]]],
    policy: RetrievalPolicy,
) -> None:
    """Record dense hits that were computed elsewhere: `(view, concept_id, rank)`.

    The dense pass is run in bulk by the runtime so each encoder loads once for the whole
    corpus; this only folds its output into the union.
    """
    for retriever, hits in hits_by_retriever.items():
        for view, concept_id, rank in hits:
            if view not in policy.dense_views or rank > policy.dense_top_k:
                continue
            candidates.add(concept_id, retriever, view, rank)


def role_for_ontology(ontology: str) -> str:
    return ROLE_DIAGNOSIS if ontology == ONTOLOGY_ICD else ROLE_DRUG


def assert_role_allowed(retriever_roles: Sequence[str], ontology: str, key: str) -> None:
    """A retriever may only contribute to a role its registry entry declares.

    ClinLinker declares diagnosis only; this is the runtime half of that contract, so the
    policy and the registry cannot drift apart silently.
    """
    role = role_for_ontology(ontology)
    if role not in retriever_roles:
        raise ValueError(
            f"{key} is not declared for role {role!r} (declares {list(retriever_roles)}). "
            "Retriever roles come from the model registry, not from the retrieval policy."
        )


__all__ = [
    "RETRIEVER_EXACT",
    "RETRIEVER_SPARSE",
    "VIEWS",
    "VIEW_EN",
    "VIEW_RAW",
    "VIEW_VI",
    "Candidate",
    "CandidateSet",
    "RetrievalPolicy",
    "assert_role_allowed",
    "retrieve_dense",
    "retrieve_sparse",
    "role_for_ontology",
]
