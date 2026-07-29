"""Locked-snapshot candidate grounding for L5 (spec §9, §10, P7).

Principle P7 is absolute: "LLMs must not recall ontology IDs from memory. They may
only select from retrieved candidates". Spec §16 adds the fail-fast rule — a code
absent from the frozen KB snapshot stops the file rather than being silently
repaired. This module is where both are enforced, for either ontology, independent
of which retriever or which model is in play:

    1  exact normalized alias lookup      (highest precision; wins outright)
    2  lexical / sparse retrieval
    3  dense retrieval
    4  optional reranking

Fallback order: an exact high-confidence alias hit is emitted alone; otherwise the
calibrated top-k; otherwise **empty**. An empty candidate list is a correct
statement that retrieval found nothing, and it is strictly safer than a plausible
invented code — a language model asked for an ICD-10 code will happily produce a
well-formed one that does not exist.

Membership in the locked snapshot is checked twice: once as each retrieval stage is
read, so an index that has drifted from the snapshot cannot smuggle a code through,
and once again before anything leaves this module.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.structured_output import normalize_json_completion
from ..schemas.constants import CANDIDATE_ONTOLOGY_BY_TYPE

SNAPSHOT_LINKING_CONTRACT_VERSION = "snapshot-linking-v1"

SOURCE_ALIAS_EXACT = "alias_exact"
SOURCE_LEXICAL = "lexical"
SOURCE_DENSE = "dense"
SOURCE_RERANKER = "reranker"
RETRIEVAL_SOURCES: tuple[str, ...] = (
    SOURCE_ALIAS_EXACT, SOURCE_LEXICAL, SOURCE_DENSE, SOURCE_RERANKER)

DEFAULT_TOP_K = 5
# An exact alias hit at or above this is emitted alone: the ontology already
# answered the question and a model reordering it can only make it worse.
DEFAULT_ALIAS_CONFIDENCE = 0.99


class SnapshotLinkingError(ValueError):
    """Raised when linking would emit a code the locked snapshot does not contain."""


def normalize_alias(text: str) -> str:
    """Casefold + NFC, whitespace-collapsed. Offsets are never touched."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieved ontology candidate with its provenance and score."""

    code: str
    ontology: str
    score: float
    source: str
    rank: int

    def __post_init__(self) -> None:
        if self.source not in RETRIEVAL_SOURCES:
            raise SnapshotLinkingError(f"unknown retrieval source {self.source!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "ontology": self.ontology, "score": self.score,
                "source": self.source, "rank": self.rank}


@dataclass(frozen=True, slots=True)
class OntologySnapshot:
    """A frozen ontology. Membership is the final authority on any code."""

    ontology: str
    snapshot_id: str
    codes: frozenset[str]
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def contains(self, code: str) -> bool:
        return code in self.codes

    def alias_hits(self, text: str) -> tuple[str, ...]:
        return tuple(self.aliases.get(normalize_alias(text), ()))

    def as_dict(self) -> dict[str, Any]:
        return {"ontology": self.ontology, "snapshot_id": self.snapshot_id,
                "code_count": len(self.codes), "alias_count": len(self.aliases)}


def ontology_for_type(entity_type: str) -> str | None:
    """Which ontology a type carries, if any. Cross-links are forbidden (spec §7.3)."""
    return CANDIDATE_ONTOLOGY_BY_TYPE.get(entity_type)


def generate_candidates(
    *,
    mention_text: str,
    entity_type: str,
    snapshot: OntologySnapshot,
    lexical: Sequence[tuple[str, float]] = (),
    dense: Sequence[tuple[str, float]] = (),
    reranked: Sequence[tuple[str, float]] = (),
    top_k: int = DEFAULT_TOP_K,
    alias_confidence: float = DEFAULT_ALIAS_CONFIDENCE,
) -> tuple[Candidate, ...]:
    """Snapshot-grounded candidates, in the documented order."""
    expected = ontology_for_type(entity_type)
    if expected is None:
        return ()
    if expected != snapshot.ontology:
        raise SnapshotLinkingError(
            f"{entity_type} takes {expected} candidates but the snapshot is "
            f"{snapshot.ontology}; cross-ontology linking is forbidden (spec §7.3)")

    exact = [c for c in snapshot.alias_hits(mention_text) if snapshot.contains(c)]
    if exact:
        return tuple(
            Candidate(code=code, ontology=snapshot.ontology,
                      score=alias_confidence, source=SOURCE_ALIAS_EXACT, rank=rank)
            for rank, code in enumerate(sorted(exact), start=1))

    pooled: dict[str, tuple[float, str]] = {}
    for source, results in ((SOURCE_LEXICAL, lexical), (SOURCE_DENSE, dense),
                            (SOURCE_RERANKER, reranked)):
        for code, score in results:
            if not snapshot.contains(code):
                continue
            current = pooled.get(code)
            if current is None or float(score) > current[0]:
                pooled[code] = (float(score), source)
    ordered = sorted(pooled.items(), key=lambda item: (-item[1][0], item[0]))
    return tuple(
        Candidate(code=code, ontology=snapshot.ontology, score=score,
                  source=source, rank=rank)
        for rank, (code, (score, source)) in enumerate(ordered[:top_k], start=1))


# Spec §12.1's constrained output shape for candidate selection. The model chooses
# from an offered set; it has no field with which to introduce a code.
SELECTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["selected"],
    "properties": {
        "selected": {"type": "array", "items": {"type": "string"},
                     "uniqueItems": True},
    },
    "additionalProperties": False,
}


def constrain_selection(
    payload: str, candidates: Sequence[Candidate], *, snapshot: OntologySnapshot,
) -> tuple[tuple[Candidate, ...], tuple[str, ...]]:
    """Constrain a model's selection to the offered set (spec P7, §12.1).

    Returns ``(selected, rejected_codes)``. A code the model returns that was not
    offered is dropped and reported — never emitted, **even if it exists in the
    snapshot**, because it was not retrieved for this mention. A malformed payload
    falls back to the retrieved order unchanged rather than to an empty answer.
    """
    import json

    offered = {candidate.code: candidate for candidate in candidates}
    try:
        document = json.loads(normalize_json_completion(payload))
        proposed = document["selected"]
        if not isinstance(proposed, list):
            raise TypeError("selected is not a list")
    except (json.JSONDecodeError, TypeError, KeyError):
        return tuple(candidates), ()

    selected: list[Candidate] = []
    rejected: list[str] = []
    for code in proposed:
        if not isinstance(code, str) or code not in offered:
            rejected.append(str(code))
            continue
        selected.append(offered[code])
    # Final authority: the snapshot, again.
    verified = tuple(c for c in selected if snapshot.contains(c.code))
    rejected.extend(c.code for c in selected if not snapshot.contains(c.code))
    return verified, tuple(rejected)


def assert_codes_in_snapshot(
    codes: Sequence[str], *, snapshot: OntologySnapshot,
) -> None:
    """Every emitted code must exist in the locked snapshot (spec §16 fail-fast)."""
    missing = [code for code in codes if not snapshot.contains(code)]
    if missing:
        raise SnapshotLinkingError(
            f"{sorted(missing)} are not in {snapshot.ontology} snapshot "
            f"{snapshot.snapshot_id}; codes are retrieved, never generated")


@dataclass(frozen=True, slots=True)
class SnapshotLinkingResult:
    """Candidates for one mention, with full retrieval provenance."""

    entity_type: str
    ontology: str | None
    candidates: tuple[Candidate, ...]
    rejected_codes: tuple[str, ...] = ()
    fallback: str = ""

    def codes(self) -> tuple[str, ...]:
        return tuple(candidate.code for candidate in self.candidates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": SNAPSHOT_LINKING_CONTRACT_VERSION,
            "entity_type": self.entity_type,
            "ontology": self.ontology,
            "candidates": [c.as_dict() for c in self.candidates],
            "codes": list(self.codes()),
            "rejected_codes": list(self.rejected_codes),
            "fallback": self.fallback,
            "codes_were_generated_by_a_model": False,
        }


def link_against_snapshot(
    *,
    mention_text: str,
    entity_type: str,
    snapshot: OntologySnapshot | None,
    lexical: Sequence[tuple[str, float]] = (),
    dense: Sequence[tuple[str, float]] = (),
    reranked: Sequence[tuple[str, float]] = (),
    selection_payload: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> SnapshotLinkingResult:
    """Full candidate pipeline for one mention, ending in the snapshot check."""
    ontology = ontology_for_type(entity_type)
    if ontology is None or snapshot is None:
        return SnapshotLinkingResult(
            entity_type=entity_type, ontology=ontology, candidates=(),
            fallback="type_carries_no_candidates")
    candidates = generate_candidates(
        mention_text=mention_text, entity_type=entity_type, snapshot=snapshot,
        lexical=lexical, dense=dense, reranked=reranked, top_k=top_k)
    if not candidates:
        return SnapshotLinkingResult(
            entity_type=entity_type, ontology=ontology, candidates=(),
            fallback="no_candidate_retrieved")
    if candidates[0].source == SOURCE_ALIAS_EXACT:
        assert_codes_in_snapshot([c.code for c in candidates], snapshot=snapshot)
        return SnapshotLinkingResult(
            entity_type=entity_type, ontology=ontology, candidates=candidates,
            fallback="exact_alias")
    rejected: tuple[str, ...] = ()
    if selection_payload:
        candidates, rejected = constrain_selection(
            selection_payload, candidates, snapshot=snapshot)
    assert_codes_in_snapshot([c.code for c in candidates], snapshot=snapshot)
    return SnapshotLinkingResult(
        entity_type=entity_type, ontology=ontology, candidates=candidates,
        rejected_codes=rejected, fallback="calibrated_top_k")


__all__ = [
    "DEFAULT_ALIAS_CONFIDENCE",
    "DEFAULT_TOP_K",
    "RETRIEVAL_SOURCES",
    "SELECTION_SCHEMA",
    "SNAPSHOT_LINKING_CONTRACT_VERSION",
    "SOURCE_ALIAS_EXACT",
    "SOURCE_DENSE",
    "SOURCE_LEXICAL",
    "SOURCE_RERANKER",
    "Candidate",
    "OntologySnapshot",
    "SnapshotLinkingError",
    "SnapshotLinkingResult",
    "assert_codes_in_snapshot",
    "constrain_selection",
    "generate_candidates",
    "link_against_snapshot",
    "normalize_alias",
    "ontology_for_type",
]
