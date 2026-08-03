"""GraphCENT orchestration: type gate, retrieval, disambiguation, tier gate (0080).

One inference pass produces every output variant. The tier of each selected candidate is
recorded per mention, so `allnull`, `tierA`, `tierAB` and `tierABC` are four serializations of
the same run rather than four GPU runs.

The organizer type from E3 is a hard gate, in the MedType spirit: CHẨN_ĐOÁN reaches ICD only,
THUỐC reaches RxNorm only, and everything else never enters candidate linking at all. Nothing
downstream may retype an entity or mutate an assertion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..reasoner.validator import ASSERTION_TYPES, CANDIDATE_TYPES
from .disambiguation import (
    NULL_DECISION,
    TIER_NONE,
    Decision,
    TierPolicy,
    classify,
    emit_for_variant,
)
from .ontology import ONTOLOGY_ICD, ONTOLOGY_RXNORM, IcdFacts, RxNormFacts
from .retrieval import CandidateEvidence
from .spans import SpanAlternative

#: MedType-style hard gate. The organizer's predicted type decides the ontology.
ONTOLOGY_FOR_TYPE: dict[str, str] = {
    "CHẨN_ĐOÁN": ONTOLOGY_ICD,
    "THUỐC": ONTOLOGY_RXNORM,
}


def linkable(entity_type: str) -> bool:
    """Only diagnoses and drugs are linked. Symptoms and lab types never are."""
    return entity_type in ONTOLOGY_FOR_TYPE


@dataclass
class MentionResult:
    """Per-mention evidence and outcome. Carries no chain of thought."""

    document: str
    text: str
    entity_type: str
    position: tuple[int, int]
    span_provenance: str
    candidates: list[CandidateEvidence] = field(default_factory=list)
    decision: Decision = NULL_DECISION
    tiers: dict[str, str] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)

    def emitted_for(self, variant: str) -> list[str]:
        """Governed ids this variant emits, in the model's selection order."""
        return [
            concept_id
            for concept_id in self.decision.candidate_ids
            if emit_for_variant(self.tiers.get(concept_id, TIER_NONE), variant)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "text": self.text,
            "type": self.entity_type,
            "position": list(self.position),
            "span_provenance": self.span_provenance,
            "candidates": [
                {"candidate_index": index, **candidate.as_dict()}
                for index, candidate in enumerate(self.candidates)
            ],
            "decision": self.decision.as_dict(),
            "tiers": dict(sorted(self.tiers.items())),
            "facts": self.facts,
            "emitted": {v: self.emitted_for(v) for v in ("tierA", "tierAB", "tierABC")},
        }


def resolve_tiers(
    result: MentionResult,
    facts_for: Callable[[str], IcdFacts | RxNormFacts | None],
    mention_has_structure: bool,
    policy: TierPolicy,
) -> None:
    """Classify every selected candidate. Unselected candidates are never emitted."""
    by_id = {c.concept_id: c for c in result.candidates}
    for concept_id in result.decision.candidate_ids:
        evidence = by_id.get(concept_id)
        if evidence is None:
            result.tiers[concept_id] = TIER_NONE
            continue
        result.tiers[concept_id] = classify(
            evidence, facts_for(concept_id), mention_has_structure, policy
        )


def serialize_document(
    seed: list[dict[str, Any]],
    results: list[MentionResult],
    variant: str,
    *,
    apply_spans: bool,
) -> list[dict[str, Any]]:
    """Organizer JSON for one variant.

    Types and assertions are copied from the E3 seed untouched; only `candidates` - and, in
    `joint_safe_span` mode, an allowed exact-source boundary - may differ.
    """
    by_position = {(int(e["position"][0]), int(e["position"][1])): e for e in seed}
    replacements: dict[tuple[int, int], MentionResult] = {}
    for result in results:
        if apply_spans and result.span_provenance != "e3_original":
            replacements[result.position] = result

    out: list[dict[str, Any]] = []
    consumed: set[tuple[int, int]] = set()
    for result in results:
        original = by_position.get(result.position)
        if original is None and result.position in replacements:
            original = None
        if original is not None:
            consumed.add(result.position)

    for entity in seed:
        key = (int(entity["position"][0]), int(entity["position"][1]))
        row: dict[str, Any] = {
            "text": entity["text"],
            "type": entity["type"],
            "position": list(entity["position"]),
        }
        if entity["type"] in ASSERTION_TYPES:
            row["assertions"] = list(entity.get("assertions") or [])  # never mutated
        if entity["type"] in CANDIDATE_TYPES:
            row["candidates"] = []
            match = next(
                (
                    r
                    for r in results
                    if r.position == key
                    or (
                        apply_spans
                        and r.span_provenance != "e3_original"
                        and r.position[0] == key[0]
                    )
                ),
                None,
            )
            if match is not None:
                row["candidates"] = match.emitted_for(variant)
                if apply_spans and match.span_provenance != "e3_original":
                    row["text"] = match.text
                    row["position"] = [match.position[0], match.position[1]]
        out.append(row)
    out.sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
    return out


__all__ = [
    "ONTOLOGY_FOR_TYPE",
    "MentionResult",
    "SpanAlternative",
    "linkable",
    "resolve_tiers",
    "serialize_document",
]
