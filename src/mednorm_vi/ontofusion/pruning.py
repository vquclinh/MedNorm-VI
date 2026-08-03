"""Semantic, type and structured pruning before disambiguation (0082).

MedType's contribution here is the type gate, not a new classifier: the organizer type of a
verified mention already decides which ontology may answer, so a diagnosis can never be
linked to a drug and a lab entity is never linked at all.

Beyond that gate, hard pruning is allowed **only when governed metadata proves a conflict**.
The rule this module is built around is that absence of evidence is not evidence: if a
candidate records no strength, a mention that states one does not conflict with it - it is
merely unspecific, and the set-wise reranker is the right place to weigh that. Every drop is
counted with its reason, so an over-eager rule shows up as a number rather than as quiet
recall loss.

Nothing here contains a disease, a drug, a code or any term taken from a test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graphcent.ontology import ONTOLOGY_ICD, ONTOLOGY_RXNORM
from ..kb.rxnorm.structured import StructuredDrug, parse_strengths
from ..reasoner.validator import CANDIDATE_TYPES
from .retrieval import Candidate

#: The only mapping from organizer type to ontology. Anything not here is not linkable.
ONTOLOGY_FOR_TYPE: dict[str, str] = {
    "CHẨN_ĐOÁN": ONTOLOGY_ICD,
    "THUỐC": ONTOLOGY_RXNORM,
}

PRUNE_WRONG_ONTOLOGY = "wrong_ontology"
PRUNE_NOT_GOVERNED = "not_governed"
PRUNE_STRENGTH_CONFLICT = "strength_conflict"
PRUNE_FORM_CONFLICT = "dose_form_conflict"
PRUNE_ROUTE_CONFLICT = "route_conflict"

#: Generic dose-form and route vocabulary. These are ordinary pharmaceutical words, not
#: entities: they say how a drug is given, never which drug it is.
_FORM_WORDS: dict[str, tuple[str, ...]] = {
    "tablet": ("viên nén", "viên"),
    "capsule": ("viên nang", "nang"),
    "injection": ("tiêm", "ống", "lọ tiêm"),
    "solution": ("dung dịch", "siro", "sirô"),
    "cream": ("kem", "mỡ", "thuốc mỡ"),
    "suppository": ("viên đặt", "đặt"),
}
_ROUTE_WORDS: dict[str, tuple[str, ...]] = {
    "oral": ("uống", "po", "đường uống"),
    "injection": ("tiêm", "iv", "im", "truyền"),
    "topical": ("bôi", "thoa", "ngoài da"),
    "rectal": ("đặt hậu môn",),
    "inhalation": ("xịt", "khí dung", "hít"),
}


def linkable(entity_type: str) -> bool:
    """Only the two candidate-bearing organizer types ever reach a linker."""
    return entity_type in CANDIDATE_TYPES and entity_type in ONTOLOGY_FOR_TYPE


def ontology_for(entity_type: str) -> str:
    return ONTOLOGY_FOR_TYPE.get(entity_type, "")


def _stated(mention: str, vocabulary: dict[str, tuple[str, ...]]) -> set[str]:
    """Which generic categories the mention explicitly states."""
    lowered = f" {' '.join(mention.casefold().split())} "
    return {
        category
        for category, words in vocabulary.items()
        if any(f" {word} " in lowered for word in words)
    }


def _candidate_categories(text: str, vocabulary: dict[str, tuple[str, ...]]) -> set[str]:
    lowered = (text or "").casefold()
    return {category for category in vocabulary if category in lowered}


@dataclass
class PruneOutcome:
    kept: list[Candidate] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)

    def drop(self, candidate: Candidate, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.dropped[candidate.concept_id] = reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": [c.concept_id for c in self.kept],
            "reasons": dict(self.reasons),
            "dropped": dict(self.dropped),
        }


def strength_conflict(mention: str, drug: StructuredDrug | None) -> bool:
    """True only when both sides state a strength and they disagree.

    A candidate with no recorded strength is unspecific, not contradictory - dropping it
    here would be fabricating the attribute the source declined to record.
    """
    if drug is None:
        return False
    stated = {s.key for s in parse_strengths(mention)}
    recorded = drug.strength_keys
    if not stated or not recorded:
        return False
    return not (stated & recorded)


def form_conflict(mention: str, drug: StructuredDrug | None) -> bool:
    """True only when the mention names a dose form and the candidate records another."""
    if drug is None:
        return False
    recorded_text = " ".join((*drug.dose_forms, drug.rxterm_form)).strip()
    if not recorded_text:
        return False
    stated = _stated(mention, _FORM_WORDS)
    recorded = _candidate_categories(recorded_text, _FORM_WORDS)
    if not stated or not recorded:
        return False
    return not (stated & recorded)


def route_conflict(mention: str, drug: StructuredDrug | None) -> bool:
    """True only when the mention names a route and governed metadata records another.

    RxNorm does not carry a route attribute in the recovered 0074 set, so this fires from
    the dose form's implied route only, and only when both sides are explicit.
    """
    if drug is None:
        return False
    recorded_text = " ".join((*drug.dose_forms, drug.rxterm_form)).strip()
    if not recorded_text:
        return False
    stated = _stated(mention, _ROUTE_WORDS)
    recorded = _candidate_categories(recorded_text, _ROUTE_WORDS)
    if not stated or not recorded:
        return False
    return not (stated & recorded)


def prune(
    candidates: list[Candidate],
    *,
    entity_type: str,
    mention: str,
    governed: frozenset[str],
    structured: dict[str, StructuredDrug] | None = None,
) -> PruneOutcome:
    """Type gate, governance gate, then structured conflicts. Everything else survives."""
    outcome = PruneOutcome()
    ontology = ontology_for(entity_type)
    drugs = structured or {}

    for candidate in candidates:
        if candidate.ontology != ontology:
            outcome.drop(candidate, PRUNE_WRONG_ONTOLOGY)
            continue
        if candidate.concept_id not in governed:
            outcome.drop(candidate, PRUNE_NOT_GOVERNED)
            continue
        if ontology == ONTOLOGY_RXNORM:
            drug = drugs.get(candidate.concept_id)
            if strength_conflict(mention, drug):
                outcome.drop(candidate, PRUNE_STRENGTH_CONFLICT)
                continue
            if form_conflict(mention, drug):
                outcome.drop(candidate, PRUNE_FORM_CONFLICT)
                continue
            if route_conflict(mention, drug):
                outcome.drop(candidate, PRUNE_ROUTE_CONFLICT)
                continue
        outcome.kept.append(candidate)
    return outcome


__all__ = [
    "ONTOLOGY_FOR_TYPE",
    "PRUNE_FORM_CONFLICT",
    "PRUNE_NOT_GOVERNED",
    "PRUNE_ROUTE_CONFLICT",
    "PRUNE_STRENGTH_CONFLICT",
    "PRUNE_WRONG_ONTOLOGY",
    "PruneOutcome",
    "form_conflict",
    "linkable",
    "ontology_for",
    "prune",
    "route_conflict",
    "strength_conflict",
]
