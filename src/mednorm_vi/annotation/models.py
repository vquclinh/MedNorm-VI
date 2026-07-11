"""Typed contracts for team-owned annotation data.

Annotations are the ONLY legitimate source of development ground truth. The
organizer competition test set has no labels; a dataset that includes organizer
test inputs must declare ``labeled: false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..evaluation.models import Provenance  # shared provenance vocabulary


class ReviewStatus(str, Enum):
    DRAFT = "DRAFT"
    SINGLE_REVIEWED = "SINGLE_REVIEWED"
    DOUBLE_REVIEWED = "DOUBLE_REVIEWED"
    ADJUDICATED = "ADJUDICATED"
    REJECTED = "REJECTED"


#: Statuses that may be treated as final development gold by default.
ACCEPTED_GOLD_STATUSES: frozenset[str] = frozenset({ReviewStatus.ADJUDICATED.value})


class IntendedUse(str, Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    STRESS_TESTING = "stress_testing"
    LEADERBOARD_PRECHECK = "leaderboard_precheck"


@dataclass(frozen=True, slots=True)
class AnnotationEntity:
    """A single team-owned annotation (organizer-facing type + review metadata)."""

    document_id: str
    text: str
    type: str  # organizer Vietnamese label
    start: int
    end: int
    assertions: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    annotator_id: str = "anon"
    timestamp: str | None = None
    provenance: Provenance = Provenance.GOLD
    review_status: ReviewStatus = ReviewStatus.DRAFT
    notes: str | None = None
    source_dataset_id: str | None = None
    guideline_version: str | None = None
    route_case: str | None = None
    section: str | None = None

    @property
    def position(self) -> tuple[int, int]:
        return (self.start, self.end)

    def is_accepted_gold(self) -> bool:
        return self.review_status.value in ACCEPTED_GOLD_STATUSES


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Versioned manifest for an internal dataset."""

    dataset_id: str
    version: str
    provenance: str
    labeled: bool
    n_documents: int = 0
    n_entities: int = 0
    license_note: str | None = None
    entity_type_distribution: dict[str, int] = field(default_factory=dict)
    assertion_distribution: dict[str, int] = field(default_factory=dict)
    candidate_coverage: dict[str, int] = field(default_factory=dict)
    section_distribution: dict[str, int] = field(default_factory=dict)
    case_distribution: dict[str, int] = field(default_factory=dict)
    guideline_version: str | None = None
    annotator_count: int = 0
    review_status: str | None = None
    sha256: str | None = None
    intended_use: tuple[str, ...] = ()
    includes_organizer_test_inputs: bool = False


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    """Inter-annotator agreement for one document (annotator A vs B)."""

    document_id: str
    n_a: int
    n_b: int
    n_span_agree: int
    n_type_agree: int
    n_assertion_agree: int
    n_candidate_agree: int

    @property
    def span_agreement(self) -> float:
        denom = max(self.n_a, self.n_b)
        return self.n_span_agree / denom if denom else 1.0

    @property
    def type_agreement(self) -> float:
        denom = max(self.n_a, self.n_b)
        return self.n_type_agree / denom if denom else 1.0

    @property
    def assertion_agreement(self) -> float:
        return self.n_assertion_agree / self.n_span_agree if self.n_span_agree else 1.0

    @property
    def candidate_agreement(self) -> float:
        return self.n_candidate_agree / self.n_span_agree if self.n_span_agree else 1.0


@dataclass(frozen=True, slots=True)
class AdjudicationRecord:
    """A preserved disagreement + its adjudicated resolution."""

    document_id: str
    annotator_a: str
    annotator_b: str
    disagreement_reason: str
    adjudicated_result: AnnotationEntity
    adjudicator_id: str
    guideline_version: str
    notes: str | None = None


__all__ = [
    "ReviewStatus",
    "ACCEPTED_GOLD_STATUSES",
    "IntendedUse",
    "AnnotationEntity",
    "DatasetManifest",
    "AgreementSummary",
    "AdjudicationRecord",
    "Provenance",
]
