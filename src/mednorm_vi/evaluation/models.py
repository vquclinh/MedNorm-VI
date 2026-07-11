"""Typed data contracts for the PROVISIONAL LOCAL EVALUATOR.

These are pure dataclasses/enums — no scoring logic. The evaluator scores
team-owned labeled data (GOLD/SILVER/SYNTHETIC/ORGANIZER_PUBLISHED_EXAMPLE/
EXTERNAL_PERMITTED). The organizer competition test set has **no ground truth**;
``ORGANIZER_TEST`` is never a valid labeled provenance.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provenance(str, Enum):
    """Trust/source level of team-owned labeled evaluation data."""

    GOLD = "GOLD"
    SILVER = "SILVER"
    SYNTHETIC = "SYNTHETIC"
    ORGANIZER_PUBLISHED_EXAMPLE = "ORGANIZER_PUBLISHED_EXAMPLE"
    EXTERNAL_PERMITTED = "EXTERNAL_PERMITTED"


#: The string the organizer test inputs use. It is NEVER a labeled provenance.
ORGANIZER_TEST = "ORGANIZER_TEST"

#: All provenance levels that may carry labels.
LABELED_PROVENANCE: frozenset[str] = frozenset(p.value for p in Provenance)


def parse_provenance(value: str) -> Provenance:
    """Parse a provenance string; reject ``ORGANIZER_TEST`` and unknowns."""
    if value == ORGANIZER_TEST:
        raise ValueError(
            "ORGANIZER_TEST is input-only and can never be a labeled provenance; "
            "the organizer competition test set has no ground truth."
        )
    try:
        return Provenance(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown provenance {value!r}; allowed: {sorted(LABELED_PROVENANCE)}"
        ) from exc


# --- Entities & documents -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvaluationEntity:
    """One organizer-facing entity in a labeled document or a prediction.

    ``type`` is the organizer Vietnamese label. ``route_case`` (C1-C7) and
    ``section`` are optional metadata used only for diagnostic grouping.
    """

    text: str
    type: str
    start: int
    end: int
    assertions: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    route_case: str | None = None
    section: str | None = None

    @property
    def position(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """A labeled document (ground truth) or a prediction document."""

    document_id: str
    entities: tuple[EvaluationEntity, ...]
    provenance: Provenance | None = None
    original_text: str | None = None


# --- Matching -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MatchingDecision:
    """A single matcher decision linking a GT index to a prediction index."""

    gt_index: int
    pred_index: int
    strategy: str
    cost: float


@dataclass(frozen=True, slots=True)
class EntityPair:
    """A matched (GT, prediction) pair with everything needed for replay."""

    document_id: str
    gt_index: int
    pred_index: int
    entity_type: str
    gt_text: str
    pred_text: str
    gt_position: tuple[int, int]
    pred_position: tuple[int, int]
    strategy: str
    cost: float


@dataclass(frozen=True, slots=True)
class UnmatchedGroundTruth:
    document_id: str
    gt_index: int
    entity_type: str
    text: str
    position: tuple[int, int]


@dataclass(frozen=True, slots=True)
class UnmatchedPrediction:
    document_id: str
    pred_index: int
    entity_type: str
    text: str
    position: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MatchingResult:
    """Output of a matcher for one document."""

    pairs: tuple[MatchingDecision, ...]
    unmatched_gt: tuple[int, ...]
    unmatched_pred: tuple[int, ...]
    strategy: str


# --- Score breakdowns ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WERBreakdown:
    """Token-sequence Levenshtein WER breakdown for one matched pair."""

    tokenization: str
    substitutions: int
    deletions: int
    insertions: int
    ref_tokens: int
    raw_wer: float
    raw_text_score: float  # 1 - raw_wer (may be negative when WER > 1)
    clipping_enabled: bool
    clipped_text_score: float | None = None

    @property
    def text_score(self) -> float:
        """The score used downstream: clipped if enabled, else raw."""
        if self.clipping_enabled and self.clipped_text_score is not None:
            return self.clipped_text_score
        return self.raw_text_score


@dataclass(frozen=True, slots=True)
class SetSimilarityBreakdown:
    """Deterministic Jaccard breakdown over assertions or candidates."""

    kind: str  # "assertions" | "candidates"
    gt_ordered: tuple[str, ...]
    pred_ordered: tuple[str, ...]
    gt_deduped: tuple[str, ...]
    pred_deduped: tuple[str, ...]
    intersection: tuple[str, ...]
    union: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    jaccard: float
    gt_duplicates: tuple[str, ...] = ()
    pred_duplicates: tuple[str, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PerEntityScore:
    """Per-entity (or per-slot) score, including unmatched slots."""

    document_id: str
    slot_kind: str  # "matched" | "missing" | "spurious"
    entity_type: str
    text_score: float
    candidate_weight: float
    diagnostics: tuple[str, ...] = ()
    pair: EntityPair | None = None
    wer: WERBreakdown | None = None
    assertions: SetSimilarityBreakdown | None = None
    candidates: SetSimilarityBreakdown | None = None
    route_case: str | None = None
    section: str | None = None
    provenance: Provenance | None = None
    assertions_eligible: bool = False
    candidates_eligible: bool = False


@dataclass(frozen=True, slots=True)
class PerDocumentScore:
    document_id: str
    provenance: Provenance | None
    text_score: float
    assertions_score: float
    candidates_score: float
    final_score: float
    n_gt: int
    n_pred: int
    n_matched: int
    n_missing: int
    n_spurious: int
    per_entity: tuple[PerEntityScore, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusScore:
    text_score: float
    assertions_score: float
    candidates_score: float
    final_score: float
    aggregation_policy: str
    n_documents: int
    n_entities: int
    n_matched: int
    n_missing: int
    n_spurious: int
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    per_case: dict[str, dict[str, float]] = field(default_factory=dict)


# --- Config & run metadata ----------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Resolved evaluator configuration (see configs/evaluation/provisional_v1.yaml)."""

    evaluator_version: str
    matching_strategy: str
    tokenization: str
    aggregation_policy: str
    clipping_enabled: bool
    cost_weights: dict[str, float]
    max_matching_cost: float
    allowed_provenance: tuple[str, ...]
    deterministic: bool = True
    emit_html: bool = True
    weight_text: float = 0.3
    weight_assertions: float = 0.3
    weight_candidates: float = 0.4

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> EvaluationConfig:
        weights = data.get("final_weights", {}) or {}
        cost = data.get("cost_weights", {}) or {}
        report = data.get("report", {}) or {}
        return EvaluationConfig(
            evaluator_version=str(data.get("evaluator_version", "provisional-v1")),
            matching_strategy=str(data.get("matching_strategy", "exact-text-occurrence")),
            tokenization=str(data.get("tokenization", "whitespace-punctuation")),
            aggregation_policy=str(data.get("aggregation_policy", "provisional-v1")),
            clipping_enabled=bool(data.get("clip_text_score", False)),
            cost_weights={str(k): float(v) for k, v in cost.items()},
            max_matching_cost=float(data.get("max_matching_cost", 1.0)),
            allowed_provenance=tuple(
                str(p) for p in data.get("allowed_provenance", sorted(LABELED_PROVENANCE))
            ),
            deterministic=bool(data.get("deterministic", True)),
            emit_html=bool(report.get("emit_html", True)),
            weight_text=float(weights.get("text", 0.3)),
            weight_assertions=float(weights.get("assertions", 0.3)),
            weight_candidates=float(weights.get("candidates", 0.4)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationRunMetadata:
    evaluator_version: str
    timestamp_utc: str
    python_version: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None


# --- Diagnostics --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One structured diagnostic finding for later error analysis."""

    category: str
    document_id: str
    entity_type: str | None = None
    gt_index: int | None = None
    pred_index: int | None = None
    route_case: str | None = None
    section: str | None = None
    provenance: Provenance | None = None
    experiment_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationDiagnostics:
    """Aggregated diagnostics for a whole run."""

    items: tuple[Diagnostic, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.items:
            out[d.category] = out.get(d.category, 0) + 1
        return dict(sorted(out.items()))


# --- Replay -------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReplayManifest:
    evaluator_version: str
    config_hash: str
    ground_truth_dir: str
    predictions_dir: str
    ground_truth_file_hashes: dict[str, str]
    prediction_file_hashes: dict[str, str]
    ground_truth_dir_hash: str
    predictions_dir_hash: str
    python_version: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None
    matching_strategy: str
    tokenization: str
    aggregation_policy: str
    clipping_enabled: bool
    ground_truth_provenance: str
    prediction_experiment_id: str | None
    timestamp_utc: str
    report_file_hashes: dict[str, str] = field(default_factory=dict)


# --- Serialization helper -----------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums/tuples to JSON-serializable data.

    Deterministic: dict keys are emitted in their existing order (callers are
    expected to build ordered dicts).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
