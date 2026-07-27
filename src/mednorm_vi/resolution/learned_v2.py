"""Learned L4 Boundary & Type Resolver v2.

The resolver v2 code here implements the trainable contract from architecture
§7 without doing local training. It builds deterministic supervised examples
from governed train/validation lattices, records leakage-safe split identities,
and provides a read-only linear checkpoint runtime for future Colab-trained
weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..lattice.models import SpanLattice
from ..lattice.models import SpanProposal as LatticeProposal
from ..schemas.constants import ENTITY_TYPES
from ..schemas.hypotheses import TypedHypothesis
from ..schemas.spans import Span, SpanCoordinates

RESOLVER_V2_VERSION = "learned-l4-boundary-type-resolver-v2"
BOUNDARY_KEEP = "KEEP"
BOUNDARY_SELECT_EXISTING = "SELECT_EXISTING"
BOUNDARY_OFFSET = "OFFSET"
BOUNDARY_DROP = "DROP"
TYPE_DROP = "DROP"
SUPPORTED_BOUNDARY_ACTIONS: tuple[str, ...] = (
    BOUNDARY_KEEP,
    BOUNDARY_SELECT_EXISTING,
    BOUNDARY_OFFSET,
    BOUNDARY_DROP,
)
TYPE_ORDER: tuple[str, ...] = (
    "MEDICATION",
    "DIAGNOSIS",
    "SYMPTOM",
    "TEST_NAME",
    "TEST_RESULT",
)


class ResolverV2Error(ValueError):
    """Raised when learned-L4 training or inference contracts are violated."""


class ResolverV2UnavailableError(RuntimeError):
    """Raised when learned-L4 v2 is enabled without a valid local checkpoint."""


@dataclass(frozen=True, slots=True)
class GoldMention:
    document_id: str
    start: int
    end: int
    text: str
    entity_type: str
    source_group: str = ""

    def validate_against(self, original_text: str, document_id: str) -> None:
        if self.document_id != document_id:
            raise ResolverV2Error("gold mention document_id does not match lattice")
        if self.entity_type not in ENTITY_TYPES:
            raise ResolverV2Error(f"unsupported gold type {self.entity_type!r}")
        if self.end <= self.start:
            raise ResolverV2Error(f"invalid gold offsets {self.start}:{self.end}")
        if original_text[self.start:self.end] != self.text:
            raise ResolverV2Error("gold mention text is not an exact original_text slice")


@dataclass(frozen=True, slots=True)
class ResolverV2Config:
    enabled: bool = False
    checkpoint_path: str = ""
    expected_checkpoint_sha256: str = ""
    max_boundary_delta: int = 12
    keep_threshold: float = 0.5
    abstain_margin: float = 0.05
    wrong_type_risk_threshold: float = 0.35
    config_version: str = RESOLVER_V2_VERSION


@dataclass(frozen=True, slots=True)
class BoundaryTarget:
    action: str
    selected_proposal_id: str = ""
    start_delta: int = 0
    end_delta: int = 0
    token_iou: float = 0.0
    wer_boundary_surrogate: float = 0.0

    def __post_init__(self) -> None:
        if self.action not in SUPPORTED_BOUNDARY_ACTIONS:
            raise ResolverV2Error(f"unsupported boundary target action {self.action!r}")


@dataclass(frozen=True, slots=True)
class TypeTarget:
    entity_type: str
    wrong_type_cost: float
    diagnosis_symptom_pair: bool = False

    def __post_init__(self) -> None:
        if self.entity_type != TYPE_DROP and self.entity_type not in ENTITY_TYPES:
            raise ResolverV2Error(f"unsupported type target {self.entity_type!r}")


@dataclass(frozen=True, slots=True)
class ResolverV2TrainingExample:
    example_id: str
    document_id: str
    proposal_id: str
    proposal_start: int
    proposal_end: int
    proposal_text: str
    source_group: str
    split: str
    features: Mapping[str, float]
    boundary_target: BoundaryTarget
    type_target: TypeTarget
    config_sha256: str


@dataclass(frozen=True, slots=True)
class ResolverV2Thresholds:
    keep_threshold: float
    abstain_margin: float
    wrong_type_risk_threshold: float


@dataclass(frozen=True, slots=True)
class ResolverV2CheckpointMetadata:
    stage: str
    expert: str
    config_sha256: str
    corpus_sha256: str
    model_revision: str
    seed: int
    git_commit: str
    checkpoint_sha256: str
    parameter_count: int
    train_split_id: str
    validation_split_id: str
    internal_test_accessed: bool

    def validate(self) -> None:
        if self.expert != RESOLVER_V2_VERSION:
            raise ResolverV2Error("learned L4 checkpoint metadata expert mismatch")
        if self.internal_test_accessed:
            raise ResolverV2Error("learned L4 checkpoint must not access internal_test")


@dataclass(frozen=True, slots=True)
class ResolverV2Checkpoint:
    metadata: ResolverV2CheckpointMetadata
    action_weights: Mapping[str, Mapping[str, float]]
    type_weights: Mapping[str, Mapping[str, float]]
    wrong_type_weights: Mapping[str, float]
    offset_start_weights: Mapping[str, float] = field(default_factory=dict)
    offset_end_weights: Mapping[str, float] = field(default_factory=dict)
    thresholds: ResolverV2Thresholds = field(
        default_factory=lambda: ResolverV2Thresholds(0.5, 0.05, 0.35)
    )

    def validate(self) -> None:
        self.metadata.validate()
        for action in self.action_weights:
            if action not in SUPPORTED_BOUNDARY_ACTIONS:
                raise ResolverV2Error(f"checkpoint has unknown boundary action {action!r}")
        for entity_type in self.type_weights:
            if entity_type != TYPE_DROP and entity_type not in ENTITY_TYPES:
                raise ResolverV2Error(f"checkpoint has unknown type {entity_type!r}")


@dataclass(frozen=True, slots=True)
class ResolverV2Decision:
    hypothesis_id: str
    proposal_id: str
    status: str
    boundary_action: str
    original_start: int
    original_end: int
    start: int
    end: int
    entity_type: str
    calibrated_score: float
    wrong_type_risk: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "boundary_action": self.boundary_action,
            "original_span": [self.original_start, self.original_end],
            "span": [self.start, self.end],
            "entity_type": self.entity_type,
            "calibrated_score": round(self.calibrated_score, 6),
            "wrong_type_risk": round(self.wrong_type_risk, 6),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ResolverV2Result:
    document_id: str
    hypotheses: tuple[TypedHypothesis, ...]
    decisions: tuple[ResolverV2Decision, ...]
    config_sha256: str
    checkpoint_sha256: str

    def accepted(self) -> tuple[TypedHypothesis, ...]:
        return tuple(hypothesis for hypothesis in self.hypotheses if not hypothesis.abstained)

    def determinism_hash(self) -> str:
        payload = json.dumps(
            [decision.as_dict() for decision in self.decisions],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_sha256(config: ResolverV2Config) -> str:
    payload = {
        "max_boundary_delta": config.max_boundary_delta,
        "keep_threshold": config.keep_threshold,
        "abstain_margin": config.abstain_margin,
        "wrong_type_risk_threshold": config.wrong_type_risk_threshold,
        "config_version": config.config_version,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def span_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return 0.0 if union <= 0 else intersection / union


def _best_gold(
    proposal: LatticeProposal, gold_mentions: Sequence[GoldMention]
) -> tuple[GoldMention | None, float]:
    ranked = sorted(
        (
            (gold, span_iou(proposal.coordinates, (gold.start, gold.end)))
            for gold in gold_mentions
        ),
        key=lambda item: (-item[1], item[0].start, item[0].end, item[0].entity_type),
    )
    if not ranked or ranked[0][1] == 0.0:
        return None, 0.0
    return ranked[0]


def _source_group(lattice: SpanLattice, fallback: str) -> str:
    if fallback:
        return fallback
    return lattice.document_id


def _best_existing_for_gold(lattice: SpanLattice, gold: GoldMention) -> str:
    for proposal in lattice.proposals:
        if proposal.start == gold.start and proposal.end == gold.end:
            return "|".join(source.proposal_id for source in proposal.sources)
    return ""


def _nearby_char_features(original_text: str, start: int, end: int) -> dict[str, float]:
    before = original_text[start - 1] if start > 0 else ""
    after = original_text[end] if end < len(original_text) else ""
    punctuation = set(string.punctuation + ":;,.")
    return {
        "left_is_space": float(before.isspace()) if before else 1.0,
        "right_is_space": float(after.isspace()) if after else 1.0,
        "left_is_punct": float(before in punctuation) if before else 0.0,
        "right_is_punct": float(after in punctuation) if after else 0.0,
        "left_is_digit": float(before.isdigit()) if before else 0.0,
        "right_is_digit": float(after.isdigit()) if after else 0.0,
    }


def proposal_features(proposal: LatticeProposal, lattice: SpanLattice) -> dict[str, float]:
    """Feature vector specified by §7: evidence, boundaries, route, section, text."""
    sources = proposal.sources
    expert_ids = {source.expert_id for source in sources}
    best_type = proposal.best_type()
    type_values = [float(proposal.type_scores.get(entity_type, 0.0)) for entity_type in TYPE_ORDER]
    max_type_score = max(type_values) if type_values else 0.0
    type_support = sum(1 for value in type_values if value > 0.0)
    overlapping = [
        other for other in lattice.proposals if other is not proposal and proposal.overlaps(other)
    ]
    features: dict[str, float] = {
        "bias": 1.0,
        "start": float(proposal.start),
        "end": float(proposal.end),
        "candidate_length": float(proposal.length),
        "local_score_max": proposal.local_score(),
        "expert_count": float(len(expert_ids)),
        "source_count": float(len(sources)),
        "type_support_count": float(type_support),
        "max_type_score": max_type_score,
        "diagnosis_score": float(proposal.type_scores.get("DIAGNOSIS", 0.0)),
        "symptom_score": float(proposal.type_scores.get("SYMPTOM", 0.0)),
        "diagnosis_minus_symptom": float(
            proposal.type_scores.get("DIAGNOSIS", 0.0)
            - proposal.type_scores.get("SYMPTOM", 0.0)
        ),
        "has_route": float(bool(proposal.routes)),
        "route_c1": float("C1" in proposal.routes),
        "route_c2": float("C2" in proposal.routes),
        "route_c3": float("C3" in proposal.routes),
        "section_laboratory": float(proposal.section == "laboratory"),
        "section_medication": float(proposal.section == "medication"),
        "normalized_length_delta": float(len(proposal.normalized_view) - len(proposal.text)),
        "overlap_count": float(len(overlapping)),
        "max_overlap_iou": max(
            (span_iou(proposal.coordinates, other.coordinates) for other in overlapping),
            default=0.0,
        ),
        "ontology_placeholder_med_or_diag": float(best_type in {"MEDICATION", "DIAGNOSIS"}),
    }
    features.update(_nearby_char_features(lattice.original_text, proposal.start, proposal.end))
    for entity_type in TYPE_ORDER:
        features[f"type_score_{entity_type}"] = float(proposal.type_scores.get(entity_type, 0.0))
    for source in sources:
        features[f"has_{source.expert_id}"] = 1.0
        features["grammar_completeness"] = max(
            features.get("grammar_completeness", 0.0),
            float(source.features.get("grammar_component_count", 0.0)),
        )
        features["laboratory_structure"] = max(
            features.get("laboratory_structure", 0.0),
            float(source.features.get("lab_value_with_unit", 0.0)),
        )
    features.setdefault("grammar_completeness", 0.0)
    features.setdefault("laboratory_structure", 0.0)
    return features


def _boundary_target(
    proposal: LatticeProposal,
    gold: GoldMention | None,
    iou: float,
    lattice: SpanLattice,
    config: ResolverV2Config,
) -> BoundaryTarget:
    if gold is None:
        return BoundaryTarget(BOUNDARY_DROP, token_iou=0.0, wer_boundary_surrogate=1.0)
    if proposal.start == gold.start and proposal.end == gold.end:
        return BoundaryTarget(BOUNDARY_KEEP, token_iou=1.0, wer_boundary_surrogate=0.0)
    existing_id = _best_existing_for_gold(lattice, gold)
    if existing_id:
        return BoundaryTarget(
            BOUNDARY_SELECT_EXISTING,
            selected_proposal_id=existing_id,
            token_iou=iou,
            wer_boundary_surrogate=1.0 - iou,
        )
    start_delta = gold.start - proposal.start
    end_delta = gold.end - proposal.end
    if (
        abs(start_delta) <= config.max_boundary_delta
        and abs(end_delta) <= config.max_boundary_delta
    ):
        return BoundaryTarget(
            BOUNDARY_OFFSET,
            start_delta=start_delta,
            end_delta=end_delta,
            token_iou=iou,
            wer_boundary_surrogate=1.0 - iou,
        )
    return BoundaryTarget(BOUNDARY_DROP, token_iou=iou, wer_boundary_surrogate=1.0 - iou)


def _wrong_type_cost(proposal: LatticeProposal, gold: GoldMention | None) -> TypeTarget:
    if gold is None:
        return TypeTarget(TYPE_DROP, wrong_type_cost=0.0)
    predicted = proposal.best_type()
    diagnosis_symptom_pair = {predicted, gold.entity_type} == {"DIAGNOSIS", "SYMPTOM"}
    if predicted == gold.entity_type:
        cost = 0.0
    elif diagnosis_symptom_pair:
        cost = 2.0
    else:
        cost = 1.5
    return TypeTarget(
        gold.entity_type,
        wrong_type_cost=cost,
        diagnosis_symptom_pair=diagnosis_symptom_pair,
    )


def build_training_examples(
    lattice: SpanLattice,
    gold_mentions: Sequence[GoldMention],
    *,
    split: str,
    source_group: str = "",
    config: ResolverV2Config | None = None,
) -> tuple[ResolverV2TrainingExample, ...]:
    """Build deterministic supervised examples from governed train/validation only."""
    if split not in {"train", "validation"}:
        raise ResolverV2Error("learned L4 examples may only be built from train/validation")
    cfg = config or ResolverV2Config()
    for gold_mention in gold_mentions:
        gold_mention.validate_against(lattice.original_text, lattice.document_id)
    group = _source_group(lattice, source_group)
    cfg_hash = config_sha256(cfg)
    examples: list[ResolverV2TrainingExample] = []
    for proposal in lattice.proposals:
        matched_gold, iou = _best_gold(proposal, gold_mentions)
        source_id = "|".join(source.proposal_id for source in proposal.sources)
        boundary_target = _boundary_target(proposal, matched_gold, iou, lattice, cfg)
        type_target = _wrong_type_cost(proposal, matched_gold)
        raw_id = f"{lattice.document_id}:{source_id}:{split}:{cfg_hash}"
        example_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
        examples.append(
            ResolverV2TrainingExample(
                example_id=example_id,
                document_id=lattice.document_id,
                proposal_id=source_id,
                proposal_start=proposal.start,
                proposal_end=proposal.end,
                proposal_text=proposal.text,
                source_group=group,
                split=split,
                features=proposal_features(proposal, lattice),
                boundary_target=boundary_target,
                type_target=type_target,
                config_sha256=cfg_hash,
            )
        )
    return tuple(
        sorted(examples, key=lambda ex: (ex.document_id, ex.proposal_start, ex.proposal_id))
    )


def grouped_train_validation_split(
    examples: Sequence[ResolverV2TrainingExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[ResolverV2TrainingExample, ...], tuple[ResolverV2TrainingExample, ...], str]:
    """Leakage-safe split: one source/document group cannot cross folds."""
    if not 0.0 < validation_fraction < 1.0:
        raise ResolverV2Error("validation_fraction must be in (0, 1)")
    groups = sorted({example.source_group for example in examples})
    if len(groups) < 2:
        raise ResolverV2Error("at least two source groups are required for grouped split")
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    validation_count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation_groups = set(sorted(shuffled[:validation_count]))
    train = tuple(example for example in examples if example.source_group not in validation_groups)
    validation = tuple(example for example in examples if example.source_group in validation_groups)
    train_groups = {example.source_group for example in train}
    if train_groups & validation_groups:
        raise ResolverV2Error("grouped split leaked a source group")
    identity_raw = json.dumps(
        {"seed": seed, "validation_groups": sorted(validation_groups)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return train, validation, hashlib.sha256(identity_raw.encode("utf-8")).hexdigest()


def _dot(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(float(weight) * float(features.get(name, 0.0)) for name, weight in weights.items())


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def _clamped_offset(weights: Mapping[str, float], features: Mapping[str, float], limit: int) -> int:
    value = round(_dot(weights, features))
    return max(-limit, min(limit, value))


def _required_int(raw: Mapping[str, object], key: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ResolverV2Error(f"checkpoint metadata field {key!r} must be an integer")
    return int(value)


def _load_metadata(raw: Mapping[str, object]) -> ResolverV2CheckpointMetadata:
    return ResolverV2CheckpointMetadata(
        stage=str(raw["stage"]),
        expert=str(raw["expert"]),
        config_sha256=str(raw["config_sha256"]),
        corpus_sha256=str(raw["corpus_sha256"]),
        model_revision=str(raw["model_revision"]),
        seed=_required_int(raw, "seed"),
        git_commit=str(raw["git_commit"]),
        checkpoint_sha256=str(raw["checkpoint_sha256"]),
        parameter_count=_required_int(raw, "parameter_count"),
        train_split_id=str(raw["train_split_id"]),
        validation_split_id=str(raw["validation_split_id"]),
        internal_test_accessed=bool(raw["internal_test_accessed"]),
    )


def _nested_float_mapping(raw: object) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        raise ResolverV2Error("checkpoint weights must be mappings")
    out: dict[str, dict[str, float]] = {}
    for key, values in raw.items():
        if not isinstance(key, str) or not isinstance(values, Mapping):
            raise ResolverV2Error("checkpoint weights must be string -> mapping")
        out[key] = {str(name): float(value) for name, value in values.items()}
    return out


def _float_mapping(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ResolverV2Error("checkpoint weights must be mappings")
    return {str(name): float(value) for name, value in raw.items()}


def load_checkpoint(path: str | Path, *, expected_sha256: str = "") -> ResolverV2Checkpoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise ResolverV2UnavailableError(f"learned L4 v2 checkpoint missing: {checkpoint_path}")
    raw_bytes = checkpoint_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ResolverV2UnavailableError("learned L4 v2 checkpoint SHA-256 mismatch")
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ResolverV2Error("learned L4 checkpoint must be a JSON object")
    metadata_raw = payload.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        raise ResolverV2Error("checkpoint metadata must be a mapping")
    thresholds_raw = payload.get("thresholds", {})
    if not isinstance(thresholds_raw, Mapping):
        raise ResolverV2Error("checkpoint thresholds must be a mapping")
    checkpoint = ResolverV2Checkpoint(
        metadata=_load_metadata(metadata_raw),
        action_weights=_nested_float_mapping(payload.get("action_weights", {})),
        type_weights=_nested_float_mapping(payload.get("type_weights", {})),
        wrong_type_weights=_float_mapping(payload.get("wrong_type_weights", {})),
        offset_start_weights=_float_mapping(payload.get("offset_start_weights", {})),
        offset_end_weights=_float_mapping(payload.get("offset_end_weights", {})),
        thresholds=ResolverV2Thresholds(
            keep_threshold=float(thresholds_raw.get("keep_threshold", 0.5)),
            abstain_margin=float(thresholds_raw.get("abstain_margin", 0.05)),
            wrong_type_risk_threshold=float(
                thresholds_raw.get("wrong_type_risk_threshold", 0.35)
            ),
        ),
    )
    checkpoint.validate()
    return checkpoint


def validate_runtime_checkpoint(config: ResolverV2Config) -> ResolverV2Checkpoint:
    if not config.enabled:
        raise ResolverV2UnavailableError("learned L4 v2 is disabled by profile")
    if not config.checkpoint_path:
        raise ResolverV2UnavailableError("learned L4 v2 has no configured checkpoint")
    return load_checkpoint(
        config.checkpoint_path,
        expected_sha256=config.expected_checkpoint_sha256,
    )


def _choose_action(
    checkpoint: ResolverV2Checkpoint, features: Mapping[str, float]
) -> tuple[str, float]:
    scores = {
        action: _dot(weights, features)
        for action, weights in checkpoint.action_weights.items()
    }
    if not scores:
        return BOUNDARY_DROP, 0.0
    action = min(scores.items(), key=lambda item: (-item[1], item[0]))[0]
    return action, _sigmoid(scores[action])


def _choose_type(
    checkpoint: ResolverV2Checkpoint, features: Mapping[str, float]
) -> tuple[str, dict[str, float]]:
    scores = {
        entity_type: _dot(weights, features)
        for entity_type, weights in checkpoint.type_weights.items()
    }
    distribution = _softmax(scores)
    if not distribution:
        return TYPE_DROP, {}
    entity_type = min(distribution.items(), key=lambda item: (-item[1], item[0]))[0]
    return entity_type, distribution


def _select_existing_boundary(proposal: LatticeProposal, lattice: SpanLattice) -> tuple[int, int]:
    competitors = [
        other for other in lattice.proposals if other is not proposal and proposal.overlaps(other)
    ]
    if not competitors:
        return proposal.start, proposal.end
    winner = sorted(
        competitors,
        key=lambda item: (-item.local_score(), item.start, item.end, item.best_type()),
    )[0]
    return winner.start, winner.end


def resolve_lattice(
    lattice: SpanLattice,
    checkpoint: ResolverV2Checkpoint,
    config: ResolverV2Config,
) -> ResolverV2Result:
    """Read-only learned L4 inference over an L3 lattice."""
    checkpoint.validate()
    cfg_hash = config_sha256(config)
    hypotheses: list[TypedHypothesis] = []
    decisions: list[ResolverV2Decision] = []
    for ordinal, proposal in enumerate(lattice.proposals, start=1):
        features = proposal_features(proposal, lattice)
        action, keep_score = _choose_action(checkpoint, features)
        start, end = proposal.start, proposal.end
        if action == BOUNDARY_SELECT_EXISTING:
            start, end = _select_existing_boundary(proposal, lattice)
        elif action == BOUNDARY_OFFSET:
            start += _clamped_offset(
                checkpoint.offset_start_weights, features, config.max_boundary_delta
            )
            end += _clamped_offset(
                checkpoint.offset_end_weights, features, config.max_boundary_delta
            )
        if start < 0 or end <= start or end > len(lattice.original_text):
            action = BOUNDARY_DROP
            start = proposal.start
            end = proposal.start
        entity_type, distribution = _choose_type(checkpoint, features)
        wrong_type_risk = _sigmoid(_dot(checkpoint.wrong_type_weights, features))
        abstain = (
            action == BOUNDARY_DROP
            or entity_type == TYPE_DROP
            or keep_score < checkpoint.thresholds.keep_threshold
            or wrong_type_risk > checkpoint.thresholds.wrong_type_risk_threshold
        )
        reason = "accepted"
        if abstain:
            reason = "abstained_by_threshold_or_drop_action"
        text = "" if end <= start else lattice.original_text[start:end]
        hypothesis_id = f"l4v2-{lattice.document_id}-{ordinal:04d}"
        source_ids = tuple(source.proposal_id for source in proposal.sources)
        if distribution and TYPE_DROP in distribution:
            distribution = {
                key: value for key, value in distribution.items() if key != TYPE_DROP
            }
        hypothesis = TypedHypothesis(
            hypothesis_id=hypothesis_id,
            text=text,
            coords=SpanCoordinates(absolute=Span(start, end)),
            type_distribution=distribution,
            calibrated_score=keep_score,
            abstained=abstain,
            source_proposal_ids=source_ids,
            evidence_ids=source_ids,
            features={
                "wrong_type_risk": wrong_type_risk,
                "learned_l4_keep_score": keep_score,
                "proposal_start_delta": float(start - proposal.start),
                "proposal_end_delta": float(end - proposal.end),
            },
        )
        hypotheses.append(hypothesis)
        decisions.append(
            ResolverV2Decision(
                hypothesis_id=hypothesis_id,
                proposal_id="|".join(source_ids),
                status="abstained" if abstain else "accepted",
                boundary_action=action,
                original_start=proposal.start,
                original_end=proposal.end,
                start=start,
                end=end,
                entity_type=entity_type,
                calibrated_score=keep_score,
                wrong_type_risk=wrong_type_risk,
                reason=reason,
            )
        )
    return ResolverV2Result(
        document_id=lattice.document_id,
        hypotheses=tuple(hypotheses),
        decisions=tuple(decisions),
        config_sha256=cfg_hash,
        checkpoint_sha256=checkpoint.metadata.checkpoint_sha256,
    )


__all__ = [
    "BOUNDARY_DROP",
    "BOUNDARY_KEEP",
    "BOUNDARY_OFFSET",
    "BOUNDARY_SELECT_EXISTING",
    "GoldMention",
    "RESOLVER_V2_VERSION",
    "ResolverV2Checkpoint",
    "ResolverV2CheckpointMetadata",
    "ResolverV2Config",
    "ResolverV2Decision",
    "ResolverV2Error",
    "ResolverV2Result",
    "ResolverV2Thresholds",
    "ResolverV2TrainingExample",
    "ResolverV2UnavailableError",
    "TYPE_DROP",
    "TYPE_ORDER",
    "BoundaryTarget",
    "TypeTarget",
    "build_training_examples",
    "config_sha256",
    "grouped_train_validation_split",
    "load_checkpoint",
    "proposal_features",
    "resolve_lattice",
    "span_iou",
    "validate_runtime_checkpoint",
]
