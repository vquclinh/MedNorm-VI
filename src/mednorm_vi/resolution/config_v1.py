"""The single tracked configuration for the L4 Boundary & Type Resolver v1.

Every threshold and weight the resolver uses lives in
``configs/resolution/boundary_type_resolver_v1.yaml``. This module loads it and
records the **SHA-256 of the file bytes**, which is what reports quote: a number
produced by an edited config can therefore never be confused with a number
produced by the tracked one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas.constants import CANDIDATE_ONTOLOGY_BY_TYPE, ENTITY_TYPES

DEFAULT_CONFIG_PATH = "configs/resolution/boundary_type_resolver_v1.yaml"


class ResolverConfigError(ValueError):
    """Raised when the tracked resolver config is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class TypeWeights:
    neural_span_type: float = 1.0
    grammar_completeness: float = 0.6
    laboratory_evidence: float = 0.6
    section_prior: float = 0.25
    route_prior: float = 0.2
    expert_agreement: float = 0.3
    ontology_evidence: float = 0.25
    delimiter_shape: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "neural_span_type": self.neural_span_type,
            "grammar_completeness": self.grammar_completeness,
            "laboratory_evidence": self.laboratory_evidence,
            "section_prior": self.section_prior,
            "route_prior": self.route_prior,
            "expert_agreement": self.expert_agreement,
            "ontology_evidence": self.ontology_evidence,
            "delimiter_shape": self.delimiter_shape,
        }


# Default trim character class, shared by the dataclass default and the loader so
# neither can drift from the other. ``slots=True`` forbids reading a field default
# off the class, which is why this is a module constant.
DEFAULT_TRIM_CHARACTERS = " \t\n\r.,;:!?()[]{}\"'`«»…-–—/\\|*+•"


@dataclass(frozen=True, slots=True)
class BoundaryPolicy:
    enable_trim: bool = True
    enable_expand: bool = True
    max_trim_chars: int = 4
    trim_characters: str = DEFAULT_TRIM_CHARACTERS
    trim_leading_list_markers: bool = True
    expand_requires_grammar_completeness: float = 0.6
    max_expand_chars: int = 40
    # Preferred boundary kind per specialist family, keyed "medication" /
    # "test_result" and matched against ``resolution.features.boundary_kind``.
    group_preference: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OverlapPolicy:
    # Migrated from the retired Phase-1C-A `resolver_v1.yaml:abstain_on_conflict`
    # (Audit 0055). When a same-type near-complete competition is TIED, mark both
    # sides UNRESOLVED instead of picking one. Abstention is not rejection.
    abstain_on_conflict: bool = False
    near_complete_iou: float = 0.6
    competition_penalty: float = 0.2
    suppress_cross_type: bool = False
    protect_has_result_pairs: bool = True
    strong_has_result_score: float = 0.5


@dataclass(frozen=True, slots=True)
class AbstentionPolicy:
    min_type_utility: float = 0.3
    min_type_margin: float = 0.05
    abstain_on_ontology_conflict: bool = True
    abstain_on_flagged_deterministic_evidence: bool = True


@dataclass(frozen=True, slots=True)
class ResolverV1Config:
    """Resolved L4 v1 configuration plus the digest of the file it came from."""

    config_version: str
    type_weights: TypeWeights
    boundary: BoundaryPolicy
    overlap: OverlapPolicy
    abstention: AbstentionPolicy
    section_priors: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    route_priors: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    allowed_ontology_evidence: Mapping[str, str] = field(default_factory=dict)
    forbidden_ontology_evidence: Mapping[str, str] = field(default_factory=dict)
    expert_disagreement_penalty: float = 0.15
    config_sha256: str = ""
    config_path: str = ""

    def section_prior(self, section: str, entity_type: str) -> float:
        return float(self.section_priors.get(section, {}).get(entity_type, 0.0))

    def route_prior(self, routes: tuple[str, ...], entity_type: str) -> float:
        return max(
            (float(self.route_priors.get(route, {}).get(entity_type, 0.0))
             for route in routes),
            default=0.0)

    def ontology_evidence_allowed(self, entity_type: str, ontology: str) -> bool:
        """Spec §7.3: ICD may never reinforce MEDICATION, RxNorm never DIAGNOSIS."""
        if self.forbidden_ontology_evidence.get(entity_type) == ontology:
            return False
        allowed = self.allowed_ontology_evidence.get(entity_type)
        return allowed is not None and allowed == ontology

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "type_weights": self.type_weights.as_dict(),
            "expert_disagreement_penalty": self.expert_disagreement_penalty,
            "abstention": {
                "min_type_utility": self.abstention.min_type_utility,
                "min_type_margin": self.abstention.min_type_margin,
                "abstain_on_ontology_conflict": self.abstention.abstain_on_ontology_conflict,
                "abstain_on_flagged_deterministic_evidence":
                    self.abstention.abstain_on_flagged_deterministic_evidence,
            },
            "boundary": {
                "enable_trim": self.boundary.enable_trim,
                "enable_expand": self.boundary.enable_expand,
                "max_trim_chars": self.boundary.max_trim_chars,
                "trim_leading_list_markers": self.boundary.trim_leading_list_markers,
                "max_expand_chars": self.boundary.max_expand_chars,
            },
            "overlap": {
                "abstain_on_conflict": self.overlap.abstain_on_conflict,
                "near_complete_iou": self.overlap.near_complete_iou,
                "competition_penalty": self.overlap.competition_penalty,
                "suppress_cross_type": self.overlap.suppress_cross_type,
                "protect_has_result_pairs": self.overlap.protect_has_result_pairs,
                "strong_has_result_score": self.overlap.strong_has_result_score,
            },
            "section_priors": {k: dict(v) for k, v in sorted(self.section_priors.items())},
            "route_priors": {k: dict(v) for k, v in sorted(self.route_priors.items())},
            "ontology": {
                "allowed_evidence": dict(sorted(self.allowed_ontology_evidence.items())),
                "forbidden_evidence": dict(sorted(self.forbidden_ontology_evidence.items())),
            },
        }


def _typed_priors(raw: Any, *, label: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, mapping in (raw or {}).items():
        bucket: dict[str, float] = {}
        for entity_type, value in (mapping or {}).items():
            if entity_type not in ENTITY_TYPES:
                raise ResolverConfigError(
                    f"{label} {key!r} names an unsupported entity type {entity_type!r}")
            bucket[str(entity_type)] = float(value)
        out[str(key)] = bucket
    return out


def load_resolver_v1_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ResolverV1Config:
    """Load and validate the tracked config, recording the digest of its bytes."""
    import yaml

    target = Path(path)
    raw_bytes = target.read_bytes()
    document: dict[str, Any] = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    digest = hashlib.sha256(raw_bytes).hexdigest()

    weights_raw = document.get("type_weights", {}) or {}
    boundary_raw = document.get("boundary", {}) or {}
    overlap_raw = document.get("overlap", {}) or {}
    abstention_raw = document.get("abstention", {}) or {}
    ontology_raw = document.get("ontology", {}) or {}

    allowed = {str(k): str(v) for k, v in (ontology_raw.get("allowed_evidence", {}) or {}).items()}
    forbidden = {
        str(k): str(v) for k, v in (ontology_raw.get("forbidden_evidence", {}) or {}).items()
    }
    for entity_type, ontology in allowed.items():
        if entity_type not in ENTITY_TYPES:
            raise ResolverConfigError(f"unknown entity type in ontology.allowed: {entity_type!r}")
        expected = CANDIDATE_ONTOLOGY_BY_TYPE.get(entity_type)
        if expected != ontology:
            raise ResolverConfigError(
                f"ontology.allowed pairs {entity_type} with {ontology}, but the frozen "
                f"schema contract says {expected}; spec §7.3 forbids cross-links")
    for entity_type, ontology in forbidden.items():
        if allowed.get(entity_type) == ontology:
            raise ResolverConfigError(
                f"{ontology} is both allowed and forbidden for {entity_type}")

    config = ResolverV1Config(
        config_version=str(document.get("config_version", "boundary-type-resolver-v1")),
        type_weights=TypeWeights(
            neural_span_type=float(weights_raw.get("neural_span_type", 1.0)),
            grammar_completeness=float(weights_raw.get("grammar_completeness", 0.6)),
            laboratory_evidence=float(weights_raw.get("laboratory_evidence", 0.6)),
            section_prior=float(weights_raw.get("section_prior", 0.25)),
            route_prior=float(weights_raw.get("route_prior", 0.2)),
            expert_agreement=float(weights_raw.get("expert_agreement", 0.3)),
            ontology_evidence=float(weights_raw.get("ontology_evidence", 0.25)),
            delimiter_shape=float(weights_raw.get("delimiter_shape", 0.1)),
        ),
        boundary=BoundaryPolicy(
            enable_trim=bool(boundary_raw.get("enable_trim", True)),
            enable_expand=bool(boundary_raw.get("enable_expand", True)),
            max_trim_chars=int(boundary_raw.get("max_trim_chars", 4)),
            trim_characters=str(boundary_raw.get(
                "trim_characters", DEFAULT_TRIM_CHARACTERS)),
            trim_leading_list_markers=bool(boundary_raw.get("trim_leading_list_markers", True)),
            expand_requires_grammar_completeness=float(
                boundary_raw.get("expand_requires_grammar_completeness", 0.6)),
            max_expand_chars=int(boundary_raw.get("max_expand_chars", 40)),
            group_preference={
                str(k): str(v)
                for k, v in (boundary_raw.get("group_preference", {}) or {}).items()
            },
        ),
        overlap=OverlapPolicy(
            abstain_on_conflict=bool(overlap_raw.get("abstain_on_conflict", False)),
            near_complete_iou=float(overlap_raw.get("near_complete_iou", 0.6)),
            competition_penalty=float(overlap_raw.get("competition_penalty", 0.2)),
            suppress_cross_type=bool(overlap_raw.get("suppress_cross_type", False)),
            protect_has_result_pairs=bool(overlap_raw.get("protect_has_result_pairs", True)),
            strong_has_result_score=float(overlap_raw.get("strong_has_result_score", 0.5)),
        ),
        abstention=AbstentionPolicy(
            min_type_utility=float(abstention_raw.get("min_type_utility", 0.3)),
            min_type_margin=float(abstention_raw.get("min_type_margin", 0.05)),
            abstain_on_ontology_conflict=bool(
                abstention_raw.get("abstain_on_ontology_conflict", True)),
            abstain_on_flagged_deterministic_evidence=bool(
                abstention_raw.get("abstain_on_flagged_deterministic_evidence", True)),
        ),
        section_priors=_typed_priors(document.get("section_priors"), label="section_priors"),
        route_priors=_typed_priors(document.get("route_priors"), label="route_priors"),
        allowed_ontology_evidence=allowed,
        forbidden_ontology_evidence=forbidden,
        expert_disagreement_penalty=float(document.get("expert_disagreement_penalty", 0.15)),
        config_sha256=digest,
        config_path=str(target),
    )
    if config.boundary.max_trim_chars < 0:
        raise ResolverConfigError("boundary.max_trim_chars must be >= 0")
    if not 0.0 <= config.overlap.near_complete_iou <= 1.0:
        raise ResolverConfigError("overlap.near_complete_iou must be in [0, 1]")
    return config


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_TRIM_CHARACTERS",
    "AbstentionPolicy",
    "BoundaryPolicy",
    "OverlapPolicy",
    "ResolverConfigError",
    "ResolverV1Config",
    "TypeWeights",
    "load_resolver_v1_config",
]
