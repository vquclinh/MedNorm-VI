"""ZS0 zero-shot baseline profile (Audit 0048).

ZS0 is a **pure zero-shot / pretrained** end-to-end baseline. Its purpose is to
produce a valid organizer `output.zip` using only deterministic logic and pinned
pretrained checkpoints — nothing this project trained.

That constraint is the whole point, so it is enforced rather than documented:
:func:`assert_zs0_components_allowed` refuses every forbidden component by name,
and a test asserts each one is refused.

Three arms, evaluated once each on the governed validation split:

    ZS0-A   E1 + E2                                   deterministic only
    ZS0-B   E1 + E2 + pretrained GLiNER
    ZS0-C   E1 + E2 + pretrained GLiNER + constrained Qwen on uncertain segments

No E3, E4 or E5 in any arm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ZS0_PROFILE_VERSION = "zs0-baseline-v1"

ZS0_A = "ZS0-A"
ZS0_B = "ZS0-B"
ZS0_C = "ZS0-C"
ZS0_ARMS: tuple[str, ...] = (ZS0_A, ZS0_B, ZS0_C)

ORGANIZER_AUTHORIZATION = "I_AUTHORIZE_ZS0_ORGANIZER_INFERENCE_AND_PACKAGE"

# ---------------------------------------------------------------------------
# What ZS0 may and may not contain
# ---------------------------------------------------------------------------
#
# "Zero-shot" here means: no weights this project fitted, and no head this
# project initialized. A randomly initialized task head is not zero-shot — it is
# untrained, which is worse than absent because it emits confident noise.
FORBIDDEN_COMPONENTS: Mapping[str, str] = {
    "E3_vihealthbert_span_type":
        "fine-tuned in this project (Audit 0031); ZS0 uses no fitted weights",
    "e3_vihealthbert_span_type":
        "fine-tuned in this project (Audit 0031); ZS0 uses no fitted weights",
    "E4_phobert_w2ner":
        "retired from the active stack (Audit 0048)",
    "e4_phobert_w2ner":
        "retired from the active stack (Audit 0048)",
    "E5_xlmr_mrc_ner":
        "its MRC task head is randomly initialized; an untrained head emits "
        "confident noise and is not zero-shot",
    "e5_xlmr_mrc_ner":
        "its MRC task head is randomly initialized; an untrained head emits "
        "confident noise and is not zero-shot",
    "l4_learned_resolver_v2":
        "no trained checkpoint exists; ZS0 uses the deterministic resolver",
    "s2_assertion_head":
        "the governed corpus has zero assertion supervision (Audit 0042)",
}

FORBIDDEN_FEATURE_FLAGS: tuple[str, ...] = (
    "enable_e3_vihealthbert",
    "enable_e4_phobert_w2ner",
    "enable_e5_xlmr_mrc",
    "enable_l4_learned_v2",
)

# Everything ZS0 is permitted to use. Deterministic components carry no weights.
ALLOWED_DETERMINISTIC: tuple[str, ...] = (
    "E1_medication_grammar",
    "E2_laboratory_parser",
    "l4_conservative_zero_shot_resolver",
    "l6_assertion_cues",
    "l8_deterministic_calibration",
    "l9_organizer_policy",
    "ontology_alias_index",
    "ontology_lexical_index",
)
ALLOWED_PRETRAINED: tuple[str, ...] = (
    "E6_gliner_open_type",
    "E7_qwen_cascade",
    "dense_embedder",
    "reranker",
)


class ZS0ProfileError(ValueError):
    """Raised when a ZS0 profile violates the zero-shot contract."""


@dataclass(frozen=True, slots=True)
class ZS0Arm:
    """One evaluation arm."""

    name: str
    uses_gliner: bool
    uses_qwen_proposer: bool

    def __post_init__(self) -> None:
        if self.name not in ZS0_ARMS:
            raise ZS0ProfileError(f"unknown ZS0 arm {self.name!r}")
        if self.uses_qwen_proposer and not self.uses_gliner:
            raise ZS0ProfileError(
                "the Qwen proposer only runs on segments the other experts left "
                "uncertain, so it presupposes GLiNER")

    @property
    def components(self) -> tuple[str, ...]:
        parts = ["E1_medication_grammar", "E2_laboratory_parser"]
        if self.uses_gliner:
            parts.append("E6_gliner_open_type")
        if self.uses_qwen_proposer:
            parts.append("E7_qwen_cascade")
        return tuple(parts)

    @property
    def neural_components(self) -> tuple[str, ...]:
        return tuple(c for c in self.components if c in ALLOWED_PRETRAINED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "components": list(self.components),
            "neural_components": list(self.neural_components),
            "uses_gliner": self.uses_gliner,
            "uses_qwen_proposer": self.uses_qwen_proposer,
            "uses_fine_tuned_weights": False,
            "uses_random_task_head": False,
        }


def build_arm(name: str) -> ZS0Arm:
    if name == ZS0_A:
        return ZS0Arm(name=ZS0_A, uses_gliner=False, uses_qwen_proposer=False)
    if name == ZS0_B:
        return ZS0Arm(name=ZS0_B, uses_gliner=True, uses_qwen_proposer=False)
    if name == ZS0_C:
        return ZS0Arm(name=ZS0_C, uses_gliner=True, uses_qwen_proposer=True)
    raise ZS0ProfileError(f"unknown ZS0 arm {name!r}; expected one of {ZS0_ARMS}")


def all_arms() -> tuple[ZS0Arm, ...]:
    return tuple(build_arm(name) for name in ZS0_ARMS)


def assert_zs0_components_allowed(components: Sequence[str]) -> None:
    """Refuse any forbidden component, by name, with the reason."""
    for component in components:
        reason = FORBIDDEN_COMPONENTS.get(component)
        if reason is not None:
            raise ZS0ProfileError(
                f"{component} is forbidden in ZS0: {reason}")
        if component not in ALLOWED_DETERMINISTIC and component not in ALLOWED_PRETRAINED:
            raise ZS0ProfileError(
                f"{component} is not an approved ZS0 component; allowed are "
                f"{list(ALLOWED_DETERMINISTIC) + list(ALLOWED_PRETRAINED)}")


def assert_zs0_feature_flags(feature_flags: Mapping[str, Any]) -> None:
    """Every trained-expert flag must be false in a ZS0 profile."""
    enabled = [flag for flag in FORBIDDEN_FEATURE_FLAGS
               if bool(feature_flags.get(flag, False))]
    if enabled:
        raise ZS0ProfileError(
            f"ZS0 requires {enabled} to be false; it uses no weights this "
            "project fitted and no head this project initialized")


def assert_no_training_executed(record: Mapping[str, Any]) -> None:
    """ZS0 is inference only: no backward pass, optimizer or scheduler."""
    for key in ("backward_passes", "optimizer_steps", "scheduler_steps"):
        value = int(record.get(key, 0) or 0)
        if value:
            raise ZS0ProfileError(
                f"ZS0 performed {value} {key}; it is an inference-only baseline")
    if record.get("optimizer_constructed") or record.get("scheduler_constructed"):
        raise ZS0ProfileError("ZS0 must not construct an optimizer or scheduler")


def assert_no_external_api(configuration: Mapping[str, Any]) -> None:
    """All neural inference is self-hosted; no external API, ever."""
    for key in ("openai_api_key", "anthropic_api_key", "external_api_base",
                "remote_inference_endpoint"):
        if configuration.get(key):
            raise ZS0ProfileError(
                f"{key} is configured; ZS0 forbids any external model API and "
                "runs every model self-hosted")


__all__ = [
    "ALLOWED_DETERMINISTIC",
    "ALLOWED_PRETRAINED",
    "FORBIDDEN_COMPONENTS",
    "FORBIDDEN_FEATURE_FLAGS",
    "ORGANIZER_AUTHORIZATION",
    "ZS0_A",
    "ZS0_ARMS",
    "ZS0_B",
    "ZS0_C",
    "ZS0_PROFILE_VERSION",
    "ZS0Arm",
    "ZS0ProfileError",
    "all_arms",
    "assert_no_external_api",
    "assert_no_training_executed",
    "assert_zs0_components_allowed",
    "assert_zs0_feature_flags",
    "build_arm",
]
