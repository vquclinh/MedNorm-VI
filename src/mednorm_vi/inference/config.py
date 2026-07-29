"""Pipeline configuration loading and readiness validation.

Loading is the enforcement point for retirement (Audit 0051). A profile that still
declares a retired expert's feature flag, or requires its checkpoint, is refused at
load time rather than silently normalized to ``False`` — a config carrying a dead
flag is a config someone can flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..governance.e4_retirement import (
    assert_e4_absent_from_flags,
    assert_no_e4_checkpoint_required,
)

DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "enable_e1_medication_grammar": True,
    "enable_e2_laboratory_parser": True,
    "enable_e3_vihealthbert": True,
    "enable_e5_xlmr_mrc": False,
    "enable_e6_gliner": False,
    "enable_e7_qwen_proposer": False,
    "enable_l4_deterministic_v1": False,
    "enable_l4_learned_v2": False,
}

CHECKPOINT_BY_FEATURE_FLAG: dict[str, str] = {
    "enable_e3_vihealthbert": "mention/vihealthbert",
    "enable_e5_xlmr_mrc": "mention/xlmr_mrc",
    "enable_e6_gliner": "mention/gliner",
    "enable_e7_qwen_proposer": "mention/qwen3_1_7b_proposer",
    "enable_l4_learned_v2": "resolution/learned_l4_v2",
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    l1_config: str
    router_config: str
    medication_config: str
    laboratory_config: str
    resolver_config: str
    icd_index: str = ""
    rxnorm_index: str = ""
    checkpoint_root: str = "models/checkpoints/full_v1"
    full_requires_checkpoints: tuple[str, ...] = field(default_factory=tuple)
    specialist_requires_checkpoints: tuple[str, ...] = field(default_factory=tuple)
    allow_specialist_fallback: bool = True
    expected_documents: int = 100
    feature_flags: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_FEATURE_FLAGS))

    @staticmethod
    def load(path: str | Path) -> PipelineConfig:
        import yaml

        doc: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        profile_name = str(doc.get("name", Path(path).stem))
        feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        raw_flags = doc.get("feature_flags", {})
        if isinstance(raw_flags, dict):
            feature_flags.update({str(key): bool(value) for key, value in raw_flags.items()})

        # Retirement is enforced here, and in every nested per-mode profile, so a
        # dead flag cannot survive anywhere in the tree waiting to be flipped.
        assert_e4_absent_from_flags(feature_flags, profile=profile_name)
        raw_profiles = doc.get("profiles", {})
        if isinstance(raw_profiles, dict):
            for mode, spec in raw_profiles.items():
                if isinstance(spec, dict) and isinstance(spec.get("feature_flags"), dict):
                    assert_e4_absent_from_flags(
                        spec["feature_flags"], profile=f"{profile_name}:{mode}")
        for key in ("full_requires_checkpoints", "specialist_requires_checkpoints"):
            assert_no_e4_checkpoint_required(
                [str(v) for v in doc.get(key, [])], profile=f"{profile_name}.{key}")

        return PipelineConfig(
            l1_config=str(doc.get("l1_config", "configs/document_intelligence/base.yaml")),
            router_config=str(doc.get("router_config", "configs/case_router/base.yaml")),
            medication_config=str(
                doc.get("medication_config", "configs/medication/grammar_v1.yaml")
            ),
            laboratory_config=str(
                doc.get("laboratory_config", "configs/laboratory/parser_v1.yaml")
            ),
            resolver_config=str(doc.get("resolver_config", "configs/resolution/resolver_v1.yaml")),
            icd_index=str(doc.get("icd_index", "")),
            rxnorm_index=str(doc.get("rxnorm_index", "")),
            checkpoint_root=str(doc.get("checkpoint_root", "models/checkpoints/full_v1")),
            full_requires_checkpoints=tuple(
                str(v) for v in doc.get("full_requires_checkpoints", [])
            ),
            specialist_requires_checkpoints=tuple(
                str(v) for v in doc.get("specialist_requires_checkpoints", [])
            ),
            allow_specialist_fallback=bool(doc.get("allow_specialist_fallback", True)),
            expected_documents=int(doc.get("expected_documents", 100)),
            feature_flags=feature_flags,
        )


def validate_readiness(config: PipelineConfig, *, mode: str) -> tuple[str, ...]:
    """Return readiness errors for the selected mode."""
    errors: list[str] = []
    if mode not in {"deterministic", "specialist", "full"}:
        errors.append(f"unknown_mode:{mode}")
    if mode == "full":
        for rel in config.full_requires_checkpoints:
            path = Path(config.checkpoint_root) / rel
            if not path.exists():
                errors.append(f"missing_checkpoint:{path}")
        for flag, rel in CHECKPOINT_BY_FEATURE_FLAG.items():
            if not config.feature_flags.get(flag, False):
                continue
            path = Path(config.checkpoint_root) / rel
            if not path.exists():
                errors.append(f"enabled_feature_missing_checkpoint:{flag}:{path}")
    if mode == "specialist" and not config.allow_specialist_fallback:
        for rel in config.specialist_requires_checkpoints:
            path = Path(config.checkpoint_root) / rel
            if not path.exists():
                errors.append(f"missing_checkpoint:{path}")
    for index_name, index_path in (
        ("icd_index", config.icd_index),
        ("rxnorm_index", config.rxnorm_index),
    ):
        if index_path and not Path(index_path).is_file():
            errors.append(f"missing_{index_name}:{index_path}")
    return tuple(errors)


__all__ = [
    "CHECKPOINT_BY_FEATURE_FLAG",
    "DEFAULT_FEATURE_FLAGS",
    "PipelineConfig",
    "validate_readiness",
]
