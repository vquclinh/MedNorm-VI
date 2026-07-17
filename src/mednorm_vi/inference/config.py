"""Pipeline configuration loading and readiness validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    @staticmethod
    def load(path: str | Path) -> PipelineConfig:
        import yaml  # type: ignore[import-untyped]

        doc: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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


__all__ = ["PipelineConfig", "validate_readiness"]
