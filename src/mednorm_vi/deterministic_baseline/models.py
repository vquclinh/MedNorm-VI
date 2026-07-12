"""Phase 1B orchestration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..case_router.models import NodeRouting
from ..case_router.signals import RouterConfig, load_router_config
from ..mention_factory.laboratory.lexicon import LabLexicon, load_lab_lexicon
from ..mention_factory.medication.lexicon import MedicationLexicon, load_medication_lexicon
from ..mention_factory.merge import MergeDiagnostics
from ..mention_factory.models import RelationProposal, SpanProposal
from ..validator.results import ValidationIssue


@dataclass(frozen=True, slots=True)
class Phase1BConfig:
    """Resolved Phase 1B configuration (router + both specialist lexicons)."""

    router: RouterConfig
    medication: MedicationLexicon
    laboratory: LabLexicon
    medication_config_version: str
    laboratory_config_version: str

    @staticmethod
    def load(
        router_config: str | Path,
        medication_config: str | Path,
        laboratory_config: str | Path,
    ) -> Phase1BConfig:
        med = load_medication_lexicon(medication_config)
        lab = load_lab_lexicon(laboratory_config)
        return Phase1BConfig(
            router=load_router_config(router_config),
            medication=med,
            laboratory=lab,
            medication_config_version=med.grammar_version,
            laboratory_config_version=lab.parser_version,
        )


@dataclass(frozen=True, slots=True)
class Phase1BResult:
    """Everything Phase 1B produced for one document (proposals, not finals)."""

    document_id: str
    routings: tuple[NodeRouting, ...]
    proposals: tuple[SpanProposal, ...]
    relations: tuple[RelationProposal, ...]
    merge_diagnostics: MergeDiagnostics
    warnings: tuple[str, ...] = field(default_factory=tuple)
    l1_valid: bool = True
    proposals_valid: bool = True
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def medication_proposals(self) -> tuple[SpanProposal, ...]:
        return tuple(p for p in self.proposals if p.source_specialist == "medication")

    def laboratory_proposals(self) -> tuple[SpanProposal, ...]:
        return tuple(p for p in self.proposals if p.source_specialist == "laboratory")


__all__ = ["Phase1BConfig", "Phase1BResult"]
