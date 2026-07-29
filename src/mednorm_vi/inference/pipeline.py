"""The canonical L1-L9 runner (spec §16). One runner, one path, one L4.

Flow, after Audit 0052::

    L1 DocumentGraph
      -> L2 route tags
      -> L3  E1 + E2 deterministic  +  every ENABLED registered expert
      -> ONE unified SpanLattice          (lattice.builder.build_span_lattice)
      -> ONE canonical L4                 (resolution.canonical)
      -> L5 assertions + ICD/RxNorm linking
      -> L6 evidence graph -> L7 cascade -> L8 decoder -> L9 validate + package

Before Audit 0052 this runner passed Phase-1B proposals straight to
``resolution.resolver``. It never built a lattice, so E3 — the project's only
trained model — could not run in production, and each new expert would have meant
editing this file. Experts now arrive through
``mention_factory.registry``: this module names no expert except the deterministic
E1/E2 pair that Phase 1B owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..confidence_cascade import apply_confidence_cascade
from ..deterministic_baseline.models import Phase1BConfig
from ..deterministic_baseline.pipeline import run_phase1b
from ..document_intelligence import analyze_document
from ..document_intelligence.builder import load_config as load_l1_config
from ..evidence_graph import ClinicalEvidenceGraph, build_evidence_graph
from ..kb.indexing.retrieval import LocalIndex, load_index
from ..lattice.builder import build_span_lattice, lattice_config_hash
from ..lattice.models import SpanLattice
from ..linking.icd10 import link_icd10
from ..linking.models import LinkerResult
from ..linking.rxnorm import link_rxnorm
from ..mention_factory import experts as _registered_experts  # noqa: F401
from ..mention_factory.registry import (
    ExpertRegistry,
    ExpertRunRecord,
    L3Expert,
    run_registered_experts,
)
from ..metric_decoder import decode_expected_jaccard
from ..resolution.canonical import resolution_counts, resolve_lattice_to_hypotheses
from ..resolution.config_v1 import load_resolver_v1_config
from ..resolution.models import EntityHypothesis
from ..schemas.prediction import EntityPrediction
from ..specialists.assertion import resolve_assertions
from .config import PipelineConfig, flags_for_mode, validate_readiness
from .packaging import package_output_zip, write_output_directory
from .serialization import to_entity_predictions, to_submission_json

# Importing `mention_factory.experts` is what registers E3 (and any future expert)
# into the default registry. It is imported for its side effect, which is why the
# `noqa: F401` above is deliberate rather than an oversight.


@dataclass(frozen=True, slots=True)
class PipelineResult:
    document_id: str
    predictions: tuple[EntityPrediction, ...]
    evidence_graph: ClinicalEvidenceGraph
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # What actually happened at L3/L4 on this document. Reported, never inferred:
    # a profile's claim about which experts ran is only as good as this record.
    lattice: SpanLattice | None = None
    expert_records: tuple[ExpertRunRecord, ...] = field(default_factory=tuple)
    l4_counts: dict[str, int] = field(default_factory=dict)

    def experts_that_ran(self) -> tuple[str, ...]:
        return tuple(
            record.expert_id for record in self.expert_records if record.executed)

    def proposal_counts_by_expert(self) -> dict[str, int]:
        counts = {
            record.expert_id: record.proposal_count
            for record in self.expert_records if record.executed
        }
        if self.lattice is not None:
            for proposal in self.lattice.proposals:
                for source in proposal.sources:
                    if source.expert_id.startswith(("E1", "E2")):
                        counts[source.expert_id] = counts.get(source.expert_id, 0) + 1
        return dict(sorted(counts.items()))


def _indexes(config: PipelineConfig) -> tuple[LocalIndex | None, LocalIndex | None]:
    icd = load_index(config.icd_index) if config.icd_index else None
    rxn = load_index(config.rxnorm_index) if config.rxnorm_index else None
    return icd, rxn


def _link(
    hypotheses: tuple[EntityHypothesis, ...],
    *,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
) -> tuple[LinkerResult, ...]:
    results: list[LinkerResult] = []
    for h in hypotheses:
        if h.entity_type == "CHẨN_ĐOÁN" and icd_index is not None:
            results.append(link_icd10(h, icd_index))
        elif h.entity_type == "THUỐC" and rxnorm_index is not None:
            results.append(link_rxnorm(h, rxnorm_index))
    return tuple(results)


def run_document(
    path: str | Path,
    config: PipelineConfig,
    *,
    mode: str,
    registry: ExpertRegistry | None = None,
    prepared_experts: dict[str, L3Expert] | None = None,
) -> PipelineResult:
    """Run one document through the canonical L1-L9 path.

    ``prepared_experts`` is an optional cache so a multi-document run loads each
    neural expert once. ``registry`` is injectable for tests; production uses the
    default registry, which is populated by importing ``mention_factory.experts``.
    """
    readiness = validate_readiness(config, mode=mode)
    if readiness:
        raise RuntimeError(f"pipeline not ready for mode={mode}: {', '.join(readiness)}")

    # --- L1 -----------------------------------------------------------------
    l1_config, l1_lexicon = load_l1_config(config.l1_config)
    graph = analyze_document(path, config=l1_config, lexicon=l1_lexicon)

    # --- L2 + L3 (E1/E2 deterministic) --------------------------------------
    phase1b = run_phase1b(
        graph,
        Phase1BConfig.load(
            config.router_config,
            config.medication_config,
            config.laboratory_config,
        ),
    )

    # --- L3 (registered experts) -------------------------------------------
    # Every ENABLED expert runs. An enabled-but-unready expert raises
    # `ExpertNotReady` from here, naming its role and exact missing path, rather
    # than silently contributing nothing.
    # Mode-scoped: `deterministic` runs E1+E2 even when the profile enables E3.
    scoped_flags = flags_for_mode(config, mode)
    expert_proposals, expert_records = run_registered_experts(
        graph,
        phase1b.routings,
        feature_flags=scoped_flags,
        settings=config.expert_settings,
        registry=registry,
        prepared=prepared_experts,
    )

    # --- L3 unified span lattice -------------------------------------------
    lattice = build_span_lattice(
        graph.document_id,
        graph.original_text,
        routings=phase1b.routings,
        specialist_proposals=phase1b.proposals,
        expert_spans=expert_proposals,
        relations=phase1b.relations,
        config_hash=lattice_config_hash({
            "mode": mode,
            "feature_flags": dict(sorted(scoped_flags.items())),
            "resolver_config": config.resolver_config,
        }),
    )

    # --- L4 (one canonical entry point over the lattice) -------------------
    resolved = resolve_lattice_to_hypotheses(
        lattice,
        load_resolver_v1_config(config.l4_config),
        relations=phase1b.relations,
    )
    accepted = resolved.accepted()

    # --- L5 - L9 (unchanged) -----------------------------------------------
    assertions = resolve_assertions(graph.original_text, accepted)
    icd_index, rxnorm_index = _indexes(config)
    links = _link(accepted, icd_index=icd_index, rxnorm_index=rxnorm_index)
    cascade = apply_confidence_cascade(accepted)
    decoded = decode_expected_jaccard(accepted, cascade, assertions, links)
    predictions = to_entity_predictions(decoded)
    evidence_graph = build_evidence_graph(graph.document_id, accepted, assertions, links)
    return PipelineResult(
        document_id=Path(path).stem,
        predictions=predictions,
        evidence_graph=evidence_graph,
        warnings=tuple(phase1b.warnings) + tuple(resolved.warnings),
        lattice=lattice,
        expert_records=expert_records,
        l4_counts=resolution_counts(resolved),
    )


def run_input_dir(
    input_dir: str | Path,
    *,
    output_zip: str | Path,
    config: PipelineConfig,
    mode: str,
) -> tuple[PipelineResult, ...]:
    """Run all ``*.txt`` inputs and package deterministic organizer JSON."""
    readiness = validate_readiness(config, mode=mode)
    if readiness:
        raise RuntimeError(f"pipeline not ready for mode={mode}: {', '.join(readiness)}")
    inputs = sorted(
        Path(input_dir).glob("*.txt"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    # One prepared-expert cache for the whole run: each neural expert loads once,
    # not once per document.
    prepared: dict[str, L3Expert] = {}
    results = tuple(
        run_document(path, config, mode=mode, prepared_experts=prepared)
        for path in inputs
    )
    payloads = {
        f"{result.document_id}.json": to_submission_json(result.predictions)
        for result in results
    }
    output_dir = Path(output_zip).with_suffix("")
    if output_dir.name != "output":
        output_dir = Path(output_zip).parent / "output"
    write_output_directory(payloads, output_dir)
    package_output_zip(output_dir, output_zip, expected_count=config.expected_documents)
    return results


__all__ = ["PipelineResult", "run_document", "run_input_dir"]
