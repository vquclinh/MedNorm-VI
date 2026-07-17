"""Shared L1-L9 pipeline implementation for deterministic, specialist, and full modes."""

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
from ..linking.icd10 import link_icd10
from ..linking.models import LinkerResult
from ..linking.rxnorm import link_rxnorm
from ..metric_decoder import decode_expected_jaccard
from ..resolution import ResolverConfig, resolve
from ..resolution.models import EntityHypothesis
from ..schemas.prediction import EntityPrediction
from ..specialists.assertion import resolve_assertions
from .config import PipelineConfig, validate_readiness
from .packaging import package_output_zip, write_output_directory
from .serialization import to_entity_predictions, to_submission_json


@dataclass(frozen=True, slots=True)
class PipelineResult:
    document_id: str
    predictions: tuple[EntityPrediction, ...]
    evidence_graph: ClinicalEvidenceGraph
    warnings: tuple[str, ...] = field(default_factory=tuple)


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


def run_document(path: str | Path, config: PipelineConfig, *, mode: str) -> PipelineResult:
    """Run a single document through the shared L1-L9 contracts."""
    readiness = validate_readiness(config, mode=mode)
    if readiness:
        raise RuntimeError(f"pipeline not ready for mode={mode}: {', '.join(readiness)}")
    l1_config, l1_lexicon = load_l1_config(config.l1_config)
    graph = analyze_document(path, config=l1_config, lexicon=l1_lexicon)
    phase1b = run_phase1b(
        graph,
        Phase1BConfig.load(
            config.router_config,
            config.medication_config,
            config.laboratory_config,
        ),
    )
    resolved = resolve(
        graph.document_id,
        graph.original_text,
        list(phase1b.proposals),
        list(phase1b.relations),
        ResolverConfig.load(config.resolver_config),
    )
    accepted = resolved.accepted()
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
        warnings=tuple(phase1b.warnings + resolved.warnings),
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
    results = tuple(run_document(path, config, mode=mode) for path in inputs)
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
