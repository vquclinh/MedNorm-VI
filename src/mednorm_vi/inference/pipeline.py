"""The canonical L1-L9 runner (spec §16). One runner, one path, one L4.

Flow, after Audit 0052::

    L1 DocumentGraph
      -> L2 route tags
      -> L3  E1 + E2 deterministic  +  every ENABLED registered expert
      -> ONE unified SpanLattice          (lattice.builder.build_span_lattice)
      -> ONE canonical L4                 (resolution.canonical)
      -> L5 assertions + ICD/RxNorm linking
      -> L6 evidence graph -> L7 cascade -> L8 decoder -> L9 validate + package

Before Audit 0052 this runner passed Phase-1B proposals straight to a second,
non-canonical L4 (``resolution/resolver.py``, since **deleted** in Audit 0055). It
never built a lattice, so E3 — the project's only trained model — could not run in
production, and each new expert would have meant editing this file. Experts now
arrive through ``mention_factory.registry``: this module names no expert except the
deterministic E1/E2 pair that Phase 1B owns, and there is exactly one L4 entry point
(``resolution.canonical.resolve_lattice_to_hypotheses``).
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..confidence_cascade import apply_confidence_cascade
from ..confidence_cascade.escalation import CascadeReport, run_cascade_escalation
from ..deterministic_baseline.models import Phase1BConfig
from ..deterministic_baseline.pipeline import run_phase1b
from ..document_intelligence import analyze_document
from ..document_intelligence.builder import load_config as load_l1_config
from ..document_intelligence.models import NodeKind
from ..evidence_graph import (
    ClinicalEvidenceGraph,
    GraphConsistencyReport,
    build_evidence_graph,
    evaluate_consistency,
)
from ..kb.competition.topk import resolve_decoding_profile
from ..kb.indexing.retrieval import LocalIndex, load_index
from ..lattice.builder import build_span_lattice, lattice_config_hash
from ..lattice.models import SpanLattice
from ..linking.icd10 import link_icd10
from ..linking.models import LinkerResult
from ..linking.rxnorm import governed_bridge_notes, link_rxnorm
from ..mention_factory import experts as _registered_experts  # noqa: F401
from ..mention_factory.registry import (
    ExpertRegistry,
    ExpertRunRecord,
    L3Expert,
    run_registered_experts,
)
from ..mention_factory.route_gate import (
    RouteGateDiagnostics,
    build_route_gate_diagnostics,
)
from ..metric_decoder import decode_entities
from ..resolution.canonical import resolution_counts, resolve_lattice_to_hypotheses
from ..resolution.config_v1 import load_resolver_v1_config
from ..resolution.learned_dispatch import (
    LearnedL4Backend,
    build_learned_l4_backend,
    load_learned_l4_config,
)
from ..resolution.models import EntityHypothesis
from ..schemas.constants import CANDIDATE_ONTOLOGY_BY_TYPE, ORGANIZER_LABEL_BY_TYPE
from ..schemas.prediction import EntityPrediction
from ..specialists.assertion import resolve_assertions
from ..validator.final_gate import (
    FinalDocument,
    FinalValidationError,
    gate_final_documents,
)
from ..validator.kb_membership import KbMembershipViolation, LockedSnapshots
from .config import (
    L4_ROUTE_LEARNED_V2,
    PipelineConfig,
    evaluate_readiness,
    flags_for_mode,
    select_l4_route,
    validate_readiness,
)
from .manifest import (
    CheckpointRecord,
    SnapshotRecord,
    build_run_manifest,
    peak_memory_gib,
    role_safe_path,
    write_run_manifest,
)
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
    # Route eligibility per document (Audit 0053). Reported so a route defect is
    # visible without reading parser source — which is how the Audit-0052 C2
    # over-routing stayed hidden.
    route_gate: RouteGateDiagnostics | None = None
    # L6/L7 typed public contracts (Audit 0054). Reported so a consistency
    # contradiction or an escalation is visible without re-deriving it downstream.
    consistency: GraphConsistencyReport | None = None
    escalation: CascadeReport | None = None
    # The SOURCE document, carried so the final L9 gate can verify
    # `original_text[start:end] == text` on the serialized payload (Audit 0056a).
    # Without it the invariant was checked four times in memory and never once on
    # the bytes that would be submitted.
    original_text: str = ""
    # Per serialized-entity index, the codes L5 linking OFFERED for that mention.
    # Spec P7: a model may select from these, never introduce a code. Index i here
    # is index i of the serialized list, because `to_entity_predictions` preserves
    # the decoder's order and `to_submission_json` preserves the list.
    offered_codes_by_index: dict[int, tuple[str, ...]] = field(default_factory=dict)

    def experts_that_ran(self) -> tuple[str, ...]:
        return tuple(record.expert_id for record in self.expert_records if record.executed)

    def proposal_counts_by_expert(self) -> dict[str, int]:
        counts = {
            record.expert_id: record.proposal_count
            for record in self.expert_records
            if record.executed
        }
        if self.lattice is not None:
            for proposal in self.lattice.proposals:
                for source in proposal.sources:
                    if source.expert_id.startswith(("E1", "E2")):
                        counts[source.expert_id] = counts.get(source.expert_id, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(slots=True)
class PreparedIndexes:
    """Loaded KB indices reused across the documents of ONE run (Audit 0060 §10).

    Audit 0059 measured `run_document` reloading both indices per document — 0.33 s for
    ICD and 2.56 s for RxNorm, about 4.8 minutes of pure waste in a 100-document run.

    This is a **caller-owned** cache, not a process-wide singleton. Identity is the
    exact configured path, so two configs cannot contaminate each other and a fixture
    index stays injectable; a run that wants fresh state simply constructs a new one.
    A module-level cache would have been fewer lines and would have made every test
    that swaps indices order-dependent.
    """

    icd_path: str = ""
    rxnorm_path: str = ""
    icd: LocalIndex | None = None
    rxnorm: LocalIndex | None = None
    loads: int = 0

    def for_config(self, config: PipelineConfig) -> tuple[LocalIndex | None, LocalIndex | None]:
        if config.icd_index != self.icd_path:
            self.icd = load_index(config.icd_index) if config.icd_index else None
            self.icd_path = config.icd_index
            self.loads += 1
        if config.rxnorm_index != self.rxnorm_path:
            self.rxnorm = load_index(config.rxnorm_index) if config.rxnorm_index else None
            self.rxnorm_path = config.rxnorm_index
            self.loads += 1
        return self.icd, self.rxnorm


def _indexes(
    config: PipelineConfig, prepared: PreparedIndexes | None = None
) -> tuple[LocalIndex | None, LocalIndex | None]:
    """Load the configured indices, reusing ``prepared`` when the caller supplied one."""
    if prepared is not None:
        return prepared.for_config(config)
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


def resolve_l4_backend(
    config: PipelineConfig,
    *,
    learned_backend: LearnedL4Backend | None = None,
) -> LearnedL4Backend | None:
    """The L4 backend this profile selected, or ``None`` for the deterministic route.

    ``None`` means "run the deterministic canonical L4", and it is returned **only**
    when the profile did not select the learned route. When the learned route is
    selected, this either returns a usable backend or raises: there is no path on
    which an explicitly enabled learned L4 quietly becomes the deterministic one.
    """
    if select_l4_route(config.feature_flags) != L4_ROUTE_LEARNED_V2:
        return None
    return build_learned_l4_backend(
        load_learned_l4_config(config.learned_l4_config), backend=learned_backend
    )


def run_document(
    path: str | Path,
    config: PipelineConfig,
    *,
    mode: str,
    registry: ExpertRegistry | None = None,
    prepared_experts: dict[str, L3Expert] | None = None,
    learned_l4_backend: LearnedL4Backend | None = None,
    prepared_indexes: PreparedIndexes | None = None,
) -> PipelineResult:
    """Run one document through the canonical L1-L9 path.

    ``prepared_experts`` is an optional cache so a multi-document run loads each
    neural expert once. ``registry`` is injectable for tests; production uses the
    default registry, which is populated by importing ``mention_factory.experts``.
    ``learned_l4_backend`` is the injection point for the learned L4 route; it is
    consulted only when the profile selected that route.
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
        config_hash=lattice_config_hash(
            {
                "mode": mode,
                "feature_flags": dict(sorted(scoped_flags.items())),
                "l4_config": config.l4_config,
            }
        ),
    )

    # --- L4 (one canonical entry point over the lattice) -------------------
    # The route is the profile's, not this function's: `resolve_l4_backend` returns
    # None for the deterministic route and a verified backend for the learned one,
    # raising rather than falling back if the learned route cannot run.
    resolved = resolve_lattice_to_hypotheses(
        lattice,
        load_resolver_v1_config(config.l4_config),
        relations=phase1b.relations,
        learned_backend=resolve_l4_backend(config, learned_backend=learned_l4_backend),
    )
    accepted = resolved.accepted()

    # --- L5 - L9 (unchanged) -----------------------------------------------
    assertions = resolve_assertions(graph.original_text, accepted)
    icd_index, rxnorm_index = _indexes(config, prepared_indexes)
    links = _link(accepted, icd_index=icd_index, rxnorm_index=rxnorm_index)
    cascade = apply_confidence_cascade(accepted)

    # --- L6: graph + typed consistency ---------------------------------------
    evidence_graph = build_evidence_graph(
        graph.document_id, accepted, assertions, links, document=graph, relations=phase1b.relations
    )
    section_categories = {
        node.node_id: str(node.category or "")
        for node in graph.nodes
        if node.kind == NodeKind.SECTION
    }
    consistency = evaluate_consistency(
        graph.document_id,
        evidence_graph,
        accepted,
        assertions,
        links,
        section_categories=section_categories,
    )

    # --- L7: locked-option escalation (no model; deterministic fallback) ------
    escalation = run_cascade_escalation(
        graph.document_id,
        accepted,
        cascade=cascade,
        assertions=assertions,
        link_results=links,
        consistency=consistency,
        ontology_by_type={
            # Keyed by the ORGANIZER-facing label, because that is what
            # `EntityHypothesis.entity_type` carries (see `consistency.ontology_for`).
            ORGANIZER_LABEL_BY_TYPE[internal]: ontology
            for internal, ontology in CANDIDATE_ONTOLOGY_BY_TYPE.items()
            if ontology is not None and internal in ORGANIZER_LABEL_BY_TYPE
        },
    )

    # --- L8: decode, now consuming L6 and L7 ---------------------------------
    decoded = decode_entities(
        accepted,
        cascade,
        assertions,
        links,
        consistency=consistency,
        escalation=escalation,
        topk_policy=resolve_decoding_profile(config.decoding_profile),
    )
    predictions = to_entity_predictions(decoded)
    # Spec P7's offered set, captured at the ONLY place that knows it: L5 linking.
    # Keyed by serialized index, which is the decoder's order (see PipelineResult).
    # Taking it from the linker rather than from L8's surviving candidates is what
    # makes a post-L8 injection detectable — a code L8 invented would be absent here.
    offered_by_mention = {
        result.mention_id: tuple(c.code for c in result.candidates) for result in links
    }
    offered_codes_by_index = {
        index: offered_by_mention.get(entity.hypothesis.hypothesis_id, ())
        for index, entity in enumerate(decoded)
    }
    proposals_by_expert: dict[str, int] = {}
    for proposal in lattice.proposals:
        for source in proposal.sources:
            proposals_by_expert[source.expert_id] = proposals_by_expert.get(source.expert_id, 0) + 1

    return PipelineResult(
        document_id=Path(path).stem,
        predictions=predictions,
        evidence_graph=evidence_graph,
        # Governed INN-bridge evidence is lifted from L5 into the document-level
        # warning channel (Audit 0058B). Audit 0058A found the note stopped at the
        # linker report, so a run could use a governed bridge with nothing above L5
        # recording it. This is observability only: it reads `links`, which L8 has
        # already consumed, and adds nothing to `predictions`.
        warnings=(
            tuple(phase1b.warnings)
            + tuple(resolved.warnings)
            + governed_bridge_notes(note for link in links for note in link.warnings)
        ),
        lattice=lattice,
        expert_records=expert_records,
        l4_counts=resolution_counts(resolved),
        route_gate=build_route_gate_diagnostics(
            graph.document_id, phase1b.routings, proposals_by_expert=proposals_by_expert
        ),
        consistency=consistency,
        escalation=escalation,
        original_text=graph.original_text,
        offered_codes_by_index=offered_codes_by_index,
    )


def _report_progress(ordinal: int, total: int, result: PipelineResult, *, started: float) -> None:
    """One flushed progress line per document, on stderr (Audit 0060 §11).

    stderr, not stdout: stdout carries the CLI's machine-readable result lines, and a
    progress stream interleaved into them would break anything parsing them. Carries
    counts only — never clinical text, and not even the document\'s entity texts.
    """
    elapsed = time.monotonic() - started
    rate = elapsed / max(ordinal, 1)
    eta = rate * (total - ordinal)
    print(
        f"[{ordinal}/{total}] document={result.document_id} "
        f"entities={len(result.predictions)} warnings={len(result.warnings)} "
        f"elapsed={elapsed:.1f}s eta={eta:.0f}s",
        file=sys.stderr,
        flush=True,
    )


def _gate_final_output(
    results: Sequence[PipelineResult],
    payloads: dict[str, str],
    config: PipelineConfig,
    prepared_indexes: PreparedIndexes | None = None,
) -> dict[str, int]:
    """The canonical final L9 gate on the packaging path (spec §16, Appendix A).

    Audit 0053 wired a *candidate-membership* gate here; Audit 0056a replaced it with
    the complete one. The membership gate never saw ``original_text`` and never
    received the offered sets, so two of Appendix A's requirements — "every entity
    satisfies ``original_text[start:end] == text``" and spec P7's "a model may not
    introduce a code" — were unenforced on the emitted payload.

    It runs on the **serialized organizer JSON**, because that is what the organizer
    receives: validating an in-memory model would leave the serializer unchecked. On
    any violation it raises, so no output directory and no ``output.zip`` is written.
    """
    icd_index, rxnorm_index = _indexes(config, prepared_indexes)
    snapshots = LockedSnapshots(icd10=icd_index, rxnorm=rxnorm_index)
    by_document = {result.document_id: result for result in results}
    documents: list[FinalDocument] = []
    for name in sorted(
        payloads, key=lambda n: (0, int(Path(n).stem)) if Path(n).stem.isdigit() else (1, n)
    ):
        document_id = Path(name).stem
        result = by_document.get(document_id)
        if result is None:
            # A payload with no run behind it cannot be validated against a source
            # document, and packaging something unvalidatable is exactly what this
            # gate exists to prevent.
            raise FinalValidationError([f"{name}:final.no_pipeline_result"])
        documents.append(
            FinalDocument(
                document_id=document_id,
                filename=name,
                original_text=result.original_text,
                payload=payloads[name],
                offered_codes_by_index=dict(result.offered_codes_by_index),
            )
        )
    return gate_final_documents(documents, snapshots)


def run_input_dir(
    input_dir: str | Path,
    *,
    output_zip: str | Path,
    config: PipelineConfig,
    mode: str,
    run_manifest_path: str | Path | None = None,
    learned_l4_backend: LearnedL4Backend | None = None,
) -> tuple[PipelineResult, ...]:
    """Run all ``*.txt`` inputs and package deterministic organizer JSON.

    ``run_manifest_path`` is opt-in (the CLI's ``--run-manifest``). Nothing is written
    unless it is given, and the manifest is written **even when L9 stops the run** —
    Audit 0053 §11 recorded that a detected violation left no record at all.
    """
    started = time.monotonic()
    readiness = validate_readiness(config, mode=mode)
    if readiness:
        raise RuntimeError(f"pipeline not ready for mode={mode}: {', '.join(readiness)}")
    inputs = sorted(
        Path(input_dir).glob("*.txt"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    # One prepared-expert cache for the whole run: each neural expert loads once,
    # not once per document. The L4 backend is resolved once for the same reason —
    # and, when the learned route is selected, its failure stops the run here rather
    # than on the first document.
    prepared: dict[str, L3Expert] = {}
    prepared_indexes = PreparedIndexes()
    l4_backend = resolve_l4_backend(config, learned_backend=learned_l4_backend)
    collected: list[PipelineResult] = []
    total = len(inputs)
    for ordinal, path in enumerate(inputs, start=1):
        result = run_document(
            path,
            config,
            mode=mode,
            prepared_experts=prepared,
            learned_l4_backend=l4_backend,
            prepared_indexes=prepared_indexes,
        )
        collected.append(result)
        _report_progress(ordinal, total, result, started=started)
    results = tuple(collected)
    payloads = {
        f"{result.document_id}.json": to_submission_json(result.predictions) for result in results
    }
    # The final L9 gate runs BEFORE anything is written: a violating run must not
    # leave a partial package on disk for someone to submit by mistake.
    try:
        l9_counts = _gate_final_output(results, payloads, config, prepared_indexes)
    except (FinalValidationError, KbMembershipViolation) as violation:
        if run_manifest_path is not None:
            _write_manifest(
                results,
                mode=mode,
                config=config,
                path=run_manifest_path,
                started=started,
                l9_issue_counts=_violation_counts(violation),
                l9_stopped=True,
            )
        raise
    output_dir = Path(output_zip).with_suffix("")
    if output_dir.name != "output":
        output_dir = Path(output_zip).parent / "output"
    write_output_directory(payloads, output_dir)
    package_output_zip(output_dir, output_zip, expected_count=config.expected_documents)
    if run_manifest_path is not None:
        _write_manifest(
            results,
            mode=mode,
            config=config,
            path=run_manifest_path,
            started=started,
            l9_issue_counts=l9_counts,
            l9_stopped=False,
            output_zip=output_zip,
        )
    return results


def _violation_counts(
    violation: FinalValidationError | KbMembershipViolation,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in violation.violations:
        code = entry.rsplit(":", 1)[-1]
        counts[code] = counts.get(code, 0) + 1
    return counts


def _write_manifest(
    results: Sequence[PipelineResult],
    *,
    mode: str,
    config: PipelineConfig,
    path: str | Path,
    started: float,
    l9_issue_counts: Mapping[str, int],
    l9_stopped: bool,
    output_zip: str | Path | None = None,
) -> None:
    """Assemble and write the run manifest. Only called when a path was requested."""
    icd_index, rxnorm_index = _indexes(config)
    snapshots = [
        SnapshotRecord(
            role=role,
            index_type=index.index_type,
            snapshot_id=index.source_snapshot_id,
            concept_count=len(index.records),
            safe_path=role_safe_path(source, role=role),
        )
        for role, index, source in (
            ("icd10_index", icd_index, config.icd_index),
            ("rxnorm_index", rxnorm_index, config.rxnorm_index),
        )
        if index is not None and source
    ]
    # The checkpoint SHA-256 the expert itself verified is reused rather than
    # recomputed: E3's checkpoint is 1.6 GB and the adapter already hashed it.
    checkpoints = [
        CheckpointRecord(
            role=record.expert_id,
            present=bool(record.path),
            sha256=record.checkpoint_sha256 or None,
            model_revision=record.model_revision,
            parameter_count_status="NOT_VERIFIED",
            safe_path=role_safe_path(record.path, role=record.expert_id)
            if record.path
            else f"role:{record.expert_id}",
        )
        for result in results[:1]
        for record in result.expert_records
        if record.executed
    ]
    manifest = build_run_manifest(
        results,
        mode=mode,
        config=config,
        readiness=evaluate_readiness(config, mode=mode),
        architecture_pdf="docs/MedNorm-VI_Architecture.pdf",
        checkpoints=checkpoints,
        snapshots=snapshots,
        l9_issue_counts=l9_issue_counts,
        l9_stopped=l9_stopped,
        runtime_seconds=time.monotonic() - started,
        peak_memory_gib=peak_memory_gib(),
        output_zip_sha256=(_sha256_of(output_zip) if output_zip is not None else None),
    )
    write_run_manifest(manifest, path)


def _sha256_of(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FinalValidationError",
    "KbMembershipViolation",
    "PipelineResult",
    "resolve_l4_backend",
    "run_document",
    "run_input_dir",
]
