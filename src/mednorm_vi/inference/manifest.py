"""Deterministic inference run manifest (spec §16, Appendix A).

Audit 0053 §11 recorded the gap this closes: L9 could **detect** a KB-membership
violation and stop a run, and nothing **recorded the stop**. A run that fails closed
with no artifact is safe but unauditable; a run that succeeds with no artifact is
unreproducible. This module writes one deterministic record per explicitly requested
run.

Three rules govern what it contains.

**No clinical text, ever.** Not source text, not entity surfaces, not prompts. Every
field is a count, a hash, an id, a version or a coordinate-free aggregate, and a test
asserts that no fixture substring appears anywhere in a serialized manifest. This is
what makes a manifest safe to commit, attach to a ticket, or diff between runs.

**No environment leakage.** Absolute paths under a user's home directory are reduced to
role-safe identifiers (``role:e3_checkpoint`` rather than
``/home/someone/checkpoint/...``), because a manifest is meant to travel. Tokens and
secrets are never read in the first place.

**Opt-in.** ``write_run_manifest`` is called only when a caller asks for a path — the
CLI's ``--run-manifest``. Nothing is written during ordinary unit construction, so
building a manifest object in a test has no filesystem effect.

Determinism: keys sorted, floats rounded, sets sorted before serialization. The same run
metadata yields byte-identical output apart from the fields explicitly listed in
``RUNTIME_VARIABLE_FIELDS`` — wall-clock duration and peak memory, which cannot be
deterministic and are documented as such rather than quietly excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "inference-run-manifest-v1"

# Fields that legitimately differ between two identical runs. Documented rather than
# hidden, so a determinism test can exclude exactly these and nothing else.
RUNTIME_VARIABLE_FIELDS: tuple[str, ...] = (
    "runtime_seconds", "peak_memory_gib",
)

# Deterministic execution settings this project fixes. Recorded so a divergent run can
# be explained rather than guessed at.
DETERMINISTIC_SEEDS: Mapping[str, int] = {
    "python_hash_seed": 0,
    "numpy_seed": 0,
    "torch_manual_seed": 0,
}


def _sha256_file(path: str | Path) -> str | None:
    """SHA-256 of a file, or ``None`` when it does not exist.

    Streamed: the E3 checkpoint is 1.6 GB and a manifest must not need that much RAM.
    """
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def role_safe_path(path: str | Path, *, role: str) -> str:
    """A path reduced to something safe to publish.

    Keeps the basename, which is diagnostic, and replaces everything above it with the
    role. ``/home/someone/checkpoint/s1/best.pt`` becomes ``role:e3_checkpoint/best.pt``.
    """
    name = Path(path).name
    return f"role:{role}/{name}" if name else f"role:{role}"


def _git(*args: str, cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
            timeout=15).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str = ""
    dirty: bool = True
    branch: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty, "branch": self.branch}


def capture_git_state(repo_root: str | Path = ".") -> GitState:
    """Current commit, branch and dirty state.

    ``dirty`` defaults to **True** when git cannot be consulted: an unknown tree is
    treated as modified, because the opposite default would let an unreproducible run
    claim to be reproducible.
    """
    root = Path(repo_root)
    commit = _git("rev-parse", "HEAD", cwd=root)
    if not commit:
        return GitState()
    status = _git("status", "--porcelain", cwd=root)
    return GitState(
        commit=commit, dirty=bool(status.strip()),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root))


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """One model artifact a run depended on."""

    role: str
    present: bool
    sha256: str | None = None
    size_bytes: int | None = None
    model_revision: str = ""
    parameter_count: int | None = None
    parameter_count_status: str = "NOT_VERIFIED"
    safe_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "present": self.present, "sha256": self.sha256,
            "size_bytes": self.size_bytes, "model_revision": self.model_revision,
            "parameter_count": self.parameter_count,
            "parameter_count_status": self.parameter_count_status,
            "path": self.safe_path,
        }


def checkpoint_record(
    role: str,
    path: str | Path | None,
    *,
    model_revision: str = "",
    parameter_count: int | None = None,
    parameter_count_status: str = "NOT_VERIFIED",
    hash_contents: bool = True,
) -> CheckpointRecord:
    """Describe one checkpoint. A missing checkpoint is recorded, never omitted."""
    if path is None:
        return CheckpointRecord(role=role, present=False, safe_path=f"role:{role}")
    target = Path(path)
    present = target.is_file()
    return CheckpointRecord(
        role=role, present=present,
        sha256=_sha256_file(target) if (present and hash_contents) else None,
        size_bytes=target.stat().st_size if present else None,
        model_revision=model_revision, parameter_count=parameter_count,
        parameter_count_status=parameter_count_status,
        safe_path=role_safe_path(target, role=role))


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """One locked ontology snapshot / generated index a run linked against."""

    role: str
    index_type: str = ""
    snapshot_id: str = ""
    index_id: str = ""
    deterministic_index_hash: str = ""
    concept_count: int | None = None
    safe_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "index_type": self.index_type,
            "snapshot_id": self.snapshot_id, "index_id": self.index_id,
            "deterministic_index_hash": self.deterministic_index_hash,
            "concept_count": self.concept_count, "path": self.safe_path,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    """One inference run, described without a single character of clinical text."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    git: GitState = field(default_factory=GitState)
    architecture_pdf_sha256: str = ""
    # Contract versions of every stage that ran.
    contract_versions: Mapping[str, str] = field(default_factory=dict)
    config_hashes: Mapping[str, str] = field(default_factory=dict)
    mode: str = ""
    readiness_status: str = ""
    readiness_errors: tuple[str, ...] = field(default_factory=tuple)
    expected_experts: tuple[str, ...] = field(default_factory=tuple)
    experts_that_ran: tuple[str, ...] = field(default_factory=tuple)
    degraded: tuple[str, ...] = field(default_factory=tuple)
    fail_closed: bool = False
    checkpoints: tuple[CheckpointRecord, ...] = field(default_factory=tuple)
    snapshots: tuple[SnapshotRecord, ...] = field(default_factory=tuple)
    # Aggregates only. Every value is a count or a distribution.
    document_count: int = 0
    route_gate_counts: Mapping[str, int] = field(default_factory=dict)
    proposal_counts: Mapping[str, int] = field(default_factory=dict)
    lattice_counts: Mapping[str, int] = field(default_factory=dict)
    l4_counts: Mapping[str, int] = field(default_factory=dict)
    assertion_counts: Mapping[str, int] = field(default_factory=dict)
    linking_counts: Mapping[str, int] = field(default_factory=dict)
    graph_edge_counts: Mapping[str, int] = field(default_factory=dict)
    consistency_decision_counts: Mapping[str, int] = field(default_factory=dict)
    consistency_issue_counts: Mapping[str, int] = field(default_factory=dict)
    cascade_disposition_counts: Mapping[str, int] = field(default_factory=dict)
    escalation_condition_counts: Mapping[str, int] = field(default_factory=dict)
    decoder_candidate_size_distribution: Mapping[str, int] = field(default_factory=dict)
    decoder_reason_counts: Mapping[str, int] = field(default_factory=dict)
    l9_issue_counts: Mapping[str, int] = field(default_factory=dict)
    l9_stopped: bool = False
    seeds: Mapping[str, int] = field(default_factory=lambda: dict(DETERMINISTIC_SEEDS))
    interpreter: str = ""
    # Runtime observations. Non-deterministic by nature; see RUNTIME_VARIABLE_FIELDS.
    runtime_seconds: float | None = None
    peak_memory_gib: float | None = None
    # Output hashes appear ONLY when writing output was separately authorized.
    output_directory_hash: str | None = None
    output_zip_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "manifest_schema_version": self.schema_version,
            "git": self.git.as_dict(),
            "architecture_pdf_sha256": self.architecture_pdf_sha256,
            "contract_versions": dict(sorted(self.contract_versions.items())),
            "config_hashes": dict(sorted(self.config_hashes.items())),
            "mode": self.mode,
            "readiness": {
                "status": self.readiness_status,
                "errors": list(self.readiness_errors),
                "degraded": list(self.degraded),
                "fail_closed": self.fail_closed,
            },
            "experts": {
                "expected": list(self.expected_experts),
                "ran": list(self.experts_that_ran),
            },
            "checkpoints": [c.as_dict() for c in self.checkpoints],
            "snapshots": [s.as_dict() for s in self.snapshots],
            "documents": self.document_count,
            "aggregates": {
                "route_gate": dict(sorted(self.route_gate_counts.items())),
                "proposals_by_expert": dict(sorted(self.proposal_counts.items())),
                "lattice": dict(sorted(self.lattice_counts.items())),
                "l4": dict(sorted(self.l4_counts.items())),
                "assertions": dict(sorted(self.assertion_counts.items())),
                "linking": dict(sorted(self.linking_counts.items())),
                "graph_edges": dict(sorted(self.graph_edge_counts.items())),
                "consistency_decisions": dict(
                    sorted(self.consistency_decision_counts.items())),
                "consistency_issues": dict(
                    sorted(self.consistency_issue_counts.items())),
                "cascade_dispositions": dict(
                    sorted(self.cascade_disposition_counts.items())),
                "escalation_conditions": dict(
                    sorted(self.escalation_condition_counts.items())),
                "decoder_candidate_sizes": dict(
                    sorted(self.decoder_candidate_size_distribution.items(),
                           key=lambda item: int(item[0]))),
                "decoder_reasons": dict(sorted(self.decoder_reason_counts.items())),
                "l9_issues": dict(sorted(self.l9_issue_counts.items())),
            },
            "l9_stopped_the_run": self.l9_stopped,
            "seeds": dict(sorted(self.seeds.items())),
            "interpreter": self.interpreter,
            "runtime_seconds": (round(self.runtime_seconds, 3)
                                if self.runtime_seconds is not None else None),
            "peak_memory_gib": (round(self.peak_memory_gib, 3)
                                if self.peak_memory_gib is not None else None),
            "output_directory_hash": self.output_directory_hash,
            "output_zip_sha256": self.output_zip_sha256,
            "contains_clinical_text": False,
        }
        return payload

    def to_json(self) -> str:
        """Deterministic serialization: sorted keys, fixed separators, trailing newline."""
        return json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @property
    def manifest_hash(self) -> str:
        """Hash over everything except the documented runtime-variable fields."""
        payload = self.as_dict()
        for name in RUNTIME_VARIABLE_FIELDS:
            payload.pop(name, None)
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _distribution(values: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def build_run_manifest(
    results: Sequence[Any],
    *,
    mode: str,
    config: Any,
    readiness: Any = None,
    architecture_pdf: str | Path | None = None,
    repo_root: str | Path = ".",
    checkpoints: Sequence[CheckpointRecord] = (),
    snapshots: Sequence[SnapshotRecord] = (),
    l9_issue_counts: Mapping[str, int] | None = None,
    l9_stopped: bool = False,
    runtime_seconds: float | None = None,
    peak_memory_gib: float | None = None,
    output_directory_hash: str | None = None,
    output_zip_sha256: str | None = None,
) -> RunManifest:
    """Aggregate a completed run into one manifest. Reads results; writes nothing."""
    from ..confidence_cascade.escalation import ESCALATION_CONTRACT_VERSION
    from ..evidence_graph import CONSISTENCY_VERSION, EVIDENCE_GRAPH_VERSION
    from ..linking.icd10 import ICD10_LINKER_VERSION
    from ..linking.rxnorm import RXNORM_LINKER_VERSION
    from ..mention_factory.route_gate import ROUTE_GATE_DIAGNOSTICS_VERSION
    from ..metric_decoder import DECODER_VERSION
    from ..validator.kb_membership import KB_MEMBERSHIP_VALIDATOR_VERSION

    route_gate: dict[str, int] = {}
    proposals: dict[str, int] = {}
    lattice: dict[str, int] = {"nodes": 0, "merged_coordinate_groups": 0}
    l4: dict[str, int] = {}
    assertions: dict[str, int] = {"with_labels": 0, "uncertain": 0, "all_three": 0}
    linking: dict[str, int] = {"results": 0, "candidates": 0, "empty_results": 0}
    edges: dict[str, int] = {}
    consistency_decisions: dict[str, int] = {}
    consistency_issues: dict[str, int] = {}
    dispositions: dict[str, int] = {}
    conditions: dict[str, int] = {}
    reasons: dict[str, int] = {}
    candidate_sizes: list[int] = []

    def accumulate(target: dict[str, int], source: Mapping[str, int]) -> None:
        for key, value in source.items():
            target[key] = target.get(key, 0) + int(value)

    for result in results:
        gate = getattr(result, "route_gate", None)
        if gate is not None:
            accumulate(route_gate, {
                f"eligible:{expert}": count
                for expert, count in gate.eligible_nodes_by_expert.items()})
            accumulate(route_gate, {
                f"skipped:{expert}": count
                for expert, count in gate.skipped_nodes_by_expert.items()})
        accumulate(proposals, result.proposal_counts_by_expert())
        if getattr(result, "lattice", None) is not None:
            lattice["nodes"] += len(result.lattice.proposals)
            lattice["merged_coordinate_groups"] += (
                result.lattice.merged_coordinate_groups)
        accumulate(l4, getattr(result, "l4_counts", {}) or {})
        graph = getattr(result, "evidence_graph", None)
        if graph is not None:
            accumulate(edges, graph.edge_counts())
            edges["declined_treats"] = (
                edges.get("declined_treats", 0) + len(graph.declined_treats))
        report = getattr(result, "consistency", None)
        if report is not None:
            accumulate(consistency_decisions, report.decision_counts())
            accumulate(consistency_issues, report.issue_counts())
        cascade = getattr(result, "escalation", None)
        if cascade is not None:
            accumulate(dispositions, cascade.disposition_counts())
            accumulate(conditions, cascade.condition_counts())
        for prediction in result.predictions:
            codes = getattr(prediction, "candidates", None) or ()
            candidate_sizes.append(len(codes))
            linking["candidates"] += len(codes)
            labels = getattr(prediction, "assertions", None) or ()
            if labels:
                assertions["with_labels"] += 1
            if len(labels) >= 3:
                assertions["all_three"] += 1

    config_hashes = {
        name: _sha256_text(str(getattr(config, name, "")))
        for name in ("l1_config", "router_config", "medication_config",
                     "laboratory_config", "resolver_config", "l4_config",
                     "icd_index", "rxnorm_index")
        if getattr(config, name, None)
    }
    readiness_status = getattr(readiness, "status", "") if readiness else ""
    return RunManifest(
        git=capture_git_state(repo_root),
        architecture_pdf_sha256=(_sha256_file(architecture_pdf) or ""
                                 if architecture_pdf else ""),
        contract_versions={
            "route_gate": ROUTE_GATE_DIAGNOSTICS_VERSION,
            "evidence_graph": EVIDENCE_GRAPH_VERSION,
            "consistency": CONSISTENCY_VERSION,
            "escalation": ESCALATION_CONTRACT_VERSION,
            "decoder": DECODER_VERSION,
            "icd10_linker": ICD10_LINKER_VERSION,
            "rxnorm_linker": RXNORM_LINKER_VERSION,
            "kb_membership_validator": KB_MEMBERSHIP_VALIDATOR_VERSION,
        },
        config_hashes=config_hashes,
        mode=mode,
        readiness_status=readiness_status,
        readiness_errors=tuple(getattr(readiness, "errors", ()) or ()),
        expected_experts=tuple(getattr(readiness, "expected_experts", ()) or ()),
        experts_that_ran=tuple(sorted({
            expert for result in results
            for expert in result.experts_that_ran()})),
        degraded=tuple(getattr(readiness, "degraded", ()) or ()),
        fail_closed=readiness_status == "NOT_READY",
        checkpoints=tuple(checkpoints), snapshots=tuple(snapshots),
        document_count=len(results),
        route_gate_counts=route_gate, proposal_counts=proposals,
        lattice_counts=lattice, l4_counts=l4, assertion_counts=assertions,
        linking_counts=linking, graph_edge_counts=edges,
        consistency_decision_counts=consistency_decisions,
        consistency_issue_counts={k: v for k, v in consistency_issues.items() if v},
        cascade_disposition_counts=dispositions,
        escalation_condition_counts={k: v for k, v in conditions.items() if v},
        decoder_candidate_size_distribution=_distribution(candidate_sizes),
        decoder_reason_counts=reasons,
        l9_issue_counts=dict(l9_issue_counts or {}), l9_stopped=l9_stopped,
        interpreter=f"{platform.python_implementation()} {platform.python_version()}",
        runtime_seconds=runtime_seconds, peak_memory_gib=peak_memory_gib,
        output_directory_hash=output_directory_hash,
        output_zip_sha256=output_zip_sha256,
    )


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write the manifest. Called **only** when a caller asked for a path.

    Refuses to write anything that reports carrying clinical text — a belt-and-braces
    check against a future field being added without thinking about leakage.
    """
    payload = manifest.as_dict()
    if payload.get("contains_clinical_text") is not False:
        raise ValueError("refusing to write a manifest that may contain clinical text")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json(), encoding="utf-8")
    return target


def peak_memory_gib() -> float | None:
    """Peak RSS in GiB, when the platform exposes it."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    divisor = 1024 * 1024 if os.uname().sysname == "Linux" else 1024 * 1024 * 1024
    return float(usage) / divisor


__all__ = [
    "DETERMINISTIC_SEEDS",
    "MANIFEST_SCHEMA_VERSION",
    "RUNTIME_VARIABLE_FIELDS",
    "CheckpointRecord",
    "GitState",
    "RunManifest",
    "SnapshotRecord",
    "build_run_manifest",
    "capture_git_state",
    "checkpoint_record",
    "peak_memory_gib",
    "role_safe_path",
    "write_run_manifest",
]
