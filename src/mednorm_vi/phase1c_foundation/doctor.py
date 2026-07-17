"""Phase 1C-A ``doctor`` — deterministic readiness report (offline).

Aggregates the organizer-policy registry, position policies, resource-manifest
governance, local KB snapshot availability, public-NER manifests, and resolver
readiness into one structured, deterministic report. Performs NO network access
and reads only explicit local paths (with sensible in-repo defaults).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kb.rxnorm import discover_rrf
from ..organizer_policy import load_organizer_registry
from ..position import load_position_registry
from ..resolution import ResolverConfig
from ..resources import load_manifest, validate_manifest
from ..resources.ner import load_ner_manifest, validate_ner_manifest


@dataclass(frozen=True, slots=True)
class DoctorPaths:
    organizer_dir: Path = Path("configs/organizer")
    position_config: Path = Path("configs/organizer/position_policies_v1.yaml")
    resolver_config: Path = Path("configs/resolution/resolver_v1.yaml")
    manifests_dir: Path = Path("data/manifests")
    resource_templates_dir: Path = Path("configs/resources")
    ner_manifests_dir: Path = Path("configs/resources/ner")
    rxnorm_dir: Path = Path("data/external/rxnorm")
    icd_dir: Path = Path("data/external/icd10_vi")


@dataclass(frozen=True, slots=True)
class DoctorReport:
    data: dict[str, Any]
    missing_local_resources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def determinism_hash(self) -> str:
        canonical = json.dumps(self.data, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_manifests(directory: Path, pattern: str = "*.yaml") -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if directory.is_dir():
        for p in sorted(directory.glob(pattern)):
            try:
                m = load_manifest(p)
                vr = validate_manifest(m)
                results.append({"file": p.name, "resource_id": m.resource_id,
                                "usable": m.is_usable, "errors": len(vr.errors),
                                "warnings": len(vr.warnings)})
            except (ValueError, KeyError) as exc:  # malformed manifest
                results.append({"file": p.name, "error": str(exc)})
    return {"count": len(results), "manifests": results}


def _ner_manifests(directory: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if directory.is_dir():
        for p in sorted(directory.glob("*.yaml")):
            try:
                m = load_ner_manifest(p)
                vr = validate_ner_manifest(m)
                results.append({"file": p.name, "dataset_id": m.dataset_id,
                                "license_status": m.license.status,
                                "errors": len(vr.errors)})
            except (ValueError, KeyError) as exc:
                results.append({"file": p.name, "error": str(exc)})
    return {"count": len(results), "manifests": results}


def _relative_names(root: Path, paths: list[Path]) -> list[str]:
    out: list[str] = []
    for path in paths:
        try:
            out.append(path.relative_to(root).as_posix())
        except ValueError:
            out.append(path.as_posix())
    return sorted(out)


def build_report(paths: DoctorPaths | None = None) -> DoctorReport:
    p = paths or DoctorPaths()
    missing: list[str] = []

    registry = load_organizer_registry(p.organizer_dir)
    pos = load_position_registry(p.position_config)

    rrf = discover_rrf(p.rxnorm_dir)
    rxnorm_available = rrf.has_conso
    if not rxnorm_available:
        missing.append(f"RxNorm snapshot under {p.rxnorm_dir}/ (RXNCONSO.RRF)")

    icd_tables = sorted(p.icd_dir.rglob("*.csv")) if p.icd_dir.is_dir() else []
    icd_files = _relative_names(p.icd_dir, icd_tables)
    icd_available = bool(icd_files)
    if not icd_available:
        missing.append(f"ICD-10 VI normalized table under {p.icd_dir}/ (*.csv)")

    icd_pdf_paths = sorted(p.icd_dir.rglob("*.pdf")) if p.icd_dir.is_dir() else []
    icd_pdf_files = _relative_names(p.icd_dir, icd_pdf_paths)
    required_icd_pdfs = {"06-byt.pdf", "06-byt-kem.pdf"}
    present_pdf_names = {path.name for path in icd_pdf_paths}
    icd_source_available = required_icd_pdfs.issubset(present_pdf_names)
    if not icd_source_available:
        missing.append(f"ICD-10 VI official source PDFs under {p.icd_dir}/")

    resolver_ready = p.resolver_config.is_file()
    if resolver_ready:
        ResolverConfig.load(p.resolver_config)  # validates loadability

    data: dict[str, Any] = {
        "phase": "1C-A",
        "organizer_registry": {
            "config_hash": registry.config_hash,
            "confirmed_facts": len(registry.confirmed_facts),
            "unresolved_policies": len(registry.unresolved_policies),
            "open_policies": registry.open_policy_count,
            "total_hypotheses": registry.hypothesis_count,
        },
        "position_policies": {
            "config_hash": pos.config_hash,
            "registered": list(pos.policy_ids),
            "default": pos.default_policy_id,
        },
        "resource_manifests": _validate_manifests(p.manifests_dir),
        "resource_templates": _validate_manifests(p.resource_templates_dir, "*.manifest*.yaml"),
        "public_ner_manifests": _ner_manifests(p.ner_manifests_dir),
        "rxnorm_snapshot": {"available": rxnorm_available,
                            "missing_files": list(rrf.missing()),
                            "root": str(rrf.root) if rrf.root else None},
        "rxnorm_prescribable_snapshot": {
            "available": rxnorm_available,
            "root": str(rrf.root) if rrf.root else None,
            "rxnconso": str(rrf.conso) if rrf.conso else None,
            "missing_files": list(rrf.missing()),
        },
        "rxnorm_full_snapshot": {
            "available": False,
            "status": "MISSING / awaiting UMLS approval",
        },
        "icd_source_artifacts": {"available": icd_source_available,
                                 "files": icd_pdf_files},
        "icd_snapshot": {"available": icd_available, "files": icd_files},
        "resolver": {"ready": resolver_ready,
                     "config": str(p.resolver_config) if resolver_ready else None},
        "no_network": True,
    }
    return DoctorReport(data=data, missing_local_resources=tuple(missing))


def render_report(report: DoctorReport) -> str:
    d = report.data
    lines = ["MedNorm-VI Phase 1C-A — doctor", "=" * 46]
    org = d["organizer_registry"]
    lines.append(f"organizer registry     : {org['confirmed_facts']} facts, "
                 f"{org['open_policies']}/{org['unresolved_policies']} open policies, "
                 f"{org['total_hypotheses']} hypotheses")
    pos = d["position_policies"]
    lines.append(f"position policies      : {len(pos['registered'])} registered "
                 f"(default {pos['default']})")
    lines.append(f"resource manifests     : {d['resource_manifests']['count']} tracked, "
                 f"{d['resource_templates']['count']} templates")
    lines.append(f"public NER manifests   : {d['public_ner_manifests']['count']}")
    rxp = d["rxnorm_prescribable_snapshot"]
    lines.append(f"RxNorm Prescribable    : "
                 f"{'available' if rxp['available'] else 'MISSING (local)'}")
    lines.append("RxNorm Full 2026       : MISSING / awaiting UMLS approval")
    icd_src = d["icd_source_artifacts"]
    lines.append(f"ICD-10 VI source PDFs  : "
                 f"{'available' if icd_src['available'] else 'MISSING (local)'}")
    lines.append(f"ICD-10 VI table        : "
                 f"{'available' if d['icd_snapshot']['available'] else 'NOT PREPARED / MISSING'}")
    lines.append(f"RxNorm snapshot        : "
                 f"{'available' if d['rxnorm_snapshot']['available'] else 'MISSING (local)'}")
    lines.append(f"resolver               : {'ready' if d['resolver']['ready'] else 'NOT READY'}")
    lines.append("network access         : none")
    if report.missing_local_resources:
        lines.append("missing local resources:")
        for m in report.missing_local_resources:
            lines.append(f"  - {m}")
    lines.append(f"determinism hash       : {report.determinism_hash}")
    return "\n".join(lines) + "\n"


__all__ = ["DoctorPaths", "DoctorReport", "build_report", "render_report"]
