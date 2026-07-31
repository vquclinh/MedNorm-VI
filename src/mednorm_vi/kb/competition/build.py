"""Build the competition candidate v3 snapshot (Audit 0058 §8).

One staged snapshot directory, built from local frozen 3A artifacts and local RRF
files only. Nothing is downloaded, no model is loaded and
``data/derived/kb_freeze/0057_candidate_v2/`` is opened read-only.

**Resume safety is a first-class feature here, not a convenience.** The RxNorm pass
streams two 500 MB+ relation files, so an interrupted run is a realistic event rather
than a theoretical one. Every artifact records its row count and both digests into
``build_state_v1.json`` as soon as it is written; a later invocation re-verifies each
recorded artifact against the file on disk and rebuilds only what is missing, altered
or unverifiable. That is why this builder appends to a directory instead of refusing
to touch an existing one the way :mod:`mednorm_vi.kb.freeze` does — a half-finished
3.2 GB build is worth resuming, and a silently reused *corrupt* artifact is not.

Every table carries two digests, for two different questions:

    artifact_sha256            did these bytes change?
    canonical_content_sha256   did the logical rows change?

The second is computed over sorted-key JSON of each row, so a header reordering or a
line-ending change moves the first hash and leaves the second alone. Rebuild
determinism is asserted against the second.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..freeze import sha256_file
from ..indexing.normalization import normalize_text
from .crosswalk import (
    BRIDGE_FIELDS,
    CANONICAL_CROSSWALK_SCHEMA_VERSION,
    CrosswalkAdjudication,
    normalize_crosswalk,
)
from .icd10_view import (
    ICD10_COMPETITION_VIEW_SCHEMA,
    ICD_CONCEPT_FIELDS,
    ICD_HIERARCHY_FIELDS,
    Icd10CompetitionView,
    build_icd10_competition_view,
)
from .policy import COMPETITION_POLICY_VERSION
from .rxnorm_view import (
    CONCEPT_SEARCHABLE,
    RXNORM_COMPETITION_VIEW_SCHEMA,
    RxNormClosureReport,
    build_rxnorm_competition_view,
)
from .topk import COMPETITION_TOPK_VERSION

COMPETITION_BUILDER_VERSION = "competition-kb-builder-v1"
COMPETITION_MANIFEST_SCHEMA = "competition-snapshot-manifest-schema-v1"

_BUILD_STATE = "build_state_v1.json"
_MANIFEST = "competition_snapshot_manifest_v1.json"
_VALIDATION = "competition_validation_report_v1.json"

_SYNONYM_FIELDS = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "concept_id",
    "surface",
    "normalized_surface",
    "term_type",
    "language",
    "source_vocabulary",
    "suppress",
    "searchable",
    "source_record_id",
)

_CANDIDATE_INDEX_FIELDS = (
    "schema_version",
    "ontology",
    "code",
    "snapshot_id",
    "canonical_display_name",
    "source_concept_type",
    "runtime_role",
    "final_candidate_eligible",
    "deterministic_ordering_key",
    "exclusion_reason",
)

_PROVENANCE_FIELDS = (
    "schema_version",
    "artifact",
    "source_file",
    "source_sha256",
    "extraction_method",
    "builder_version",
)

_RXN_CONSO = {
    "RXCUI": 0,
    "LAT": 1,
    "ISPREF": 6,
    "RXAUI": 7,
    "SAB": 11,
    "TTY": 12,
    "STR": 14,
    "SUPPRESS": 16,
}


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """Row count plus both digests for one emitted table."""

    name: str
    rows: int
    artifact_sha256: str
    canonical_content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.name,
            "rows": self.rows,
            "artifact_sha256": self.artifact_sha256,
            "canonical_content_sha256": self.canonical_content_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompetitionSnapshotResult:
    """Everything Audit 0058 needs to report about one v3 build."""

    snapshot_id: str
    output_dir: Path
    artifacts: tuple[ArtifactDigest, ...]
    icd_counts: Mapping[str, int]
    closure: RxNormClosureReport
    crosswalk: CrosswalkAdjudication
    validation_status: str
    reused_artifacts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "output_dir": self.output_dir.as_posix(),
            "validation_status": self.validation_status,
            "reused_artifacts": list(self.reused_artifacts),
            "icd_counts": dict(self.icd_counts),
            "rxnorm_closure": self.closure.as_dict(),
            "crosswalk": {
                "source_rows": self.crosswalk.source_row_count,
                "canonical_bridges": self.crosswalk.bridge_count,
                "unique_surfaces": self.crosswalk.unique_normalized_surfaces,
                "unique_rxcuis": self.crosswalk.unique_rxcuis,
                "decisions": dict(self.crosswalk.decision_counts),
                "canonical_content_sha256": self.crosswalk.canonical_content_sha256,
            },
            "artifacts": [a.as_dict() for a in self.artifacts],
        }


def _canonical_csv_hash(path: Path) -> tuple[int, str]:
    """Row count and content digest, computed by streaming the written CSV."""
    digest = hashlib.sha256()
    rows = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            payload = json.dumps(
                {k: (v or "") for k, v in row.items()},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update((payload + "\n").encode("utf-8"))
            rows += 1
    return rows, digest.hexdigest()


def _digest_for(output_dir: Path, name: str) -> ArtifactDigest:
    path = output_dir / name
    rows, canonical = _canonical_csv_hash(path)
    return ArtifactDigest(name, rows, sha256_file(path), canonical)


def _write_csv(
    output_dir: Path, name: str, fields: Sequence[str], rows: Iterable[Mapping[str, str]]
) -> ArtifactDigest:
    path = output_dir / name
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fields})
    return _digest_for(output_dir, name)


def _load_state(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / _BUILD_STATE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # An unreadable state file means "nothing is proven", not "nothing exists".
        return {}
    artifacts = payload.get("artifacts")
    return artifacts if isinstance(artifacts, dict) else {}


def _save_state(output_dir: Path, state: Mapping[str, Mapping[str, Any]]) -> None:
    (output_dir / _BUILD_STATE).write_text(
        json.dumps(
            {"builder_version": COMPETITION_BUILDER_VERSION, "artifacts": dict(state)},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _verified_artifact(
    output_dir: Path, name: str, state: Mapping[str, Any]
) -> ArtifactDigest | None:
    """Re-verify a previously recorded artifact against the bytes on disk."""
    recorded = state.get(name)
    if not isinstance(recorded, dict):
        return None
    path = output_dir / name
    if not path.is_file():
        return None
    if sha256_file(path) != recorded.get("artifact_sha256"):
        return None
    rows, canonical = _canonical_csv_hash(path)
    if canonical != recorded.get("canonical_content_sha256") or rows != recorded.get("rows"):
        return None
    return ArtifactDigest(name, rows, str(recorded["artifact_sha256"]), canonical)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as fh:
        return tuple(dict(row) for row in csv.DictReader(fh))


def _rrf_rows(path: Path) -> Iterable[tuple[int, list[str]]]:
    with path.open(encoding="utf-8") as fh:
        for number, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            cols = raw.rstrip("\n").split("|")
            if cols and cols[-1] == "":
                cols = cols[:-1]
            yield number, cols


def _get(cols: Sequence[str], index: int) -> str:
    return cols[index] if index < len(cols) else ""


def _synonym_rows(conso: Path, snapshot_id: str) -> Iterable[dict[str, str]]:
    """Governed synonym rows, streamed. Every searchable surface with its provenance."""
    for number, cols in _rrf_rows(conso):
        rxcui = _get(cols, _RXN_CONSO["RXCUI"])
        rxaui = _get(cols, _RXN_CONSO["RXAUI"])
        surface = _get(cols, _RXN_CONSO["STR"])
        if not rxcui or not rxaui or not surface:
            continue
        suppress = _get(cols, _RXN_CONSO["SUPPRESS"]) or "N"
        yield {
            "schema_version": RXNORM_COMPETITION_VIEW_SCHEMA,
            "ontology": "rxnorm",
            "snapshot_id": snapshot_id,
            "concept_id": rxcui,
            "surface": surface,
            "normalized_surface": normalize_text(surface, strip_accents=True),
            "term_type": _get(cols, _RXN_CONSO["TTY"]),
            "language": _get(cols, _RXN_CONSO["LAT"]),
            "source_vocabulary": _get(cols, _RXN_CONSO["SAB"]),
            "suppress": suppress,
            "searchable": "true" if suppress.upper() in {"N", ""} else "false",
            "source_record_id": f"RXNCONSO.RRF:{number}:RXAUI:{rxaui}",
        }


def _candidate_index_rows(
    icd_view: Icd10CompetitionView, output_dir: Path, snapshot_id: str
) -> Iterable[dict[str, str]]:
    """Every code the runtime may offer, with its eligibility stated explicitly.

    ICD first, then RxNorm, each in deterministic code order. This is the table L9's
    membership gate and the candidate-eligibility checks read.
    """
    for concept in icd_view.concepts:
        yield {
            "schema_version": "candidate-representation-schema-v1",
            "ontology": "icd10_vi",
            "code": concept.code,
            "snapshot_id": snapshot_id,
            "canonical_display_name": concept.source_name,
            "source_concept_type": concept.name_quality,
            "runtime_role": concept.runtime_role,
            "final_candidate_eligible": "true" if concept.searchable else "false",
            "deterministic_ordering_key": f"icd10_vi:{concept.code}",
            "exclusion_reason": concept.exclusion_reason,
        }
    with (output_dir / "rxnorm_competition_concepts_v1.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            searchable = row.get("membership") == CONCEPT_SEARCHABLE
            yield {
                "schema_version": "candidate-representation-schema-v1",
                "ontology": "rxnorm",
                "code": row.get("rxcui", ""),
                "snapshot_id": snapshot_id,
                "canonical_display_name": row.get("preferred_name", ""),
                "source_concept_type": row.get("tty_values", ""),
                "runtime_role": row.get("runtime_role", ""),
                "final_candidate_eligible": "true" if searchable else "false",
                "deterministic_ordering_key": f"rxnorm:{row.get('rxcui', '')}",
                "exclusion_reason": "" if searchable else "closure_only_not_searchable",
            }


def build_competition_snapshot(
    *,
    candidate_v2_dir: str | Path,
    prescribable_rrf: str | Path,
    full_rrf: str | Path,
    output_dir: str | Path,
    icd_source_manifest: str | Path,
    rxnorm_source_manifest: str | Path,
) -> CompetitionSnapshotResult:
    """Build or resume the competition candidate v3 snapshot."""
    v2 = Path(candidate_v2_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pre_conso = Path(prescribable_rrf) / "RXNCONSO.RRF"
    pre_rel = Path(prescribable_rrf) / "RXNREL.RRF"
    full_conso = Path(full_rrf) / "RXNCONSO.RRF"
    full_rel = Path(full_rrf) / "RXNREL.RRF"

    icd_dir = v2 / "icd10_vi_tt06_2026"
    rx_dir = v2 / "rxnorm_prescribable_2026_07_06"

    source_hashes = {
        (icd_dir / "icd10_concepts_v1.csv").as_posix(): sha256_file(
            icd_dir / "icd10_concepts_v1.csv"
        ),
        (icd_dir / "icd10_hierarchy_edges_v1.csv").as_posix(): sha256_file(
            icd_dir / "icd10_hierarchy_edges_v1.csv"
        ),
        (rx_dir / "inn_rxnorm_crosswalk_v1.csv").as_posix(): sha256_file(
            rx_dir / "inn_rxnorm_crosswalk_v1.csv"
        ),
        pre_conso.as_posix(): sha256_file(pre_conso),
        pre_rel.as_posix(): sha256_file(pre_rel),
        full_conso.as_posix(): sha256_file(full_conso),
        full_rel.as_posix(): sha256_file(full_rel),
    }
    snapshot_id = (
        "competition-kb-v3-candidate-"
        + hashlib.sha256(
            "|".join(f"{k}={v}" for k, v in sorted(source_hashes.items())).encode("ascii")
        ).hexdigest()[:16]
    )

    state = _load_state(out)
    artifacts: dict[str, ArtifactDigest] = {}
    reused: list[str] = []

    def emit(
        name: str, fields: Sequence[str], rows_factory: Any, *, always_rebuild: bool = False
    ) -> ArtifactDigest:
        if not always_rebuild:
            verified = _verified_artifact(out, name, state)
            if verified is not None:
                reused.append(name)
                artifacts[name] = verified
                return verified
        digest = _write_csv(out, name, fields, rows_factory())
        artifacts[name] = digest
        state[name] = digest.as_dict()
        _save_state(out, state)
        return digest

    # --- ICD competition view ---------------------------------------------------
    icd_view = build_icd10_competition_view(
        _read_csv(icd_dir / "icd10_concepts_v1.csv"),
        _read_csv(icd_dir / "icd10_hierarchy_edges_v1.csv"),
    )
    emit(
        "icd10_competition_concepts_v1.csv",
        ICD_CONCEPT_FIELDS,
        lambda: (c.as_row(snapshot_id=snapshot_id) for c in icd_view.concepts),
    )
    emit(
        "icd10_advisory_hierarchy_v1.csv",
        ICD_HIERARCHY_FIELDS,
        lambda: (e.as_row(snapshot_id=snapshot_id) for e in icd_view.edges),
    )

    # --- canonical crosswalk ----------------------------------------------------
    adjudication = normalize_crosswalk(_read_csv(rx_dir / "inn_rxnorm_crosswalk_v1.csv"))
    emit(
        "inn_rxnorm_canonical_bridges_v1.csv",
        BRIDGE_FIELDS,
        lambda: (b.as_row(snapshot_id=snapshot_id) for b in adjudication.bridges),
    )
    (out / "inn_rxnorm_bridge_adjudication_v1.json").write_text(
        json.dumps(adjudication.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- RxNorm searchable view, retained relations and endpoint closure ---------
    rx_concepts = "rxnorm_competition_concepts_v1.csv"
    rx_relations = "rxnorm_competition_relations_v1.csv"
    verified_concepts = _verified_artifact(out, rx_concepts, state)
    verified_relations = _verified_artifact(out, rx_relations, state)
    closure_path = out / "rxnorm_closure_report_v1.json"
    if verified_concepts is not None and verified_relations is not None and closure_path.is_file():
        reused.extend((rx_concepts, rx_relations))
        artifacts[rx_concepts] = verified_concepts
        artifacts[rx_relations] = verified_relations
        closure = RxNormClosureReport(**json.loads(closure_path.read_text(encoding="utf-8")))
    else:
        closure = build_rxnorm_competition_view(
            prescribable_rrf=prescribable_rrf,
            full_rrf=full_rrf,
            output_dir=out,
            snapshot_id=snapshot_id,
        )
        closure_path.write_text(
            json.dumps(asdict(closure), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in (rx_concepts, rx_relations):
            digest = _digest_for(out, name)
            artifacts[name] = digest
            state[name] = digest.as_dict()
        _save_state(out, state)

    # --- governed synonyms and the runtime candidate index ----------------------
    emit(
        "rxnorm_governed_synonyms_v1.csv",
        _SYNONYM_FIELDS,
        lambda: _synonym_rows(pre_conso, snapshot_id),
    )
    emit(
        "competition_candidate_index_v1.csv",
        _CANDIDATE_INDEX_FIELDS,
        lambda: _candidate_index_rows(icd_view, out, snapshot_id),
        # Depends on the RxNorm concept table, so it is rebuilt whenever that is.
        always_rebuild=rx_concepts not in reused,
    )
    # Provenance links each emitted artifact to the exact sources it derives from,
    # so a reader can answer "where did this table come from?" without reading code.
    artifact_sources: Mapping[str, tuple[Path, ...]] = {
        "icd10_competition_concepts_v1.csv": (icd_dir / "icd10_concepts_v1.csv",),
        "icd10_advisory_hierarchy_v1.csv": (
            icd_dir / "icd10_concepts_v1.csv",
            icd_dir / "icd10_hierarchy_edges_v1.csv",
        ),
        "inn_rxnorm_canonical_bridges_v1.csv": (rx_dir / "inn_rxnorm_crosswalk_v1.csv",),
        "rxnorm_competition_concepts_v1.csv": (pre_conso, full_conso, pre_rel, full_rel),
        "rxnorm_competition_relations_v1.csv": (pre_rel, full_rel, pre_conso, full_conso),
        "rxnorm_governed_synonyms_v1.csv": (pre_conso,),
        "competition_candidate_index_v1.csv": (icd_dir / "icd10_concepts_v1.csv", pre_conso),
    }
    emit(
        "competition_provenance_v1.csv",
        _PROVENANCE_FIELDS,
        lambda: (
            {
                "schema_version": "kb-provenance-schema-v1",
                "artifact": name,
                "source_file": source.as_posix(),
                "source_sha256": source_hashes[source.as_posix()],
                "extraction_method": "local_frozen_artifact_policy_reread",
                "builder_version": COMPETITION_BUILDER_VERSION,
            }
            for name in sorted(artifact_sources)
            for source in artifact_sources[name]
        ),
    )

    # --- validation --------------------------------------------------------------
    violations: list[str] = []
    if closure.missing_endpoints_after_closure:
        violations.append(
            f"{closure.missing_endpoints_after_closure} retained relations lack an endpoint"
        )
    if adjudication.decision_counts.get("CLINICALLY_APPROVED"):
        violations.append("an automated decision claimed clinical approval")
    bridge_keys = [b.key for b in adjudication.bridges]
    if len(set(bridge_keys)) != len(bridge_keys):
        violations.append("duplicate canonical crosswalk bridge key")
    if any(b.may_emit_final_candidate for b in adjudication.bridges):
        violations.append("a crosswalk bridge claimed final-candidate eligibility")
    if icd_view.counts.get("hierarchy_may_offer_candidate"):
        violations.append("an advisory ICD hierarchy edge claimed candidate authority")

    validation_status = "VALID_COMPETITION_CANDIDATE" if not violations else "INVALID"
    validation = {
        "validation_status": validation_status,
        "violations": violations,
        "policy_version": COMPETITION_POLICY_VERSION,
        "topk_policy_version": COMPETITION_TOPK_VERSION,
        "icd": dict(icd_view.counts),
        "rxnorm_closure": closure.as_dict(),
        "crosswalk": {
            "source_rows": adjudication.source_row_count,
            "canonical_bridges": adjudication.bridge_count,
            "unique_normalized_surfaces": adjudication.unique_normalized_surfaces,
            "unique_rxcuis": adjudication.unique_rxcuis,
            "decision_counts": dict(adjudication.decision_counts),
            "runtime_role_counts": dict(adjudication.runtime_role_counts),
            "canonical_content_sha256": adjudication.canonical_content_sha256,
        },
    }
    (out / _VALIDATION).write_text(
        json.dumps(validation, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    ordered = tuple(artifacts[name] for name in sorted(artifacts))
    manifest = {
        "schema_version": COMPETITION_MANIFEST_SCHEMA,
        "snapshot_id": snapshot_id,
        "builder_version": COMPETITION_BUILDER_VERSION,
        "policy_version": COMPETITION_POLICY_VERSION,
        "schema_versions": {
            "icd10_view": ICD10_COMPETITION_VIEW_SCHEMA,
            "rxnorm_view": RXNORM_COMPETITION_VIEW_SCHEMA,
            "crosswalk": CANONICAL_CROSSWALK_SCHEMA_VERSION,
            "candidate": "candidate-representation-schema-v1",
            "manifest": COMPETITION_MANIFEST_SCHEMA,
        },
        "source_manifests": [
            Path(icd_source_manifest).as_posix(),
            Path(rxnorm_source_manifest).as_posix(),
        ],
        "source_files": source_hashes,
        "predecessor_snapshot_dir": v2.as_posix(),
        "artifacts": [a.as_dict() for a in ordered],
        "counts": {
            "icd": dict(icd_view.counts),
            "rxnorm": closure.as_dict(),
            "crosswalk_source_rows": adjudication.source_row_count,
            "crosswalk_canonical_bridges": adjudication.bridge_count,
        },
        "validation_status": validation_status,
        "validation_report_sha256": sha256_file(out / _VALIDATION),
        "known_limitations": [
            "Crosswalk bridges are RETRIEVAL_ONLY; none is clinically adjudicated.",
            "ICD hierarchy remains advisory prefix inference and may not offer codes.",
            "Candidate accuracy is unmeasurable: the governed corpus carries no codes.",
        ],
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_external_data_statement": (
            "Built from existing local repository resources only; no internet, "
            "external API, model, notebook, Docker, training, internal_test or "
            "organizer-test input was used."
        ),
        "build_command": "python -m mednorm_vi.kb.competition.build",
    }
    (out / _MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return CompetitionSnapshotResult(
        snapshot_id=snapshot_id,
        output_dir=out,
        artifacts=ordered,
        icd_counts=icd_view.counts,
        closure=closure,
        crosswalk=adjudication,
        validation_status=validation_status,
        reused_artifacts=tuple(sorted(set(reused))),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the competition candidate v3 snapshot.")
    parser.add_argument("--candidate-v2-dir", required=True)
    parser.add_argument("--prescribable-rrf", required=True)
    parser.add_argument("--full-rrf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--icd-source-manifest", required=True)
    parser.add_argument("--rxnorm-source-manifest", required=True)
    args = parser.parse_args(argv)
    result = build_competition_snapshot(
        candidate_v2_dir=args.candidate_v2_dir,
        prescribable_rrf=args.prescribable_rrf,
        full_rrf=args.full_rrf,
        output_dir=args.output_dir,
        icd_source_manifest=args.icd_source_manifest,
        rxnorm_source_manifest=args.rxnorm_source_manifest,
    )
    json.dump(result.as_dict(), sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0 if result.validation_status == "VALID_COMPETITION_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPETITION_BUILDER_VERSION",
    "COMPETITION_MANIFEST_SCHEMA",
    "ArtifactDigest",
    "CompetitionSnapshotResult",
    "build_competition_snapshot",
    "main",
]
