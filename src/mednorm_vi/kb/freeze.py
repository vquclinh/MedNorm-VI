"""Deterministic KB freeze builders for Milestone 3A.

The existing runtime consumes compact ``index.json`` views. Those views are useful
for lookup, but they are not rich enough to be the authoritative KB record: ICD
parents are inferred in the legacy artifacts and RxNorm relation labels are not
present in the runtime graph. This module writes the audited, versioned tables that
own provenance, review state and validation; runtime views can be derived from
those tables, but the canonical KB evidence lives here.

The builders are offline and local-only. They never download data and never load
models. RxNorm files are read line by line; the only bounded in-memory structure is
the concept/synonym aggregate needed to validate relation endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from .indexing.normalization import normalize_text
from .rxnorm.models import STRUCTURE_RELAS

KB_FREEZE_BUILDER_VERSION = "kb-freeze-builder-v1"

ICD10_KB_SCHEMA_VERSION = "icd10-kb-schema-v1"
RXNORM_KB_SCHEMA_VERSION = "rxnorm-kb-schema-v1"
KB_PROVENANCE_SCHEMA_VERSION = "kb-provenance-schema-v1"
GOVERNED_SYNONYM_SCHEMA_VERSION = "governed-synonym-schema-v1"
INN_RXNORM_CROSSWALK_SCHEMA_VERSION = "inn-rxnorm-crosswalk-schema-v1"
CANDIDATE_REPRESENTATION_SCHEMA_VERSION = "candidate-representation-schema-v1"
KB_SNAPSHOT_MANIFEST_SCHEMA_VERSION = "kb-snapshot-manifest-schema-v1"

SOURCE_VERBATIM = "SOURCE_VERBATIM"
NORMALIZED_FROM_SOURCE = "NORMALIZED_FROM_SOURCE"
SUSPECT_TRUNCATION = "SUSPECT_TRUNCATION"
MISSING_SOURCE_NAME = "MISSING_SOURCE_NAME"
REQUIRES_REVIEW = "REQUIRES_REVIEW"
REVIEWED_APPROVED = "REVIEWED_APPROVED"
REVIEWED_REJECTED = "REVIEWED_REJECTED"
AMBIGUOUS = "AMBIGUOUS"
UNMAPPED = "UNMAPPED"

REVIEW_STATES: tuple[str, ...] = (
    SOURCE_VERBATIM,
    NORMALIZED_FROM_SOURCE,
    SUSPECT_TRUNCATION,
    MISSING_SOURCE_NAME,
    REQUIRES_REVIEW,
    REVIEWED_APPROVED,
    REVIEWED_REJECTED,
    AMBIGUOUS,
    UNMAPPED,
)

PROPOSED = "PROPOSED"
APPROVED = "APPROVED"
REJECTED = "REJECTED"

CROSSWALK_STATES: tuple[str, ...] = (
    PROPOSED,
    REQUIRES_REVIEW,
    APPROVED,
    REJECTED,
    AMBIGUOUS,
    UNMAPPED,
)

ONTOLOGY_ICD10 = "icd10_vi"
ONTOLOGY_RXNORM = "rxnorm"

_RXN_CONSO = {
    "RXCUI": 0,
    "LAT": 1,
    "TS": 2,
    "ISPREF": 6,
    "RXAUI": 7,
    "SAB": 11,
    "TTY": 12,
    "CODE": 13,
    "STR": 14,
    "SUPPRESS": 16,
}
_RXN_REL = {
    "RXCUI1": 0,
    "REL": 3,
    "RXCUI2": 4,
    "RELA": 7,
    "RUI": 8,
    "SAB": 10,
    "SL": 11,
    "RG": 12,
    "DIR": 13,
    "SUPPRESS": 14,
}

_ICD_CONCEPT_FIELDS = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "code",
    "display_code",
    "lookup_code",
    "canonical_display_name",
    "original_source_name",
    "source_version",
    "source_file",
    "source_record_id",
    "extraction_method",
    "provenance_id",
    "review_status",
    "quality_flags",
    "runtime_authoritative",
)

_ICD_EDGE_FIELDS = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "parent_code",
    "child_code",
    "relation_label",
    "direction",
    "source_evidence",
    "provenance_id",
    "review_status",
    "quality_flags",
    "runtime_authoritative",
)

_RXNORM_CONCEPT_FIELDS = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "rxcui",
    "canonical_display_name",
    "original_source_name",
    "normalized_name",
    "tty_values",
    "source_vocabularies",
    "suppress_values",
    "preferred_rxaui",
    "atom_count",
    "active_atom_count",
    "source_version",
    "source_file",
    "source_record_id",
    "extraction_method",
    "provenance_id",
    "review_status",
    "quality_flags",
    "runtime_authoritative",
)

_RXNORM_RELATION_FIELDS = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "source_rxcui",
    "target_rxcui",
    "rel",
    "rela",
    "source_vocabulary",
    "direction",
    "rui",
    "suppress",
    "source_row",
    "endpoint_status",
    "provenance_id",
    "review_status",
    "quality_flags",
    "runtime_authoritative",
)

_SYNONYM_FIELDS = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "concept_id",
    "source_record_id",
    "surface",
    "normalized_surface",
    "language",
    "term_type",
    "synonym_kind",
    "source_vocabulary",
    "provenance_id",
    "review_status",
    "quality_flags",
    "runtime_authoritative",
)

_CROSSWALK_FIELDS = (
    "schema_version",
    "snapshot_id",
    "local_inn_surface",
    "normalized_local_surface",
    "rxnorm_name",
    "rxcui",
    "rxnorm_canonical_name",
    "rxnorm_tty",
    "mapping_type",
    "evidence_source",
    "provenance_ids",
    "reviewer",
    "review_timestamp",
    "review_decision",
    "ambiguity_notes",
    "runtime_authoritative",
)

_PROVENANCE_FIELDS = (
    "schema_version",
    "provenance_id",
    "ontology",
    "source_file",
    "source_record_id",
    "source_sha256",
    "source_version",
    "extraction_method",
    "builder_version",
)

_REVIEW_QUEUE_FIELDS = (
    "queue_id",
    "ontology",
    "snapshot_id",
    "record_id",
    "issue_code",
    "severity",
    "source_file",
    "source_record_id",
    "evidence",
    "required_action",
)

_LEGACY_INN_TO_RXNORM_NAME: Mapping[str, str] = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "lignocaine": "lidocaine",
    "frusemide": "furosemide",
    "glibenclamide": "glyburide",
    "amoxycillin": "amoxicillin",
    "cephalexin": "cefalexin",
    "rifampicin": "rifampin",
    "trimethoprim-sulfamethoxazole": "sulfamethoxazole / trimethoprim",
    "co-trimoxazole": "sulfamethoxazole / trimethoprim",
}


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """Byte and logical-content digests for one emitted table."""

    path: str
    rows: int
    artifact_sha256: str
    canonical_content_sha256: str


@dataclass(frozen=True, slots=True)
class KbFreezeResult:
    """Result of one ontology freeze build."""

    ontology: str
    snapshot_id: str
    output_dir: Path
    manifest_path: Path
    validation_status: str
    counts: Mapping[str, object]
    artifacts: tuple[ArtifactDigest, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "ontology": self.ontology,
            "snapshot_id": self.snapshot_id,
            "output_dir": self.output_dir.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "validation_status": self.validation_status,
            "counts": dict(self.counts),
            "artifacts": [
                {
                    "path": item.path,
                    "rows": item.rows,
                    "artifact_sha256": item.artifact_sha256,
                    "canonical_content_sha256": item.canonical_content_sha256,
                }
                for item in self.artifacts
            ],
        }


@dataclass(slots=True)
class _RxConceptAggregate:
    rxcui: str
    strings: set[str] = field(default_factory=set)
    normalized_strings: set[str] = field(default_factory=set)
    tty_values: set[str] = field(default_factory=set)
    source_vocabularies: set[str] = field(default_factory=set)
    suppress_values: set[str] = field(default_factory=set)
    preferred_name: str = ""
    preferred_rxaui: str = ""
    first_source_row: int = 0
    atom_count: int = 0
    active_atom_count: int = 0


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 for local artifact integrity."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_line_hash(row: Mapping[str, str]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]
) -> ArtifactDigest:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {field: str(row.get(field, "")) for field in fieldnames}
            writer.writerow(normalized)
            canonical.update(_json_line_hash(normalized))
            count += 1
    return ArtifactDigest(
        path=path.as_posix(),
        rows=count,
        artifact_sha256=sha256_file(path),
        canonical_content_sha256=canonical.hexdigest(),
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _staging_dir_for(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp-{os.getpid()}")


def _begin_staged_output(target: Path) -> Path:
    staging = _staging_dir_for(target)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite existing KB freeze output directory: {target}"
        )
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    return staging


def _promote_staged_output(staging: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(
            f"refusing to promote over existing KB freeze output directory: {target}"
        )
    staging.rename(target)


def _retarget_artifacts(
    artifacts: Sequence[ArtifactDigest], *, staging: Path, target: Path
) -> tuple[ArtifactDigest, ...]:
    out: list[ArtifactDigest] = []
    for artifact in artifacts:
        path = Path(artifact.path)
        try:
            relative = path.relative_to(staging)
            display_path = (target / relative).as_posix()
        except ValueError:
            display_path = artifact.path
        out.append(
            ArtifactDigest(
                path=display_path,
                rows=artifact.rows,
                artifact_sha256=artifact.artifact_sha256,
                canonical_content_sha256=artifact.canonical_content_sha256,
            )
        )
    return tuple(out)


def _source_record_id(row: Mapping[str, str]) -> str:
    return f"page:{row.get('source_page', '')}:row:{row.get('source_row', '')}"


def _provenance_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"prov-{digest}"


def _pipe(values: Iterable[str]) -> str:
    return "|".join(sorted(v for v in values if v))


def _snapshot_suffix(*hashes: str) -> str:
    digest = hashlib.sha256("|".join(hashes).encode("ascii")).hexdigest()[:16]
    return digest


def _icd_quality_flags(row: Mapping[str, str]) -> tuple[str, ...]:
    label = row.get("vietnamese_label", "")
    flags: set[str] = {f for f in row.get("flags", "").split("|") if f}
    stripped = label.strip()
    lower = normalize_text(stripped, strip_accents=True)
    if not stripped:
        flags.add("missing_source_name")
    if stripped.startswith(("-", "+", "++", "+-", "(")):
        flags.add("suspect_pdf_fragment:leading_marker")
    if "†" in stripped or "*" in stripped:
        flags.add("suspect_pdf_fragment:dagger_asterisk")
    if lower.startswith(("loai tru", "bao gom")) or " loai tru:" in lower:
        flags.add("suspect_guidance_text")
    if lower.endswith((" do", " cua", " voi", " de", " gom", " va", " va/hoac")):
        flags.add("suspect_trailing_function_word")
    return tuple(sorted(flags))


def _review_status_for_icd(flags: Sequence[str]) -> str:
    if any(flag == "missing_source_name" for flag in flags):
        return MISSING_SOURCE_NAME
    if any(flag.startswith("suspect_") for flag in flags):
        return SUSPECT_TRUNCATION
    if any(flag == "duplicate_code_rows_collapsed" for flag in flags):
        return REQUIRES_REVIEW
    return NORMALIZED_FROM_SOURCE


def _read_icd_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as fh:
        return tuple(dict(row) for row in csv.DictReader(fh))


def _has_cycle(parent_by_child: Mapping[str, str]) -> bool:
    for code in parent_by_child:
        seen: set[str] = set()
        current = code
        while current in parent_by_child:
            current = parent_by_child[current]
            if current in seen:
                return True
            seen.add(current)
    return False


def build_icd10_freeze(
    *,
    normalized_csv: str | Path,
    source_manifest: str | Path,
    output_dir: str | Path,
    source_version: str = "TT06-2026 official source PDFs",
) -> KbFreezeResult:
    """Build a provenance-rich ICD-10 candidate freeze from local derived CSV."""
    source_path = Path(normalized_csv)
    target = Path(output_dir)
    out = _begin_staged_output(target)
    source_hash = sha256_file(source_path)
    snapshot_id = f"icd10-vi-tt06-2026-kb-v1-candidate-{_snapshot_suffix(source_hash)}"
    rows = _read_icd_rows(source_path)
    known = {row.get("undotted_code", "") for row in rows if row.get("undotted_code")}
    parent_by_child = {
        row.get("undotted_code", ""): row.get("parent", "")
        for row in rows
        if row.get("undotted_code") and row.get("parent")
    }

    concept_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item.get("undotted_code", "")):
        code = row.get("undotted_code", "")
        supplied = row.get("supplied_code", "")
        source_record = _source_record_id(row)
        provenance = _provenance_id("icd10", source_hash, source_record, code)
        flags = _icd_quality_flags(row)
        review_status = _review_status_for_icd(flags)
        concept_rows.append(
            {
                "schema_version": ICD10_KB_SCHEMA_VERSION,
                "ontology": ONTOLOGY_ICD10,
                "snapshot_id": snapshot_id,
                "code": code,
                "display_code": row.get("dotted_code", supplied),
                "lookup_code": normalize_text(code),
                "canonical_display_name": row.get("vietnamese_label", ""),
                "original_source_name": row.get("vietnamese_label", ""),
                "source_version": source_version,
                "source_file": source_path.as_posix(),
                "source_record_id": source_record,
                "extraction_method": "legacy_pdf_text_conversion_preserved",
                "provenance_id": provenance,
                "review_status": review_status,
                "quality_flags": "|".join(flags),
                "runtime_authoritative": "false"
                if review_status != NORMALIZED_FROM_SOURCE
                else "true",
            }
        )
        provenance_rows.append(
            {
                "schema_version": KB_PROVENANCE_SCHEMA_VERSION,
                "provenance_id": provenance,
                "ontology": ONTOLOGY_ICD10,
                "source_file": source_path.as_posix(),
                "source_record_id": source_record,
                "source_sha256": source_hash,
                "source_version": source_version,
                "extraction_method": "legacy_pdf_text_conversion_preserved",
                "builder_version": KB_FREEZE_BUILDER_VERSION,
            }
        )
        for flag in flags:
            if flag.startswith("suspect_") or flag in {
                "missing_source_name",
                "duplicate_code_rows_collapsed",
            }:
                review_rows.append(
                    {
                        "queue_id": _provenance_id("icd10-review", code, flag),
                        "ontology": ONTOLOGY_ICD10,
                        "snapshot_id": snapshot_id,
                        "record_id": code,
                        "issue_code": flag,
                        "severity": "warning",
                        "source_file": source_path.as_posix(),
                        "source_record_id": source_record,
                        "evidence": row.get("vietnamese_label", ""),
                        "required_action": "human_review_against_authorized_source",
                    }
                )

    edge_rows: list[dict[str, str]] = []
    for child, parent in sorted(parent_by_child.items()):
        provenance = _provenance_id("icd10-edge", source_hash, parent, child)
        endpoint_flags: list[str] = []
        if parent not in known:
            endpoint_flags.append("missing_parent_endpoint")
        if child == parent:
            endpoint_flags.append("self_parent")
        edge_rows.append(
            {
                "schema_version": ICD10_KB_SCHEMA_VERSION,
                "ontology": ONTOLOGY_ICD10,
                "snapshot_id": snapshot_id,
                "parent_code": parent,
                "child_code": child,
                "relation_label": "parent_of",
                "direction": "parent_to_child",
                "source_evidence": "legacy_code_prefix_inference",
                "provenance_id": provenance,
                "review_status": REQUIRES_REVIEW,
                "quality_flags": "|".join(
                    ("inferred_parent_requires_source_review", *endpoint_flags)
                ),
                "runtime_authoritative": "false",
            }
        )
        review_rows.append(
            {
                "queue_id": _provenance_id("icd10-parent-review", parent, child),
                "ontology": ONTOLOGY_ICD10,
                "snapshot_id": snapshot_id,
                "record_id": child,
                "issue_code": "inferred_parent_requires_source_review",
                "severity": "warning",
                "source_file": source_path.as_posix(),
                "source_record_id": "",
                "evidence": f"{parent}->{child}",
                "required_action": "confirm_parent_from_authorized_source_structure",
            }
        )

    alias_rows: tuple[dict[str, str], ...] = ()
    validation = {
        "duplicate_codes": len(rows) - len(known),
        "concepts": len(concept_rows),
        "hierarchy_edges": len(edge_rows),
        "hierarchy_cycles": int(_has_cycle(parent_by_child)),
        "missing_parent_edges": sum(1 for edge in edge_rows if edge["parent_code"] not in known),
        "source_supported_parent_edges": 0,
        "inferred_parent_edges_requiring_review": len(edge_rows),
        "suspect_name_records": sum(
            1 for row in concept_rows if row["review_status"] == SUSPECT_TRUNCATION
        ),
        "missing_name_records": sum(
            1 for row in concept_rows if row["review_status"] == MISSING_SOURCE_NAME
        ),
        "runtime_authoritative_concepts": sum(
            1 for row in concept_rows if row["runtime_authoritative"] == "true"
        ),
        "review_queue_rows": len(review_rows),
        "validation_result": "CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED",
    }

    staged_artifacts = (
        _write_csv(out / "icd10_concepts_v1.csv", _ICD_CONCEPT_FIELDS, concept_rows),
        _write_csv(out / "icd10_hierarchy_edges_v1.csv", _ICD_EDGE_FIELDS, edge_rows),
        _write_csv(out / "icd10_aliases_v1.csv", _SYNONYM_FIELDS, alias_rows),
        _write_csv(out / "icd10_provenance_v1.csv", _PROVENANCE_FIELDS, provenance_rows),
        _write_csv(out / "icd10_review_queue_v1.csv", _REVIEW_QUEUE_FIELDS, review_rows),
    )
    artifacts = _retarget_artifacts(staged_artifacts, staging=out, target=target)
    validation_sha = _write_json(out / "icd10_validation_report_v1.json", validation)
    manifest = _manifest_payload(
        ontology=ONTOLOGY_ICD10,
        snapshot_id=snapshot_id,
        source_manifest=Path(source_manifest),
        source_files={source_path.as_posix(): source_hash},
        artifacts=artifacts,
        counts=validation,
        known_limitations=(
            "Legacy ICD hierarchy edges are retained only as review-required "
            "prefix inferences because the current derived CSV lost source "
            "hierarchy evidence.",
            "Suspect PDF-fragment names are flagged and not silently repaired.",
        ),
        validation_sha256=validation_sha,
    )
    manifest_path = out / "kb_snapshot_manifest_v1.json"
    _write_json(manifest_path, manifest)
    _promote_staged_output(out, target)
    return KbFreezeResult(
        ontology=ONTOLOGY_ICD10,
        snapshot_id=snapshot_id,
        output_dir=target,
        manifest_path=target / "kb_snapshot_manifest_v1.json",
        validation_status=str(validation["validation_result"]),
        counts=validation,
        artifacts=artifacts,
    )


def _rrf_rows(path: Path) -> Iterable[tuple[int, list[str]]]:
    with path.open(encoding="utf-8") as fh:
        for row_number, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            cols = raw.rstrip("\n").split("|")
            if cols and cols[-1] == "":
                cols = cols[:-1]
            yield row_number, cols


def _rrf_get(cols: Sequence[str], index: int) -> str:
    return cols[index] if index < len(cols) else ""


def _numeric_key(value: str) -> tuple[int, int, str]:
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def _ingest_rxnorm_concepts(
    conso_path: Path,
) -> tuple[dict[str, _RxConceptAggregate], Counter[str]]:
    concepts: dict[str, _RxConceptAggregate] = {}
    tty_counter: Counter[str] = Counter()
    for row_number, cols in _rrf_rows(conso_path):
        rxcui = _rrf_get(cols, _RXN_CONSO["RXCUI"])
        rxaui = _rrf_get(cols, _RXN_CONSO["RXAUI"])
        surface = _rrf_get(cols, _RXN_CONSO["STR"])
        if not rxcui or not rxaui or not surface:
            continue
        tty = _rrf_get(cols, _RXN_CONSO["TTY"])
        sab = _rrf_get(cols, _RXN_CONSO["SAB"])
        suppress = _rrf_get(cols, _RXN_CONSO["SUPPRESS"]) or "N"
        agg = concepts.setdefault(rxcui, _RxConceptAggregate(rxcui=rxcui))
        if agg.first_source_row == 0:
            agg.first_source_row = row_number
        agg.atom_count += 1
        if suppress.upper() in {"N", ""}:
            agg.active_atom_count += 1
        agg.strings.add(surface)
        agg.normalized_strings.add(normalize_text(surface, strip_accents=True))
        agg.tty_values.add(tty)
        agg.source_vocabularies.add(sab)
        agg.suppress_values.add(suppress)
        tty_counter[tty] += 1
        if _rrf_get(cols, _RXN_CONSO["ISPREF"]) == "Y" and not agg.preferred_name:
            agg.preferred_name = surface
            agg.preferred_rxaui = rxaui
    return concepts, tty_counter


def _synonym_rows(
    *,
    conso_path: Path,
    snapshot_id: str,
    source_hash: str,
) -> Iterable[dict[str, str]]:
    for row_number, cols in _rrf_rows(conso_path):
        rxcui = _rrf_get(cols, _RXN_CONSO["RXCUI"])
        rxaui = _rrf_get(cols, _RXN_CONSO["RXAUI"])
        surface = _rrf_get(cols, _RXN_CONSO["STR"])
        if not rxcui or not rxaui or not surface:
            continue
        suppress = _rrf_get(cols, _RXN_CONSO["SUPPRESS"]) or "N"
        yield {
            "schema_version": GOVERNED_SYNONYM_SCHEMA_VERSION,
            "ontology": ONTOLOGY_RXNORM,
            "snapshot_id": snapshot_id,
            "concept_id": rxcui,
            "source_record_id": f"RXNCONSO.RRF:{row_number}:RXAUI:{rxaui}",
            "surface": surface,
            "normalized_surface": normalize_text(surface, strip_accents=True),
            "language": _rrf_get(cols, _RXN_CONSO["LAT"]),
            "term_type": _rrf_get(cols, _RXN_CONSO["TTY"]),
            "synonym_kind": "source_native_rxnorm_string",
            "source_vocabulary": _rrf_get(cols, _RXN_CONSO["SAB"]),
            "provenance_id": _provenance_id(
                "rxnorm-conso", source_hash, str(row_number), rxcui, rxaui
            ),
            "review_status": SOURCE_VERBATIM,
            "quality_flags": f"suppress:{suppress}",
            "runtime_authoritative": "true" if suppress.upper() in {"N", ""} else "false",
        }


def _conso_provenance_rows(
    *, conso_path: Path, source_hash: str, source_version: str
) -> Iterable[dict[str, str]]:
    for row_number, cols in _rrf_rows(conso_path):
        rxcui = _rrf_get(cols, _RXN_CONSO["RXCUI"])
        rxaui = _rrf_get(cols, _RXN_CONSO["RXAUI"])
        surface = _rrf_get(cols, _RXN_CONSO["STR"])
        if not rxcui or not rxaui or not surface:
            continue
        yield {
            "schema_version": KB_PROVENANCE_SCHEMA_VERSION,
            "provenance_id": _provenance_id(
                "rxnorm-conso", source_hash, str(row_number), rxcui, rxaui
            ),
            "ontology": ONTOLOGY_RXNORM,
            "source_file": conso_path.as_posix(),
            "source_record_id": f"RXNCONSO.RRF:{row_number}:RXAUI:{rxaui}",
            "source_sha256": source_hash,
            "source_version": source_version,
            "extraction_method": "rrf_source_verbatim",
            "builder_version": KB_FREEZE_BUILDER_VERSION,
        }


def _rxnorm_concept_rows(
    concepts: Mapping[str, _RxConceptAggregate],
    *,
    snapshot_id: str,
    source_version: str,
    conso_path: Path,
    source_hash: str,
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for rxcui, agg in sorted(concepts.items(), key=lambda item: _numeric_key(item[0])):
        canonical = agg.preferred_name or sorted(agg.strings)[0]
        suppress_values = _pipe(agg.suppress_values)
        flags: list[str] = []
        if agg.active_atom_count == 0:
            flags.append("all_atoms_suppressed")
        if len(agg.tty_values) > 1:
            flags.append("multiple_tty_values")
        rows.append(
            {
                "schema_version": RXNORM_KB_SCHEMA_VERSION,
                "ontology": ONTOLOGY_RXNORM,
                "snapshot_id": snapshot_id,
                "rxcui": rxcui,
                "canonical_display_name": canonical,
                "original_source_name": canonical,
                "normalized_name": normalize_text(canonical, strip_accents=True),
                "tty_values": _pipe(agg.tty_values),
                "source_vocabularies": _pipe(agg.source_vocabularies),
                "suppress_values": suppress_values,
                "preferred_rxaui": agg.preferred_rxaui,
                "atom_count": str(agg.atom_count),
                "active_atom_count": str(agg.active_atom_count),
                "source_version": source_version,
                "source_file": conso_path.as_posix(),
                "source_record_id": f"RXNCONSO.RRF:{agg.first_source_row}",
                "extraction_method": "rrf_source_verbatim",
                "provenance_id": _provenance_id("rxnorm-concept", source_hash, rxcui),
                "review_status": SOURCE_VERBATIM,
                "quality_flags": "|".join(flags),
                "runtime_authoritative": "true" if agg.active_atom_count else "false",
            }
        )
    return tuple(rows)


def _relation_rows(
    *,
    rel_path: Path,
    snapshot_id: str,
    source_hash: str,
    known_rxcuis: frozenset[str],
    relation_counter: Counter[str],
    rela_counter: Counter[str],
    endpoint_counter: Counter[str],
) -> Iterable[dict[str, str]]:
    for row_number, cols in _rrf_rows(rel_path):
        source = _rrf_get(cols, _RXN_REL["RXCUI1"])
        target = _rrf_get(cols, _RXN_REL["RXCUI2"])
        rel = _rrf_get(cols, _RXN_REL["REL"])
        rela = _rrf_get(cols, _RXN_REL["RELA"])
        sab = _rrf_get(cols, _RXN_REL["SAB"])
        rui = _rrf_get(cols, _RXN_REL["RUI"])
        suppress = _rrf_get(cols, _RXN_REL["SUPPRESS"]) or "N"
        endpoint_status, flags = _relation_endpoint_status(source, target, known_rxcuis)
        relation_counter[rel] += 1
        rela_counter[rela or "<empty>"] += 1
        endpoint_counter[endpoint_status] += 1
        provenance = _provenance_id("rxnorm-rel", source_hash, str(row_number), rui, source, target)
        yield {
            "schema_version": RXNORM_KB_SCHEMA_VERSION,
            "ontology": ONTOLOGY_RXNORM,
            "snapshot_id": snapshot_id,
            "source_rxcui": source,
            "target_rxcui": target,
            "rel": rel,
            "rela": rela,
            "source_vocabulary": sab,
            "direction": "source_to_target",
            "rui": rui,
            "suppress": suppress,
            "source_row": str(row_number),
            "endpoint_status": endpoint_status,
            "provenance_id": provenance,
            "review_status": SOURCE_VERBATIM,
            "quality_flags": "|".join(flags),
            "runtime_authoritative": "true" if endpoint_status == "ok" else "false",
        }


def _relation_endpoint_status(
    source: str, target: str, known_rxcuis: frozenset[str]
) -> tuple[str, tuple[str, ...]]:
    endpoint_status = "ok"
    flags: list[str] = []
    if source not in known_rxcuis:
        endpoint_status = "missing_source"
        flags.append("missing_source_endpoint")
    if target not in known_rxcuis:
        endpoint_status = (
            "missing_both" if endpoint_status == "missing_source" else "missing_target"
        )
        flags.append("missing_target_endpoint")
    if source == target:
        flags.append("self_relation")
    return endpoint_status, tuple(flags)


def _relation_review_rows(
    *,
    rel_path: Path,
    snapshot_id: str,
    known_rxcuis: frozenset[str],
) -> Iterable[dict[str, str]]:
    for row_number, cols in _rrf_rows(rel_path):
        source = _rrf_get(cols, _RXN_REL["RXCUI1"])
        target = _rrf_get(cols, _RXN_REL["RXCUI2"])
        rel = _rrf_get(cols, _RXN_REL["REL"])
        rela = _rrf_get(cols, _RXN_REL["RELA"])
        rui = _rrf_get(cols, _RXN_REL["RUI"])
        _endpoint_status, flags = _relation_endpoint_status(source, target, known_rxcuis)
        for flag in flags:
            yield {
                "queue_id": _provenance_id("rxnorm-rel-review", source, target, rel, rela, flag),
                "ontology": ONTOLOGY_RXNORM,
                "snapshot_id": snapshot_id,
                "record_id": rui or f"{source}->{target}",
                "issue_code": flag,
                "severity": "warning",
                "source_file": rel_path.as_posix(),
                "source_record_id": f"RXNREL.RRF:{row_number}:RUI:{rui}",
                "evidence": f"{source}|{rel}|{rela}|{target}",
                "required_action": "review_relation_endpoint_or_self_relation",
            }


def _relation_provenance_rows(
    *, rel_path: Path, source_hash: str, source_version: str
) -> Iterable[dict[str, str]]:
    for row_number, cols in _rrf_rows(rel_path):
        source = _rrf_get(cols, _RXN_REL["RXCUI1"])
        target = _rrf_get(cols, _RXN_REL["RXCUI2"])
        rui = _rrf_get(cols, _RXN_REL["RUI"])
        yield {
            "schema_version": KB_PROVENANCE_SCHEMA_VERSION,
            "provenance_id": _provenance_id(
                "rxnorm-rel", source_hash, str(row_number), rui, source, target
            ),
            "ontology": ONTOLOGY_RXNORM,
            "source_file": rel_path.as_posix(),
            "source_record_id": f"RXNREL.RRF:{row_number}:RUI:{rui}",
            "source_sha256": source_hash,
            "source_version": source_version,
            "extraction_method": "rrf_source_verbatim",
            "builder_version": KB_FREEZE_BUILDER_VERSION,
        }


def _crosswalk_rows(
    *,
    snapshot_id: str,
    name_to_concepts: Mapping[str, Sequence[tuple[str, str, str]]],
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for local_surface, rxnorm_name in sorted(_LEGACY_INN_TO_RXNORM_NAME.items()):
        normalized_rxnorm = normalize_text(rxnorm_name, strip_accents=True)
        matches = tuple(set(name_to_concepts.get(normalized_rxnorm, ())))
        if not matches:
            rows.append(
                _crosswalk_row(
                    snapshot_id=snapshot_id,
                    local_surface=local_surface,
                    rxnorm_name=rxnorm_name,
                    rxcui="",
                    canonical_name="",
                    tty="",
                    notes="no_exact_rxnorm_name_match_in_local_snapshot",
                )
            )
            continue
        for rxcui, canonical_name, tty in sorted(matches, key=lambda item: _numeric_key(item[0])):
            rows.append(
                _crosswalk_row(
                    snapshot_id=snapshot_id,
                    local_surface=local_surface,
                    rxnorm_name=rxnorm_name,
                    rxcui=rxcui,
                    canonical_name=canonical_name,
                    tty=tty,
                    notes=(
                        "legacy_runtime_bridge_inventory;"
                        "requires_clinical_review_before_authoritative_use"
                    ),
                )
            )
    return tuple(rows)


def _crosswalk_row(
    *,
    snapshot_id: str,
    local_surface: str,
    rxnorm_name: str,
    rxcui: str,
    canonical_name: str,
    tty: str,
    notes: str,
) -> dict[str, str]:
    provenance = _provenance_id("inn-rxnorm-crosswalk", snapshot_id, local_surface, rxcui)
    return {
        "schema_version": INN_RXNORM_CROSSWALK_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "local_inn_surface": local_surface,
        "normalized_local_surface": normalize_text(local_surface, strip_accents=True),
        "rxnorm_name": rxnorm_name,
        "rxcui": rxcui,
        "rxnorm_canonical_name": canonical_name,
        "rxnorm_tty": tty,
        "mapping_type": "legacy_inn_to_rxnorm_name_candidate",
        "evidence_source": "local_runtime_bridge_audit_0054",
        "provenance_ids": provenance,
        "reviewer": "",
        "review_timestamp": "",
        "review_decision": REQUIRES_REVIEW,
        "ambiguity_notes": notes,
        "runtime_authoritative": "false",
    }


def build_rxnorm_freeze(
    *,
    rrf_root: str | Path,
    source_manifest: str | Path,
    output_dir: str | Path,
    source_version: str,
    snapshot_label: str,
) -> KbFreezeResult:
    """Build a provenance-rich RxNorm freeze from local RRF assets."""
    root = Path(rrf_root)
    conso = root / "RXNCONSO.RRF"
    rel = root / "RXNREL.RRF"
    if not conso.is_file():
        raise FileNotFoundError(conso)
    if not rel.is_file():
        raise FileNotFoundError(rel)
    conso_hash = sha256_file(conso)
    rel_hash = sha256_file(rel)
    snapshot_id = (
        f"rxnorm-{snapshot_label}-kb-v1-candidate-{_snapshot_suffix(conso_hash, rel_hash)}"
    )
    target = Path(output_dir)
    out = _begin_staged_output(target)
    concepts, tty_counter = _ingest_rxnorm_concepts(conso)
    concept_rows = _rxnorm_concept_rows(
        concepts,
        snapshot_id=snapshot_id,
        source_version=source_version,
        conso_path=conso,
        source_hash=conso_hash,
    )
    name_to_concepts: dict[str, list[tuple[str, str, str]]] = {}
    for row in concept_rows:
        for name in (row["canonical_display_name"], row["original_source_name"]):
            name_to_concepts.setdefault(normalize_text(name, strip_accents=True), []).append(
                (row["rxcui"], row["canonical_display_name"], row["tty_values"])
            )
    target_names = {
        normalize_text(name, strip_accents=True) for name in _LEGACY_INN_TO_RXNORM_NAME.values()
    }
    for synonym in _synonym_rows(
        conso_path=conso,
        snapshot_id=snapshot_id,
        source_hash=conso_hash,
    ):
        if synonym["normalized_surface"] not in target_names:
            continue
        name_to_concepts.setdefault(synonym["normalized_surface"], []).append(
            (synonym["concept_id"], synonym["surface"], synonym["term_type"])
        )

    crosswalk_review_rows: list[dict[str, str]] = []
    relation_counter: Counter[str] = Counter()
    rela_counter: Counter[str] = Counter()
    endpoint_counter: Counter[str] = Counter()
    known = frozenset(concepts)
    artifacts: list[ArtifactDigest] = [
        _write_csv(out / "rxnorm_concepts_v1.csv", _RXNORM_CONCEPT_FIELDS, concept_rows),
        _write_csv(
            out / "rxnorm_synonyms_v1.csv",
            _SYNONYM_FIELDS,
            _synonym_rows(
                conso_path=conso,
                snapshot_id=snapshot_id,
                source_hash=conso_hash,
            ),
        ),
    ]
    relation_artifact = _write_csv(
        out / "rxnorm_relations_v1.csv",
        _RXNORM_RELATION_FIELDS,
        _relation_rows(
            rel_path=rel,
            snapshot_id=snapshot_id,
            source_hash=rel_hash,
            known_rxcuis=known,
            relation_counter=relation_counter,
            rela_counter=rela_counter,
            endpoint_counter=endpoint_counter,
        ),
    )
    artifacts.append(relation_artifact)
    crosswalk = _crosswalk_rows(snapshot_id=snapshot_id, name_to_concepts=name_to_concepts)
    for row in crosswalk:
        crosswalk_review_rows.append(
            {
                "queue_id": _provenance_id(
                    "crosswalk-review", row["local_inn_surface"], row["rxcui"]
                ),
                "ontology": ONTOLOGY_RXNORM,
                "snapshot_id": snapshot_id,
                "record_id": row["local_inn_surface"],
                "issue_code": "inn_rxnorm_mapping_requires_clinical_review",
                "severity": "warning",
                "source_file": "src/mednorm_vi/linking/structured_medication.py",
                "source_record_id": row["local_inn_surface"],
                "evidence": f"{row['local_inn_surface']}->{row['rxnorm_name']}->{row['rxcui']}",
                "required_action": "clinical_review_before_authoritative_crosswalk_use",
            }
        )
    artifacts.extend(
        (
            _write_csv(
                out / "inn_rxnorm_crosswalk_v1.csv",
                _CROSSWALK_FIELDS,
                crosswalk,
            ),
            _write_csv(
                out / "rxnorm_provenance_v1.csv",
                _PROVENANCE_FIELDS,
                chain(
                    _conso_provenance_rows(
                        conso_path=conso,
                        source_hash=conso_hash,
                        source_version=source_version,
                    ),
                    _relation_provenance_rows(
                        rel_path=rel,
                        source_hash=rel_hash,
                        source_version=source_version,
                    ),
                ),
            ),
            _write_csv(
                out / "rxnorm_review_queue_v1.csv",
                _REVIEW_QUEUE_FIELDS,
                chain(
                    _relation_review_rows(
                        rel_path=rel,
                        snapshot_id=snapshot_id,
                        known_rxcuis=known,
                    ),
                    crosswalk_review_rows,
                ),
            ),
        )
    )
    retargeted_artifacts = _retarget_artifacts(artifacts, staging=out, target=target)
    counts: dict[str, int] = {
        "concepts": len(concept_rows),
        "synonyms": sum(agg.atom_count for agg in concepts.values()),
        "relations": relation_artifact.rows,
        "relation_labels_preserved": relation_artifact.rows,
        "distinct_rel": len(relation_counter),
        "distinct_rela": len(rela_counter),
        "relation_endpoint_ok": endpoint_counter["ok"],
        "relation_missing_endpoint": sum(
            count for key, count in endpoint_counter.items() if key != "ok"
        ),
        "crosswalk_rows": len(crosswalk),
        "crosswalk_APPROVED": sum(1 for row in crosswalk if row["review_decision"] == APPROVED),
        "crosswalk_REQUIRES_REVIEW": sum(
            1 for row in crosswalk if row["review_decision"] == REQUIRES_REVIEW
        ),
        "review_queue_rows": artifacts[-1].rows,
        "tty_IN": tty_counter["IN"],
        "tty_PIN": tty_counter["PIN"],
        "tty_SCD": tty_counter["SCD"],
        "tty_SBD": tty_counter["SBD"],
    }
    validation_result = (
        "CANDIDATE_FOR_FREEZE_REVIEW_REQUIRED"
        if counts["relation_missing_endpoint"] or counts["crosswalk_REQUIRES_REVIEW"]
        else "VALID"
    )
    validation = {
        "validation_result": validation_result,
        "counts": counts,
        "tty_counts": dict(sorted(tty_counter.items())),
        "rel_counts": dict(sorted(relation_counter.items())),
        "rela_counts": dict(sorted(rela_counter.items())),
        "endpoint_counts": dict(sorted(endpoint_counter.items())),
        "structure_relas_preserved": {
            rela: rela_counter[rela] for rela in sorted(STRUCTURE_RELAS) if rela_counter[rela]
        },
        "crosswalk_policy": "only APPROVED rows are runtime-authoritative; none are approved",
    }
    validation_sha = _write_json(out / "rxnorm_validation_report_v1.json", validation)
    manifest = _manifest_payload(
        ontology=ONTOLOGY_RXNORM,
        snapshot_id=snapshot_id,
        source_manifest=Path(source_manifest),
        source_files={conso.as_posix(): conso_hash, rel.as_posix(): rel_hash},
        artifacts=retargeted_artifacts,
        counts=counts,
        known_limitations=(
            "Vietnamese INN bridge rows are inventoried as review-required and "
            "not runtime-authoritative.",
            "The existing active runtime index still uses a compact unlabeled "
            "adjacency view until a migration is explicitly approved.",
        ),
        validation_sha256=validation_sha,
    )
    manifest_path = out / "kb_snapshot_manifest_v1.json"
    _write_json(manifest_path, manifest)
    _promote_staged_output(out, target)
    return KbFreezeResult(
        ontology=ONTOLOGY_RXNORM,
        snapshot_id=snapshot_id,
        output_dir=target,
        manifest_path=target / "kb_snapshot_manifest_v1.json",
        validation_status=validation_result,
        counts=counts,
        artifacts=retargeted_artifacts,
    )


def _manifest_payload(
    *,
    ontology: str,
    snapshot_id: str,
    source_manifest: Path,
    source_files: Mapping[str, str],
    artifacts: Sequence[ArtifactDigest],
    counts: Mapping[str, object],
    known_limitations: Sequence[str],
    validation_sha256: str,
) -> dict[str, object]:
    artifact_payload = [
        {
            "path": artifact.path,
            "rows": artifact.rows,
            "artifact_sha256": artifact.artifact_sha256,
            "canonical_content_sha256": artifact.canonical_content_sha256,
        }
        for artifact in artifacts
    ]
    return {
        "schema_version": KB_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "schema_versions": {
            "icd10": ICD10_KB_SCHEMA_VERSION,
            "rxnorm": RXNORM_KB_SCHEMA_VERSION,
            "provenance": KB_PROVENANCE_SCHEMA_VERSION,
            "synonym": GOVERNED_SYNONYM_SCHEMA_VERSION,
            "crosswalk": INN_RXNORM_CROSSWALK_SCHEMA_VERSION,
            "candidate": CANDIDATE_REPRESENTATION_SCHEMA_VERSION,
            "manifest": KB_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        },
        "builder_version": KB_FREEZE_BUILDER_VERSION,
        "source_manifest": source_manifest.as_posix(),
        "source_files": dict(source_files),
        "counts": dict(counts),
        "artifacts": artifact_payload,
        "validation_report_sha256": validation_sha256,
        "known_limitations": list(known_limitations),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_external_data_statement": (
            "Built from existing local repository resources only; no internet, "
            "external API, model, notebook, Docker, training, internal_test or "
            "organizer-test input was used."
        ),
        "build_command": "python -m mednorm_vi.kb.freeze",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build audited KB freeze artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)
    icd = sub.add_parser("build-icd10", help="build ICD-10 freeze tables")
    icd.add_argument("--normalized-csv", required=True)
    icd.add_argument("--source-manifest", required=True)
    icd.add_argument("--output-dir", required=True)
    rxn = sub.add_parser("build-rxnorm", help="build RxNorm freeze tables")
    rxn.add_argument("--rrf-root", required=True)
    rxn.add_argument("--source-manifest", required=True)
    rxn.add_argument("--output-dir", required=True)
    rxn.add_argument("--source-version", required=True)
    rxn.add_argument("--snapshot-label", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-icd10":
        result = build_icd10_freeze(
            normalized_csv=args.normalized_csv,
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
        )
    elif args.command == "build-rxnorm":
        result = build_rxnorm_freeze(
            rrf_root=args.rrf_root,
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
            source_version=args.source_version,
            snapshot_label=args.snapshot_label,
        )
    else:
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROVED",
    "CANDIDATE_REPRESENTATION_SCHEMA_VERSION",
    "CROSSWALK_STATES",
    "GOVERNED_SYNONYM_SCHEMA_VERSION",
    "ICD10_KB_SCHEMA_VERSION",
    "INN_RXNORM_CROSSWALK_SCHEMA_VERSION",
    "KB_FREEZE_BUILDER_VERSION",
    "KB_PROVENANCE_SCHEMA_VERSION",
    "KB_SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "KbFreezeResult",
    "ONTOLOGY_ICD10",
    "ONTOLOGY_RXNORM",
    "REQUIRES_REVIEW",
    "REVIEW_STATES",
    "RXNORM_KB_SCHEMA_VERSION",
    "ArtifactDigest",
    "build_icd10_freeze",
    "build_rxnorm_freeze",
    "main",
    "sha256_file",
]
