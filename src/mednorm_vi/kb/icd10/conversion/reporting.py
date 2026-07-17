"""Writers for deterministic ICD conversion artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .normalization import NormalizedIcdRecord
from .pdf_inspection import PdfInspection
from .validation import ConversionValidation


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(records: tuple[NormalizedIcdRecord, ...]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in records:
        out.append(
            {
                "supplied_code": r.supplied_code,
                "dotted_code": r.dotted_code,
                "undotted_code": r.undotted_code,
                "vietnamese_label": r.vietnamese_label,
                "english_label": r.english_label,
                "aliases": "|".join(r.aliases),
                "chapter": r.chapter,
                "block": r.block,
                "parent": r.parent,
                "children": "|".join(r.children),
                "specificity": str(r.specificity),
                "source_page": str(r.source_page),
                "source_row": str(r.source_row),
                "source_document_sha256": r.source_document_sha256,
                "status": r.status,
                "flags": "|".join(r.flags),
            }
        )
    return out


def write_artifacts(
    output_dir: str | Path,
    *,
    inspection: PdfInspection,
    records: tuple[NormalizedIcdRecord, ...],
    validation: ConversionValidation,
) -> dict[str, Any]:
    """Write normalized, alias, hierarchy, manifest, and report artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized = out / "icd10_vi_normalized.csv"
    aliases = out / "icd10_vi_aliases.csv"
    hierarchy = out / "icd10_vi_hierarchy.csv"
    manifest = out / "conversion_manifest.yaml"
    report = out / "conversion_report.json"

    fields = [
        "supplied_code",
        "dotted_code",
        "undotted_code",
        "vietnamese_label",
        "english_label",
        "aliases",
        "chapter",
        "block",
        "parent",
        "children",
        "specificity",
        "source_page",
        "source_row",
        "source_document_sha256",
        "status",
        "flags",
    ]
    rows = _rows(records)
    _write_csv(normalized, fields, rows)
    _write_csv(
        aliases,
        ["undotted_code", "alias", "source"],
        [
            {"undotted_code": r.undotted_code, "alias": alias, "source": "official_pdf"}
            for r in records
            for alias in r.aliases
        ],
    )
    _write_csv(
        hierarchy,
        ["parent", "child", "relationship"],
        [
            {"parent": r.parent, "child": r.undotted_code, "relationship": "inferred_parent"}
            for r in records
            if r.parent
        ],
    )
    artifact_hashes = {
        normalized.name: _file_sha256(normalized),
        aliases.name: _file_sha256(aliases),
        hierarchy.name: _file_sha256(hierarchy),
    }
    payload: dict[str, Any] = {
        "resource_id": "icd10-vi-tt06-2026-derived",
        "source_pdf": inspection.path,
        "source_pdf_sha256": inspection.sha256,
        "source_pdf_md5": inspection.md5,
        "conversion": "embedded-text extraction; no OCR; conservative row parser",
        "row_count": len(records),
        "validation_ok": validation.ok,
        "artifact_hashes": artifact_hashes,
    }
    manifest.write_text(
        yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    report_payload: dict[str, Any] = {
        "inspection": asdict(inspection),
        "row_count": len(records),
        "validation": asdict(validation),
        "label_missing_count": sum(1 for r in records if not r.vietnamese_label),
        "chapter_count": len({r.chapter for r in records if r.chapter}),
        "block_count": len({r.block for r in records if r.block}),
        "artifact_hashes": artifact_hashes,
    }
    report.write_text(
        json.dumps(report_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_payload
