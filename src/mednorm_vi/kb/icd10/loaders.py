"""Configurable tabular loaders for a local Vietnamese ICD-10 table.

The organizer's exact source/format is undisclosed, so the loader is driven by a
``ColumnMap`` rather than hard-coded columns. Supports CSV/TSV with a header row.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Maps canonical fields to source column names (header-based)."""

    code: str
    label_vi: str
    label_en: str | None = None
    parent: str | None = None
    aliases: str | None = None
    chapter: str | None = None
    status: str | None = None
    alias_separator: str = "|"

    @staticmethod
    def from_mapping(d: dict[str, Any]) -> ColumnMap:
        return ColumnMap(
            code=str(d["code"]), label_vi=str(d["label_vi"]),
            label_en=(str(d["label_en"]) if d.get("label_en") else None),
            parent=(str(d["parent"]) if d.get("parent") else None),
            aliases=(str(d["aliases"]) if d.get("aliases") else None),
            chapter=(str(d["chapter"]) if d.get("chapter") else None),
            status=(str(d["status"]) if d.get("status") else None),
            alias_separator=str(d.get("alias_separator", "|")))


@dataclass(frozen=True, slots=True)
class RawIcdRow:
    code: str
    label_vi: str
    label_en: str = ""
    parent: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    chapter: str = ""
    status: str = ""


def _delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() in (".tsv", ".tab") else ","


def _col(rec: dict[str, str], name: str | None) -> str:
    return (rec.get(name) or "").strip() if name else ""


def load_rows(path: str | Path, column_map: ColumnMap) -> list[RawIcdRow]:
    """Read a CSV/TSV ICD table into canonical raw rows via ``column_map``."""
    p = Path(path)
    rows: list[RawIcdRow] = []
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=_delimiter(p))
        for rec in reader:
            code = (rec.get(column_map.code) or "").strip()
            if not code:
                continue
            aliases: tuple[str, ...] = ()
            if column_map.aliases and rec.get(column_map.aliases):
                aliases = tuple(
                    a.strip() for a in rec[column_map.aliases].split(column_map.alias_separator)
                    if a.strip())
            rows.append(RawIcdRow(
                code=code, label_vi=_col(rec, column_map.label_vi),
                label_en=_col(rec, column_map.label_en), parent=_col(rec, column_map.parent),
                aliases=aliases, chapter=_col(rec, column_map.chapter),
                status=_col(rec, column_map.status)))
    return rows


__all__ = ["ColumnMap", "RawIcdRow", "load_rows"]
