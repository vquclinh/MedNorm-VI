"""Normalize parsed ICD rows into CSV-ready records."""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import normalization as code_norm
from .row_parser import ParsedIcdRow


@dataclass(frozen=True, slots=True)
class NormalizedIcdRecord:
    supplied_code: str
    dotted_code: str
    undotted_code: str
    vietnamese_label: str
    english_label: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    chapter: str = ""
    block: str = ""
    parent: str = ""
    children: tuple[str, ...] = field(default_factory=tuple)
    specificity: int = 0
    source_page: int = 0
    source_row: int = 0
    source_document_sha256: str = ""
    status: str = "active"
    flags: tuple[str, ...] = field(default_factory=tuple)


def normalize_rows(
    rows: tuple[ParsedIcdRow, ...], *, source_document_sha256: str
) -> tuple[NormalizedIcdRecord, ...]:
    """Add reversible code forms, inferred hierarchy, and source provenance."""
    deduped = _dedupe_rows(rows)
    undotted = {code_norm.to_undotted(row.code_supplied) for row in deduped}
    children: dict[str, list[str]] = {}
    for code in undotted:
        parent = code_norm.parent_code(code)
        while parent is not None and parent not in undotted:
            parent = code_norm.parent_code(parent)
        if parent is not None:
            children.setdefault(parent, []).append(code)

    records: list[NormalizedIcdRecord] = []
    for row in deduped:
        u = code_norm.to_undotted(row.code_supplied)
        parent = code_norm.parent_code(u)
        while parent is not None and parent not in undotted:
            parent = code_norm.parent_code(parent)
        records.append(
            NormalizedIcdRecord(
                supplied_code=row.code_supplied,
                dotted_code=code_norm.to_dotted(row.code_supplied),
                undotted_code=u,
                vietnamese_label=" ".join(row.label_vi.split()),
                english_label=row.label_en,
                aliases=row.aliases,
                chapter=row.chapter or code_norm.chapter_letter(u),
                block=row.block,
                parent=parent or "",
                children=tuple(sorted(children.get(u, []))),
                specificity=code_norm.specificity(u),
                source_page=row.source_page,
                source_row=row.source_row,
                source_document_sha256=source_document_sha256,
                flags=row.flags,
            )
        )
    records.sort(key=lambda r: (r.undotted_code, r.source_row, r.source_page))
    return tuple(records)


def _dedupe_rows(rows: tuple[ParsedIcdRow, ...]) -> tuple[ParsedIcdRow, ...]:
    grouped: dict[str, list[ParsedIcdRow]] = {}
    for row in rows:
        grouped.setdefault(code_norm.to_undotted(row.code_supplied), []).append(row)
    out: list[ParsedIcdRow] = []
    for _code, group in sorted(grouped.items()):
        group.sort(
            key=lambda row: (
                not bool(row.label_vi),
                "label_not_reconstructed" in row.flags,
                row.source_row,
                row.source_page,
            )
        )
        chosen = group[0]
        flags = list(chosen.flags)
        if len(group) > 1:
            flags.append("duplicate_code_rows_collapsed")
        out.append(
            ParsedIcdRow(
                code_supplied=chosen.code_supplied,
                label_vi=chosen.label_vi,
                label_en=chosen.label_en,
                aliases=chosen.aliases,
                chapter=chosen.chapter,
                block=chosen.block,
                parent=chosen.parent,
                source_page=chosen.source_page,
                source_row=chosen.source_row,
                flags=tuple(sorted(set(flags))),
            )
        )
    return tuple(out)
