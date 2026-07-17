"""Build a deterministic local ICD-10 snapshot from a configurable table."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import normalization as norm
from .loaders import ColumnMap, RawIcdRow, load_rows
from .models import IcdConcept, IcdSnapshot


def _derive_snapshot_id(rows: list[RawIcdRow], source: str, version: str) -> str:
    h = hashlib.sha256(f"{source}|{version}\n".encode())
    for r in sorted(rows, key=lambda x: norm.to_undotted(x.code)):
        h.update(f"{norm.to_undotted(r.code)}|{r.label_vi}\n".encode())
    return f"icd10-vi-local-{h.hexdigest()[:16]}"


def build_snapshot_from_rows(
    rows: list[RawIcdRow], *, source: str, version: str, snapshot_id: str = "",
) -> IcdSnapshot:
    """Assemble concepts, inferring dotted/undotted, parents, children, chapters."""
    # First pass: build concept skeletons keyed by undotted code.
    by_undotted: dict[str, RawIcdRow] = {}
    for r in rows:
        by_undotted[norm.to_undotted(r.code)] = r

    children: dict[str, list[str]] = {}
    for u in by_undotted:
        parent = norm.parent_code(u)
        # walk up to the nearest existing ancestor
        while parent is not None and parent not in by_undotted:
            parent = norm.parent_code(parent)
        if parent is not None:
            children.setdefault(parent, []).append(u)

    concepts: list[IcdConcept] = []
    for u, r in by_undotted.items():
        declared_parent = norm.to_undotted(r.parent) if r.parent else None
        parent = declared_parent if (declared_parent and declared_parent in by_undotted) else None
        if parent is None:
            p = norm.parent_code(u)
            while p is not None and p not in by_undotted:
                p = norm.parent_code(p)
            parent = p
        concepts.append(IcdConcept(
            code_supplied=r.code, dotted=norm.to_dotted(r.code), undotted=u,
            label_vi=r.label_vi, label_en=r.label_en, aliases=r.aliases, parent=parent,
            children=tuple(sorted(children.get(u, []))),
            chapter=r.chapter or norm.chapter_letter(u), source=source, version=version,
            status=(r.status or "active")))
    concepts.sort(key=lambda c: c.undotted)
    sid = snapshot_id or _derive_snapshot_id(rows, source, version)
    return IcdSnapshot(
        snapshot_id=sid, source=source, version=version, concepts=tuple(concepts),
        _by_undotted={c.undotted: c for c in concepts})


def build_snapshot(
    path: str | Path, column_map: ColumnMap, *, source: str, version: str,
    snapshot_id: str = "",
) -> IcdSnapshot:
    """Load a local ICD-10 table and build a snapshot."""
    rows = load_rows(path, column_map)
    return build_snapshot_from_rows(rows, source=source, version=version, snapshot_id=snapshot_id)


__all__ = ["build_snapshot", "build_snapshot_from_rows"]
