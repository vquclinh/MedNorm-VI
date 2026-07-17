"""Build a deterministic local RxNorm snapshot from RRF files (offline)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .discovery import discover_rrf
from .models import (
    RxnormAtom,
    RxnormAttribute,
    RxnormRelation,
    RxnormSemanticType,
    RxnormSnapshot,
)
from .rrf_reader import read_conso, read_rel, read_sat, read_sty


def _index_atoms(atoms: tuple[RxnormAtom, ...]) -> dict[str, tuple[RxnormAtom, ...]]:
    out: dict[str, list[RxnormAtom]] = {}
    for a in atoms:
        out.setdefault(a.rxcui, []).append(a)
    return {k: tuple(v) for k, v in out.items()}


def _index_rels(rels: tuple[RxnormRelation, ...]) -> dict[str, tuple[RxnormRelation, ...]]:
    out: dict[str, list[RxnormRelation]] = {}
    for r in rels:
        out.setdefault(r.rxcui1, []).append(r)
    return {k: tuple(v) for k, v in out.items()}


def build_snapshot(
    directory: str | Path, *, snapshot_id: str = "", release_version: str = "",
    release_date: str = "",
) -> RxnormSnapshot:
    """Ingest local RRF files under ``directory`` into an :class:`RxnormSnapshot`."""
    files = discover_rrf(directory)
    atoms = tuple(read_conso(files.conso)) if files.conso else ()
    rels = tuple(read_rel(files.rel)) if files.rel else ()
    sats = tuple(read_sat(files.sat)) if files.sat else ()
    stys = tuple(read_sty(files.sty)) if files.sty else ()

    sid = snapshot_id or _derive_snapshot_id(atoms, rels, sats, stys)
    return RxnormSnapshot(
        snapshot_id=sid, release_version=release_version, release_date=release_date,
        source_dir=str(directory), atoms=atoms, relations=rels, attributes=sats,
        semantic_types=stys, _atoms_by_cui=_index_atoms(atoms), _rels_by_cui=_index_rels(rels))


def _derive_snapshot_id(
    atoms: tuple[RxnormAtom, ...], rels: tuple[RxnormRelation, ...],
    sats: tuple[RxnormAttribute, ...], stys: tuple[RxnormSemanticType, ...],
) -> str:
    """Content-derived id so two identical snapshots hash identically."""
    h = hashlib.sha256()
    for a in atoms:
        h.update(f"C|{a.rxcui}|{a.rxaui}|{a.tty}|{a.sab}|{a.string}|{a.suppress}\n".encode())
    for r in rels:
        h.update(f"R|{r.rxcui1}|{r.rel}|{r.rela}|{r.rxcui2}\n".encode())
    for s in sats:
        h.update(f"A|{s.rxcui}|{s.atn}|{s.atv}\n".encode())
    for st in stys:
        h.update(f"S|{st.rxcui}|{st.tui}\n".encode())
    return f"rxnorm-local-{h.hexdigest()[:16]}"


__all__ = ["build_snapshot"]
