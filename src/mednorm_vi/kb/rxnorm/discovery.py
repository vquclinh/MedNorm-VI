"""Locate local RxNorm RRF files in a snapshot directory (offline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_KNOWN = {
    "conso": "RXNCONSO.RRF",
    "rel": "RXNREL.RRF",
    "sat": "RXNSAT.RRF",
    "sty": "RXNSTY.RRF",
}


@dataclass(frozen=True, slots=True)
class RrfFileSet:
    """Discovered RRF files (missing ones are ``None``)."""

    conso: Path | None
    rel: Path | None
    sat: Path | None
    sty: Path | None
    root: Path | None = None

    @property
    def has_conso(self) -> bool:
        return self.conso is not None

    def missing(self) -> tuple[str, ...]:
        out = []
        for name in ("conso", "rel", "sat", "sty"):
            if getattr(self, name) is None:
                out.append(_KNOWN[name])
        return tuple(out)


def discover_rrf(directory: str | Path) -> RrfFileSet:
    """Find standard RRF files under ``directory`` (case-insensitive match).

    Direct children are preferred for backward compatibility with synthetic
    fixtures. If no direct RXNCONSO is present, search nested directories and
    choose the directory with RXNCONSO and the most known RRF companions.
    """
    root = Path(directory)
    if not root.is_dir():
        return RrfFileSet(conso=None, rel=None, sat=None, sty=None, root=None)

    def from_mapping(parent: Path, by_name: dict[str, Path]) -> RrfFileSet:
        found: dict[str, Path] = {}
        for key, fname in _KNOWN.items():
            hit = by_name.get(fname.upper())
            if hit is not None:
                found[key] = hit
        return RrfFileSet(conso=found.get("conso"), rel=found.get("rel"),
                          sat=found.get("sat"), sty=found.get("sty"), root=parent)

    direct = from_mapping(root, {p.name.upper(): p for p in sorted(root.iterdir()) if p.is_file()})
    if direct.has_conso:
        return direct

    candidates: dict[Path, dict[str, Path]] = {}
    known_names = {fname.upper(): key for key, fname in _KNOWN.items()}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        key = known_names.get(p.name.upper())
        if key is not None:
            candidates.setdefault(p.parent, {})[p.name.upper()] = p
    ranked = sorted(
        (from_mapping(parent, by_name) for parent, by_name in candidates.items()),
        key=lambda files: (
            not files.has_conso,
            -sum(x is not None for x in (files.conso, files.rel, files.sat, files.sty)),
            str(files.root or ""),
        ),
    )
    return ranked[0] if ranked else RrfFileSet(conso=None, rel=None, sat=None, sty=None, root=None)


__all__ = ["RrfFileSet", "discover_rrf"]
