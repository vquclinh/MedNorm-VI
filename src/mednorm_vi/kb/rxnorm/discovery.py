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
    """Find standard RRF files under ``directory`` (case-insensitive match)."""
    root = Path(directory)
    found: dict[str, Path] = {}
    if root.is_dir():
        by_name = {p.name.upper(): p for p in sorted(root.iterdir()) if p.is_file()}
        for key, fname in _KNOWN.items():
            hit = by_name.get(fname.upper())
            if hit is not None:
                found[key] = hit
    return RrfFileSet(conso=found.get("conso"), rel=found.get("rel"),
                      sat=found.get("sat"), sty=found.get("sty"))


__all__ = ["RrfFileSet", "discover_rrf"]
