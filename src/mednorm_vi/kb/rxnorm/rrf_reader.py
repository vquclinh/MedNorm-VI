"""Deterministic RRF (pipe-delimited) readers for a local RxNorm snapshot.

RRF rows are ``|``-delimited with a trailing ``|``. Column orders follow the
standard RxNorm files. These readers are pure and offline; they parse whatever
local RRF the caller points at (tiny synthetic fixtures in tests).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .models import RxnormAtom, RxnormAttribute, RxnormRelation, RxnormSemanticType

# Column indices (0-based) for the columns we consume.
_CONSO = {"RXCUI": 0, "LAT": 1, "TS": 2, "ISPREF": 6, "RXAUI": 7,
          "SAB": 11, "TTY": 12, "CODE": 13, "STR": 14, "SUPPRESS": 16}
_REL = {"RXCUI1": 0, "REL": 3, "RXCUI2": 4, "RELA": 7, "RUI": 8, "SAB": 10, "SUPPRESS": 14}
_SAT = {"RXCUI": 0, "ATN": 8, "ATV": 9, "SAB": 10, "SUPPRESS": 11}
_STY = {"RXCUI": 0, "TUI": 1, "STY": 3}


def _rows(path: Path) -> Iterator[list[str]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        cols = raw.split("|")
        if cols and cols[-1] == "":  # trailing delimiter
            cols = cols[:-1]
        yield cols


def _get(cols: list[str], idx: int) -> str:
    return cols[idx] if idx < len(cols) else ""


def read_conso(path: str | Path) -> list[RxnormAtom]:
    out: list[RxnormAtom] = []
    for c in _rows(Path(path)):
        out.append(RxnormAtom(
            rxcui=_get(c, _CONSO["RXCUI"]), rxaui=_get(c, _CONSO["RXAUI"]),
            lat=_get(c, _CONSO["LAT"]), tty=_get(c, _CONSO["TTY"]),
            sab=_get(c, _CONSO["SAB"]), code=_get(c, _CONSO["CODE"]),
            string=_get(c, _CONSO["STR"]),
            is_pref=_get(c, _CONSO["ISPREF"]) == "Y",
            suppress=_get(c, _CONSO["SUPPRESS"]) or "N"))
    return out


def read_rel(path: str | Path) -> list[RxnormRelation]:
    out: list[RxnormRelation] = []
    for c in _rows(Path(path)):
        out.append(RxnormRelation(
            rxcui1=_get(c, _REL["RXCUI1"]), rel=_get(c, _REL["REL"]),
            rxcui2=_get(c, _REL["RXCUI2"]), rela=_get(c, _REL["RELA"]),
            sab=_get(c, _REL["SAB"]), rui=_get(c, _REL["RUI"]),
            suppress=_get(c, _REL["SUPPRESS"]) or "N"))
    return out


def read_sat(path: str | Path) -> list[RxnormAttribute]:
    out: list[RxnormAttribute] = []
    for c in _rows(Path(path)):
        out.append(RxnormAttribute(
            rxcui=_get(c, _SAT["RXCUI"]), atn=_get(c, _SAT["ATN"]),
            atv=_get(c, _SAT["ATV"]), sab=_get(c, _SAT["SAB"]),
            suppress=_get(c, _SAT["SUPPRESS"]) or "N"))
    return out


def read_sty(path: str | Path) -> list[RxnormSemanticType]:
    out: list[RxnormSemanticType] = []
    for c in _rows(Path(path)):
        out.append(RxnormSemanticType(
            rxcui=_get(c, _STY["RXCUI"]), tui=_get(c, _STY["TUI"]),
            sty=_get(c, _STY["STY"])))
    return out


__all__ = ["read_conso", "read_rel", "read_sat", "read_sty"]
