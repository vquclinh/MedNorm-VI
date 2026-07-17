"""Local RxNorm snapshot contracts (Phase 1C-A).

Typed, offline model of a *locally acquired* RxNorm RRF snapshot. No network, no
real RxNorm content in the repo — only tiny synthetic RRF fixtures exercise it.
The exact organizer 2026 release/distribution is undisclosed (see the organizer
policy registry); a snapshot is NEVER treated as organizer-exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# RELA values we treat as "this RXCUI was replaced/remapped to another".
# RxNorm's exact retirement mechanism is release-specific; this set is a
# documented, configurable approximation for the local snapshot interface.
REMAP_RELAS: frozenset[str] = frozenset({"replaced_by", "remapped_to", "mapped_to"})
# Relationship RELAs that express drug structure (used by decoding hypotheses).
STRUCTURE_RELAS: frozenset[str] = frozenset({
    "has_ingredient", "ingredient_of", "has_precise_ingredient",
    "consists_of", "constitutes", "has_dose_form", "dose_form_of",
    "has_form", "form_of", "tradename_of", "has_tradename",
})


@dataclass(frozen=True, slots=True)
class RxnormAtom:
    """One RXNCONSO atom (a string for a concept in a source vocabulary)."""

    rxcui: str
    rxaui: str
    lat: str  # language (ENG, ...)
    tty: str  # term type (IN, PIN, SCD, SBD, BN, DF, ...)
    sab: str  # source vocabulary
    code: str
    string: str
    is_pref: bool = False
    suppress: str = "N"  # N | O | Y | E

    @property
    def normalized(self) -> str:
        return " ".join(self.string.lower().split())

    @property
    def is_suppressed(self) -> bool:
        return self.suppress not in ("N", "")


@dataclass(frozen=True, slots=True)
class RxnormRelation:
    """One RXNREL relationship between two concepts."""

    rxcui1: str
    rel: str  # RB/RN/RO/PAR/CHD/...
    rxcui2: str
    rela: str = ""  # more specific relationship
    sab: str = ""
    rui: str = ""
    suppress: str = "N"


@dataclass(frozen=True, slots=True)
class RxnormAttribute:
    """One RXNSAT attribute (e.g. RXN_STRENGTH, RXN_AVAILABLE_STRENGTH)."""

    rxcui: str
    atn: str
    atv: str
    sab: str = ""
    suppress: str = "N"


@dataclass(frozen=True, slots=True)
class RxnormSemanticType:
    """One RXNSTY semantic type."""

    rxcui: str
    tui: str
    sty: str


@dataclass(frozen=True, slots=True)
class RxnormSnapshot:
    """A local RxNorm snapshot: atoms + relations + attributes + semantic types.

    Index dicts are prebuilt by :func:`build_snapshot` for deterministic lookup.
    """

    snapshot_id: str
    release_version: str
    release_date: str
    source_dir: str
    atoms: tuple[RxnormAtom, ...]
    relations: tuple[RxnormRelation, ...] = field(default_factory=tuple)
    attributes: tuple[RxnormAttribute, ...] = field(default_factory=tuple)
    semantic_types: tuple[RxnormSemanticType, ...] = field(default_factory=tuple)
    _atoms_by_cui: dict[str, tuple[RxnormAtom, ...]] = field(default_factory=dict)
    _rels_by_cui: dict[str, tuple[RxnormRelation, ...]] = field(default_factory=dict)

    def rxcuis(self) -> tuple[str, ...]:
        return tuple(sorted(self._atoms_by_cui))

    def atoms_for(self, rxcui: str) -> tuple[RxnormAtom, ...]:
        return self._atoms_by_cui.get(rxcui, ())

    def ttys(self, rxcui: str) -> tuple[str, ...]:
        return tuple(sorted({a.tty for a in self.atoms_for(rxcui)}))

    def preferred_atom(self, rxcui: str) -> RxnormAtom | None:
        atoms = self.atoms_for(rxcui)
        if not atoms:
            return None
        # deterministic: prefer non-suppressed, is_pref, then (tty, rxaui)
        return sorted(atoms, key=lambda a: (a.is_suppressed, not a.is_pref, a.tty, a.rxaui))[0]

    def preferred_string(self, rxcui: str) -> str | None:
        atom = self.preferred_atom(rxcui)
        return atom.string if atom is not None else None

    def is_suppressed(self, rxcui: str) -> bool:
        """True when every atom of the concept is suppressed (or it is unknown)."""
        atoms = self.atoms_for(rxcui)
        return bool(atoms) and all(a.is_suppressed for a in atoms)

    def relations_from(self, rxcui: str) -> tuple[RxnormRelation, ...]:
        return self._rels_by_cui.get(rxcui, ())

    def remap_targets(self, rxcui: str) -> tuple[str, ...]:
        """RXCUIs this concept was replaced/remapped to (deterministic order)."""
        out = {r.rxcui2 for r in self.relations_from(rxcui) if r.rela in REMAP_RELAS}
        return tuple(sorted(out))


__all__ = [
    "REMAP_RELAS",
    "STRUCTURE_RELAS",
    "RxnormAtom",
    "RxnormRelation",
    "RxnormAttribute",
    "RxnormSemanticType",
    "RxnormSnapshot",
]
