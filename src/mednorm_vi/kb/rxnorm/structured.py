"""Structured RxNorm attributes recovered from the governed local source (Audit 0074).

Audit 0071 found the competition RxNorm index carries only `tty`, `tty_values`, `membership`,
`runtime_role` and `suppress` - no ingredient, strength, dose form or brand - so the semantic
document for a drug was little more than its name. A retriever cannot tell *amoxicillin 500 mg
capsule* from *amoxicillin 250 mg tablet* when neither strength nor form is in the text it
embeds.

Those fields are not missing from the source, only from the index. The governed local RxNorm
release carries them in RXNSAT and RXNREL, and this module recovers them **with provenance**:
every field records which RRF file, which attribute name or relation, and which release it
came from. Nothing is inferred. A drug with no recorded strength gets no strength, not a
guessed one.

Normalization is deterministic and conservative. `500 MG` and `500 mg` are the same strength;
`500 MG` and `0.5 G` are **not** unified, because unit conversion is a clinical judgement the
source does not make for us and a wrong equivalence would create false exact matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STRUCTURED_VERSION = "rxnorm-structured-v1"

#: RXNSAT attribute names (SAB=RXNORM) recovered for governed concepts, with the governed
#: coverage measured in Audit 0074 §3 recorded beside each.
ATN_AVAILABLE_STRENGTH = "RXN_AVAILABLE_STRENGTH"  # 52,318 governed
ATN_STRENGTH = "RXN_STRENGTH"  # 23,418
ATN_RXTERM_FORM = "RXTERM_FORM"  # 49,187
ATN_BOSS_NUM_VALUE = "RXN_BOSS_STRENGTH_NUM_VALUE"  # 14,920
ATN_BOSS_NUM_UNIT = "RXN_BOSS_STRENGTH_NUM_UNIT"
ATN_BOSS_DENOM_VALUE = "RXN_BOSS_STRENGTH_DENOM_VALUE"
ATN_BOSS_DENOM_UNIT = "RXN_BOSS_STRENGTH_DENOM_UNIT"
ATN_QUANTITY = "RXN_QUANTITY"  # 7,066
ATN_HUMAN_DRUG = "RXN_HUMAN_DRUG"  # 23,879

RECOVERED_ATTRIBUTES: tuple[str, ...] = (
    ATN_AVAILABLE_STRENGTH,
    ATN_STRENGTH,
    ATN_RXTERM_FORM,
    ATN_BOSS_NUM_VALUE,
    ATN_BOSS_NUM_UNIT,
    ATN_BOSS_DENOM_VALUE,
    ATN_BOSS_DENOM_UNIT,
    ATN_QUANTITY,
    ATN_HUMAN_DRUG,
)

#: RXNREL relations (SAB=RXNORM) with both endpoints inside the governed KB.
REL_HAS_INGREDIENT = "has_ingredient"  # 134,452 governed pairs
REL_HAS_PRECISE_INGREDIENT = "has_precise_ingredient"  # 11,706
REL_HAS_DOSE_FORM = "has_dose_form"  # 95,730
REL_TRADENAME_OF = "tradename_of"  # 104,677
REL_CONSISTS_OF = "consists_of"  # 104,454
REL_ISA = "isa"  # 206,815

RECOVERED_RELATIONS: tuple[str, ...] = (
    REL_HAS_INGREDIENT,
    REL_HAS_PRECISE_INGREDIENT,
    REL_HAS_DOSE_FORM,
    REL_TRADENAME_OF,
    REL_CONSISTS_OF,
    REL_ISA,
)

#: `500 MG`, `0.5 MG/ML`, `12.5 MG / 5 ML`. Captures value and unit without converting either.
_STRENGTH = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>[A-Za-z%µ]+(?:/[A-Za-z0-9.%µ]+)?)")
_SPACE = re.compile(r"\s+")


def normalize_unit(unit: str) -> str:
    """Case-fold and compact a unit. Never converts between units."""
    return _SPACE.sub("", (unit or "").strip().casefold())


def normalize_value(value: str) -> str:
    """Canonical numeric text: `0.50` and `.5` both become `0.5`; non-numerics pass through."""
    text = (value or "").strip().replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return text.casefold()
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


@dataclass(frozen=True, slots=True)
class Strength:
    """One parsed strength component. `raw` is always kept so nothing is lost."""

    value: str
    unit: str
    raw: str = ""

    @property
    def key(self) -> str:
        return f"{normalize_value(self.value)}{normalize_unit(self.unit)}"

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "raw": self.raw, "key": self.key}


def parse_strengths(text: str) -> tuple[Strength, ...]:
    """Every `<value> <unit>` pair in a strength string, in source order."""
    out: list[Strength] = []
    for match in _STRENGTH.finditer(text or ""):
        out.append(
            Strength(
                value=normalize_value(match.group("value")),
                unit=normalize_unit(match.group("unit")),
                raw=match.group(0).strip(),
            )
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one recovered field came from. Recorded per concept, never inferred."""

    release: str
    rrf_file: str
    field_name: str

    def as_dict(self) -> dict[str, Any]:
        return {"release": self.release, "rrf_file": self.rrf_file, "field": self.field_name}


@dataclass(frozen=True, slots=True)
class StructuredDrug:
    """Recovered structure for one governed RxCUI. Absent fields stay absent."""

    rxcui: str
    tty: str = ""
    name: str = ""
    ingredients: tuple[str, ...] = field(default_factory=tuple)
    precise_ingredients: tuple[str, ...] = field(default_factory=tuple)
    dose_forms: tuple[str, ...] = field(default_factory=tuple)
    brands: tuple[str, ...] = field(default_factory=tuple)
    strengths: tuple[Strength, ...] = field(default_factory=tuple)
    available_strength: str = ""
    rxterm_form: str = ""
    quantity: str = ""
    human_drug: bool = False
    provenance: tuple[Provenance, ...] = field(default_factory=tuple)

    @property
    def strength_keys(self) -> frozenset[str]:
        return frozenset(s.key for s in self.strengths)

    @property
    def has_structure(self) -> bool:
        return bool(
            self.ingredients or self.dose_forms or self.strengths or self.brands or self.rxterm_form
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rxcui": self.rxcui,
            "tty": self.tty,
            "name": self.name,
            "ingredients": list(self.ingredients),
            "precise_ingredients": list(self.precise_ingredients),
            "dose_forms": list(self.dose_forms),
            "brands": list(self.brands),
            "strengths": [s.as_dict() for s in self.strengths],
            "available_strength": self.available_strength,
            "rxterm_form": self.rxterm_form,
            "quantity": self.quantity,
            "human_drug": self.human_drug,
            "provenance": [p.as_dict() for p in self.provenance],
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> StructuredDrug:
        return StructuredDrug(
            rxcui=str(row.get("rxcui", "")),
            tty=str(row.get("tty", "")),
            name=str(row.get("name", "")),
            ingredients=tuple(row.get("ingredients") or ()),
            precise_ingredients=tuple(row.get("precise_ingredients") or ()),
            dose_forms=tuple(row.get("dose_forms") or ()),
            brands=tuple(row.get("brands") or ()),
            strengths=tuple(
                Strength(value=s.get("value", ""), unit=s.get("unit", ""), raw=s.get("raw", ""))
                for s in (row.get("strengths") or ())
            ),
            available_strength=str(row.get("available_strength", "")),
            rxterm_form=str(row.get("rxterm_form", "")),
            quantity=str(row.get("quantity", "")),
            human_drug=bool(row.get("human_drug", False)),
            provenance=tuple(
                Provenance(
                    release=p.get("release", ""),
                    rrf_file=p.get("rrf_file", ""),
                    field_name=p.get("field", ""),
                )
                for p in (row.get("provenance") or ())
            ),
        )


def semantic_text(drug: StructuredDrug, *, alias_limit: int = 6) -> str:
    """Bounded structured document text for dense retrieval and reranking.

    Field order is fixed so the string is stable, and each field is labelled so the encoder
    sees *what* a token is rather than a bag of words. Only recovered fields appear.
    """
    parts = [f"RxCUI {drug.rxcui}"]
    if drug.name:
        parts.append(drug.name)
    if drug.tty:
        parts.append(f"TTY: {drug.tty}")
    if drug.ingredients:
        parts.append("ingredient: " + "; ".join(drug.ingredients[:alias_limit]))
    if drug.precise_ingredients:
        parts.append("precise ingredient: " + "; ".join(drug.precise_ingredients[:alias_limit]))
    if drug.strengths:
        parts.append("strength: " + "; ".join(s.raw or s.key for s in drug.strengths[:4]))
    elif drug.available_strength:
        parts.append(f"strength: {drug.available_strength}")
    if drug.dose_forms:
        parts.append("dose form: " + "; ".join(drug.dose_forms[:alias_limit]))
    elif drug.rxterm_form:
        parts.append(f"dose form: {drug.rxterm_form}")
    if drug.brands:
        parts.append("brand: " + "; ".join(drug.brands[:alias_limit]))
    if drug.quantity:
        parts.append(f"quantity: {drug.quantity}")
    return " | ".join(parts)


__all__ = [
    "ATN_AVAILABLE_STRENGTH",
    "ATN_BOSS_DENOM_UNIT",
    "ATN_BOSS_DENOM_VALUE",
    "ATN_BOSS_NUM_UNIT",
    "ATN_BOSS_NUM_VALUE",
    "ATN_HUMAN_DRUG",
    "ATN_QUANTITY",
    "ATN_RXTERM_FORM",
    "ATN_STRENGTH",
    "RECOVERED_ATTRIBUTES",
    "RECOVERED_RELATIONS",
    "REL_CONSISTS_OF",
    "REL_HAS_DOSE_FORM",
    "REL_HAS_INGREDIENT",
    "REL_HAS_PRECISE_INGREDIENT",
    "REL_ISA",
    "REL_TRADENAME_OF",
    "STRUCTURED_VERSION",
    "Provenance",
    "Strength",
    "StructuredDrug",
    "normalize_unit",
    "normalize_value",
    "parse_strengths",
    "semantic_text",
]
