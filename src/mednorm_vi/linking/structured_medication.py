"""Structured medication representation for RxNorm linking (spec §10.1).

Spec §10.1 asks the RxNorm linker to work from a structured drug description
``D = {ingredient, strength, dose_form, route, ...}`` rather than from a raw surface
string. Audit 0052 preserved exactly that evidence — E1's ``ComponentSpan`` records —
all the way through the lattice and L4 into ``EntityHypothesis.components``, and Audit
0053 recorded the fact that the linker still ignored it. This module is the missing
adapter: it turns those components into one typed representation the linker can reason
over.

Three properties matter more than completeness here.

**Nothing is invented.** Every field is either taken from a ComponentSpan that E1
actually emitted, or it is absent. There is no dictionary of default routes, no
"probably a tablet", no inferred brand. A field with no evidence lands in
``unresolved_fields`` and stays out of every compatibility decision.

**Missing evidence is not negative evidence.** "This mention does not state a dose
form" and "this mention states a dose form that conflicts with the candidate" are
different facts with different consequences, and conflating them is how a linker
starts rejecting correct candidates. ``StructuredField`` is present-or-absent;
conflicts are computed later, by the linker, against a candidate.

**Normalization is conservative and never destroys the surface.** ``500mg`` and
``500 MG`` normalize to the same ``(500.0, "mg")`` pair for comparison, but the
original text and its absolute offsets survive on the field, because L9 re-verifies
offsets and a human reading an audit needs the words that were actually written.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..resolution.models import EntityHypothesis

STRUCTURED_MEDICATION_VERSION = "structured-medication-v1"

# Roles E1's grammar emits (verified against the tracked medication fixture, not
# assumed from the spec). Anything outside this set is carried as an extra role
# rather than dropped, so a grammar extension does not silently lose evidence.
ROLE_NAME = "name"
ROLE_SALT = "salt"
ROLE_STRENGTH_VALUE = "strength_value"
ROLE_STRENGTH_UNIT = "strength_unit"
ROLE_CONCENTRATION = "concentration"
ROLE_DOSE_FORM = "dose_form"
ROLE_RELEASE = "release"
ROLE_BRAND = "brand"
ROLE_ROUTE = "route"
ROLE_FREQUENCY = "frequency"
ROLE_PRN = "prn"
ROLE_DURATION = "duration"

STRUCTURED_ROLES: tuple[str, ...] = (
    ROLE_NAME,
    ROLE_SALT,
    ROLE_STRENGTH_VALUE,
    ROLE_STRENGTH_UNIT,
    ROLE_CONCENTRATION,
    ROLE_DOSE_FORM,
    ROLE_RELEASE,
    ROLE_BRAND,
    ROLE_ROUTE,
    ROLE_FREQUENCY,
    ROLE_PRN,
    ROLE_DURATION,
)

# Conservative unit normalization. Only forms actually seen in Vietnamese clinical
# text and in the locked RxNorm snapshot are mapped; an unknown unit is kept verbatim
# and marked unnormalized rather than guessed into a neighbour.
_UNIT_CANONICAL: Mapping[str, str] = {
    "mg": "mg",
    "milligram": "mg",
    "miligram": "mg",
    "mgs": "mg",
    "g": "g",
    "gam": "g",
    "gram": "g",
    "gr": "g",
    "mcg": "mcg",
    "ug": "mcg",
    "µg": "mcg",
    "μg": "mcg",
    "microgram": "mcg",
    "ml": "ml",
    "millilitre": "ml",
    "milliliter": "ml",
    "l": "l",
    "lit": "l",
    "litre": "l",
    "ui": "unit",
    "iu": "unit",
    "u": "unit",
    "unit": "unit",
    "units": "unit",
    "meq": "meq",
    "mmol": "mmol",
    "mol": "mol",
    "%": "%",
}

# Unit families that may never be compared against each other. Comparing a mass to a
# volume is not a near miss, it is a category error, and the linker must say so.
_UNIT_FAMILY: Mapping[str, str] = {
    "mg": "mass",
    "g": "mass",
    "mcg": "mass",
    "ml": "volume",
    "l": "volume",
    "unit": "activity",
    "meq": "amount",
    "mmol": "amount",
    "mol": "amount",
    "%": "ratio",
}

# Mass conversion to milligrams, for comparing `1 g` against `1000 mg`. Only exact
# decimal factors, so no floating-point tolerance games.
_TO_MG: Mapping[str, float] = {"mg": 1.0, "g": 1000.0, "mcg": 0.001}


# --- Legacy INN -> RxNorm-name bridge inventory ---------------------------------
#
# A governed-data gap found by measurement, not by reading the spec: RxNorm is a US
# vocabulary and names ingredients by USAN, while Vietnamese clinical text uses the
# INN/WHO name. `paracetamol` occurs in **zero** records of the locked prescribable
# snapshot — not as a canonical name and not as an alias — so a mention of
# `paracetamol 500mg` is unlinkable through name matching alone. Without this bridge
# the old linker answered such mentions with trigram noise (`ALCOHOL 0.7 L in 1 L
# TOPICAL GEL` scored highest for "paracetamol"), which is worse than answering
# nothing.
#
# Milestone 3A freezes the safer contract: unreviewed lexical bridges are retained as
# an inventory/review queue input, but they are not runtime-authoritative and must not
# widen candidate retrieval. Only APPROVED rows from a governed crosswalk may do that.
INN_BRIDGE_STATUS = "REQUIRES_CLINICAL_REVIEW"
LEGACY_INN_BRIDGE_RUNTIME_ENABLED = False

INN_TO_RXNORM_NAME: Mapping[str, str] = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "lignocaine": "lidocaine",
    "frusemide": "furosemide",
    "glibenclamide": "glyburide",
    "amoxycillin": "amoxicillin",
    "cephalexin": "cefalexin",
    "rifampicin": "rifampin",
    "trimethoprim-sulfamethoxazole": "sulfamethoxazole / trimethoprim",
    "co-trimoxazole": "sulfamethoxazole / trimethoprim",
}


def bridged_ingredient_names(surface: str) -> tuple[str, ...]:
    """Approved RxNorm-side names to also retrieve on, for an INN surface form.

    The legacy bridge above is currently ``REQUIRES_CLINICAL_REVIEW`` inventory.
    Until an approved governed crosswalk is wired in, it must not widen runtime
    retrieval or silently turn an unreviewed mapping into an authoritative candidate.
    """
    if not LEGACY_INN_BRIDGE_RUNTIME_ENABLED:
        return ()
    key = normalize_surface(surface)
    target = INN_TO_RXNORM_NAME.get(key)
    return (target,) if target else ()


def normalize_surface(text: str) -> str:
    """Casefolded, accent-stripped, whitespace-collapsed comparison key."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(unicodedata.normalize("NFKC", stripped).casefold().split())


def canonical_unit(raw: str) -> str | None:
    """Canonical unit token, or ``None`` when the unit is not recognised."""
    key = normalize_surface(raw).replace(".", "")
    return _UNIT_CANONICAL.get(key)


def unit_family(unit: str | None) -> str | None:
    return _UNIT_FAMILY.get(unit) if unit else None


def strength_in_mg(value: float, unit: str | None) -> float | None:
    """Mass strength expressed in mg, or ``None`` when not a mass."""
    factor = _TO_MG.get(unit or "")
    return None if factor is None else value * factor


def _parse_number(raw: str) -> float | None:
    """Parse a clinical numeral. `0,5` is European decimal notation, not a list."""
    text = normalize_surface(raw).replace(" ", "")
    if not text:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class StructuredField:
    """One piece of structured evidence, with the surface it came from.

    Presence is meaningful: a ``StructuredField`` exists only when E1 emitted a
    component for it. There is no "empty" instance, so ``None`` unambiguously means
    "the mention does not state this".
    """

    role: str
    text: str
    start: int
    end: int
    normalized: str
    # Set only for the numeric roles; ``None`` on textual roles by construction.
    value: float | None = None
    unit: str | None = None
    # True when the raw form was recognised by the conservative normalizer. A false
    # value is not an error — it means comparisons on this field must be textual.
    unit_recognised: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "normalized": self.normalized,
            "value": self.value,
            "unit": self.unit,
            "unit_recognised": self.unit_recognised,
        }


@dataclass(frozen=True, slots=True)
class StructuredMedicationMention:
    """Spec §10.1's structured drug description, built only from E1 evidence."""

    hypothesis_id: str
    document_id: str
    start: int
    end: int
    text: str
    ingredient: StructuredField | None = None
    salt: StructuredField | None = None
    strength_value: StructuredField | None = None
    strength_unit: StructuredField | None = None
    concentration: StructuredField | None = None
    dose_form: StructuredField | None = None
    release: StructuredField | None = None
    brand: StructuredField | None = None
    route: StructuredField | None = None
    frequency: StructuredField | None = None
    prn: StructuredField | None = None
    duration: StructuredField | None = None
    # Roles with no evidence in this mention. Explicit, because "absent" must be
    # readable from the object rather than inferred from a pile of Nones.
    unresolved_fields: tuple[str, ...] = field(default_factory=tuple)
    # Structured evidence that is present but internally incoherent — e.g. a strength
    # unit E1 found but the normalizer does not recognise. Recorded, never repaired.
    incoherent_fields: tuple[str, ...] = field(default_factory=tuple)
    provenance: tuple[str, ...] = field(default_factory=tuple)
    version: str = STRUCTURED_MEDICATION_VERSION

    @property
    def has_ingredient_evidence(self) -> bool:
        return self.ingredient is not None

    @property
    def strength_mg(self) -> float | None:
        """Strength in mg when both a value and a mass unit are stated."""
        if self.strength_value is None or self.strength_value.value is None:
            return None
        unit = self.strength_unit.unit if self.strength_unit else None
        return strength_in_mg(self.strength_value.value, unit)

    def query_terms(self) -> tuple[str, ...]:
        """Retrieval keys, most specific first, all drawn from stated evidence.

        The ingredient name alone is included deliberately: retrieving on the full
        mention surface (`amlodipine besylate 10 mg po daily`) is what limited the old
        linker to n-gram noise, because no RxNorm concept is named that way.
        """
        terms: list[str] = []
        if self.brand is not None:
            terms.append(self.brand.text)
        if self.ingredient is not None:
            terms.append(self.ingredient.text)
            terms.extend(bridged_ingredient_names(self.ingredient.text))
            if self.salt is not None:
                terms.append(f"{self.ingredient.text} {self.salt.text}")
        terms.append(self.text)
        seen: set[str] = set()
        ordered: list[str] = []
        for term in terms:
            key = normalize_surface(term)
            if key and key not in seen:
                seen.add(key)
                ordered.append(term)
        return tuple(ordered)

    def as_dict(self) -> dict[str, Any]:
        """Serialization for audits and the run manifest. Carries mention text.

        This is the one structure in the L5 chain that legitimately holds mention
        text, because it *is* the mention. The run manifest must therefore summarise
        it, never embed it — see ``manifest`` for the aggregate-only contract.
        """
        payload: dict[str, Any] = {
            "structured_medication_version": self.version,
            "hypothesis_id": self.hypothesis_id,
            "document_id": self.document_id,
            "position": [self.start, self.end],
            "unresolved_fields": list(self.unresolved_fields),
            "incoherent_fields": list(self.incoherent_fields),
            "provenance": list(self.provenance),
        }
        for role in STRUCTURED_ROLES:
            attr = "ingredient" if role == ROLE_NAME else role
            value = getattr(self, attr, None)
            payload[attr] = value.as_dict() if isinstance(value, StructuredField) else None
        return payload


def _field_from_component(role: str, component: Mapping[str, Any]) -> StructuredField:
    text = str(component.get("text", ""))
    normalized = str(component.get("normalized") or "") or normalize_surface(text)
    value: float | None = None
    unit: str | None = None
    recognised = True
    if role == ROLE_STRENGTH_VALUE:
        value = _parse_number(text)
        recognised = value is not None
    elif role == ROLE_STRENGTH_UNIT:
        unit = canonical_unit(text)
        recognised = unit is not None
    return StructuredField(
        role=role,
        text=text,
        start=int(component.get("start", 0)),
        end=int(component.get("end", 0)),
        normalized=normalize_surface(normalized),
        value=value,
        unit=unit,
        unit_recognised=recognised,
    )


def build_structured_mention(
    hypothesis: EntityHypothesis,
) -> StructuredMedicationMention:
    """Build spec §10.1's structured representation from E1's preserved components.

    Only the *first* component per role is taken. A second component for the same
    role is real evidence of a mention this grammar did not fully decompose (a
    combination product, say), so it is recorded in ``incoherent_fields`` rather than
    silently overwriting or silently merging.
    """
    grouped = hypothesis.components_by_role()
    fields: dict[str, StructuredField] = {}
    incoherent: list[str] = []
    provenance: list[str] = [f"hypothesis:{hypothesis.hypothesis_id}"]
    provenance.extend(f"proposal:{pid}" for pid in hypothesis.source_proposal_ids)
    provenance.extend(f"expert:{eid}" for eid in hypothesis.expert_ids)

    for role in STRUCTURED_ROLES:
        components = grouped.get(role, ())
        if not components:
            continue
        built = _field_from_component(role, components[0])
        fields[role] = built
        if len(components) > 1:
            incoherent.append(f"{role}:multiple_components:{len(components)}")
        if not built.unit_recognised:
            incoherent.append(f"{role}:unnormalized_surface")

    # A strength value without a unit, or a unit without a value, is incomplete
    # structured evidence. Both halves are needed for any strength comparison, and
    # pretending otherwise is how a linker matches `500` to `500 mL`.
    if (ROLE_STRENGTH_VALUE in fields) != (ROLE_STRENGTH_UNIT in fields):
        incoherent.append("strength:value_unit_pair_incomplete")

    unresolved = tuple(role for role in STRUCTURED_ROLES if role not in fields)
    return StructuredMedicationMention(
        hypothesis_id=hypothesis.hypothesis_id,
        document_id=hypothesis.document_id,
        start=hypothesis.start,
        end=hypothesis.end,
        text=hypothesis.text,
        ingredient=fields.get(ROLE_NAME),
        salt=fields.get(ROLE_SALT),
        strength_value=fields.get(ROLE_STRENGTH_VALUE),
        strength_unit=fields.get(ROLE_STRENGTH_UNIT),
        concentration=fields.get(ROLE_CONCENTRATION),
        dose_form=fields.get(ROLE_DOSE_FORM),
        release=fields.get(ROLE_RELEASE),
        brand=fields.get(ROLE_BRAND),
        route=fields.get(ROLE_ROUTE),
        frequency=fields.get(ROLE_FREQUENCY),
        prn=fields.get(ROLE_PRN),
        duration=fields.get(ROLE_DURATION),
        unresolved_fields=unresolved,
        incoherent_fields=tuple(incoherent),
        provenance=tuple(provenance),
    )


def build_structured_mentions(
    hypotheses: Sequence[EntityHypothesis],
) -> tuple[StructuredMedicationMention, ...]:
    """Structured representations for every medication hypothesis, in input order."""
    return tuple(build_structured_mention(h) for h in hypotheses if h.entity_type == "THUỐC")


__all__ = [
    "ROLE_BRAND",
    "ROLE_CONCENTRATION",
    "ROLE_DOSE_FORM",
    "ROLE_DURATION",
    "ROLE_FREQUENCY",
    "ROLE_NAME",
    "ROLE_PRN",
    "ROLE_RELEASE",
    "ROLE_ROUTE",
    "ROLE_SALT",
    "ROLE_STRENGTH_UNIT",
    "ROLE_STRENGTH_VALUE",
    "STRUCTURED_MEDICATION_VERSION",
    "STRUCTURED_ROLES",
    "StructuredField",
    "StructuredMedicationMention",
    "build_structured_mention",
    "build_structured_mentions",
    "canonical_unit",
    "normalize_surface",
    "strength_in_mg",
    "unit_family",
]
