"""Synthetic laboratory stress cases with exact offsets **by construction**.

Spec §14.1 requires synthetic data whose spans are exact by construction rather
than re-found by string search, and §14.2 names the laboratory variations that
must be covered: "Decimal comma/dot, units, missing units, H/L flags, reordered
columns, multiline formats". §18.3 adds the offset stress suite.

Every case here is assembled by a builder that *emits* text and records the span
it just wrote. No case ever calls ``str.find``, so an offset can never drift and
a repeated test name at two positions is unambiguous by construction.

**These are synthetic cases.** Nothing measured on them is a claim about real
TEST_NAME / TEST_RESULT performance — the governed corpus contains no
supervision for those two types at all. Results must always be reported under a
separate, clearly synthetic label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# TEST_RESULT gold boundary convention used by these cases.
#
# The organizer's convention for a numeric result with a unit is UNRESOLVED. The
# tracked deterministic default (`configs/resolution/resolver_v1.yaml`,
# `test_result_boundary: value_only`) selects the value alone, so the gold below
# follows that policy and says so. It is a *policy*, not a confirmed fact.
RESULT_GOLD_BOUNDARY_POLICY = "value_only"

FAMILY_COLON = "colon_name_value"
FAMILY_UNIT = "name_value_unit"
FAMILY_DECIMAL_COMMA = "decimal_comma"
FAMILY_DECIMAL_POINT = "decimal_point"
FAMILY_FLAG = "flag_high_low"
FAMILY_QUALITATIVE = "qualitative_positive_negative"
FAMILY_REFERENCE_RANGE = "reference_range"
FAMILY_MISSING_UNIT = "missing_unit"
FAMILY_MULTILINE = "multiline_rows"
FAMILY_SEMICOLON = "semicolon_row"
FAMILY_TABLE = "table_like"
FAMILY_REPEATED = "repeated_test"
FAMILY_DISTRACTOR = "distractor_values"

FAMILIES: tuple[str, ...] = (
    FAMILY_COLON, FAMILY_UNIT, FAMILY_DECIMAL_COMMA, FAMILY_DECIMAL_POINT,
    FAMILY_FLAG, FAMILY_QUALITATIVE, FAMILY_REFERENCE_RANGE, FAMILY_MISSING_UNIT,
    FAMILY_MULTILINE, FAMILY_SEMICOLON, FAMILY_TABLE, FAMILY_REPEATED,
    FAMILY_DISTRACTOR,
)


@dataclass(frozen=True, slots=True)
class SyntheticEntity:
    """One gold mention whose offsets were recorded as the text was written."""

    start: int
    end: int
    entity_type: str
    text: str
    role: str = ""          # "name" | "value" | "value_unit"

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.entity_type)


@dataclass(frozen=True, slots=True)
class SyntheticRelation:
    """A gold ``TEST_NAME --has_result--> TEST_RESULT`` edge, by entity index."""

    name_index: int
    result_index: int


@dataclass(frozen=True, slots=True)
class SyntheticLabCase:
    """One synthetic laboratory document with exact gold spans and relations."""

    case_id: str
    family: str
    text: str
    entities: tuple[SyntheticEntity, ...]
    relations: tuple[SyntheticRelation, ...] = field(default_factory=tuple)
    note: str = ""

    def gold_pairs(self) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        """Gold has_result pairs as ``((name span), (result span))``."""
        return tuple(
            ((self.entities[r.name_index].start, self.entities[r.name_index].end),
             (self.entities[r.result_index].start, self.entities[r.result_index].end))
            for r in self.relations)

    def validate(self) -> None:
        """Prove every recorded offset still slices its own text out of ``text``."""
        for entity in self.entities:
            if self.text[entity.start:entity.end] != entity.text:
                raise ValueError(
                    f"{self.case_id}: constructed offset [{entity.start}, "
                    f"{entity.end}) does not slice out its own text")
            if entity.end <= entity.start:
                raise ValueError(f"{self.case_id}: empty span")
        for relation in self.relations:
            if not (0 <= relation.name_index < len(self.entities)
                    and 0 <= relation.result_index < len(self.entities)):
                raise ValueError(f"{self.case_id}: relation index out of range")


class _CaseBuilder:
    """Writes text while recording the exact span of every gold fragment."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.entities: list[SyntheticEntity] = []
        self.relations: list[SyntheticRelation] = []

    def write(self, fragment: str) -> None:
        """Append plain, unlabelled text (delimiters, units, noise)."""
        self._parts.append(fragment)
        self._length += len(fragment)

    def label(self, fragment: str, entity_type: str, *, role: str = "") -> int:
        """Append text AND record it as a gold mention. Returns its index."""
        start = self._length
        self.write(fragment)
        self.entities.append(SyntheticEntity(
            start, start + len(fragment), entity_type, fragment, role))
        return len(self.entities) - 1

    def pair(self, name_index: int, result_index: int) -> None:
        self.relations.append(SyntheticRelation(name_index, result_index))

    def build(self, case_id: str, family: str, note: str = "") -> SyntheticLabCase:
        case = SyntheticLabCase(
            case_id=case_id, family=family, text="".join(self._parts),
            entities=tuple(self.entities), relations=tuple(self.relations), note=note)
        case.validate()
        return case


def _row(builder: _CaseBuilder, name: str, value: str, *, unit: str = "",
         suffix: str = "", separator: str = ": ") -> None:
    """One ``NAME<sep>VALUE[ UNIT][suffix]`` row with a paired gold relation."""
    name_index = builder.label(name, "TEST_NAME", role="name")
    builder.write(separator)
    result_index = builder.label(value, "TEST_RESULT", role="value")
    if unit:
        builder.write(" " + unit)
    if suffix:
        builder.write(suffix)
    builder.pair(name_index, result_index)


def build_cases() -> tuple[SyntheticLabCase, ...]:
    """The full synthetic laboratory stress suite."""
    cases: list[SyntheticLabCase] = []

    # 1. name:value, no space after the colon.
    b = _CaseBuilder()
    _row(b, "WBC", "14.43", separator=":")
    cases.append(b.build("lab-colon-01", FAMILY_COLON,
                         "colon with no whitespace, decimal point"))

    # 2. name: value unit.
    b = _CaseBuilder()
    _row(b, "Glucose", "5.6", unit="mmol/L")
    cases.append(b.build("lab-unit-01", FAMILY_UNIT, "value followed by a unit"))

    # 3. decimal comma (Vietnamese convention).
    b = _CaseBuilder()
    _row(b, "Creatinine", "88,5", unit="µmol/L")
    cases.append(b.build("lab-decimal-comma-01", FAMILY_DECIMAL_COMMA,
                         "decimal comma must not split the value"))

    # 4. decimal point alongside a percent unit.
    b = _CaseBuilder()
    _row(b, "NEUT%", "76.4", unit="%")
    cases.append(b.build("lab-decimal-point-01", FAMILY_DECIMAL_POINT,
                         "percent-suffixed test name plus a percent unit"))

    # 5. H / L flags after the value.
    b = _CaseBuilder()
    _row(b, "HGB", "11.2", unit="g/dL", suffix=" L")
    b.write("\n")
    _row(b, "CRP", "48", unit="mg/dL", suffix=" H")
    cases.append(b.build("lab-flag-01", FAMILY_FLAG,
                         "flags are evidence, never part of the value span"))

    # 6. qualitative positive/negative results.
    b = _CaseBuilder()
    _row(b, "HbA1c", "dương tính")
    b.write("\n")
    _row(b, "CRP", "âm tính")
    cases.append(b.build("lab-qualitative-01", FAMILY_QUALITATIVE,
                         "bare qualitative rows: no number, no unit, no lab "
                         "section header — the L2 router requires lab-specific "
                         "evidence, so C2 does not fire and E2 never runs"))

    # 6b. the same qualitative rows UNDER a laboratory section header. The header
    #     supplies the C2 evidence a bare qualitative row lacks, so this case
    #     isolates exactly what the router needs (see the note on the bare case).
    b = _CaseBuilder()
    b.write("Xét nghiệm:\n")
    _row(b, "HbA1c", "dương tính")
    b.write("\n")
    _row(b, "CRP", "âm tính")
    cases.append(b.build("lab-qualitative-02", FAMILY_QUALITATIVE,
                         "qualitative rows under a laboratory section header"))

    # 7. reference range in parentheses.
    b = _CaseBuilder()
    _row(b, "Natri", "138", unit="mmol/L", suffix=" (135-145)")
    cases.append(b.build("lab-reference-01", FAMILY_REFERENCE_RANGE,
                         "the reference range must not be swallowed by the value"))

    # 8. missing unit.
    b = _CaseBuilder()
    _row(b, "PLT", "250")
    cases.append(b.build("lab-missing-unit-01", FAMILY_MISSING_UNIT,
                         "a unit-less result is still a result"))

    # 9. multiline rows.
    b = _CaseBuilder()
    b.write("Xét nghiệm:\n")
    _row(b, "WBC", "10,2", unit="10^9/L")
    b.write("\n")
    _row(b, "RBC", "4,5", unit="10^12/L")
    b.write("\n")
    _row(b, "PLT", "210", unit="10^9/L")
    cases.append(b.build("lab-multiline-01", FAMILY_MULTILINE,
                         "three rows under one section header"))

    # 10. semicolon-separated row.
    b = _CaseBuilder()
    _row(b, "AST", "45", unit="U/L")
    b.write("; ")
    _row(b, "ALT", "52", unit="U/L")
    cases.append(b.build("lab-semicolon-01", FAMILY_SEMICOLON,
                         "two tests in one physical line, semicolon separated"))

    # 11. table-like layout with tab cells.
    b = _CaseBuilder()
    _row(b, "Kali", "3,9", unit="mmol/L", separator="\t")
    b.write("\n")
    _row(b, "Clo", "101", unit="mmol/L", separator="\t")
    cases.append(b.build("lab-table-01", FAMILY_TABLE, "tab-delimited cells"))

    # 12. the SAME test twice with different values — two distinct mentions at
    #     different offsets, which must never be deduplicated by text (spec §5 C7).
    b = _CaseBuilder()
    _row(b, "Glucose", "5,6", unit="mmol/L")
    b.write("\n")
    _row(b, "Glucose", "7,8", unit="mmol/L")
    cases.append(b.build("lab-repeated-01", FAMILY_REPEATED,
                         "repeated test name at two offsets stays two mentions"))

    # 13. distractors: a date, an age, a phone number and a medication dose must
    #     produce NO laboratory mention at all. This case has zero gold entities.
    b = _CaseBuilder()
    b.write("Ngày vào viện: 12/03/2026\n")
    b.write("Tuổi: 54\n")
    b.write("Điện thoại: 0912345678\n")
    b.write("Paracetamol 500 mg\n")
    cases.append(b.build("lab-distractor-01", FAMILY_DISTRACTOR,
                         "structure that looks like a lab row but is not one"))

    return tuple(cases)


def suite_summary(cases: tuple[SyntheticLabCase, ...]) -> dict[str, int]:
    """Case, mention and relation counts per family, for the report header."""
    out: dict[str, int] = {
        "cases": len(cases),
        "gold_mentions": sum(len(c.entities) for c in cases),
        "gold_relations": sum(len(c.relations) for c in cases),
        "families": len({c.family for c in cases}),
    }
    for family in FAMILIES:
        out[f"cases_{family}"] = sum(1 for c in cases if c.family == family)
    return out


__all__ = [
    "FAMILIES",
    "FAMILY_COLON",
    "FAMILY_DECIMAL_COMMA",
    "FAMILY_DECIMAL_POINT",
    "FAMILY_DISTRACTOR",
    "FAMILY_FLAG",
    "FAMILY_MISSING_UNIT",
    "FAMILY_MULTILINE",
    "FAMILY_QUALITATIVE",
    "FAMILY_REFERENCE_RANGE",
    "FAMILY_REPEATED",
    "FAMILY_SEMICOLON",
    "FAMILY_TABLE",
    "FAMILY_UNIT",
    "RESULT_GOLD_BOUNDARY_POLICY",
    "SyntheticEntity",
    "SyntheticLabCase",
    "SyntheticRelation",
    "build_cases",
    "suite_summary",
]
