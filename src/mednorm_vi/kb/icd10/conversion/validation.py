"""Validation for derived ICD-10 Vietnamese conversion artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import normalization as code_norm
from .normalization import NormalizedIcdRecord


@dataclass(frozen=True, slots=True)
class ConversionValidation:
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _has_cycle(records: tuple[NormalizedIcdRecord, ...]) -> bool:
    parent = {record.undotted_code: record.parent for record in records if record.parent}
    for code in parent:
        seen: set[str] = set()
        cur = code
        while cur in parent:
            cur = parent[cur]
            if cur in seen:
                return True
            seen.add(cur)
    return False


def validate_rows(records: tuple[NormalizedIcdRecord, ...]) -> ConversionValidation:
    """Validate code forms, labels, duplicates, and inferred hierarchy."""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.undotted_code in seen:
            errors.append(f"duplicate_code:{record.undotted_code}")
        seen.add(record.undotted_code)
        if not code_norm.is_wellformed(record.undotted_code):
            errors.append(f"malformed_code:{record.supplied_code}")
        if code_norm.to_undotted(record.dotted_code) != record.undotted_code:
            errors.append(f"nonreversible_code:{record.supplied_code}")
        if not record.vietnamese_label:
            warnings.append(f"missing_label:{record.undotted_code}")
        if record.parent and record.parent not in seen and record.parent not in {
            r.undotted_code for r in records
        }:
            warnings.append(f"missing_parent:{record.undotted_code}->{record.parent}")
    if _has_cycle(records):
        errors.append("hierarchy_cycle")
    return ConversionValidation(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
