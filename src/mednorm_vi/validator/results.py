"""Shared result types for the deterministic validator foundation (L9).

Validators return structured, aggregatable results instead of raising, so a
whole document (or the whole submission) can be checked and reported. Callers
that need fail-fast behaviour can inspect ``ValidationResult.ok`` and stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation finding."""

    code: str  # stable machine-readable id, e.g. "offset.substring_mismatch"
    message: str
    severity: Severity = Severity.ERROR
    # Optional locators for pinpointing the offending item.
    document_id: str | None = None
    entity_index: int | None = None
    context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregated validation findings."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """True when there are no ERROR-severity issues (warnings are allowed)."""
        return not self.errors

    def merged_with(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(issues=self.issues + other.issues)

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> ValidationResult:
        return cls(issues=tuple(issues))
