"""Phase 1B validation composition + a fail-fast guard."""

from __future__ import annotations

from ..mention_factory.validation import validate_proposals
from ..validator.results import ValidationResult
from .models import Phase1BResult


class Phase1BValidationError(Exception):
    """Raised by :func:`assert_valid` when a Phase 1B invariant is violated."""


def result_validation(result: Phase1BResult, original_text: str) -> ValidationResult:
    """Re-validate the proposals/relations of a result against original_text."""
    return validate_proposals(list(result.proposals), list(result.relations), original_text)


def assert_valid(result: Phase1BResult, original_text: str) -> None:
    """Fail fast if any proposal/relation invariant is violated."""
    res = result_validation(result, original_text)
    if not res.ok:
        codes = ", ".join(sorted({i.code for i in res.errors}))
        raise Phase1BValidationError(f"Phase 1B validation failed: {codes}")


__all__ = ["Phase1BValidationError", "result_validation", "assert_valid"]
