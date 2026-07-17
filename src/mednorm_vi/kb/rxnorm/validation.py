"""Integrity validation for a local RxNorm snapshot (offline)."""

from __future__ import annotations

from ...validator.results import Severity, ValidationIssue, ValidationResult
from .models import RxnormSnapshot


def validate_snapshot(snapshot: RxnormSnapshot) -> ValidationResult:
    """Structural integrity checks — never a claim about organizer correctness."""
    issues: list[ValidationIssue] = []

    def err(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, msg))

    def warn(code: str, msg: str) -> None:
        issues.append(ValidationIssue(code, msg, severity=Severity.WARNING))

    if not snapshot.atoms:
        err("rxnorm.empty", "snapshot has no RXNCONSO atoms")
    for a in snapshot.atoms:
        if not a.rxcui:
            err("rxnorm.atom_no_rxcui", f"atom {a.rxaui!r} has no RXCUI")
        if not a.string:
            warn("rxnorm.atom_no_string", f"RXCUI {a.rxcui} atom {a.rxaui} has empty STR")

    known = set(snapshot.rxcuis())
    for r in snapshot.relations:
        if r.rxcui1 and r.rxcui1 not in known:
            warn("rxnorm.rel_dangling_source", f"relation source {r.rxcui1} not in snapshot")
        if r.rxcui2 and r.rxcui2 not in known:
            warn("rxnorm.rel_dangling_target", f"relation target {r.rxcui2} not in snapshot")
    return ValidationResult.from_issues(issues)


__all__ = ["validate_snapshot"]
