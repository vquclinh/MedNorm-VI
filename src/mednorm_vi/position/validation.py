"""Validation for submission-position encoding (Phase 1C-A).

Guarantees:
- the internal raw span is exact (``original_text[raw_start:raw_end] == text``);
- the encoded position came from a registered policy;
- a reversible round-trip exists where the policy claims reversibility;
- unsupported / ambiguous mappings fail clearly (never silently rounded);
- encoding never edits entity text.
"""

from __future__ import annotations

from ..validator.results import ValidationIssue, ValidationResult
from .encoders import PositionEncodingError
from .models import PositionEncodingResult
from .registry import PositionPolicyRegistry


def validate_encoding(
    registry: PositionPolicyRegistry, original_text: str, result: PositionEncodingResult,
    *, separator: str = "\n", document_id: str | None = None, entity_index: int | None = None,
) -> ValidationResult:
    """Validate one :class:`PositionEncodingResult` against the raw invariant."""
    issues: list[ValidationIssue] = []
    prov = result.provenance
    rs, re = prov.raw_start, prov.raw_end

    def _issue(code: str, msg: str) -> ValidationIssue:
        return ValidationIssue(code, msg, document_id=document_id, entity_index=entity_index)

    # 1. registered policy
    if result.policy_id not in registry.policy_ids:
        issues.append(_issue("position.unregistered_policy",
                             f"policy {result.policy_id!r} is not registered"))
        return ValidationResult.from_issues(issues)

    # 2. raw invariant: text is a byte-for-byte slice of original_text
    if not (0 <= rs <= re <= len(original_text)):
        issues.append(_issue("position.raw_out_of_bounds",
                             f"raw span [{rs},{re}) out of bounds for len {len(original_text)}"))
        return ValidationResult.from_issues(issues)
    if original_text[rs:re] != result.text:
        issues.append(_issue("position.text_offset_mismatch",
                             "original_text[raw_start:raw_end] != text (encoding edited text?)"))

    # 3. re-encode from the raw span and confirm the encoder is deterministic
    try:
        redo = registry.encode(result.policy_id, original_text, rs, re, separator=separator)
    except PositionEncodingError as exc:
        issues.append(_issue("position.encode_failed", f"re-encode failed: {exc}"))
        return ValidationResult.from_issues(issues)
    if redo.encoded_position != result.encoded_position:
        issues.append(_issue("position.nondeterministic_encoding",
                             "re-encoding the raw span produced a different position"))

    # 4. reversible round-trip where claimed
    if prov.reversible and not registry.round_trips(
            result.policy_id, original_text, rs, re, separator=separator):
        issues.append(_issue("position.irreversible",
                             f"policy {result.policy_id!r} claims reversibility but the "
                             "round-trip did not recover the raw span"))
    return ValidationResult.from_issues(issues)


__all__ = ["validate_encoding"]
