"""The final L9 gate: one check over the payload the organizer actually receives.

Spec §16 step 10 and Appendix A require the emitted output to be validated against
the source document — ``original_text[start:end] == text`` for **every** entity — and
they forbid silent repair. Until Audit 0056a the canonical packaging path did not do
this.

What was actually wired before this module existed:

* ``_gate_kb_membership`` validated candidate **codes** against the frozen snapshots;
* ``package_output_zip`` validated the **layout** (file set, UTF-8, JSON shape) and
  each entity's organizer **schema** through ``validate_organizer_document``.

Neither ever saw ``original_text``. ``validate_organizer_document`` receives only the
parsed JSON, so it structurally cannot check the offset invariant, and the composed
``validator.validate_document(original_text, entities)`` — which does — had **no
caller anywhere in the repository** outside this package. The invariant was enforced
four times upstream (expert, registry, lattice, L4) and then not once on the bytes
that would be submitted, which is precisely the stage those four checks do not cover:
serialization.

The second gap was spec P7. ``validate_document_candidates`` has always accepted an
``offered_codes_by_index`` argument implementing "a model may not introduce a code" —
and the canonical caller never passed it, so the rule was a unit-test contract that
never ran on real output.

This module closes both, in one gate, over the **serialized** payload:

1. safe filename (``{n}.json``, no path separators, no traversal);
2. UTF-8 decode and JSON parse; root must be a list;
3. organizer schema — type vocabulary, exact required/forbidden field set per type,
   assertion applicability, candidate syntax and ontology, position shape;
4. end-exclusive offsets inside the document, and exact
   ``original_text[start:end] == text``;
5. duplicate candidates within an entity;
6. snapshot membership, correct ontology per type, and **offered-set membership**;
7. deterministic emission order.

A failure raises :class:`FinalValidationError` from the caller **before** anything is
written, so a violating run leaves no ``output/`` and no ``output.zip``. Nothing here
repairs, reorders, drops or normalizes: the gate reports and the caller stops.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from ..schemas.constants import ORGANIZER_LABELS, TYPE_BY_ORGANIZER_LABEL
from .kb_membership import LockedSnapshots, validate_document_candidates
from .organizer import validate_organizer_document
from .results import Severity, ValidationIssue, ValidationResult

FINAL_GATE_CONTRACT_VERSION = "l9-final-serialized-gate-v1"


class FinalValidationError(RuntimeError):
    """Raised by L9's caller when the final serialized payload fails validation.

    Like :class:`~mednorm_vi.validator.kb_membership.KbMembershipViolation`, the
    validator reports and the *caller* stops. Spec §16's "never silently repair
    output" means a violating run must produce no package at all, rather than a
    package carrying a warning nobody reads.
    """

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            "final L9 validation failed; no output package was written: "
            + ", ".join(self.violations)
        )


@dataclass(frozen=True, slots=True)
class FinalDocument:
    """Everything the final gate needs about one document. All of it is required.

    ``offered_codes_by_index`` maps the index of an entity **in the serialized list**
    to the codes the retrieval stage offered for that mention. The alignment is by
    construction: ``to_entity_predictions`` preserves the decoder's order and
    ``to_submission_json`` preserves the list, so index *i* of the payload is index
    *i* of the decoded entities.
    """

    document_id: str
    filename: str
    original_text: str
    payload: str
    offered_codes_by_index: Mapping[int, tuple[str, ...]] = field(default_factory=dict)


def _issue(
    code: str,
    message: str,
    document_id: str,
    entity_index: int | None = None,
    **ctx: object,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=Severity.ERROR,
        document_id=document_id,
        entity_index=entity_index,
        context=ctx,
    )


def validate_safe_filename(filename: str, document_id: str) -> ValidationResult:
    """The emitted name must be a bare ``{digits}.json`` with no path component."""
    issues: list[ValidationIssue] = []
    if not filename:
        return ValidationResult.from_issues(
            [_issue("final.empty_filename", "an output filename may not be empty", document_id)]
        )
    if "/" in filename or "\\" in filename or ":" in filename:
        issues.append(
            _issue(
                "final.unsafe_filename",
                f"output filename {filename!r} contains a path separator",
                document_id,
                value=filename,
            )
        )
    parts = PurePosixPath(filename).parts
    if any(part in ("..", ".") for part in parts) or len(parts) != 1:
        issues.append(
            _issue(
                "final.unsafe_filename",
                f"output filename {filename!r} must be a single path component",
                document_id,
                value=filename,
            )
        )
    stem, _, suffix = filename.rpartition(".")
    if suffix != "json" or not stem.isdigit():
        issues.append(
            _issue(
                "final.unexpected_filename",
                f"output filename {filename!r} must be '<number>.json'",
                document_id,
                value=filename,
            )
        )
    return ValidationResult.from_issues(issues)


def _validate_offsets_against_source(
    entities: Sequence[object], original_text: str, document_id: str
) -> ValidationResult:
    """Spec §4/§16: every emitted span must slice exactly out of the source."""
    issues: list[ValidationIssue] = []
    length = len(original_text)
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            continue  # `validate_organizer_document` reports the shape error
        position = entity.get("position")
        text = entity.get("text")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            continue  # organizer validation reports the malformed position
        start, end = position
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not isinstance(text, str):
            continue
        if start < 0 or end < start:
            issues.append(
                _issue(
                    "final.invalid_span",
                    f"entity {index} has an invalid end-exclusive span [{start}, {end})",
                    document_id,
                    index,
                    start=start,
                    end=end,
                )
            )
            continue
        if end > length:
            issues.append(
                _issue(
                    "final.span_outside_document",
                    f"entity {index} span [{start}, {end}) exceeds the source document "
                    f"length {length}",
                    document_id,
                    index,
                    start=start,
                    end=end,
                    document_length=length,
                )
            )
            continue
        if original_text[start:end] != text:
            # The message deliberately reports LENGTHS and the span, never the
            # clinical text itself, so a validation log carries no patient data.
            issues.append(
                _issue(
                    "final.text_offset_mismatch",
                    f"entity {index} violates original_text[start:end] == text at "
                    f"[{start}, {end}): emitted length {len(text)}, source slice length "
                    f"{end - start}",
                    document_id,
                    index,
                    start=start,
                    end=end,
                    emitted_length=len(text),
                    source_length=end - start,
                )
            )
    return ValidationResult.from_issues(issues)


def _validate_duplicate_candidates(
    entities: Sequence[object], document_id: str
) -> ValidationResult:
    """A candidate set is a SET: a repeated code inflates nothing and hides a bug."""
    issues: list[ValidationIssue] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            continue
        candidates = entity.get("candidates")
        if not isinstance(candidates, (list, tuple)):
            continue
        seen: set[str] = set()
        for code in candidates:
            if not isinstance(code, str):
                continue
            if code in seen:
                issues.append(
                    _issue(
                        "final.duplicate_candidate",
                        f"entity {index} lists candidate {code!r} more than once",
                        document_id,
                        index,
                        value=code,
                    )
                )
            seen.add(code)
    return ValidationResult.from_issues(issues)


def _validate_deterministic_order(entities: Sequence[object], document_id: str) -> ValidationResult:
    """Entities must be emitted in non-decreasing ``(start, end)`` document order.

    This holds by construction today — ``build_span_lattice`` sorts its merged
    proposals, L4 preserves that order, and L8 and the serializer both iterate — so
    the check locks in an existing invariant rather than imposing a new one. It is
    what makes "deterministic ordering" (Appendix A) checkable on a single payload
    instead of only by running the pipeline twice.
    """
    issues: list[ValidationIssue] = []
    previous: tuple[int, int] | None = None
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            continue
        position = entity.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            continue
        start, end = position
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        current = (start, end)
        if previous is not None and current < previous:
            issues.append(
                _issue(
                    "final.nondeterministic_order",
                    f"entity {index} at {list(current)} precedes {list(previous)}; "
                    "entities must be emitted in non-decreasing (start, end) order",
                    document_id,
                    index,
                )
            )
        previous = current
    return ValidationResult.from_issues(issues)


def _validate_offered_alignment(
    entities: Sequence[object],
    offered_codes_by_index: Mapping[int, tuple[str, ...]],
    document_id: str,
) -> ValidationResult:
    """Every candidate-bearing entity must have an offered set supplied.

    Without this, a missing entry would silently disable P7 for that entity — which
    is exactly how the rule came to be a no-op in the first place. An entity that
    emits candidates but has no recorded offered set is a wiring defect, and it
    fails rather than passing by default.
    """
    issues: list[ValidationIssue] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            continue
        etype = entity.get("type")
        if etype not in ORGANIZER_LABELS:
            continue
        if "candidates" not in entity:
            continue
        candidates = entity.get("candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            continue
        if index not in offered_codes_by_index:
            issues.append(
                _issue(
                    "final.offered_set_missing",
                    f"entity {index} ({TYPE_BY_ORGANIZER_LABEL.get(str(etype), etype)}) "
                    "emits candidates but no offered set was recorded for it; spec P7 "
                    "cannot be enforced without one",
                    document_id,
                    index,
                )
            )
    return ValidationResult.from_issues(issues)


def validate_final_document(
    document: FinalDocument, snapshots: LockedSnapshots
) -> ValidationResult:
    """Run the complete final gate over one serialized document."""
    result = validate_safe_filename(document.filename, document.document_id)

    try:
        root = json.loads(document.payload)
    except json.JSONDecodeError as exc:
        return result.merged_with(
            ValidationResult.from_issues(
                [
                    _issue(
                        "final.bad_json",
                        f"final payload is not well-formed JSON: {exc}",
                        document.document_id,
                    )
                ]
            )
        )
    if not isinstance(root, list):
        return result.merged_with(
            ValidationResult.from_issues(
                [
                    _issue(
                        "final.root_not_list",
                        f"final payload root must be a JSON list, got {type(root)!r}",
                        document.document_id,
                    )
                ]
            )
        )

    # Organizer schema: type vocabulary, exact field set per type, assertion
    # applicability, candidate syntax/ontology, position shape.
    result = result.merged_with(validate_organizer_document(root, document_id=document.document_id))
    # Offsets against the ACTUAL source document.
    result = result.merged_with(
        _validate_offsets_against_source(root, document.original_text, document.document_id)
    )
    result = result.merged_with(_validate_duplicate_candidates(root, document.document_id))
    result = result.merged_with(_validate_deterministic_order(root, document.document_id))
    result = result.merged_with(
        _validate_offered_alignment(root, document.offered_codes_by_index, document.document_id)
    )
    # Snapshot membership, ontology-per-type, and spec P7 offered-set membership.
    result = result.merged_with(
        validate_document_candidates(
            root,
            snapshots,
            document_id=document.document_id,
            offered_codes_by_index=dict(document.offered_codes_by_index),
        )
    )
    return result


def gate_final_documents(
    documents: Sequence[FinalDocument], snapshots: LockedSnapshots
) -> dict[str, int]:
    """Validate every final document. Raises on the first document with errors.

    Returns issue counts by code for the run manifest. Raising rather than returning
    a status is deliberate: the caller must not be able to proceed to packaging by
    forgetting to inspect a return value.
    """
    counts: dict[str, int] = {}
    violations: list[str] = []
    for document in documents:
        result = validate_final_document(document, snapshots)
        for issue in result.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        violations.extend(f"{document.filename}:{issue.code}" for issue in result.errors)
    if violations:
        raise FinalValidationError(violations)
    return counts


__all__ = [
    "FINAL_GATE_CONTRACT_VERSION",
    "FinalDocument",
    "FinalValidationError",
    "gate_final_documents",
    "validate_final_document",
    "validate_safe_filename",
]
