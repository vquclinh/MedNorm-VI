"""Deterministic L1 validation over a built DocumentGraph.

Detects invalid ranges, text/offset mismatches, overlapping/incomplete line
coverage, broken parent containment, duplicate node ids, broken alignment maps,
and normalized spans lacking a covering original span. Uses the repository
``ValidationResult`` style; invariant violations are ERRORs (fail fast), while
uncertain structural interpretations are WARNINGs.
"""

from __future__ import annotations

from ..validator.results import Severity, ValidationIssue, ValidationResult
from .models import DocumentGraph, NodeKind


def validate_graph(graph: DocumentGraph) -> ValidationResult:
    issues: list[ValidationIssue] = []
    n = len(graph.original_text)
    index = graph.by_id()

    seen_ids: set[str] = set()
    for node in graph.nodes:
        if node.node_id in seen_ids:
            issues.append(ValidationIssue("l1.duplicate_node_id",
                                          f"duplicate node id {node.node_id!r}"))
        seen_ids.add(node.node_id)

        if not (0 <= node.start <= node.end <= n):
            issues.append(ValidationIssue(
                "l1.invalid_range",
                f"node {node.node_id} span [{node.start}, {node.end}) out of bounds (len {n})"))
            continue
        # Text/offset consistency: the slice must be recoverable (it always is by
        # construction; this guards against any future offset corruption).
        if graph.original_text[node.start : node.end] != node.text(graph.original_text):
            issues.append(ValidationIssue(
                "l1.text_offset_mismatch", f"node {node.node_id} text/offset mismatch"))

        # Parent containment.
        if node.parent_id is not None:
            parent = index.get(node.parent_id)
            if parent is None:
                issues.append(ValidationIssue(
                    "l1.missing_parent",
                    f"node {node.node_id} references missing parent {node.parent_id}"))
            elif not (parent.start <= node.start and node.end <= parent.end):
                issues.append(ValidationIssue(
                    "l1.broken_containment",
                    f"node {node.node_id} [{node.start},{node.end}) not within parent "
                    f"{parent.node_id} [{parent.start},{parent.end})"))

    # Full line coverage: LINE (non-zero) + newline delimiters must tile [0, n).
    intervals: list[tuple[int, int]] = []
    for node in graph.nodes:
        if node.kind is NodeKind.LINE and node.end > node.start:
            intervals.append((node.start, node.end))
        elif node.kind is NodeKind.DELIMITER and node.attributes.get("delimiter") == "newline":
            intervals.append((node.start, node.end))
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start < cursor:
            issues.append(ValidationIssue(
                "l1.line_overlap",
                f"line/delimiter coverage overlaps at {start} (cursor {cursor})"))
        elif start > cursor:
            issues.append(ValidationIssue(
                "l1.line_gap", f"uncovered document region [{cursor}, {start})",
                severity=Severity.WARNING))
        cursor = max(cursor, end)
    if intervals and cursor != n:
        issues.append(ValidationIssue(
            "l1.incomplete_coverage", f"coverage ends at {cursor}, expected {n}",
            severity=Severity.WARNING))

    # Alignment consistency + covering for each normalized view. Under
    # smallest-covering semantics, the full normalized range maps back to a span
    # that starts at 0 and whose image reaches the full normalized length; the
    # end may fall short of `n` only when trailing original chars were deleted
    # (e.g. collapsed trailing whitespace), which is correct, not a violation.
    for name, view in sorted(graph.normalized.items()):
        align = view.alignment
        for err in align.consistency_errors():
            issues.append(ValidationIssue("l1.broken_alignment", f"view {name!r}: {err}"))
        if align.normalized_length > 0:
            os, oe = align.normalized_span_to_original(0, align.normalized_length)
            if os != 0 or align.o2n[oe] != align.normalized_length:
                issues.append(ValidationIssue(
                    "l1.alignment_not_covering",
                    f"view {name!r} full-range covering invalid (start={os}, "
                    f"o2n[end]={align.o2n[oe]} != {align.normalized_length})"))

    return ValidationResult.from_issues(issues)


__all__ = ["validate_graph"]
