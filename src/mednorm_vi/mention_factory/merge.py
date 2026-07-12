"""Deterministic proposal collection + identity/overlap diagnostics.

Proposals are NEVER merged by text alone. Identity is
(document, start, end, proposed_types, specialist). When two proposals share the
exact same coordinates + type, this policy KEEPS BOTH (preserving each source's
provenance) and records a diagnostic — Phase 1B does no final global span
selection. Repeated identical text at different positions stays distinct.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SpanProposal


@dataclass(frozen=True, slots=True)
class MergeDiagnostics:
    duplicate_identity_groups: tuple[tuple[str, ...], ...] = ()
    overlap_pairs: tuple[tuple[str, str], ...] = ()
    repeated_surface: tuple[str, ...] = ()


def collect(proposals: list[SpanProposal]) -> tuple[list[SpanProposal], MergeDiagnostics]:
    """Return proposals in a deterministic order + overlap/duplicate diagnostics."""
    ordered = sorted(
        proposals,
        key=lambda p: (p.start, p.end, p.source_specialist, p.proposed_types, p.proposal_id),
    )

    # exact-identity duplicates (same coords + type + specialist): keep all, flag.
    by_identity: dict[tuple[str, int, int, tuple[str, ...], str], list[str]] = {}
    for p in ordered:
        by_identity.setdefault(p.identity(), []).append(p.proposal_id)
    dup_groups = tuple(tuple(v) for v in by_identity.values() if len(v) > 1)

    # overlapping (non-identical) proposals of the same type within a node.
    overlaps: list[tuple[str, str]] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.start >= a.end:
                continue
            if (a.source_node_id == b.source_node_id and a.proposed_types == b.proposed_types
                    and (a.start, a.end) != (b.start, b.end)):
                overlaps.append((a.proposal_id, b.proposal_id))

    # repeated surface text at different absolute positions.
    positions: dict[str, set[int]] = {}
    for p in ordered:
        positions.setdefault(p.text, set()).add(p.start)
    repeated = tuple(sorted(t for t, starts in positions.items() if len(starts) > 1))

    return ordered, MergeDiagnostics(dup_groups, tuple(overlaps), repeated)


__all__ = ["MergeDiagnostics", "collect"]
