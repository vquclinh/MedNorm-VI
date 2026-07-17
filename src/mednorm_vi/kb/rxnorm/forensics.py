"""Deterministic comparison of two local RxNorm snapshots (Phase 1C-A).

Reports, for two locally acquired snapshots: concepts in both, label changes, TTY
changes, active↔suppressed transitions, replaced/remapped RXCUIs, concepts unique
to one snapshot, and a representative mention lookup. Used to reason about which
undisclosed release the organizer might use — never to declare one authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compatibility import resolve_current
from .models import RxnormSnapshot


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    left_id: str
    right_id: str
    n_left: int
    n_right: int
    in_both: tuple[str, ...]
    only_left: tuple[str, ...]
    only_right: tuple[str, ...]
    label_changed: tuple[tuple[str, str, str], ...]  # (rxcui, left_str, right_str)
    tty_changed: tuple[tuple[str, str, str], ...]  # (rxcui, left_ttys, right_ttys)
    suppression_changed: tuple[tuple[str, bool, bool], ...]  # (rxcui, left_supp, right_supp)
    remapped: tuple[tuple[str, str], ...]  # (rxcui, right_resolved) where right remaps it
    notes: tuple[str, ...] = field(default_factory=tuple)


def diff_snapshots(left: RxnormSnapshot, right: RxnormSnapshot) -> SnapshotDiff:
    lcuis, rcuis = set(left.rxcuis()), set(right.rxcuis())
    both = sorted(lcuis & rcuis)
    label_changed: list[tuple[str, str, str]] = []
    tty_changed: list[tuple[str, str, str]] = []
    supp_changed: list[tuple[str, bool, bool]] = []
    for cui in both:
        ls, rs = left.preferred_string(cui) or "", right.preferred_string(cui) or ""
        if ls != rs:
            label_changed.append((cui, ls, rs))
        lt, rt = ",".join(left.ttys(cui)), ",".join(right.ttys(cui))
        if lt != rt:
            tty_changed.append((cui, lt, rt))
        lsu, rsu = left.is_suppressed(cui), right.is_suppressed(cui)
        if lsu != rsu:
            supp_changed.append((cui, lsu, rsu))

    remapped: list[tuple[str, str]] = []
    for cui in sorted(lcuis - rcuis):
        res = resolve_current(right, cui)
        if res.resolved_exists and res.resolved_rxcui != cui:
            remapped.append((cui, res.resolved_rxcui))

    return SnapshotDiff(
        left_id=left.snapshot_id, right_id=right.snapshot_id,
        n_left=len(lcuis), n_right=len(rcuis), in_both=tuple(both),
        only_left=tuple(sorted(lcuis - rcuis)), only_right=tuple(sorted(rcuis - lcuis)),
        label_changed=tuple(label_changed), tty_changed=tuple(tty_changed),
        suppression_changed=tuple(supp_changed), remapped=tuple(remapped))


def lookup_mention(snapshot: RxnormSnapshot, text: str) -> tuple[str, ...]:
    """Representative exact/normalized mention lookup → matching RXCUIs (sorted)."""
    needle = " ".join(text.lower().split())
    hits = {a.rxcui for a in snapshot.atoms if a.normalized == needle}
    return tuple(sorted(hits))


__all__ = ["SnapshotDiff", "diff_snapshots", "lookup_mention"]
