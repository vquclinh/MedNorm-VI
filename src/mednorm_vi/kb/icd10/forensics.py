"""Deterministic comparison of two local ICD-10 snapshots (Phase 1C-A).

Reports missing codes, format differences, label differences, hierarchy changes,
alias differences, and specificity differences between two locally acquired
snapshots. Never declares one snapshot the organizer's authoritative source.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import IcdSnapshot


@dataclass(frozen=True, slots=True)
class IcdSnapshotDiff:
    left_id: str
    right_id: str
    n_left: int
    n_right: int
    only_left: tuple[str, ...]
    only_right: tuple[str, ...]
    label_changed: tuple[tuple[str, str, str], ...]  # (code, left_vi, right_vi)
    parent_changed: tuple[tuple[str, str, str], ...]  # (code, left_parent, right_parent)
    alias_changed: tuple[tuple[str, str, str], ...]  # (code, left_aliases, right_aliases)
    format_changed: tuple[tuple[str, str, str], ...]  # (code, left_supplied, right_supplied)
    specificity_changed: tuple[tuple[str, int, int], ...]  # (code, left_spec, right_spec)


def diff_snapshots(left: IcdSnapshot, right: IcdSnapshot) -> IcdSnapshotDiff:
    lcodes, rcodes = set(left.codes()), set(right.codes())
    both = sorted(lcodes & rcodes)
    label_changed: list[tuple[str, str, str]] = []
    parent_changed: list[tuple[str, str, str]] = []
    alias_changed: list[tuple[str, str, str]] = []
    format_changed: list[tuple[str, str, str]] = []
    spec_changed: list[tuple[str, int, int]] = []
    for code in both:
        lc, rc = left.get(code), right.get(code)
        assert lc is not None and rc is not None
        if lc.label_vi != rc.label_vi:
            label_changed.append((code, lc.label_vi, rc.label_vi))
        if (lc.parent or "") != (rc.parent or ""):
            parent_changed.append((code, lc.parent or "", rc.parent or ""))
        la, ra = ",".join(lc.aliases), ",".join(rc.aliases)
        if la != ra:
            alias_changed.append((code, la, ra))
        if lc.code_supplied != rc.code_supplied:
            format_changed.append((code, lc.code_supplied, rc.code_supplied))
        if lc.specificity != rc.specificity:
            spec_changed.append((code, lc.specificity, rc.specificity))
    return IcdSnapshotDiff(
        left_id=left.snapshot_id, right_id=right.snapshot_id,
        n_left=len(lcodes), n_right=len(rcodes),
        only_left=tuple(sorted(lcodes - rcodes)), only_right=tuple(sorted(rcodes - lcodes)),
        label_changed=tuple(label_changed), parent_changed=tuple(parent_changed),
        alias_changed=tuple(alias_changed), format_changed=tuple(format_changed),
        specificity_changed=tuple(spec_changed))


__all__ = ["IcdSnapshotDiff", "diff_snapshots"]
