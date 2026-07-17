"""Deterministic, configurable boundary selection (Phase 1C-A).

Each logical mention has one or more boundary alternatives (medication:
name_only→full; lab result: value_only/value+unit). A configurable policy picks
one; the rest are retained. No single boundary is claimed to match the organizer
globally — though the confirmed medication example favours the full span.
"""

from __future__ import annotations

from ..mention_factory.models import SpanProposal
from .features import boundary_kind, width

# Progressive width order for medication boundary kinds (narrow -> wide).
_MED_ORDER: dict[str, int] = {
    "name_only": 0, "name_strength": 1, "name_strength_form": 2,
    "name_strength_route": 2, "full": 3,
}


def _by_deterministic(p: SpanProposal) -> tuple[int, int, str]:
    return (p.start, p.end, p.proposal_id)


def choose_boundary(
    entity_type: str, group: list[SpanProposal], *, medication_policy: str,
    test_result_policy: str,
) -> tuple[SpanProposal, tuple[str, ...], str]:
    """Return (chosen proposal, considered kinds, note) for one mention group."""
    considered = tuple(sorted({boundary_kind(p) for p in group}))
    if len(group) == 1:
        return group[0], considered, "single boundary"

    if entity_type == "THUỐC":
        return _choose_medication(group, medication_policy)
    if entity_type == "KẾT_QUẢ_XÉT_NGHIỆM":
        return _choose_result(group, test_result_policy)
    # default: widest, deterministic
    chosen = max(group, key=lambda p: (width(p), -p.start, p.proposal_id))
    return chosen, considered, "widest fallback"


def _choose_medication(
    group: list[SpanProposal], policy: str,
) -> tuple[SpanProposal, tuple[str, ...], str]:
    considered = tuple(sorted({boundary_kind(p) for p in group}))
    target = _MED_ORDER.get(policy, _MED_ORDER["full"])
    # exact policy kind if present
    exact = [p for p in group if boundary_kind(p) == policy]
    if exact:
        return min(exact, key=_by_deterministic), considered, f"policy={policy}"
    # else the widest whose order <= target, else the narrowest available
    below = [p for p in group if _MED_ORDER.get(boundary_kind(p), 99) <= target]
    if below:
        chosen = max(below, key=lambda p: (_MED_ORDER.get(boundary_kind(p), 0),
                                           width(p), -p.start))
        return chosen, considered, f"policy={policy} (fallback to widest<=target)"
    chosen = min(group, key=lambda p: (_MED_ORDER.get(boundary_kind(p), 0), _by_deterministic(p)))
    return chosen, considered, f"policy={policy} (fallback narrowest)"


def _choose_result(
    group: list[SpanProposal], policy: str,
) -> tuple[SpanProposal, tuple[str, ...], str]:
    considered = tuple(sorted({boundary_kind(p) for p in group}))
    want = "value_unit" if policy in ("value_with_unit", "value_unit") else "value_only"
    match = [p for p in group if boundary_kind(p) == want]
    if match:
        return min(match, key=_by_deterministic), considered, f"policy={policy}"
    # fall back to value_only, else narrowest
    vonly = [p for p in group if boundary_kind(p) == "value_only"]
    if vonly:
        return (min(vonly, key=_by_deterministic), considered,
                f"policy={policy} (fallback value_only)")
    return min(group, key=_by_deterministic), considered, f"policy={policy} (fallback narrowest)"


__all__ = ["choose_boundary"]
