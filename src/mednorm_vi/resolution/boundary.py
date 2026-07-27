"""Deterministic, configurable boundary selection (Phase 1C-A) and shaping (v1).

Each logical mention has one or more boundary alternatives (medication:
name_only→full; lab result: value_only/value+unit). A configurable policy picks
one; the rest are retained. No single boundary is claimed to match the organizer
globally — though the confirmed medication example favours the full span.

The second half of this module is the **v1 boundary shaping** used by the L4
Boundary & Type Resolver (spec §7.1). Two bounded operations:

* :func:`trim_span` removes leading/trailing characters the organizer's
  convention does not include — whitespace, list numbers, bullets, separators —
  never more than the configured cap and never producing an empty span;
* :func:`expand_to_competitor` adopts a boundary another expert **already
  proposed**. The resolver never invents an offset no expert put forward, which
  is what keeps ``original_text[start:end] == text`` trivially true.

Spec §7.1 also describes a *learned* boundary-offset head trained with an
exact-span/IoU/WER-surrogate loss. **That head does not exist and was not trained
here.** Everything below is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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


# ------------------------------------------------------------------------------
# v1 boundary shaping for the L4 resolver (spec §7.1)
# ------------------------------------------------------------------------------

# A leading list marker is document STRUCTURE, not entity text: "1.", "2)", "-",
# "•", "*". Spec §7.1 names swallowed list numbers and delimiters as the exact
# hard negatives the boundary ensemble must avoid.
#
# The numeric form REQUIRES trailing whitespace. Without that guard "14.43" reads
# as the list marker "14." followed by "43", and a decimal laboratory value gets
# silently shredded — which is exactly what the synthetic laboratory stress suite
# caught. A bullet may be followed immediately by its content.
LIST_MARKER_PATTERN = re.compile(r"^(?:\(?\d{1,2}[.)\]]\s+|[-–—•*+·▪]\s*)")


@dataclass(frozen=True, slots=True)
class BoundaryShape:
    """The outcome of shaping one span, with every action recorded for replay."""

    start: int
    end: int
    actions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return bool(self.actions)


def trim_span(
    original_text: str, start: int, end: int, *,
    trim_characters: str, max_trim_chars: int, trim_leading_list_markers: bool,
) -> BoundaryShape:
    """Bounded trim of a span's edges. Never empties the span, never expands it.

    Trimming is capped on each side independently, so a badly-formed proposal can
    shrink a little but can never be shaved down to an unrelated substring.
    """
    if end <= start:
        raise ValueError(f"cannot trim the empty span [{start}, {end})")
    actions: list[str] = []
    new_start, new_end = start, end

    if trim_leading_list_markers:
        match = LIST_MARKER_PATTERN.match(original_text[new_start:new_end])
        if match and match.end() > 0 and new_start + match.end() < new_end:
            if match.end() <= max_trim_chars:
                new_start += match.end()
                actions.append(f"trim_left_list_marker:{match.end()}")

    removed = 0
    while (new_start < new_end - 1 and removed < max_trim_chars
           and original_text[new_start] in trim_characters):
        new_start += 1
        removed += 1
    if removed:
        actions.append(f"trim_left:{removed}")

    removed = 0
    while (new_end - 1 > new_start and removed < max_trim_chars
           and original_text[new_end - 1] in trim_characters):
        new_end -= 1
        removed += 1
    if removed:
        actions.append(f"trim_right:{removed}")

    if new_end <= new_start:  # pragma: no cover - the loop guards forbid this
        raise ValueError("trimming produced an empty span")
    return BoundaryShape(new_start, new_end, tuple(actions))


def expand_to_competitor(
    start: int, end: int, competitors: tuple[tuple[int, int, float], ...], *,
    min_grammar_completeness: float, max_expand_chars: int,
) -> BoundaryShape:
    """Adopt a wider boundary another expert already proposed, if it is supported.

    ``competitors`` is ``(start, end, grammar_completeness)``. A competitor is
    eligible only when it strictly contains the current span, shares one edge
    with it (so the mention's identity is unchanged), carries enough grammar
    completeness, and adds no more than ``max_expand_chars`` characters.
    """
    best: tuple[int, int] | None = None
    best_width = end - start
    for other_start, other_end, completeness in competitors:
        if completeness < min_grammar_completeness:
            continue
        if other_start > start or other_end < end:
            continue
        if (other_start, other_end) == (start, end):
            continue
        if other_start != start and other_end != end:
            continue  # must share an edge; otherwise it is a different mention
        added = (start - other_start) + (other_end - end)
        if added > max_expand_chars:
            continue
        if other_end - other_start > best_width:
            best = (other_start, other_end)
            best_width = other_end - other_start
    if best is None:
        return BoundaryShape(start, end, ())
    return BoundaryShape(
        best[0], best[1],
        (f"expand_to_competitor:{best[1] - best[0] - (end - start)}",))


__all__ = [
    "LIST_MARKER_PATTERN",
    "BoundaryShape",
    "choose_boundary",
    "expand_to_competitor",
    "trim_span",
]
