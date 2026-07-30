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

# Test-result kinds the policy can name, and the fallback ladder below.
RESULT_VALUE_ONLY = "value_only"
RESULT_VALUE_UNIT = "value_unit"
# `value_with_unit` is the spelling the retired Phase-1C-A config used; both are
# accepted so a migrated config cannot silently mean something else.
_RESULT_UNIT_ALIASES: frozenset[str] = frozenset({RESULT_VALUE_UNIT, "value_with_unit"})

# Rank bands for `preference_rank`. Exact policy match always wins; the bands keep
# the two fallback directions from ever interleaving.
_RANK_EXACT = 0
_RANK_AT_OR_BELOW_TARGET = 1
_RANK_ABOVE_TARGET = 100


def medication_kind_order(kind: str) -> int:
    """Narrow-to-wide position of a medication boundary kind (99 when unknown)."""
    return _MED_ORDER.get(kind, 99)


def preference_rank(entity_type: str, kind: str, preference: str) -> int:
    """Rank one boundary kind against a configured preference. Lower is better.

    This is the **single** implementation of the boundary-policy ladder, shared by
    the canonical lattice L4 (`resolver_v1._select_within_boundary_groups`) and,
    until Audit 0055 deleted it, by the retired Phase-1C-A resolver. Having one
    function is the point: two ladders that "mirror" each other in comments are two
    ladders that eventually disagree.

    The ladder reproduces the retired resolver's documented semantics exactly:

    * **medication** — the exact configured kind if the group offers it; otherwise
      the **widest kind at or below** the policy's width (a narrower span is a safe
      under-read); otherwise the **narrowest kind above** it.
    * **test_result** — the exact configured kind; otherwise ``value_only``, which
      is the conservative reading while the organizer's convention is unresolved;
      otherwise anything else.

    An empty ``preference`` means "no policy configured", and every kind ranks
    equally so the caller's utility/width tie-breaks decide alone.
    """
    if not preference:
        return _RANK_EXACT
    if entity_type in {"MEDICATION", "THUỐC"}:
        if kind == preference:
            return _RANK_EXACT
        target = medication_kind_order(preference)
        order = medication_kind_order(kind)
        if order <= target:
            # Widest at or below the target ranks best: target - order == 0 is the
            # target width itself, and each step narrower is one rank worse.
            return _RANK_AT_OR_BELOW_TARGET + (target - order)
        return _RANK_ABOVE_TARGET + order
    if entity_type in {"TEST_RESULT", "KẾT_QUẢ_XÉT_NGHIỆM"}:
        want = (RESULT_VALUE_UNIT if preference in _RESULT_UNIT_ALIASES
                else RESULT_VALUE_ONLY)
        if kind == want:
            return _RANK_EXACT
        if kind == RESULT_VALUE_ONLY:
            return _RANK_AT_OR_BELOW_TARGET
        return _RANK_ABOVE_TARGET
    # Any other type has no configured ladder; every kind is equally acceptable.
    return _RANK_EXACT


def preference_note(entity_type: str, kind: str, preference: str) -> str:
    """Human-readable record of why a kind won, for hypothesis provenance."""
    if not preference:
        return "no_boundary_policy"
    rank = preference_rank(entity_type, kind, preference)
    if rank == _RANK_EXACT:
        return f"policy={preference}:exact"
    if rank < _RANK_ABOVE_TARGET:
        return f"policy={preference}:fallback_widest_at_or_below_target:kind={kind}"
    return f"policy={preference}:fallback_narrowest_above_target:kind={kind}"


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
        return _choose_by_ladder(entity_type, group, medication_policy)
    if entity_type == "KẾT_QUẢ_XÉT_NGHIỆM":
        return _choose_by_ladder(entity_type, group, test_result_policy)
    # default: widest, deterministic
    chosen = max(group, key=lambda p: (width(p), -p.start, p.proposal_id))
    return chosen, considered, "widest fallback"


def _choose_by_ladder(
    entity_type: str, group: list[SpanProposal], policy: str,
) -> tuple[SpanProposal, tuple[str, ...], str]:
    """Pick one alternative using the shared :func:`preference_rank` ladder.

    Both L4 implementations route through the same ranking function, so the
    Phase-1C-A foundation and the canonical lattice resolver cannot drift apart.
    """
    considered = tuple(sorted({boundary_kind(p) for p in group}))
    chosen = min(group, key=lambda p: (
        preference_rank(entity_type, boundary_kind(p), policy),
        _by_deterministic(p)))
    return chosen, considered, preference_note(
        entity_type, boundary_kind(chosen), policy)


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
    "RESULT_VALUE_ONLY",
    "RESULT_VALUE_UNIT",
    "BoundaryShape",
    "choose_boundary",
    "expand_to_competitor",
    "medication_kind_order",
    "preference_note",
    "preference_rank",
    "trim_span",
]
