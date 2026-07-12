"""Deterministic medication evidence scoring (NOT calibrated probabilities)."""

from __future__ import annotations

from .lexicon import MedicationLexicon
from .models import MedicationParse

_KIND_BONUS = {
    "name_only": 0.0,
    "name_strength": 0.04,
    "name_strength_form": 0.06,
    "name_strength_route": 0.06,
    "full": 0.08,
}


def score_medication(
    lex: MedicationLexicon,
    parse: MedicationParse,
    candidate_kind: str,
    candidate_start: int,
    candidate_end: int,
    *,
    in_med_section: bool,
    in_list: bool,
) -> float:
    """Score one boundary candidate. Component bonuses count ONLY components that
    fall inside the candidate span, so wider boundaries score differently."""
    s = lex.scoring
    total = (s.get("name_lexicon_match", 0.45) if parse.name_known
             else s.get("name_unknown_with_structure", 0.25))
    if in_med_section:
        total += s.get("section_prior", 0.20)
    if in_list:
        total += s.get("list_structure", 0.15)
    inside = [c for c in parse.components
              if c.start >= candidate_start and c.end <= candidate_end]
    roles = {c.role for c in inside}
    if roles & {"strength_value", "concentration"}:
        total += s.get("strength_present", 0.20)
    if "route" in roles:
        total += s.get("route_present", 0.10)
    if "frequency" in roles:
        total += s.get("frequency_present", 0.10)
    if "dose_form" in roles:
        total += s.get("dose_form_present", 0.10)
    total += min(0.20, s.get("completeness_per_component", 0.05) * (len(inside) - 1))
    total += _KIND_BONUS.get(candidate_kind, 0.0)
    return round(min(1.0, total), 6)


__all__ = ["score_medication"]
