"""Deterministic per-case scoring from fired signals (C1-C5 + section priors).

Score = min(1.0, sum of fired-signal weights). Deterministic heuristic evidence,
not a probability. ``derived.*`` signals (C6/C7) are scored later by the router
once specialist proposals exist.
"""

from __future__ import annotations

from .models import CaseScore, RouteSignal
from .signals import CaseSpec, LineContext, RouterConfig, evaluate_detector


def score_case(spec: CaseSpec, ctx: LineContext, config: RouterConfig) -> CaseScore:
    fired: list[RouteSignal] = []
    total = 0.0
    for sig in spec.signals:
        if sig.detector == "structural.section_prior":
            priors = config.section_priors.get(ctx.section_category or "", {})
            if spec.case in priors:
                weight = priors[spec.case]
                fired.append(RouteSignal(spec.case, sig.name, "structural", weight,
                                         detail=ctx.section_category))
                total += weight
            continue
        ok, weight_override, detail = evaluate_detector(sig.detector, ctx, config)
        if ok:
            weight = weight_override if weight_override is not None else sig.weight
            fired.append(RouteSignal(spec.case, sig.name, sig.kind, weight, detail))
            total += weight
    return CaseScore(
        case=spec.case,
        score=min(1.0, total),
        fired_signals=tuple(fired),
        activated_specialists=spec.activated_specialists,
    )


# Structured cases handled by a specialist that a parent row already covers.
_STRUCTURED_CASES = frozenset({"C1", "C2"})


def score_line(ctx: LineContext, config: RouterConfig) -> list[CaseScore]:
    """Score every non-derived case for one unit (only fired cases returned)."""
    out: list[CaseScore] = []
    for spec in config.cases:
        if any(s.detector.startswith("derived.") for s in spec.signals):
            continue  # C6/C7 handled after specialists run
        # A narrative-only unit (sentence nested in a row) is not structured-routed.
        if ctx.narrative_only and spec.case in _STRUCTURED_CASES:
            continue
        cs = score_case(spec, ctx, config)
        if cs.fired_signals:
            out.append(cs)
    return out


__all__ = ["score_case", "score_line"]
