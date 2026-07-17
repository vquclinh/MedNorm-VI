"""Deterministic numeric summaries (no numpy; stable rounding)."""

from __future__ import annotations

from .models import Distribution


def _percentile(sorted_values: list[int], q: float) -> float:
    """Linear-interpolation percentile over a sorted list (deterministic)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return round(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac, 4)


def summarize(values: list[int]) -> Distribution:
    """Summarize a list of measurements deterministically."""
    if not values:
        return Distribution(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = sorted(values)
    total = sum(s)
    return Distribution(
        n=len(s), total=total, minimum=s[0], maximum=s[-1],
        mean=round(total / len(s), 4), median=_percentile(s, 0.5),
        p25=_percentile(s, 0.25), p75=_percentile(s, 0.75), p90=_percentile(s, 0.90))


def histogram(values: list[int], buckets: tuple[int, ...]) -> list[tuple[str, int]]:
    """Bucket values into ``<=b`` ranges plus a final ``>max`` bucket."""
    out: list[tuple[str, int]] = []
    remaining = sorted(values)
    lower = 0
    for b in buckets:
        count = sum(1 for v in remaining if lower < v <= b)
        out.append((f"{lower + 1}-{b}", count))
        lower = b
    out.append((f">{lower}", sum(1 for v in remaining if v > lower)))
    return out


__all__ = ["summarize", "histogram"]
