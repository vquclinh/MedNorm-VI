"""Deterministic entity-matching strategies for the provisional evaluator.

Strategies never match across organizer types. Choose one via the evaluator
config (``matching_strategy``).
"""

from __future__ import annotations

from ..models import EvaluationConfig
from .base import Matcher
from .exact_position import ExactPositionMatcher
from .exact_text_occurrence import ExactTextOccurrenceMatcher
from .min_cost_bipartite import MinCostBipartiteMatcher

STRATEGIES = ("exact-position", "exact-text-occurrence", "min-cost-bipartite")


def build_matcher(config: EvaluationConfig) -> Matcher:
    """Construct the matcher named by ``config.matching_strategy``."""
    strategy = config.matching_strategy
    if strategy == "exact-position":
        return ExactPositionMatcher()
    if strategy == "exact-text-occurrence":
        return ExactTextOccurrenceMatcher()
    if strategy == "min-cost-bipartite":
        return MinCostBipartiteMatcher(config)
    raise KeyError(f"unknown matching strategy {strategy!r}; choose from {list(STRATEGIES)}")


__all__ = [
    "Matcher",
    "ExactPositionMatcher",
    "ExactTextOccurrenceMatcher",
    "MinCostBipartiteMatcher",
    "build_matcher",
    "STRATEGIES",
]
