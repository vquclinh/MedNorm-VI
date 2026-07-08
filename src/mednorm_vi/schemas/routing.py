"""L2 Case Router contracts (spec section 5).

The router is multi-label: a ``RouteDecision`` for one segment may carry several
``RouteTag`` s (e.g. C1 + C4 + C6). Contracts only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RouteTag:
    """One case assignment for a segment, with the score and firing signals."""

    case: str  # "C1".."C7"
    score: float
    fired_signals: tuple[str, ...] = field(default_factory=tuple)
    applied_priors: dict[str, float] = field(default_factory=dict)
    lexicon_version: int = 0


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The full multi-label routing outcome for one segment."""

    segment_id: str
    tags: tuple[RouteTag, ...] = field(default_factory=tuple)

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(tag.case for tag in self.tags)

    def has_case(self, case: str) -> bool:
        return any(tag.case == case for tag in self.tags)

    def activated_modules(self, routes_config: dict[str, Any]) -> tuple[str, ...]:
        """Union of ``activated_modules`` for this segment's cases.

        ``routes_config`` is the parsed ``configs/routes.yaml`` mapping. This is
        a convenience over the declarative config; the router itself does not
        run modules.
        """
        modules: list[str] = []
        cases_cfg = routes_config.get("cases", {})
        for case in self.cases:
            for mod in cases_cfg.get(case, {}).get("activated_modules", []) or []:
                if mod not in modules:
                    modules.append(mod)
        return tuple(modules)


# Optional per-segment priors the router may attach (e.g. section-derived).
SegmentPriors = dict[str, float]
