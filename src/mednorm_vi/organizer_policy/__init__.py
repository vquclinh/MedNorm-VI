"""Organizer-policy registry (Phase 1C-A).

Encodes the organizer's confirmed facts and intentionally unresolved policies as
data. Confirmed facts and hypotheses are kept strictly separate; a hypothesis is
never presented as a confirmed rule.
"""

from __future__ import annotations

from .models import (
    ConfirmedFact,
    OrganizerPolicyRegistry,
    PolicyHypothesis,
    PolicyOption,
)
from .registry import load_organizer_registry

__all__ = [
    "ConfirmedFact",
    "PolicyOption",
    "PolicyHypothesis",
    "OrganizerPolicyRegistry",
    "load_organizer_registry",
]
