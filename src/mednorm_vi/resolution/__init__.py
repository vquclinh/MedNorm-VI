"""L4 Boundary & Type Resolver (spec §7). **One** canonical entry point.

``resolve_lattice_to_hypotheses`` is it. Give it a ``SpanLattice`` and a
``ResolverV1Config`` and it returns a ``ResolutionResult`` of typed
``EntityHypothesis`` objects: one chosen boundary per logical mention, retained
alternatives, and accepted / rejected / unresolved status. It performs no ontology
linking, preserves exact offsets, keeps repeated occurrences distinct, and retains
``has_result`` evidence without requiring pairing. It never emits final entities.

Audit 0055 removed the second implementation. Until then this package also exported
``ResolverConfig`` / ``resolve`` — the Phase-1C-A foundation, which consumed
``mention_factory`` proposals directly and so could never see a lattice or a neural
expert. Its one unique behaviour, a configurable per-type boundary policy, now lives
in ``boundary.preference_rank`` and is reached through the config's
``boundary.group_preference``. **There is no compatibility alias**: importing the old
names fails loudly rather than resolving to something that behaves differently.

The learned slot (``learned_v2``) consumes the **same** ``SpanLattice`` contract and
stays disabled while no checkpoint exists.
"""

from __future__ import annotations

from .canonical import (
    CANONICAL_L4_VERSION,
    resolution_counts,
    resolve_lattice_to_hypotheses,
)
from .config_v1 import ResolverV1Config, load_resolver_v1_config
from .models import (
    BoundaryAlternative,
    BoundaryEvidence,
    EntityHypothesis,
    OverlapDecision,
    ResolutionResult,
    TypeEvidence,
)
from .serialization import determinism_hash, to_debug_dict, to_json
from .validation import validate_result

__all__ = [
    "CANONICAL_L4_VERSION",
    "BoundaryAlternative",
    "BoundaryEvidence",
    "EntityHypothesis",
    "OverlapDecision",
    "ResolutionResult",
    "ResolverV1Config",
    "TypeEvidence",
    "determinism_hash",
    "load_resolver_v1_config",
    "resolution_counts",
    "resolve_lattice_to_hypotheses",
    "to_debug_dict",
    "to_json",
    "validate_result",
]
