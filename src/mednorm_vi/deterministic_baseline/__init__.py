"""Phase 1B — deterministic medical structure specialists orchestration.

Ties L1 → Case Router → medication grammar → laboratory parser → proposal
merging → validation. Produces auditable PROPOSALS and internal relations only —
no final entities, assertions, ontology codes, or organizer JSON.
"""

from __future__ import annotations

from .models import Phase1BConfig, Phase1BResult
from .pipeline import run_phase1b
from .serialization import determinism_hash, to_debug_dict, to_json
from .validation import Phase1BValidationError, assert_valid

__all__ = [
    "Phase1BConfig",
    "Phase1BResult",
    "run_phase1b",
    "to_debug_dict",
    "to_json",
    "determinism_hash",
    "assert_valid",
    "Phase1BValidationError",
]
