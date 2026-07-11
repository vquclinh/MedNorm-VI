"""Structured diagnostic categories for error analysis.

Categories are stable string constants so downstream tooling can group findings
by document, entity type, routed case (C1-C7), section, provenance, and
experiment id. Diagnostic construction lives in ``scoring.py``; this module
defines the vocabulary and light helpers.
"""

from __future__ import annotations

from typing import Final

# --- Entity-level structural / matching ---
MISSING_ENTITY: Final = "missing_entity"
SPURIOUS_ENTITY: Final = "spurious_entity"
WRONG_TYPE: Final = "wrong_type"
EXACT_DUPLICATE_ENTITY: Final = "exact_duplicate_entity"
INVALID_POSITION: Final = "invalid_position"
INVALID_OUTPUT_SCHEMA: Final = "invalid_output_schema"
UNCERTAIN_MATCHING: Final = "uncertain_matching"

# --- Text ---
TEXT_BOUNDARY_ERROR: Final = "text_boundary_error"
TEXT_TOKEN_SUBSTITUTION: Final = "text_token_substitution"
TEXT_TOKEN_INSERTION: Final = "text_token_insertion"
TEXT_TOKEN_DELETION: Final = "text_token_deletion"

# --- Assertions ---
DUPLICATE_ASSERTION: Final = "duplicate_assertion"
ASSERTION_MISSING: Final = "assertion_missing"
ASSERTION_EXTRA: Final = "assertion_extra"

# --- Candidates ---
DUPLICATE_CANDIDATE: Final = "duplicate_candidate"
CANDIDATE_MISSING: Final = "candidate_missing"
CANDIDATE_EXTRA: Final = "candidate_extra"
CANDIDATE_WRONG_ONTOLOGY: Final = "candidate_wrong_ontology"
CANDIDATE_STRENGTH_CONFLICT: Final = "candidate_strength_conflict"
CANDIDATE_EMPTY_WHEN_REQUIRED: Final = "candidate_empty_when_required"

# --- Provenance warnings ---
GOLD_PROVENANCE_WARNING: Final = "gold_provenance_warning"
SILVER_LABEL_WARNING: Final = "silver_label_warning"

ALL_CATEGORIES: Final[tuple[str, ...]] = (
    MISSING_ENTITY,
    SPURIOUS_ENTITY,
    WRONG_TYPE,
    EXACT_DUPLICATE_ENTITY,
    DUPLICATE_ASSERTION,
    DUPLICATE_CANDIDATE,
    INVALID_POSITION,
    INVALID_OUTPUT_SCHEMA,
    TEXT_BOUNDARY_ERROR,
    TEXT_TOKEN_SUBSTITUTION,
    TEXT_TOKEN_INSERTION,
    TEXT_TOKEN_DELETION,
    ASSERTION_MISSING,
    ASSERTION_EXTRA,
    CANDIDATE_MISSING,
    CANDIDATE_EXTRA,
    CANDIDATE_WRONG_ONTOLOGY,
    CANDIDATE_STRENGTH_CONFLICT,
    CANDIDATE_EMPTY_WHEN_REQUIRED,
    UNCERTAIN_MATCHING,
    GOLD_PROVENANCE_WARNING,
    SILVER_LABEL_WARNING,
)


__all__ = [
    "ALL_CATEGORIES",
    "MISSING_ENTITY",
    "SPURIOUS_ENTITY",
    "WRONG_TYPE",
    "EXACT_DUPLICATE_ENTITY",
    "INVALID_POSITION",
    "INVALID_OUTPUT_SCHEMA",
    "UNCERTAIN_MATCHING",
    "TEXT_BOUNDARY_ERROR",
    "TEXT_TOKEN_SUBSTITUTION",
    "TEXT_TOKEN_INSERTION",
    "TEXT_TOKEN_DELETION",
    "DUPLICATE_ASSERTION",
    "ASSERTION_MISSING",
    "ASSERTION_EXTRA",
    "DUPLICATE_CANDIDATE",
    "CANDIDATE_MISSING",
    "CANDIDATE_EXTRA",
    "CANDIDATE_WRONG_ONTOLOGY",
    "CANDIDATE_STRENGTH_CONFLICT",
    "CANDIDATE_EMPTY_WHEN_REQUIRED",
    "GOLD_PROVENANCE_WARNING",
    "SILVER_LABEL_WARNING",
]
