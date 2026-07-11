"""Fixed vocabularies from the architecture spec (sections 1, 4.2, 7.2, 7.3).

These sets are authoritative for validation. They are intentionally small and
frozen; changing them is an architecture-sensitive change and must be justified
against ``docs/MedNorm-VI_Architecture.pdf``.
"""

from __future__ import annotations

from typing import Final

# The five entity categories (spec: "Five entity categories").
#
# These are INTERNAL English enums. Organizer-facing JSON must serialize the
# ``type`` field to the exact Vietnamese labels below (see ORGANIZER_LABEL_BY_TYPE).
ENTITY_TYPES: Final[frozenset[str]] = frozenset(
    {
        "MEDICATION",
        "DIAGNOSIS",
        "SYMPTOM",
        "TEST_NAME",
        "TEST_RESULT",
    }
)

# Internal enum -> exact organizer-facing Vietnamese label (confirmed by the
# organizer). Serialized submission JSON MUST use these strings for ``type``;
# English enum values are rejected by the organizer-facing validator.
ORGANIZER_LABEL_BY_TYPE: Final[dict[str, str]] = {
    "MEDICATION": "THUỐC",
    "DIAGNOSIS": "CHẨN_ĐOÁN",
    "SYMPTOM": "TRIỆU_CHỨNG",
    "TEST_NAME": "TÊN_XÉT_NGHIỆM",
    "TEST_RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
}

# Reverse map: organizer Vietnamese label -> internal enum.
TYPE_BY_ORGANIZER_LABEL: Final[dict[str, str]] = {
    label: internal for internal, label in ORGANIZER_LABEL_BY_TYPE.items()
}

# The set of valid organizer-facing ``type`` strings.
ORGANIZER_LABELS: Final[frozenset[str]] = frozenset(ORGANIZER_LABEL_BY_TYPE.values())

# The three assertion labels (spec section 1 / section 8). Multi-label.
ASSERTION_LABELS: Final[frozenset[str]] = frozenset(
    {
        "isNegated",
        "isHistorical",
        "isFamily",
    }
)

# Ontology used for the ``candidates`` field, per entity type (spec sections 9,
# 10, 7.3). Cross-links are forbidden: no ICD-10 on MEDICATION, no RxNorm on
# DIAGNOSIS. ``None`` means the type carries no candidate codes in round one.
CANDIDATE_ONTOLOGY_BY_TYPE: Final[dict[str, str | None]] = {
    "DIAGNOSIS": "ICD10",
    "MEDICATION": "RXNORM",
    "SYMPTOM": None,
    "TEST_NAME": None,
    "TEST_RESULT": None,
}

# Per-entity-type ORGANIZER-FACING output field policy (confirmed). The
# organizer serializer must emit ONLY these fields, in this order, for each type.
# Internal models may keep normalized always-present lists, but a field absent
# here must never reach final JSON for that type.
#   THUỐC / CHẨN_ĐOÁN      -> text, type, candidates, assertions, position
#   TRIỆU_CHỨNG            -> text, type, assertions, position
#   TÊN_XÉT_NGHIỆM /
#   KẾT_QUẢ_XÉT_NGHIỆM     -> text, type, position
ORGANIZER_FIELDS_BY_TYPE: Final[dict[str, tuple[str, ...]]] = {
    "MEDICATION": ("text", "type", "candidates", "assertions", "position"),
    "DIAGNOSIS": ("text", "type", "candidates", "assertions", "position"),
    "SYMPTOM": ("text", "type", "assertions", "position"),
    "TEST_NAME": ("text", "type", "position"),
    "TEST_RESULT": ("text", "type", "position"),
}

# Position convention for emitted spans: half-open interval [start, end).
# CONFIRMED by the organizer's published example (Python-style end-exclusive
# character offsets over Unicode code points): original_text[start:end] == text.
POSITION_IS_END_EXCLUSIVE: Final[bool] = True

# Qualification round document/file count (spec section 1).
N_SUBMISSION_DOCUMENTS: Final[int] = 100
