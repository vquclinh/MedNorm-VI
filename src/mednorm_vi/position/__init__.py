"""Submission-position policy framework (Phase 1C-A).

Internally, spans are always raw code points with ``original_text[s:e] == text``.
A :class:`PositionPolicy` re-expresses a raw span into a submission coordinate,
keeping reversible provenance. No policy is treated as the organizer's global
choice; the default is ``raw-codepoint-half-open``.
"""

from __future__ import annotations

from .encoders import (
    PositionEncodingError,
    build_forward_map,
    decode_position,
    encode_span,
)
from .forensics import ForensicsReport, Observation, PolicyStat, analyze
from .models import (
    PositionEncodingResult,
    PositionPolicy,
    PositionProvenance,
)
from .registry import PositionPolicyRegistry, load_position_registry
from .validation import validate_encoding

__all__ = [
    "PositionPolicy",
    "PositionProvenance",
    "PositionEncodingResult",
    "PositionEncodingError",
    "build_forward_map",
    "encode_span",
    "decode_position",
    "PositionPolicyRegistry",
    "load_position_registry",
    "validate_encoding",
    "Observation",
    "PolicyStat",
    "ForensicsReport",
    "analyze",
]
