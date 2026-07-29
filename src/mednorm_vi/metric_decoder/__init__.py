"""L8 — deterministic evidence-ranked set selection (spec §13).

Named for what it does. The spec §13 metric-aware decoder — expected-Jaccard
candidate sets, expected-WER boundary utility, per-label assertion thresholds — needs
calibrated probabilities that do not exist yet, so it is present only as the
fail-closed :func:`decode_expected_jaccard_calibrated` slot.
"""

from .decoder import (
    CANDIDATE_SAFETY_BOUND,
    DECODER_VERSION,
    CalibratedDecoderUnavailable,
    CandidateDecision,
    DecodedEntity,
    decode_entities,
    decode_expected_jaccard_calibrated,
    decoder_status,
)

__all__ = [
    "CANDIDATE_SAFETY_BOUND",
    "DECODER_VERSION",
    "CalibratedDecoderUnavailable",
    "CandidateDecision",
    "DecodedEntity",
    "decode_entities",
    "decode_expected_jaccard_calibrated",
    "decoder_status",
]
