"""E3 — the trained ViHealthBERT span/type expert (spec §6, expert E3).

Two deliberately separated halves:

* :mod:`.decoding` is **pure**: token-level multi-label predictions plus the
  aligned character offsets in, character-exact mention spans out. No Torch, no
  Transformers, no checkpoint — so every decoding rule is unit-testable.
* :mod:`.runtime` performs **forward-only** inference with the validated local
  checkpoint. It never trains, never calls ``backward``, never constructs an
  optimizer or scheduler, and never downloads base-model weights.

Like every other L3 expert, E3 emits proposals only. It never emits a final
organizer entity (spec §6).
"""

from __future__ import annotations

from .decoding import (
    NeuralSpan,
    decode_type_runs,
    neural_spans_to_mentions,
    validate_decoded_spans,
)

__all__ = [
    "NeuralSpan",
    "decode_type_runs",
    "neural_spans_to_mentions",
    "validate_decoded_spans",
]
