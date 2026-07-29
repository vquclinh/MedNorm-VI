"""Registered L3 experts (spec §6).

Importing this package registers every implemented expert into
``mention_factory.registry.DEFAULT_REGISTRY``. The canonical runner imports it once
and then asks the registry what to run, so adding an expert means adding a module
here — never editing ``inference/pipeline.py``.

Registered today:

    E3  ViHealthBERT span/type   trained checkpoint present, runs when enabled

Not registered yet, and why:

    E5  XLM-R MRC-NER   its task head is randomly initialized; registering it would
                        make an untrained head reachable from a profile
    E6  GLiNER          no verified local checkpoint
    E7  causal-LM       no verified local checkpoint

    E4  PhoBERT-W2NER   RETIRED_FROM_ACTIVE_ARCHITECTURE — ``ExpertRegistration``
                        refuses its id outright
"""

from __future__ import annotations

from .e3_vihealthbert import (
    E3_CHECKPOINT_KEY,
    E3_CHECKPOINT_SHA256,
    E3_FEATURE_FLAG,
    E3_PINNED_MODEL_REVISION,
    E3_REGISTRATION,
    E3_ROLE,
    E3VietHealthBertExpert,
    build_e3_expert,
)

__all__ = [
    "E3_CHECKPOINT_KEY",
    "E3_CHECKPOINT_SHA256",
    "E3_FEATURE_FLAG",
    "E3_PINNED_MODEL_REVISION",
    "E3_REGISTRATION",
    "E3_ROLE",
    "E3VietHealthBertExpert",
    "build_e3_expert",
]
