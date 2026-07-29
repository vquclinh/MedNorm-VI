"""L7 LLM cascade: local-only backends and constrained structured schemas.

Spec §19 names this module ``llm/  # critic/adjudicator, constrained schemas``.
It holds the model-independent half of the cascade contract (spec §12):

    backends           local-only lazy loaders; ``local_files_only=True`` always
    structured_output  constrained JSON decision parsing, never free text

What an LLM may do here is bounded by spec §12.1 and enforced by the callers, not
by this package: it may not change a span that is absent from the option set, and
it may not emit an ontology code outside the retrieved candidates
(``linking.snapshot.constrain_selection``). Chain-of-thought is never stored.

No model is imported at module import time, and no path in this package can
download one.
"""

from __future__ import annotations

from .backends import (
    BACKEND_CONTRACT_VERSION,
    LOCAL_FILES_ONLY,
    BackendError,
    CausalLMBackend,
    LocalCheckpointMissing,
    OpenTypeSpanBackend,
    SpanBackend,
    TextBackend,
    assert_no_external_api,
    require_local_directory,
)
from .structured_output import (
    STRUCTURED_OUTPUT_CONTRACT_VERSION,
    StructuredOutputError,
    extract_json_object,
    normalize_json_completion,
    require_json_object,
)

__all__ = [
    "BACKEND_CONTRACT_VERSION",
    "LOCAL_FILES_ONLY",
    "STRUCTURED_OUTPUT_CONTRACT_VERSION",
    "BackendError",
    "CausalLMBackend",
    "LocalCheckpointMissing",
    "OpenTypeSpanBackend",
    "SpanBackend",
    "StructuredOutputError",
    "TextBackend",
    "assert_no_external_api",
    "extract_json_object",
    "normalize_json_completion",
    "require_json_object",
    "require_local_directory",
]
