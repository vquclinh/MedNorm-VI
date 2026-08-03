"""OntoFusion: pretrained-only multi-view linking (milestone 0082).

A second linker beside GraphCENT 0080, not a replacement for it, so the three ablations stay
clean: 0080 = E3 + GraphCENT, 0081 = contextual + GraphCENT, 0082 = contextual + OntoFusion.

Nothing is trained and no model is added: the deployed stack is exactly the certified
GraphCENT stack, and sparse retrieval contributes no parameters.
"""

from .pipeline import VARIANTS, ConfidencePolicy, MentionRecord
from .retrieval import RetrievalPolicy
from .runtime import OntoFusionConfig, RunOutcome, run
from .sparse import SparseSettings

__all__ = [
    "VARIANTS",
    "ConfidencePolicy",
    "MentionRecord",
    "OntoFusionConfig",
    "RetrievalPolicy",
    "RunOutcome",
    "SparseSettings",
    "run",
]
