"""Contextual entity proposal and verification (milestone 0081).

Feeds a better-verified entity set into the unchanged GraphCENT 0080 linker. The deployed
model is the same Qwen3-8B GraphCENT already loads; nothing new is added to the budget.
"""

from .alignment import DocumentView, align
from .proposals import SOURCE_ALIAS, SOURCE_E3, SOURCE_QWEN, Proposal, ProposalPool
from .runtime import VARIANTS, ContextualConfig, DocumentOutcome, process_document

__all__ = [
    "SOURCE_ALIAS",
    "SOURCE_E3",
    "SOURCE_QWEN",
    "VARIANTS",
    "ContextualConfig",
    "DocumentOutcome",
    "DocumentView",
    "Proposal",
    "ProposalPool",
    "align",
    "process_document",
]
