"""Deployed-parameter guard for the 8B reasoner (sprint 0075).

The competition cap is a hard `< 9,000,000,000` across every learned component that runs at
inference. Qwen3-8B fits beside E3; it does **not** fit beside a second embedding stack and
reranker pair, and that combination must be impossible rather than merely discouraged.
"""

from __future__ import annotations

from typing import Any

PARAMETER_CAP = 9_000_000_000

E3_PARAMETERS = 135_002_117
QWEN3_8B = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_8B_PARAMETERS = 8_190_735_360
EMBEDDING_4B_PARAMETERS = 4_021_774_336
RERANKER_4B_PARAMETERS = 4_021_784_576


class ParameterBudgetExceeded(RuntimeError):
    """Raised when a configuration would deploy 9B or more parameters."""


def reasoner_total(*, with_semantic_s1: bool = False) -> int:
    total = E3_PARAMETERS + QWEN3_8B_PARAMETERS
    if with_semantic_s1:
        total += EMBEDDING_4B_PARAMETERS + RERANKER_4B_PARAMETERS
    return total


def assert_within_cap(*, with_semantic_s1: bool = False) -> int:
    """Fail closed. Returns the total when it is admissible."""
    total = reasoner_total(with_semantic_s1=with_semantic_s1)
    if total >= PARAMETER_CAP:
        raise ParameterBudgetExceeded(
            f"configuration deploys {total:,} parameters, cap is {PARAMETER_CAP:,}. "
            "Qwen3-8B and a 4B embedding/reranker pair cannot co-deploy; "
            "the 8B reasoner uses parameter-free lexical retrieval instead."
        )
    return total


def manifest(*, with_semantic_s1: bool = False) -> dict[str, Any]:
    return {
        "components": [
            {"model": "ViHealthBERT E3", "parameter_count": E3_PARAMETERS, "deployed": True},
            {
                "model": QWEN3_8B,
                "revision": QWEN3_8B_REVISION,
                "parameter_count": QWEN3_8B_PARAMETERS,
                "deployed": True,
            },
        ],
        "semantic_s1_co_deployed": with_semantic_s1,
        "total_deployed_parameters": reasoner_total(with_semantic_s1=with_semantic_s1),
        "parameter_cap": PARAMETER_CAP,
        "under_cap": reasoner_total(with_semantic_s1=with_semantic_s1) < PARAMETER_CAP,
        "training_performed": False,
    }


__all__ = [
    "E3_PARAMETERS",
    "EMBEDDING_4B_PARAMETERS",
    "PARAMETER_CAP",
    "QWEN3_8B",
    "QWEN3_8B_PARAMETERS",
    "QWEN3_8B_REVISION",
    "RERANKER_4B_PARAMETERS",
    "ParameterBudgetExceeded",
    "assert_within_cap",
    "manifest",
    "reasoner_total",
]
