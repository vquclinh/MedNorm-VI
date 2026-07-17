"""Optional teacher-generation contract validation.

Teacher APIs are not available to competition inference. This module validates
offline data-generation contracts only and never stores chain-of-thought.
"""

from __future__ import annotations

from .models import TeacherGenerationContract


class TeacherContractError(ValueError):
    """Raised when a teacher data-generation contract violates governance."""


def validate_teacher_contract(contract: TeacherGenerationContract) -> None:
    if contract.stores_chain_of_thought:
        raise TeacherContractError("teacher contracts must not store chain-of-thought")
    if contract.competition_inference_allowed:
        raise TeacherContractError("teacher APIs are forbidden in competition inference")


__all__ = ["TeacherContractError", "validate_teacher_contract"]
