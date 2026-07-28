"""Governance: parameter budgeting and post-stage gating (Audit 0042).

Keeps the research inventory separable from what a submission actually loads, so
the spec §17 9B limit constrains deployment without pruning the architecture.
"""

from __future__ import annotations

from .parameter_budget import (
    MAX_DEPLOYMENT_PARAMETERS,
    CandidateModel,
    CandidateRegistry,
    DeploymentBudgetExceeded,
    DeploymentBudgetReport,
    ParameterBudgetError,
    UnverifiedDeploymentComponent,
    compute_deployment_budget,
    count_from_config,
    count_parameters,
    load_candidate_registry,
    load_deployment_selection,
    render_budget_report,
)
from .post_e4_gates import (
    BLOCKED_ON_E4_FULL_ARTIFACT,
    POST_E4_TASKS,
    E4GateStatus,
    evaluate_post_e4_gate,
    post_e4_task_table,
)

__all__ = [
    "BLOCKED_ON_E4_FULL_ARTIFACT",
    "MAX_DEPLOYMENT_PARAMETERS",
    "POST_E4_TASKS",
    "CandidateModel",
    "CandidateRegistry",
    "DeploymentBudgetExceeded",
    "DeploymentBudgetReport",
    "E4GateStatus",
    "ParameterBudgetError",
    "UnverifiedDeploymentComponent",
    "compute_deployment_budget",
    "count_from_config",
    "count_parameters",
    "evaluate_post_e4_gate",
    "load_candidate_registry",
    "load_deployment_selection",
    "post_e4_task_table",
    "render_budget_report",
]
