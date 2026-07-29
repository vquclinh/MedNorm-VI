"""Governance: parameter budgeting and retirement records (Audits 0042, 0051).

Keeps the research inventory separable from what a submission actually loads, so
the spec §17 9B limit constrains deployment without pruning the architecture, and
records which researched components have been withdrawn from it.
"""

from __future__ import annotations

from .e4_retirement import (
    RETIRED_FROM_ACTIVE_ARCHITECTURE,
    E4RetirementError,
    E4RetirementRecord,
    assert_e4_absent_from_flags,
    assert_e4_absent_from_ledger,
    assert_e4_absent_from_registry,
    assert_no_e4_checkpoint_required,
)
from .parameter_budget import (
    MAX_DEPLOYMENT_PARAMETERS,
    NON_DEPLOYABLE_STATUSES,
    STATUS_RETIRED_FROM_ACTIVE_ARCHITECTURE,
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

__all__ = [
    "MAX_DEPLOYMENT_PARAMETERS",
    "NON_DEPLOYABLE_STATUSES",
    "RETIRED_FROM_ACTIVE_ARCHITECTURE",
    "STATUS_RETIRED_FROM_ACTIVE_ARCHITECTURE",
    "CandidateModel",
    "CandidateRegistry",
    "DeploymentBudgetExceeded",
    "DeploymentBudgetReport",
    "E4RetirementError",
    "E4RetirementRecord",
    "ParameterBudgetError",
    "UnverifiedDeploymentComponent",
    "assert_e4_absent_from_flags",
    "assert_e4_absent_from_ledger",
    "assert_e4_absent_from_registry",
    "assert_no_e4_checkpoint_required",
    "compute_deployment_budget",
    "count_from_config",
    "count_parameters",
    "load_candidate_registry",
    "load_deployment_selection",
    "render_budget_report",
]
