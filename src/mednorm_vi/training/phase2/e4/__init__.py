"""The current E4 PhoBERT W2NER expert (Audit 0045).

One implementation, one notebook, one configuration family. The implementation
that produced the collapsed run audited in 0043/0044 was removed rather than
patched; Audits 0043 and 0044 remain as the record of why.

Layout::

    contracts.py            identity, pinned revision, weight format, checkpoint
                            schema v3, supervision scope, manifest/config builders
    alignment.py            atomic grid words and the PhoBERT projection, carried
                            forward verbatim (Audit 0043 proved it exact)
    recipes.py              the three candidate training contracts
    sampling.py             deterministic epoch order and positive-aware sampling
    training.py             accumulation, precision, collapse guard, resume
    gates.py                the four-stage fail-closed authorization chain
    runtime_io.py           Drive health, local materialization, bounded streaming
    progress.py             heartbeats, ETA, epoch lifecycle records
    alignment_diagnostic.py full-corpus alignment preflight

Nothing in this package trains, downloads a model, or opens internal_test.
"""

from __future__ import annotations

from .contracts import (
    E4_CHECKPOINT_SCHEMA_VERSION,
    E4_EXPERT_ID,
    E4_FULL_AUTHORIZATION,
    E4_GOVERNED_TRAIN_SHA256,
    E4_GOVERNED_VALIDATION_SHA256,
    E4_INPUT_CONTRACT_VERSION,
    E4_MODEL_ID,
    E4_PINNED_MODEL_REVISION,
    E4_STAGE_ID,
    E4_SUBSET_AUTHORIZATION,
    E4_SUPERVISED_TYPES,
    E4_TINY_AUTHORIZATION,
    E4_UNSUPERVISED_TYPES,
    E4ContractError,
    reject_superseded_checkpoint,
)
from .gates import (
    STAGE_FULL,
    STAGE_SUBSET,
    STAGE_TINY,
    GateArtifact,
    GateError,
    RecipeResult,
    ReproductionCheck,
    SubsetResult,
    assert_full_training_allowed,
    assert_real_reproduction,
    assert_stage_authorized,
    select_recipe,
)
from .recipes import (
    BALANCED_FOCAL,
    RECIPE_NAMES,
    REFERENCE_CE,
    REFERENCE_CE_RESAMPLED,
    BatchGlobalAccumulator,
    OptimizerGroups,
    Recipe,
    RecipeError,
    ScheduleConfig,
    all_recipes,
    build_recipe,
)
from .sampling import ExampleIndex, build_epoch_order, measure_order
from .training import (
    AccumulationPlan,
    CollapseVerdict,
    E4TrainingError,
    MixedPrecisionPolicy,
    SchedulePlan,
    TinyEpochSignal,
    TinyOverfitStopPolicy,
    ValidationSnapshot,
    evaluate_collapse_guard,
    plan_gradient_accumulation,
    plan_schedule,
    resolve_mixed_precision_policy,
)

__all__ = [
    "BALANCED_FOCAL",
    "E4_CHECKPOINT_SCHEMA_VERSION",
    "E4_EXPERT_ID",
    "E4_FULL_AUTHORIZATION",
    "E4_GOVERNED_TRAIN_SHA256",
    "E4_GOVERNED_VALIDATION_SHA256",
    "E4_INPUT_CONTRACT_VERSION",
    "E4_MODEL_ID",
    "E4_PINNED_MODEL_REVISION",
    "E4_STAGE_ID",
    "E4_SUBSET_AUTHORIZATION",
    "E4_SUPERVISED_TYPES",
    "E4_TINY_AUTHORIZATION",
    "E4_UNSUPERVISED_TYPES",
    "RECIPE_NAMES",
    "REFERENCE_CE",
    "REFERENCE_CE_RESAMPLED",
    "STAGE_FULL",
    "STAGE_SUBSET",
    "STAGE_TINY",
    "AccumulationPlan",
    "BatchGlobalAccumulator",
    "CollapseVerdict",
    "E4ContractError",
    "E4TrainingError",
    "ExampleIndex",
    "GateArtifact",
    "GateError",
    "MixedPrecisionPolicy",
    "OptimizerGroups",
    "Recipe",
    "RecipeError",
    "RecipeResult",
    "ReproductionCheck",
    "ScheduleConfig",
    "SchedulePlan",
    "SubsetResult",
    "TinyEpochSignal",
    "TinyOverfitStopPolicy",
    "ValidationSnapshot",
    "all_recipes",
    "assert_full_training_allowed",
    "assert_real_reproduction",
    "assert_stage_authorized",
    "build_epoch_order",
    "build_recipe",
    "evaluate_collapse_guard",
    "measure_order",
    "plan_gradient_accumulation",
    "plan_schedule",
    "reject_superseded_checkpoint",
    "resolve_mixed_precision_policy",
    "select_recipe",
]
