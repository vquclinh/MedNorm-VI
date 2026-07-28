"""Generic Phase-2 training primitives shared by E4 and E5 (Audit 0045).

Gradient-accumulation accounting, mixed-precision resolution and the optimizer
signature are not specific to any one expert. They lived inside the removed E4
implementation, which meant E5 imported them from E4; when E4 was replaced they
moved here so no expert depends on another expert's training module.

Nothing here trains or imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRECISION_FP16 = "fp16"
PRECISION_BF16 = "bf16"
PRECISION_FP32 = "fp32"
SUPPORTED_PRECISION_MODES: tuple[str, ...] = (
    PRECISION_FP32, PRECISION_FP16, PRECISION_BF16)
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"


class TrainingContractError(ValueError):
    """Raised when a shared training-contract invariant is violated."""


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise TrainingContractError("denominator must be positive")
    return -(-numerator // denominator)


@dataclass(frozen=True, slots=True)
class AccumulationPlan:
    """Derived optimizer-step accounting. Every number here is arithmetic."""

    example_count: int
    micro_batch_size: int
    accumulation_steps: int
    epochs: int

    def __post_init__(self) -> None:
        if self.example_count <= 0:
            raise TrainingContractError("example_count must be positive")
        if self.micro_batch_size <= 0:
            raise TrainingContractError("micro_batch_size must be positive")
        if self.accumulation_steps <= 0:
            raise TrainingContractError("accumulation_steps must be positive")
        if self.epochs <= 0:
            raise TrainingContractError("epochs must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    @property
    def micro_batches_per_epoch(self) -> int:
        return _ceil_div(self.example_count, self.micro_batch_size)

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return _ceil_div(self.micro_batches_per_epoch, self.accumulation_steps)

    @property
    def expected_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_epoch * self.epochs

    @property
    def expected_backward_passes(self) -> int:
        return self.micro_batches_per_epoch * self.epochs

    @property
    def final_partial_group_size(self) -> int:
        """Microbatches in the last group; equals ``accumulation_steps`` if exact.

        Accounting only. The removed E4 implementation also used this to scale
        each microbatch's loss by its group size — a per-example normalization
        that is exactly the defect Audit 0044 measured, so no loss-scaling helper
        is provided here. Reduction happens once per effective batch, over valid
        cells (``e4.recipes.BatchGlobalAccumulator``).
        """
        remainder = self.micro_batches_per_epoch % self.accumulation_steps
        return remainder or self.accumulation_steps

    @property
    def has_partial_final_group(self) -> bool:
        return self.micro_batches_per_epoch % self.accumulation_steps != 0

    def group_size_for(self, micro_batch_index: int) -> int:
        """Microbatches in the accumulation group containing this index."""
        if micro_batch_index < 0 or micro_batch_index >= self.micro_batches_per_epoch:
            raise TrainingContractError("micro-batch index outside the epoch")
        group_index = micro_batch_index // self.accumulation_steps
        remaining = self.micro_batches_per_epoch - group_index * self.accumulation_steps
        return min(self.accumulation_steps, remaining)

    def is_optimizer_step_boundary(self, micro_batch_index: int) -> bool:
        if micro_batch_index < 0 or micro_batch_index >= self.micro_batches_per_epoch:
            raise TrainingContractError("micro-batch index outside the epoch")
        is_group_end = (micro_batch_index + 1) % self.accumulation_steps == 0
        is_epoch_end = micro_batch_index + 1 == self.micro_batches_per_epoch
        return is_group_end or is_epoch_end

    @property
    def signature(self) -> str:
        return (f"micro{self.micro_batch_size}-accum{self.accumulation_steps}"
                f"-effective{self.effective_batch_size}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "micro_batches_per_epoch": self.micro_batches_per_epoch,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "expected_optimizer_steps": self.expected_optimizer_steps,
            "expected_backward_passes": self.expected_backward_passes,
            "final_partial_group_size": self.final_partial_group_size,
            "has_partial_final_group": self.has_partial_final_group,
            "accumulation_signature": self.signature,
            "loss_reduction": "batch_global_valid_cell_mean",
        }


def plan_gradient_accumulation(
    example_count: int, *, micro_batch_size: int = 1,
    accumulation_steps: int, epochs: int,
) -> AccumulationPlan:
    return AccumulationPlan(
        example_count=example_count, micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps, epochs=epochs)


def assert_step_accounting(plan: AccumulationPlan, observed_steps: int) -> None:
    if observed_steps != plan.expected_optimizer_steps:
        raise TrainingContractError(
            f"optimizer step accounting mismatch: expected "
            f"{plan.expected_optimizer_steps}, observed {observed_steps}")


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    """Resolved autocast/GradScaler policy.

    T4 (compute capability 7.5) has no bf16, so the target runtime resolves to
    fp16 **with** a GradScaler. Support is read from the runtime capability, never
    from the device name.
    """

    mode: str
    device_type: str
    autocast_enabled: bool
    use_grad_scaler: bool

    @property
    def autocast_dtype_name(self) -> str:
        if not self.autocast_enabled:
            return ""
        return "torch.bfloat16" if self.mode == PRECISION_BF16 else "torch.float16"

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision_mode": self.mode,
            "precision_device_type": self.device_type,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": self.autocast_dtype_name,
            "use_grad_scaler": self.use_grad_scaler,
        }


def resolve_mixed_precision_policy(
    requested_mode: str, *, device_type: str, bf16_supported: bool = False,
) -> MixedPrecisionPolicy:
    if requested_mode not in SUPPORTED_PRECISION_MODES:
        raise TrainingContractError(
            f"unsupported precision mode {requested_mode!r}; "
            f"expected one of {SUPPORTED_PRECISION_MODES}")
    if device_type != DEVICE_CUDA:
        return MixedPrecisionPolicy(
            mode=PRECISION_FP32, device_type=device_type,
            autocast_enabled=False, use_grad_scaler=False)
    if requested_mode == PRECISION_FP32:
        return MixedPrecisionPolicy(
            mode=PRECISION_FP32, device_type=DEVICE_CUDA,
            autocast_enabled=False, use_grad_scaler=False)
    if requested_mode == PRECISION_BF16 and bf16_supported:
        return MixedPrecisionPolicy(
            mode=PRECISION_BF16, device_type=DEVICE_CUDA,
            autocast_enabled=True, use_grad_scaler=False)
    return MixedPrecisionPolicy(
        mode=PRECISION_FP16, device_type=DEVICE_CUDA,
        autocast_enabled=True, use_grad_scaler=True)


def assert_training_device(device_type: str) -> None:
    """Training requires CUDA. No particular GPU model is required."""
    if device_type != DEVICE_CUDA:
        raise TrainingContractError(
            "E4 training requires a CUDA device; the CPU path is valid only for "
            "contract tests")




def optimizer_signature(
    *, name: str, learning_rate: float, weight_decay: float, max_grad_norm: float,
) -> str:
    """Compact single-learning-rate optimizer identity.

    E5 uses one learning rate for its whole model. E4 does not — a pretrained
    backbone and a freshly initialized head must not share one, so E4 defines its
    signature over both parameter groups in ``e4.recipes.OptimizerGroups``.
    """
    return f"{name}-lr{learning_rate:g}-wd{weight_decay:g}-clip{max_grad_norm:g}"


__all__ = [
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "PRECISION_BF16",
    "PRECISION_FP16",
    "PRECISION_FP32",
    "SUPPORTED_PRECISION_MODES",
    "AccumulationPlan",
    "MixedPrecisionPolicy",
    "TrainingContractError",
    "assert_step_accounting",
    "assert_training_device",
    "optimizer_signature",
    "plan_gradient_accumulation",
    "resolve_mixed_precision_policy",
]
