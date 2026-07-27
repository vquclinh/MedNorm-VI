"""Parameter-budget validation (spec section 17).

Enforces: the SUM of ``base_params`` across models in the ACTIVE PLANNED PROFILE
must not exceed ``budget.max_total_base_params`` (9,000,000,000). Adapter / LoRA
/ head parameters (``adapter_params``) are EXCLUDED — they never count toward
the 9B cap. A soft target may raise a WARNING for margin, never an error.

Planned vs installed
--------------------
A model's membership in the planned profile is a distinct concept from whether
its weights are actually downloaded/installed. The budget sums the *profile*
status field (``budget.count_status``, default ``in_profile``); ``installed``
records runtime reality. At the bootstrap stage no model is installed.

Accepts an already-parsed config dict (see ``configs/parameter_budget.yaml``).
``load_budget_config`` is a thin YAML loader for convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .results import Severity, ValidationIssue, ValidationResult

DEFAULT_COUNT_STATUS = "in_profile"


def _as_int(value: Any) -> int | None:
    """Coerce YAML ints (possibly with underscores) to int; None on failure."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.replace("_", "").strip())
        except ValueError:
            return None
    return None


def _count_status(config: dict[str, Any]) -> str:
    budget = config.get("budget", {}) or {}
    status = budget.get("count_status", DEFAULT_COUNT_STATUS)
    return status if isinstance(status, str) else DEFAULT_COUNT_STATUS


def total_base_params(config: dict[str, Any]) -> int:
    """Sum of ``base_params`` over profile models (adapters excluded)."""
    status = _count_status(config)
    total = 0
    for model in config.get("models", []) or []:
        if not model.get(status, False):
            continue
        base = _as_int(model.get("base_params"))
        if base is not None:
            total += base
    return total


def total_adapter_params(config: dict[str, Any]) -> int:
    """Sum of ``adapter_params`` over profile models (never in the base budget)."""
    status = _count_status(config)
    total = 0
    for model in config.get("models", []) or []:
        if not model.get(status, False):
            continue
        adapter = _as_int(model.get("adapter_params"))
        if adapter is not None:
            total += adapter
    return total


def installed_models(config: dict[str, Any]) -> list[str]:
    """Names of models marked ``installed: true`` (runtime reality)."""
    return [
        str(m.get("name"))
        for m in config.get("models", []) or []
        if m.get("installed", False)
    ]


def validate_parameter_budget(config: dict[str, Any]) -> ValidationResult:
    """Validate the base-model parameter budget from a parsed config dict."""
    issues: list[ValidationIssue] = []

    budget = config.get("budget", {})
    max_total = _as_int(budget.get("max_total_base_params"))
    if max_total is None:
        issues.append(
            ValidationIssue(
                code="budget.missing_cap",
                message="budget.max_total_base_params is missing or non-integer",
            )
        )
        return ValidationResult.from_issues(issues)

    status = _count_status(config)
    required_fields = ("name", "role", "base_params", "adapter_params", status, "installed")

    # Per-model field validation.
    models = config.get("models", []) or []
    for i, model in enumerate(models):
        name = model.get("name", f"<model#{i}>")
        for field_name in required_fields:
            if field_name not in model:
                issues.append(
                    ValidationIssue(
                        code="budget.model_missing_field",
                        message=f"model {name!r} missing field {field_name!r}",
                        context={"model": name, "field": field_name},
                    )
                )
        if _as_int(model.get("base_params")) is None:
            issues.append(
                ValidationIssue(
                    code="budget.bad_base_params",
                    message=f"model {name!r} has non-integer base_params "
                    f"{model.get('base_params')!r}",
                    context={"model": name},
                )
            )
        if _as_int(model.get("adapter_params")) is None:
            issues.append(
                ValidationIssue(
                    code="budget.bad_adapter_params",
                    message=f"model {name!r} has non-integer adapter_params "
                    f"{model.get('adapter_params')!r}",
                    context={"model": name},
                )
            )

    base_total = total_base_params(config)

    # The hard constraint: planned base params must not exceed the cap. Adapters
    # are deliberately NOT added here.
    if base_total > max_total:
        issues.append(
            ValidationIssue(
                code="budget.base_exceeds_cap",
                message=(
                    f"planned base params {base_total:,} exceeds cap {max_total:,} "
                    "(adapters excluded)"
                ),
                severity=Severity.ERROR,
                context={"base_total": base_total, "cap": max_total},
            )
        )

    # Soft target is advisory only.
    soft_target = _as_int(budget.get("soft_target_base_params"))
    if soft_target is not None and base_total > soft_target and base_total <= max_total:
        issues.append(
            ValidationIssue(
                code="budget.over_soft_target",
                message=(
                    f"planned base params {base_total:,} exceeds soft target "
                    f"{soft_target:,} (still within hard cap)"
                ),
                severity=Severity.WARNING,
                context={"base_total": base_total, "soft_target": soft_target},
            )
        )

    return ValidationResult.from_issues(issues)


def load_budget_config(path: str | Path) -> dict[str, Any]:
    """Load a parameter-budget YAML file into a dict (requires PyYAML)."""
    import yaml  # local import; stubs optional

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"budget config at {path} did not parse to a mapping")
    return data


__all__ = [
    "validate_parameter_budget",
    "total_base_params",
    "total_adapter_params",
    "installed_models",
    "load_budget_config",
    "DEFAULT_COUNT_STATUS",
]
