"""Parameter-budget tests (spec section 17)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.validator import (
    load_budget_config,
    total_adapter_params,
    total_base_params,
    validate_parameter_budget,
)
from mednorm_vi.validator.budget import installed_models

REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_YAML = REPO_ROOT / "configs" / "parameter_budget.yaml"


def _config(models: list[dict], cap: int = 9_000_000_000, soft: int | None = None) -> dict:
    budget = {"max_total_base_params": cap, "count_status": "in_profile"}
    if soft is not None:
        budget["soft_target_base_params"] = soft
    return {"budget": budget, "models": models}


def _model(
    name: str,
    base: int,
    adapter: int = 0,
    in_profile: bool = True,
    installed: bool = False,
) -> dict:
    return {
        "name": name,
        "role": "test",
        "base_params": base,
        "adapter_params": adapter,
        "in_profile": in_profile,
        "installed": installed,
    }


def test_under_budget_passes() -> None:
    cfg = _config([_model("a", 4_000_000_000), _model("b", 4_000_000_000)])
    assert validate_parameter_budget(cfg).ok


def test_over_9b_base_fails() -> None:
    cfg = _config([_model("a", 5_000_000_000), _model("b", 5_000_000_000)])
    result = validate_parameter_budget(cfg)
    assert not result.ok
    assert any(i.code == "budget.base_exceeds_cap" for i in result.errors)


def test_adapters_do_not_count_toward_base_budget() -> None:
    # 9B base exactly + 200M adapters. Adapters must be excluded, so this passes
    # even though the physical total (~9.2B) exceeds 9B.
    cfg = _config([_model("big", 9_000_000_000, adapter=200_000_000)])
    result = validate_parameter_budget(cfg)
    assert result.ok
    assert total_base_params(cfg) == 9_000_000_000
    assert total_adapter_params(cfg) == 200_000_000


def test_adapters_alone_cannot_push_base_over_cap() -> None:
    # Base is under cap; huge adapters must NOT trip the base check.
    cfg = _config([_model("m", 8_000_000_000, adapter=5_000_000_000)])
    assert validate_parameter_budget(cfg).ok


def test_models_outside_profile_are_excluded() -> None:
    cfg = _config(
        [_model("in", 8_000_000_000), _model("out", 8_000_000_000, in_profile=False)]
    )
    assert validate_parameter_budget(cfg).ok
    assert total_base_params(cfg) == 8_000_000_000


def test_missing_model_field_flagged() -> None:
    cfg = _config(
        [{"name": "x", "base_params": 1, "adapter_params": 0, "in_profile": True}]
    )
    result = validate_parameter_budget(cfg)  # missing "role" and "installed"
    assert not result.ok
    assert any(i.code == "budget.model_missing_field" for i in result.errors)


def test_no_model_installed_at_bootstrap() -> None:
    cfg = load_budget_config(BUDGET_YAML)
    assert installed_models(cfg) == []  # bootstrap: nothing downloaded/enabled


def test_shipped_budget_config_is_valid_and_within_cap() -> None:
    cfg = load_budget_config(BUDGET_YAML)
    result = validate_parameter_budget(cfg)
    assert result.ok, [i.message for i in result.errors]
    # The ACTIVE plan is spec section 17's Safe-8.85B stack LESS PhoBERT-large,
    # withdrawn with retired E4 (Audits 0048, 0051): 8,852,000,000 - 370,000,000.
    # The row is still in the manifest with `in_profile: false`, so the spec's own
    # figure stays visible and only the summed active plan changed.
    assert total_base_params(cfg) <= 9_000_000_000
    assert total_base_params(cfg) == 8_482_000_000
    assert total_base_params(cfg) == 8_852_000_000 - 370_000_000


def test_default_profile_has_no_soft_target_advisory() -> None:
    # The default planned profile (8.852B) sits under the 8.9B soft target, so it
    # must NOT emit the advisory that the old 8.850B target produced.
    cfg = load_budget_config(BUDGET_YAML)
    result = validate_parameter_budget(cfg)
    assert result.warnings == ()
    assert not any(i.code == "budget.over_soft_target" for i in result.issues)
