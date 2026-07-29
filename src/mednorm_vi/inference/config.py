"""Pipeline configuration loading and readiness validation.

Loading is the enforcement point for retirement (Audit 0051). A profile that still
declares a retired expert's feature flag, or requires its checkpoint, is refused at
load time rather than silently normalized to ``False`` — a config carrying a dead
flag is a config someone can flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..governance.e4_retirement import (
    assert_e4_absent_from_flags,
    assert_no_e4_checkpoint_required,
)

DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "enable_e1_medication_grammar": True,
    "enable_e2_laboratory_parser": True,
    "enable_e3_vihealthbert": True,
    "enable_e5_xlmr_mrc": False,
    "enable_e6_gliner": False,
    "enable_e7_qwen_proposer": False,
    "enable_l4_deterministic_v1": False,
    "enable_l4_learned_v2": False,
}

CHECKPOINT_BY_FEATURE_FLAG: dict[str, str] = {
    "enable_e3_vihealthbert": "mention/vihealthbert",
    "enable_e5_xlmr_mrc": "mention/xlmr_mrc",
    "enable_e6_gliner": "mention/gliner",
    "enable_e7_qwen_proposer": "mention/qwen3_1_7b_proposer",
    "enable_l4_learned_v2": "resolution/learned_l4_v2",
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    l1_config: str
    router_config: str
    medication_config: str
    laboratory_config: str
    resolver_config: str
    # The canonical L4 configuration (spec §7). `resolver_config` remains for the
    # legacy Phase-1C-A resolver, which is no longer on the canonical path.
    l4_config: str = "configs/resolution/boundary_type_resolver_v1.yaml"
    icd_index: str = ""
    rxnorm_index: str = ""
    checkpoint_root: str = "models/checkpoints/full_v1"
    full_requires_checkpoints: tuple[str, ...] = field(default_factory=tuple)
    specialist_requires_checkpoints: tuple[str, ...] = field(default_factory=tuple)
    allow_specialist_fallback: bool = True
    expected_documents: int = 100
    feature_flags: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_FEATURE_FLAGS))
    # Per-expert settings the L3 registry passes to each factory (checkpoint paths,
    # cache dirs, thresholds). Kept opaque here so adding an expert never requires a
    # new field on this dataclass.
    expert_settings: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> PipelineConfig:
        import yaml

        doc: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        profile_name = str(doc.get("name", Path(path).stem))
        feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        raw_flags = doc.get("feature_flags", {})
        if isinstance(raw_flags, dict):
            feature_flags.update({str(key): bool(value) for key, value in raw_flags.items()})

        # Retirement is enforced here, and in every nested per-mode profile, so a
        # dead flag cannot survive anywhere in the tree waiting to be flipped.
        assert_e4_absent_from_flags(feature_flags, profile=profile_name)
        raw_profiles = doc.get("profiles", {})
        if isinstance(raw_profiles, dict):
            for mode, spec in raw_profiles.items():
                if isinstance(spec, dict) and isinstance(spec.get("feature_flags"), dict):
                    assert_e4_absent_from_flags(
                        spec["feature_flags"], profile=f"{profile_name}:{mode}")
        for key in ("full_requires_checkpoints", "specialist_requires_checkpoints"):
            assert_no_e4_checkpoint_required(
                [str(v) for v in doc.get(key, [])], profile=f"{profile_name}.{key}")

        raw_expert_settings = doc.get("expert_settings", {})
        if not isinstance(raw_expert_settings, dict):
            raw_expert_settings = {}

        return PipelineConfig(
            l1_config=str(doc.get("l1_config", "configs/document_intelligence/base.yaml")),
            router_config=str(doc.get("router_config", "configs/case_router/base.yaml")),
            medication_config=str(
                doc.get("medication_config", "configs/medication/grammar_v1.yaml")
            ),
            laboratory_config=str(
                doc.get("laboratory_config", "configs/laboratory/parser_v1.yaml")
            ),
            resolver_config=str(doc.get("resolver_config", "configs/resolution/resolver_v1.yaml")),
            l4_config=str(doc.get(
                "l4_config", "configs/resolution/boundary_type_resolver_v1.yaml")),
            icd_index=str(doc.get("icd_index", "")),
            rxnorm_index=str(doc.get("rxnorm_index", "")),
            checkpoint_root=str(doc.get("checkpoint_root", "models/checkpoints/full_v1")),
            full_requires_checkpoints=tuple(
                str(v) for v in doc.get("full_requires_checkpoints", [])
            ),
            specialist_requires_checkpoints=tuple(
                str(v) for v in doc.get("specialist_requires_checkpoints", [])
            ),
            allow_specialist_fallback=bool(doc.get("allow_specialist_fallback", True)),
            expected_documents=int(doc.get("expected_documents", 100)),
            feature_flags=feature_flags,
            expert_settings=dict(raw_expert_settings),
        )


# Which experts each named mode is *defined* to run. A mode's identity is this
# set — not a flag file — so two modes can never quietly become the same thing.
MODE_EXPERTS: dict[str, tuple[str, ...]] = {
    "deterministic": ("enable_e1_medication_grammar", "enable_e2_laboratory_parser"),
    "specialist": (
        "enable_e1_medication_grammar", "enable_e2_laboratory_parser",
        "enable_e3_vihealthbert"),
    # `full` adds whatever the profile additionally enables; its required set is
    # declared by `full_requires_checkpoints`.
    "full": (
        "enable_e1_medication_grammar", "enable_e2_laboratory_parser",
        "enable_e3_vihealthbert"),
}

READY = "READY"
DEGRADED_FALLBACK = "DEGRADED_FALLBACK"
NOT_READY = "NOT_READY"


def flags_for_mode(
    config: PipelineConfig, mode: str
) -> dict[str, bool]:
    """The feature flags that apply in ``mode``.

    A mode is a *scope*, not just a label: `deterministic` must run E1+E2 even when
    the profile also enables E3, otherwise the two named modes would behave
    identically again — the exact defect Audit 0051 found. An expert is on only when
    the profile enables it **and** the mode's definition includes it.
    """
    allowed = set(MODE_EXPERTS.get(mode, ()))
    scoped = {flag: False for flag in config.feature_flags}
    for flag, enabled in config.feature_flags.items():
        scoped[flag] = bool(enabled) and flag in allowed
    if mode == "full":
        # `full` additionally honours whatever else the profile turns on, because
        # its required set is declared by `full_requires_checkpoints`.
        for flag, enabled in config.feature_flags.items():
            if enabled and flag.startswith("enable_e"):
                scoped[flag] = True
    return scoped


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """What a mode will ACTUALLY do, not what its name suggests.

    Audit 0051 found `specialist` reporting READY while behaving identically to
    `deterministic`: `allow_specialist_fallback` skipped every checkpoint check and
    the runner had no neural path at all. A profile that cannot differ from a
    weaker one must say so, which is what `DEGRADED_FALLBACK` is for.
    """

    mode: str
    status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    degraded: tuple[str, ...] = field(default_factory=tuple)
    expected_experts: tuple[str, ...] = field(default_factory=tuple)
    runnable_experts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when the mode may run at all (READY or explicitly degraded)."""
        return self.status in (READY, DEGRADED_FALLBACK)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "errors": list(self.errors),
            "degraded": list(self.degraded),
            "expected_experts": list(self.expected_experts),
            "runnable_experts": list(self.runnable_experts),
        }


def _expert_readiness(
    config: PipelineConfig, flag: str
) -> tuple[bool, str, str]:
    """Ask the registered expert behind ``flag`` whether it could run.

    Imported lazily so this module stays importable with no expert package and no
    heavy dependency present.
    """
    from ..mention_factory.registry import DEFAULT_REGISTRY

    for registration in DEFAULT_REGISTRY.ordered():
        if registration.feature_flag != flag:
            continue
        expert = registration.factory(dict(config.expert_settings))
        ready, reason, path = expert.readiness()
        return ready, reason or "", path or ""
    # E1/E2 are deterministic and owned by Phase 1B; they need no registration.
    if flag in ("enable_e1_medication_grammar", "enable_e2_laboratory_parser"):
        return True, "", ""
    return False, "no registered expert provides this flag", ""


def evaluate_readiness(config: PipelineConfig, *, mode: str) -> ReadinessReport:
    """Report exactly what ``mode`` will run, and why anything is missing."""
    if mode not in MODE_EXPERTS:
        return ReadinessReport(
            mode=mode, status=NOT_READY, errors=(f"unknown_mode:{mode}",))

    errors: list[str] = []
    degraded: list[str] = []

    for index_name, index_path in (
        ("icd_index", config.icd_index),
        ("rxnorm_index", config.rxnorm_index),
    ):
        if index_path and not Path(index_path).is_file():
            errors.append(f"missing_{index_name}:{index_path}")

    # `full` declares its required checkpoints explicitly and fails closed on any
    # of them. This is unchanged: a submission profile may not run partially.
    if mode == "full":
        for rel in config.full_requires_checkpoints:
            path = Path(config.checkpoint_root) / rel
            if not path.exists():
                errors.append(f"missing_checkpoint:{path}")
        for flag, rel in CHECKPOINT_BY_FEATURE_FLAG.items():
            if not config.feature_flags.get(flag, False):
                continue
            path = Path(config.checkpoint_root) / rel
            if not path.exists():
                errors.append(f"enabled_feature_missing_checkpoint:{flag}:{path}")

    scoped = flags_for_mode(config, mode)
    expected = tuple(
        flag for flag in MODE_EXPERTS[mode] if scoped.get(flag, False))
    runnable: list[str] = []
    for flag in expected:
        ready, reason, detail_path = _expert_readiness(config, flag)
        if ready:
            runnable.append(flag)
            continue
        detail = f"{flag}:{reason}" + (f":{detail_path}" if detail_path else "")
        if mode == "specialist" and config.allow_specialist_fallback:
            # Allowed to continue, but the profile must SAY it is weaker.
            degraded.append(detail)
        else:
            errors.append(f"expert_not_ready:{detail}")

    # A mode whose expert set collapses onto a weaker mode's is degraded, even if
    # every individual check passed.
    if mode == "specialist" and not degraded:
        deterministic_flags = {
            flag for flag in MODE_EXPERTS["deterministic"]
            if scoped.get(flag, False)}
        if set(runnable) <= deterministic_flags:
            degraded.append(
                "specialist_runs_no_expert_beyond_deterministic:"
                "enable the trained mention expert or run deterministic mode")

    if errors:
        status = NOT_READY
    elif degraded:
        status = DEGRADED_FALLBACK
    else:
        status = READY
    return ReadinessReport(
        mode=mode, status=status, errors=tuple(errors), degraded=tuple(degraded),
        expected_experts=expected, runnable_experts=tuple(runnable))


def validate_readiness(config: PipelineConfig, *, mode: str) -> tuple[str, ...]:
    """Blocking readiness errors for ``mode``. Empty means the mode may run.

    A `DEGRADED_FALLBACK` mode returns no errors — it is allowed to run — but
    :func:`evaluate_readiness` is what tells an operator it will do less than its
    name implies.
    """
    return evaluate_readiness(config, mode=mode).errors


__all__ = [
    "CHECKPOINT_BY_FEATURE_FLAG",
    "DEFAULT_FEATURE_FLAGS",
    "DEGRADED_FALLBACK",
    "MODE_EXPERTS",
    "flags_for_mode",
    "NOT_READY",
    "READY",
    "PipelineConfig",
    "ReadinessReport",
    "evaluate_readiness",
    "validate_readiness",
]
