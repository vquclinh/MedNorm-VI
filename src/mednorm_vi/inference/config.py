"""Pipeline configuration loading and readiness validation.

Loading is the enforcement point for retirement (Audit 0051). A profile that still
declares a retired expert's feature flag, or requires its checkpoint, is refused at
load time rather than silently normalized to ``False`` — a config carrying a dead
flag is a config someone can flip.

Audit 0056a extended that principle from *retired* keys to *inert* ones. Three
defects an independent review found here, all of the same shape — a declaration
nothing read:

1. **``full`` mode could report READY for an expert that cannot execute.**
   ``expected`` was built from ``MODE_EXPERTS["full"]``, which names only E1/E2/E3,
   so :func:`_expert_readiness` — the only function that asks whether an expert can
   actually run — was never consulted for an enabled E5, E6 or E7. Their sole check
   was ``Path.exists()`` on a role directory, and ``Path.exists()`` is ``True`` for
   an **empty** directory. Three ``mkdir``s bought a READY verdict for three experts
   the runner would then silently not run, recording nothing at all.
2. **``specialist_requires_checkpoints`` was parsed and never read.** Only
   ``full_requires_checkpoints`` reached :func:`evaluate_readiness`.
3. **Both L4 feature flags were inert.** The canonical runner called the
   deterministic L4 unconditionally and read no L4 flag, so ``enable_l4_learned_v2:
   true`` changed nothing and raised nothing.

All three are closed here. Expert identity now comes from one place —
:mod:`mednorm_vi.mention_factory.expert_spec` — and readiness cross-checks it
against the live registry, so a profile flag, a specification and an actual
registration can no longer drift apart without a named error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..governance.e4_retirement import (
    assert_e4_absent_from_flags,
    assert_no_e4_checkpoint_required,
)
from ..kb.competition.topk import DEFAULT_DECODING_PROFILE
from ..mention_factory.expert_spec import (
    EXPERT_FEATURE_FLAGS,
    EXPERT_SPEC_BY_FLAG,
    IMPLEMENTATION_ABSENT,
    IMPLEMENTATION_DETERMINISTIC,
    IMPLEMENTATION_REGISTERED,
    ExpertSpec,
    artifact_is_valid,
)

# --- L4 route -----------------------------------------------------------------
# The two L4 flags select a ROUTE. Exactly one must be chosen: L4 is not optional —
# nothing downstream of it can consume a lattice — so "neither" is a configuration
# error rather than a mode. Before Audit 0056a both flags were `false` in the shipped
# profile while the deterministic resolver ran anyway, which made the profile a
# false statement about its own behaviour.
L4_ROUTE_DETERMINISTIC_V1 = "deterministic_v1"
L4_ROUTE_LEARNED_V2 = "learned_v2"

L4_FLAG_DETERMINISTIC = "enable_l4_deterministic_v1"
L4_FLAG_LEARNED = "enable_l4_learned_v2"

DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "enable_e1_medication_grammar": True,
    "enable_e2_laboratory_parser": True,
    "enable_e3_vihealthbert": True,
    "enable_e5_xlmr_mrc": False,
    "enable_e6_gliner": False,
    "enable_e7_qwen_proposer": False,
    # TRUE because this is what the canonical runner has always executed. It was
    # `False` until Audit 0056a while the deterministic resolver ran regardless.
    L4_FLAG_DETERMINISTIC: True,
    L4_FLAG_LEARNED: False,
}

#: Checkpoint roles owned by something other than an L3 expert.
CHECKPOINT_ROLE_LEARNED_L4 = "resolution/learned_l4_v2"

#: Role -> feature flag, for every role an L3 expert owns. Derived from the one
#: canonical specification rather than restated, so the two cannot disagree.
CHECKPOINT_BY_FEATURE_FLAG: dict[str, str] = {
    spec.feature_flag: spec.checkpoint_role
    for spec in EXPERT_SPEC_BY_FLAG.values()
    if spec.checkpoint_role
}
FEATURE_FLAG_BY_CHECKPOINT_ROLE: dict[str, str] = {
    role: flag for flag, role in CHECKPOINT_BY_FEATURE_FLAG.items()
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    l1_config: str
    router_config: str
    medication_config: str
    laboratory_config: str
    # The one canonical L4 configuration (spec §7). Audit 0055 removed the separate
    # `resolver_config` field: it pointed at the retired Phase-1C-A resolver's config,
    # which was deleted with that resolver. A profile still declaring the key is
    # REFUSED rather than ignored — a dead key is a key somebody can set.
    l4_config: str = "configs/resolution/boundary_type_resolver_v1.yaml"
    # The learned-L4 declaration consulted only when the learned route is selected.
    learned_l4_config: str = "configs/resolution/learned_l4_v2.yaml"
    icd_index: str = ""
    rxnorm_index: str = ""
    # Governed decoding profile name (Audit 0060 §4). Resolved by
    # `kb.competition.topk.resolve_decoding_profile`; an unknown name is refused.
    decoding_profile: str = DEFAULT_DECODING_PROFILE
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
        if "resolver_config" in doc:
            raise ValueError(
                f"{path}: `resolver_config` was removed with the Phase-1C-A resolver "
                "(Audit 0055). Delete the key and use `l4_config` "
                "(configs/resolution/boundary_type_resolver_v1.yaml); its "
                "boundary.group_preference and overlap.abstain_on_conflict keys carry "
                "the migrated medication_boundary / test_result_boundary / "
                "abstain_on_conflict policies. Silently ignoring it would let a "
                "profile believe it had configured a resolver that no longer exists."
            )
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
                        spec["feature_flags"], profile=f"{profile_name}:{mode}"
                    )
        for key in ("full_requires_checkpoints", "specialist_requires_checkpoints"):
            assert_no_e4_checkpoint_required(
                [str(v) for v in doc.get(key, [])], profile=f"{profile_name}.{key}"
            )

        # An L4 route must be selected, and exactly one. Both flags now have real
        # semantics, so an ambiguous or empty selection is refused at load time
        # rather than resolved by whichever branch happens to run first.
        select_l4_route(feature_flags, profile=profile_name)

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
            l4_config=str(
                doc.get("l4_config", "configs/resolution/boundary_type_resolver_v1.yaml")
            ),
            learned_l4_config=str(
                doc.get("learned_l4_config", "configs/resolution/learned_l4_v2.yaml")
            ),
            icd_index=str(doc.get("icd_index", "")),
            rxnorm_index=str(doc.get("rxnorm_index", "")),
            decoding_profile=str(doc.get("decoding_profile", DEFAULT_DECODING_PROFILE)),
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


def select_l4_route(feature_flags: dict[str, bool], *, profile: str = "") -> str:
    """Which L4 route the flags select. Raises on an ambiguous or empty selection.

    L4 is mandatory — L5-L9 consume ``EntityHypothesis`` and nothing else produces
    it — so "neither route" is a configuration error, not a way to skip L4.
    """
    deterministic = bool(feature_flags.get(L4_FLAG_DETERMINISTIC, False))
    learned = bool(feature_flags.get(L4_FLAG_LEARNED, False))
    where = f"{profile}: " if profile else ""
    if deterministic and learned:
        raise ValueError(
            f"{where}both {L4_FLAG_DETERMINISTIC} and {L4_FLAG_LEARNED} are enabled, "
            "so the L4 route is ambiguous. Exactly one L4 route may be selected; "
            "there is one canonical L4 entry point and it resolves each lattice once"
        )
    if learned:
        return L4_ROUTE_LEARNED_V2
    if deterministic:
        return L4_ROUTE_DETERMINISTIC_V1
    raise ValueError(
        f"{where}no L4 route is selected: both {L4_FLAG_DETERMINISTIC} and "
        f"{L4_FLAG_LEARNED} are false. L4 is not optional — L5-L9 consume "
        "EntityHypothesis and no other stage produces it. Set "
        f"{L4_FLAG_DETERMINISTIC}: true for the shipped deterministic resolver"
    )


# Which experts each named mode is *defined* to run. A mode's identity is this
# set — not a flag file — so two modes can never quietly become the same thing.
MODE_EXPERTS: dict[str, tuple[str, ...]] = {
    "deterministic": ("enable_e1_medication_grammar", "enable_e2_laboratory_parser"),
    "specialist": (
        "enable_e1_medication_grammar",
        "enable_e2_laboratory_parser",
        "enable_e3_vihealthbert",
    ),
    # `full` runs whatever the profile enables. Its members are listed for the
    # mode-collapse check below; the set actually evaluated comes from
    # `expected_expert_flags`, which reads the SCOPED flags — that is the fix for
    # E5/E6/E7 having been invisible to readiness.
    "full": (
        "enable_e1_medication_grammar",
        "enable_e2_laboratory_parser",
        "enable_e3_vihealthbert",
    ),
}

READY = "READY"
DEGRADED_FALLBACK = "DEGRADED_FALLBACK"
NOT_READY = "NOT_READY"


def flags_for_mode(config: PipelineConfig, mode: str) -> dict[str, bool]:
    """The feature flags that apply in ``mode``.

    A mode is a *scope*, not just a label: `deterministic` must run E1+E2 even when
    the profile also enables E3, otherwise the two named modes would behave
    identically again — the exact defect Audit 0051 found. An expert is on only when
    the profile enables it **and** the mode's definition includes it.

    Non-expert flags (the L4 route) are never scoped away: L4 runs in every mode.
    """
    allowed = set(MODE_EXPERTS.get(mode, ()))
    scoped = dict(config.feature_flags)
    for flag in config.feature_flags:
        if flag in EXPERT_FEATURE_FLAGS:
            scoped[flag] = bool(config.feature_flags[flag]) and flag in allowed
    if mode == "full":
        # `full` additionally honours whatever else the profile turns on. Every such
        # expert is now readiness-checked (see `expected_expert_flags`).
        for flag, enabled in config.feature_flags.items():
            if enabled and flag in EXPERT_FEATURE_FLAGS:
                scoped[flag] = True
    return scoped


def expected_expert_flags(config: PipelineConfig, mode: str) -> tuple[str, ...]:
    """Every expert flag that is ON in ``mode``, in architecture order.

    This is the Audit-0056a correction. It reads the SCOPED flags rather than
    ``MODE_EXPERTS[mode]``, so an expert a profile enables cannot be enabled for the
    runner and simultaneously invisible to readiness.
    """
    scoped = flags_for_mode(config, mode)
    return tuple(flag for flag in EXPERT_FEATURE_FLAGS if scoped.get(flag, False))


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
    l4_route: str = ""

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
            "l4_route": self.l4_route,
        }


def _registered_flags() -> set[str]:
    """Feature flags the live L3 registry actually provides.

    Imported lazily so this module stays importable with no expert package and no
    heavy dependency present.
    """
    from ..mention_factory.registry import DEFAULT_REGISTRY

    return {registration.feature_flag for registration in DEFAULT_REGISTRY.ordered()}


def _expert_readiness(config: PipelineConfig, flag: str) -> tuple[bool, str, str]:
    """Whether the expert behind ``flag`` could run. Loads no model.

    Consults the canonical specification **and** the live registry, and refuses to
    let them drift: a spec that claims ``REGISTERED`` while the registry has no such
    expert is itself a defect and is reported as one, rather than being papered over
    by falling through to "not ready".
    """
    from ..mention_factory.registry import DEFAULT_REGISTRY

    spec = EXPERT_SPEC_BY_FLAG.get(flag)
    if spec is None:
        return False, f"no canonical expert specification declares {flag}", ""

    if spec.implementation == IMPLEMENTATION_DETERMINISTIC:
        # E1/E2 are owned by Phase 1B and reached through `run_phase1b`, not the
        # registry. They need no checkpoint and no registration.
        return True, "", ""

    if spec.implementation == IMPLEMENTATION_ABSENT:
        # Enabled but deliberately not implemented/registered. This is the blocker
        # that used to be invisible in `full` mode.
        return False, spec.blocker or "no registered expert provides this flag", ""

    if spec.implementation == IMPLEMENTATION_REGISTERED:
        if flag not in _registered_flags():
            return (
                False,
                (
                    f"{spec.expert_id} is declared REGISTERED by the canonical expert "
                    "specification but no registration provides its flag; the "
                    "specification and the L3 registry have drifted"
                ),
                "",
            )
        for registration in DEFAULT_REGISTRY.ordered():
            if registration.feature_flag != flag:
                continue
            expert = registration.factory(dict(config.expert_settings))
            ready, reason, path = expert.readiness()
            return ready, reason or "", path or ""

    return False, "no registered expert provides this flag", ""


def _artifact_readiness(config: PipelineConfig, spec: ExpertSpec) -> tuple[bool, str]:
    """Whether the declared checkpoint role for ``spec`` holds a usable artifact."""
    if not spec.requires_checkpoint:
        return True, ""
    path = Path(config.checkpoint_root) / spec.checkpoint_role
    return artifact_is_valid(path, spec)


def _declared_role_readiness(config: PipelineConfig, role: str) -> tuple[bool, str]:
    """Whether a role named in ``*_requires_checkpoints`` is satisfied.

    A role an L3 expert owns is resolved through **that expert**, because the
    adapter is the authority on where its asset actually lives: E3's checkpoint is
    the configured ``e3_checkpoint_path``, not a path under ``checkpoint_root``.
    Requiring the literal role directory would make `specialist` NOT_READY while the
    real, digest-verified E3 artifact was present and usable — a false negative is
    no more honest than a false positive.

    A role no expert owns (retrieval, reranker, calibration, LoRA) is checked as an
    artifact under ``checkpoint_root``.
    """
    flag = FEATURE_FLAG_BY_CHECKPOINT_ROLE.get(role)
    if flag is not None:
        spec = EXPERT_SPEC_BY_FLAG[flag]
        if spec.implementation == IMPLEMENTATION_ABSENT:
            return False, spec.blocker or f"{role} has no implementation"
        ready, reason, detail_path = _expert_readiness(config, flag)
        if not ready:
            return False, f"{reason}{f' ({detail_path})' if detail_path else ''}"
        return True, ""
    if role == CHECKPOINT_ROLE_LEARNED_L4:
        from ..resolution.learned_dispatch import (
            learned_l4_blockers,
            load_learned_l4_config,
        )

        try:
            learned = load_learned_l4_config(config.learned_l4_config)
        except Exception as exc:  # noqa: BLE001 - reported, never raised through readiness
            return False, str(exc)
        blockers = learned_l4_blockers(learned)
        return (not blockers), "; ".join(blockers)
    artifact_path = Path(config.checkpoint_root) / role
    if not artifact_path.exists():
        return False, f"required artifact is missing: {artifact_path}"
    if artifact_path.is_dir() and not any(
        entry.is_file() and entry.stat().st_size > 0 for entry in artifact_path.iterdir()
    ):
        return False, (
            f"artifact directory {artifact_path} is empty; an empty directory does not "
            "satisfy checkpoint readiness"
        )
    if artifact_path.is_file() and artifact_path.stat().st_size == 0:
        return False, f"required artifact is empty: {artifact_path}"
    return True, ""


def _l4_route_errors(config: PipelineConfig) -> tuple[str, tuple[str, ...]]:
    """The selected L4 route and any blockers preventing it from running."""
    try:
        route = select_l4_route(config.feature_flags)
    except ValueError as exc:
        return "", (f"l4_route_invalid:{exc}",)
    if route != L4_ROUTE_LEARNED_V2:
        return route, ()
    from ..resolution.learned_dispatch import (
        LearnedL4Unavailable,
        learned_l4_blockers,
        load_learned_l4_config,
    )

    try:
        learned = load_learned_l4_config(config.learned_l4_config)
    except LearnedL4Unavailable as exc:
        return route, (f"l4_learned_unavailable:{exc.reason}",)
    blockers = learned_l4_blockers(learned)
    return route, tuple(f"l4_learned_not_ready:{blocker}" for blocker in blockers)


def evaluate_readiness(config: PipelineConfig, *, mode: str) -> ReadinessReport:
    """Report exactly what ``mode`` will run, and why anything is missing."""
    if mode not in MODE_EXPERTS:
        return ReadinessReport(mode=mode, status=NOT_READY, errors=(f"unknown_mode:{mode}",))

    errors: list[str] = []
    degraded: list[str] = []

    for index_name, index_path in (
        ("icd_index", config.icd_index),
        ("rxnorm_index", config.rxnorm_index),
    ):
        if index_path and not Path(index_path).is_file():
            errors.append(f"missing_{index_name}:{index_path}")

    # L4 is mandatory in every mode, so its route is validated in every mode.
    route, route_errors = _l4_route_errors(config)
    errors.extend(route_errors)

    # Declared per-mode checkpoint requirements. `specialist_requires_checkpoints`
    # was parsed and never read until Audit 0056a; a declared requirement that
    # nothing enforces tells an operator they configured a guarantee they did not.
    declared_roles: tuple[str, ...] = ()
    if mode == "full":
        declared_roles = config.full_requires_checkpoints
    elif mode == "specialist":
        declared_roles = config.specialist_requires_checkpoints
    for role in declared_roles:
        ok, reason = _declared_role_readiness(config, role)
        if not ok:
            errors.append(f"missing_required_checkpoint:{role}:{reason}")

    # Every ENABLED expert with a declared checkpoint role must hold a real artifact
    # in `full` mode. An empty directory no longer satisfies this.
    scoped = flags_for_mode(config, mode)
    if mode == "full":
        for flag in EXPERT_FEATURE_FLAGS:
            if not scoped.get(flag, False):
                continue
            artifact_spec = EXPERT_SPEC_BY_FLAG[flag]
            ok, reason = _artifact_readiness(config, artifact_spec)
            if not ok:
                errors.append(f"enabled_feature_missing_checkpoint:{flag}:{reason}")

    expected = expected_expert_flags(config, mode)
    runnable: list[str] = []
    for flag in expected:
        ready, reason, detail_path = _expert_readiness(config, flag)
        if ready:
            runnable.append(flag)
            continue
        detail = f"{flag}:{reason}" + (f":{detail_path}" if detail_path else "")
        candidate_spec = EXPERT_SPEC_BY_FLAG.get(flag)
        implementation_absent = (
            candidate_spec is not None and candidate_spec.implementation == IMPLEMENTATION_ABSENT
        )
        if mode == "specialist" and config.allow_specialist_fallback and not implementation_absent:
            # Allowed to continue, but the profile must SAY it is weaker. An expert
            # with NO implementation is never merely "degraded": there is nothing to
            # fall back from, so it is an error in every mode.
            degraded.append(detail)
        else:
            errors.append(f"expert_not_ready:{detail}")

    # A mode whose expert set collapses onto a weaker mode's is degraded, even if
    # every individual check passed.
    if mode == "specialist" and not degraded:
        deterministic_flags = {
            flag for flag in MODE_EXPERTS["deterministic"] if scoped.get(flag, False)
        }
        if set(runnable) <= deterministic_flags:
            degraded.append(
                "specialist_runs_no_expert_beyond_deterministic:"
                "enable the trained mention expert or run deterministic mode"
            )

    if errors:
        status = NOT_READY
    elif degraded:
        status = DEGRADED_FALLBACK
    else:
        status = READY
    return ReadinessReport(
        mode=mode,
        status=status,
        errors=tuple(errors),
        degraded=tuple(degraded),
        expected_experts=expected,
        runnable_experts=tuple(runnable),
        l4_route=route,
    )


def validate_readiness(config: PipelineConfig, *, mode: str) -> tuple[str, ...]:
    """Blocking readiness errors for ``mode``. Empty means the mode may run.

    A `DEGRADED_FALLBACK` mode returns no errors — it is allowed to run — but
    :func:`evaluate_readiness` is what tells an operator it will do less than its
    name implies.
    """
    return evaluate_readiness(config, mode=mode).errors


__all__ = [
    "CHECKPOINT_BY_FEATURE_FLAG",
    "CHECKPOINT_ROLE_LEARNED_L4",
    "DEFAULT_FEATURE_FLAGS",
    "DEGRADED_FALLBACK",
    "FEATURE_FLAG_BY_CHECKPOINT_ROLE",
    "L4_FLAG_DETERMINISTIC",
    "L4_FLAG_LEARNED",
    "L4_ROUTE_DETERMINISTIC_V1",
    "L4_ROUTE_LEARNED_V2",
    "MODE_EXPERTS",
    "flags_for_mode",
    "expected_expert_flags",
    "NOT_READY",
    "READY",
    "PipelineConfig",
    "ReadinessReport",
    "evaluate_readiness",
    "select_l4_route",
    "validate_readiness",
]
