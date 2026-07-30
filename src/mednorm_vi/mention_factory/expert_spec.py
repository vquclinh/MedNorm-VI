"""The canonical L3 expert specification (spec §6). One declaration per expert.

Audit 0056a exists because readiness, the profile flags, ``MODE_EXPERTS`` and the
registry each carried a *partial* view of what an expert is, and the four views
could disagree without anything noticing. The concrete defect: ``full`` mode built
its ``expected`` set from ``MODE_EXPERTS["full"]``, which lists only E1/E2/E3, so
:func:`~mednorm_vi.inference.config._expert_readiness` was never consulted for an
enabled E5, E6 or E7. Their only check was ``Path.exists()`` on a checkpoint role —
and ``Path.exists()`` is ``True`` for an **empty directory**. A profile could
therefore enable three experts that cannot execute, create three empty directories,
and be told ``READY`` while the runner silently ran none of them, recording nothing.

This module is the one declaration those four views now share. It states, for every
expert in ``AVAILABLE_EXPERTS``, what the expert *is* independently of whether it
happens to be registered:

* ``expert_id`` and ``feature_flag`` — the identity and the single switch;
* ``implementation`` — whether an executable adapter exists at all
  (:data:`IMPLEMENTATION_REGISTERED` / :data:`IMPLEMENTATION_ABSENT`), and if absent,
  the reason a future milestone must resolve;
* ``checkpoint_role`` — the role under ``checkpoint_root`` a profile must supply,
  empty for a deterministic expert;
* ``artifact_kind`` — what a valid artifact *is* (a file, or a directory carrying
  named entries), so an empty directory can no longer satisfy readiness;
* ``requires_model_revision`` / ``parameter_count_status`` — provenance obligations
  that the parameter ledger and the run manifest consume;
* ``blocker`` — the exact, quotable sentence readiness emits when the expert is
  enabled but cannot run.

Two rules this module enforces that the previous code could not:

**Never add a registration to make readiness pass.** E5's task head is randomly
initialized, E6 and E7 have no local weights. They are declared here with
``IMPLEMENTATION_ABSENT`` precisely so that enabling one is a *loud* NOT_READY with a
named blocker, rather than a quiet no-op. Registering them would make an untrained
head reachable from a profile, which spec §6 and the project's fail-closed rule both
forbid.

**A declaration is not an implementation.** ``expert_spec`` says what an expert
would need; :mod:`mednorm_vi.mention_factory.registry` says whether it can actually
run. Readiness consults both and refuses to let them drift — see
``test_profile_experts_cannot_drift_from_the_registry``.

E4 is absent by construction: its id is not in ``AVAILABLE_EXPERTS``, and
:func:`expert_spec_for_flag` raises for a retired flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..governance.e4_retirement import E4_FEATURE_FLAG
from ..lattice.models import (
    AVAILABLE_EXPERTS,
    EXPERT_GLINER,
    EXPERT_LABORATORY_PARSER,
    EXPERT_MEDICATION_GRAMMAR,
    EXPERT_QWEN_PROPOSER,
    EXPERT_VIHEALTHBERT,
    EXPERT_XLMR_MRC,
)

EXPERT_SPEC_CONTRACT_VERSION = "l3-expert-spec-v1"

# --- implementation state -----------------------------------------------------
# Whether an executable adapter exists AND is registered into the default registry.
IMPLEMENTATION_REGISTERED = "REGISTERED"
# A deterministic expert owned by Phase 1B. It needs no registry entry because the
# canonical runner reaches it through `run_phase1b`, not through the L3 registry.
IMPLEMENTATION_DETERMINISTIC = "DETERMINISTIC_PHASE1B"
# Declared, typed, and deliberately NOT registered. Enabling it must fail closed.
IMPLEMENTATION_ABSENT = "NOT_IMPLEMENTED_OR_NOT_REGISTERED"

# --- artifact kinds -----------------------------------------------------------
# No artifact is required (deterministic experts).
ARTIFACT_NONE = "NONE"
# A single checkpoint file must exist and be non-empty.
ARTIFACT_FILE = "FILE"
# A directory that must exist AND contain at least one of `required_entries`.
# An empty directory is NOT a valid artifact — that was the Audit-0056a defect.
ARTIFACT_DIRECTORY_WITH_ENTRIES = "DIRECTORY_WITH_ENTRIES"

# --- parameter-count provenance ----------------------------------------------
PARAM_COUNT_NOT_APPLICABLE = "NOT_APPLICABLE"
PARAM_COUNT_NOT_VERIFIED = "NOT_VERIFIED"


class ExpertSpecError(KeyError):
    """Raised when a flag or expert id has no canonical specification."""


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    """The canonical, registration-independent description of one L3 expert."""

    expert_id: str
    feature_flag: str
    role: str
    implementation: str
    checkpoint_role: str = ""
    artifact_kind: str = ARTIFACT_NONE
    # For ARTIFACT_DIRECTORY_WITH_ENTRIES: at least one of these names must be
    # present inside the directory. Names are matched as glob patterns so a
    # future checkpoint layout does not need a code change to be recognised.
    required_entries: tuple[str, ...] = field(default_factory=tuple)
    requires_model_revision: bool = False
    parameter_count_status: str = PARAM_COUNT_NOT_APPLICABLE
    # The exact sentence readiness emits when this expert is enabled but the
    # implementation is absent. Empty for an implemented expert.
    blocker: str = ""

    @property
    def is_deterministic(self) -> bool:
        return self.implementation == IMPLEMENTATION_DETERMINISTIC

    @property
    def expects_registration(self) -> bool:
        """True when the canonical runner reaches this expert via the L3 registry."""
        return self.implementation == IMPLEMENTATION_REGISTERED

    @property
    def requires_checkpoint(self) -> bool:
        return bool(self.checkpoint_role) and self.artifact_kind != ARTIFACT_NONE


#: The canonical specification, in architecture order. E4 is absent by construction.
EXPERT_SPECS: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        expert_id=EXPERT_MEDICATION_GRAMMAR,
        feature_flag="enable_e1_medication_grammar",
        role="l3_e1_medication_grammar",
        implementation=IMPLEMENTATION_DETERMINISTIC,
        parameter_count_status=PARAM_COUNT_NOT_APPLICABLE,
    ),
    ExpertSpec(
        expert_id=EXPERT_LABORATORY_PARSER,
        feature_flag="enable_e2_laboratory_parser",
        role="l3_e2_laboratory_parser",
        implementation=IMPLEMENTATION_DETERMINISTIC,
        parameter_count_status=PARAM_COUNT_NOT_APPLICABLE,
    ),
    ExpertSpec(
        expert_id=EXPERT_VIHEALTHBERT,
        feature_flag="enable_e3_vihealthbert",
        role="l3_e3_mention_span_type",
        implementation=IMPLEMENTATION_REGISTERED,
        checkpoint_role="mention/vihealthbert",
        # E3's artifact is the trained `best.pt`. The adapter resolves and digest-
        # verifies the real path itself; the role directory is the profile-level
        # declaration that the asset was supplied.
        artifact_kind=ARTIFACT_DIRECTORY_WITH_ENTRIES,
        required_entries=("*.pt", "*.safetensors", "*.json"),
        requires_model_revision=True,
        parameter_count_status=PARAM_COUNT_NOT_VERIFIED,
    ),
    ExpertSpec(
        expert_id=EXPERT_XLMR_MRC,
        feature_flag="enable_e5_xlmr_mrc",
        role="l3_e5_xlmr_mrc_ner",
        implementation=IMPLEMENTATION_ABSENT,
        checkpoint_role="mention/xlmr_mrc",
        artifact_kind=ARTIFACT_DIRECTORY_WITH_ENTRIES,
        required_entries=("*.pt", "*.safetensors", "*.bin"),
        requires_model_revision=True,
        parameter_count_status=PARAM_COUNT_NOT_VERIFIED,
        blocker=(
            "E5 has no registered expert: its MRC task head is randomly initialized "
            "and no trained checkpoint exists, so registering it would make an "
            "untrained head reachable from a profile (spec §6). Train S5/E5 and "
            "register an adapter before enabling this flag"
        ),
    ),
    ExpertSpec(
        expert_id=EXPERT_GLINER,
        feature_flag="enable_e6_gliner",
        role="l3_e6_gliner_open_type",
        implementation=IMPLEMENTATION_ABSENT,
        checkpoint_role="mention/gliner",
        artifact_kind=ARTIFACT_DIRECTORY_WITH_ENTRIES,
        required_entries=("*.pt", "*.safetensors", "*.bin"),
        requires_model_revision=True,
        parameter_count_status=PARAM_COUNT_NOT_VERIFIED,
        blocker=(
            "E6 has no registered expert: no verified local GLiNER checkpoint exists "
            "and `mention_factory/gliner.py` is a strict adapter that fails closed "
            "without one. Download and verify the approved GLiNER revision, then "
            "register an adapter before enabling this flag"
        ),
    ),
    ExpertSpec(
        expert_id=EXPERT_QWEN_PROPOSER,
        feature_flag="enable_e7_qwen_proposer",
        role="l3_e7_qwen_proposer",
        implementation=IMPLEMENTATION_ABSENT,
        checkpoint_role="mention/qwen3_1_7b_proposer",
        artifact_kind=ARTIFACT_DIRECTORY_WITH_ENTRIES,
        required_entries=("*.safetensors", "*.bin", "config.json"),
        requires_model_revision=True,
        parameter_count_status=PARAM_COUNT_NOT_VERIFIED,
        blocker=(
            "E7 has no registered expert: no local Qwen weights exist and the "
            "proposal-only runtime contract (constrained decoding plus offset "
            "re-anchoring) is not implemented. Supply approved local weights and "
            "register a proposal-only adapter before enabling this flag"
        ),
    ),
)

EXPERT_SPEC_BY_FLAG: Mapping[str, ExpertSpec] = {spec.feature_flag: spec for spec in EXPERT_SPECS}
EXPERT_SPEC_BY_ID: Mapping[str, ExpertSpec] = {spec.expert_id: spec for spec in EXPERT_SPECS}

#: Every flag this repository recognises as an L3 expert switch.
EXPERT_FEATURE_FLAGS: tuple[str, ...] = tuple(spec.feature_flag for spec in EXPERT_SPECS)


def expert_spec_for_flag(flag: str) -> ExpertSpec:
    """The canonical specification behind ``flag``. Raises for unknown/retired."""
    if flag == E4_FEATURE_FLAG:
        raise ExpertSpecError(
            f"{flag} is RETIRED_FROM_ACTIVE_ARCHITECTURE and has no specification"
        )
    try:
        return EXPERT_SPEC_BY_FLAG[flag]
    except KeyError as exc:
        raise ExpertSpecError(f"no canonical expert specification for {flag!r}") from exc


def artifact_is_valid(path: Path, spec: ExpertSpec) -> tuple[bool, str]:
    """Whether ``path`` is a usable artifact for ``spec``. Reads no weights.

    Returns ``(ok, reason)``. The reason is empty when ok.

    This is the check that closes the Audit-0056a hole: a bare ``Path.exists()``
    accepted an empty directory, so ``mkdir -p`` was enough to make a profile claim
    READY for an expert that could not execute. A directory artifact must now
    actually contain something that could be a checkpoint.
    """
    if spec.artifact_kind == ARTIFACT_NONE:
        return True, ""
    if not path.exists():
        return False, f"required artifact is missing: {path}"
    if spec.artifact_kind == ARTIFACT_FILE:
        if not path.is_file():
            return False, f"required artifact is not a file: {path}"
        if path.stat().st_size == 0:
            return False, f"required artifact is empty: {path}"
        return True, ""
    if spec.artifact_kind == ARTIFACT_DIRECTORY_WITH_ENTRIES:
        if path.is_file():
            # A single file where a role directory was declared is still a real
            # artifact, provided it carries bytes.
            if path.stat().st_size == 0:
                return False, f"required artifact is empty: {path}"
            return True, ""
        if not path.is_dir():
            return False, f"required artifact is neither a file nor a directory: {path}"
        for pattern in spec.required_entries or ("*",):
            for candidate in sorted(path.glob(pattern)):
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return True, ""
        wanted = ", ".join(spec.required_entries) or "any file"
        return False, (
            f"artifact directory {path} contains no non-empty entry matching "
            f"[{wanted}]; an empty directory does not satisfy checkpoint readiness"
        )
    return False, f"unknown artifact kind {spec.artifact_kind!r}"


def registered_flags(registrations: Sequence[object]) -> set[str]:
    """The feature flags the L3 registry actually provides.

    Takes the registrations rather than importing the registry, so this module
    stays importable with no expert package present.
    """
    return {
        str(getattr(registration, "feature_flag", ""))
        for registration in registrations
        if getattr(registration, "feature_flag", "")
    }


__all__ = [
    "ARTIFACT_DIRECTORY_WITH_ENTRIES",
    "ARTIFACT_FILE",
    "ARTIFACT_NONE",
    "EXPERT_FEATURE_FLAGS",
    "EXPERT_SPECS",
    "EXPERT_SPEC_BY_FLAG",
    "EXPERT_SPEC_BY_ID",
    "EXPERT_SPEC_CONTRACT_VERSION",
    "IMPLEMENTATION_ABSENT",
    "IMPLEMENTATION_DETERMINISTIC",
    "IMPLEMENTATION_REGISTERED",
    "PARAM_COUNT_NOT_APPLICABLE",
    "PARAM_COUNT_NOT_VERIFIED",
    "AVAILABLE_EXPERTS",
    "ExpertSpec",
    "ExpertSpecError",
    "artifact_is_valid",
    "expert_spec_for_flag",
    "registered_flags",
]
