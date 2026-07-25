"""Two-phase Colab dependency bootstrap for the S1 smoke (Audit 0023).

The S1 smoke previously installed packages in three separate cells and imported
``torch``/``transformers`` in the **same cell** as an install, with no kernel
restart. When dependency resolution rewrote NumPy on disk, the already-running
process then loaded a NumPy whose C struct layout no longer matched the
extensions compiled against the baseline, producing::

    ValueError: numpy.dtype size changed, may indicate binary incompatibility.
                Expected 96 from C header, got 88 from PyObject

This module holds the **pure, testable** decision logic for the fix: which action
the notebook should take, what pip constraints protect the inherited stack, what
the install command is, and whether the environment/readiness contracts hold.

Nothing here imports NumPy, Torch, or any scientific package, and nothing here
mutates the environment — the notebook performs the side effects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

# Bootstrap actions.
INSTALL_AND_RESTART = "INSTALL_AND_RESTART"
PROCEED = "PROCEED"

# Package specifiers that must never appear in an install command.
FORBIDDEN_INSTALL_TOKENS = ("--force-reinstall", "--upgrade")
PROTECTED_PACKAGES = ("numpy", "torch", "torchvision", "torchaudio")

# Every field a marker must carry AND match before installation may be skipped.
# A marker is a claim about a specific runtime + a specific tracked contract; a
# version string alone is far too weak to prove the environment still matches.
REQUIRED_MARKER_FIELDS = (
    "install_completed",
    "dependency_contract_version",
    "dependency_contract_sha256",
    "python_major_minor",
    "protected_baseline_versions",
    "install_requirement_hash",
)


class DependencyContractError(ValueError):
    """Raised when the dependency contract is violated or malformed."""


@dataclass(frozen=True, slots=True)
class DependencyContract:
    contract_version: str
    inherited_from_runtime: tuple[str, ...]
    install_specifiers: tuple[str, ...]
    forbidden_operations: tuple[str, ...]
    marker_path: str
    expected_passes: int
    not_installed: tuple[str, ...] = field(default_factory=tuple)
    # SHA-256 of the exact contract file bytes; changes whenever the tracked
    # contract changes in any way, forcing a fresh install+restart.
    contract_sha256: str = ""

    @property
    def install_requirement_hash(self) -> str:
        """Deterministic hash of the normalized requirements + protected policy."""
        payload = {
            "install": sorted(self.install_specifiers),
            "protected": sorted(PROTECTED_PACKAGES),
            "inherited": sorted(self.inherited_from_runtime),
            "not_installed": sorted(self.not_installed),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MarkerFingerprint:
    """The full identity a bootstrap marker must prove before skipping install."""

    dependency_contract_version: str
    dependency_contract_sha256: str
    python_major_minor: str
    protected_baseline_versions: dict[str, str]
    install_requirement_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dependency_contract_version": self.dependency_contract_version,
            "dependency_contract_sha256": self.dependency_contract_sha256,
            "python_major_minor": self.python_major_minor,
            "protected_baseline_versions": dict(self.protected_baseline_versions),
            "install_requirement_hash": self.install_requirement_hash,
        }


def compute_contract_sha256(path: str | Path) -> str:
    """SHA-256 over the exact bytes of the tracked dependency contract file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_marker_fingerprint(
    contract: DependencyContract, python_major_minor: str,
    baseline_versions: Mapping[str, str],
) -> MarkerFingerprint:
    """Fingerprint binding a marker to this contract AND this runtime."""
    protected = {
        name: str(baseline_versions.get(name, "") or "")
        for name in PROTECTED_PACKAGES
        if str(baseline_versions.get(name, "") or "")
    }
    return MarkerFingerprint(
        dependency_contract_version=contract.contract_version,
        dependency_contract_sha256=contract.contract_sha256,
        python_major_minor=str(python_major_minor),
        protected_baseline_versions=protected,
        install_requirement_hash=contract.install_requirement_hash,
    )


def load_dependency_contract(path: str | Path) -> DependencyContract:
    """Load the tracked S1 Colab dependency contract."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise DependencyContractError("dependency contract must be a mapping")
    version = str(doc.get("dependency_contract_version", ""))
    if not version:
        raise DependencyContractError("missing dependency_contract_version")
    install = doc.get("install") or []
    specifiers: list[str] = []
    for entry in install:
        specifier = str(entry.get("specifier", "")).strip()
        if not specifier:
            raise DependencyContractError(f"install entry missing specifier: {entry}")
        specifiers.append(specifier)
    if not specifiers:
        raise DependencyContractError("dependency contract installs nothing")
    restart = doc.get("restart") or {}
    return DependencyContract(
        contract_version=version,
        inherited_from_runtime=tuple(str(v) for v in doc.get("inherited_from_runtime", [])),
        install_specifiers=tuple(specifiers),
        forbidden_operations=tuple(str(v) for v in doc.get("forbidden_operations", [])),
        marker_path=str(restart.get("marker_path", "/content/.mednorm_s1_bootstrap.json")),
        expected_passes=int(restart.get("expected_passes", 2)),
        not_installed=tuple(str(e.get("name", "")) for e in (doc.get("not_installed") or [])),
        contract_sha256=compute_contract_sha256(path),
    )


def marker_mismatches(
    marker: Mapping[str, Any] | None, fingerprint: MarkerFingerprint,
) -> list[str]:
    """Reasons this marker cannot be trusted; empty means it fully matches.

    A marker is only evidence that *some* install happened. It is accepted only
    when it proves the SAME tracked contract bytes, the SAME Python major.minor,
    the SAME protected baseline versions, and the SAME normalized requirement set.
    """
    if not marker:
        return ["missing marker"]
    problems: list[str] = []
    for required in REQUIRED_MARKER_FIELDS:
        if required not in marker:
            problems.append(f"missing field: {required}")
    if problems:
        return problems                                   # legacy/incomplete marker
    if not marker.get("install_completed", False):
        problems.append("install_completed is false")
    expected = fingerprint.as_dict()
    for key, want in expected.items():
        got = marker.get(key)
        if key == "protected_baseline_versions":
            got = {str(k): str(v) for k, v in dict(got or {}).items()}
            want = {str(k): str(v) for k, v in dict(want).items()}
        if got != want:
            problems.append(f"{key} mismatch")
    return problems


def decide_bootstrap_action(
    marker: Mapping[str, Any] | None, fingerprint: MarkerFingerprint,
) -> str:
    """Decide whether to install+restart or proceed.

    Returns :data:`PROCEED` only when the marker matches the fingerprint in
    **every** field. Any missing, legacy, stale, or mismatched marker triggers
    exactly one install+restart cycle. Because the marker is written with the
    current fingerprint immediately before the restart, an exactly matching marker
    can never cause a second restart (no restart loop).
    """
    return PROCEED if not marker_mismatches(marker, fingerprint) else INSTALL_AND_RESTART


def build_pip_constraints(baseline_versions: Mapping[str, str]) -> list[str]:
    """Pin the inherited stack to the versions the runtime already provides.

    This is the key protection: rather than blindly pinning ``numpy<2``, the
    detected baseline is frozen so pip cannot silently move NumPy/Torch. An
    incompatible request then fails loudly instead of corrupting the ABI.
    """
    lines: list[str] = []
    for package in PROTECTED_PACKAGES:
        version = str(baseline_versions.get(package, "") or "").strip()
        if version:
            lines.append(f"{package}=={version}")
    if not lines:
        raise DependencyContractError(
            "no baseline versions detected for the protected packages; refusing to "
            "install without ABI protection")
    return lines


def build_install_command(
    contract: DependencyContract, constraint_file: str, python_executable: str = "python",
) -> list[str]:
    """One consolidated, constrained install transaction (no upgrades/reinstalls)."""
    return [
        python_executable, "-m", "pip", "install", "-q",
        "--constraint", str(constraint_file),
        *contract.install_specifiers,
    ]


def validate_install_command(command: Sequence[str]) -> None:
    """Reject contract-violating install commands."""
    tokens = [str(t) for t in command]
    for forbidden in FORBIDDEN_INSTALL_TOKENS:
        if forbidden in tokens:
            raise DependencyContractError(f"forbidden pip option: {forbidden}")
    if "--constraint" not in tokens:
        raise DependencyContractError("install command must use a constraints file")
    for token in tokens:
        bare = token.split("==")[0].split(">=")[0].split("<")[0].strip()
        if bare in PROTECTED_PACKAGES:
            raise DependencyContractError(
                f"{bare} is inherited from the runtime and must not be installed")


def validate_abi_report(report: Mapping[str, Any]) -> list[str]:
    """Return ABI problems; an empty list means the environment is healthy."""
    problems: list[str] = []
    if not report.get("numpy_imported", False):
        problems.append("numpy import failed")
    if not report.get("numpy_random_imported", False):
        problems.append("numpy.random.RandomState import failed")
    if not report.get("torch_imported", False):
        problems.append("torch import failed")
    if not report.get("adamw_constructed", False):
        problems.append("torch.optim.AdamW construction failed")
    if int(report.get("numpy_distribution_count", 0) or 0) > 1:
        problems.append("multiple numpy distributions installed")
    if not report.get("pip_check_passed", False):
        problems.append("pip check reported broken requirements")
    numpy_path = str(report.get("numpy_path", "") or "")
    for shadow in ("/content/drive", "/MedNorm-VI/src", "/content/MedNorm-VI"):
        if shadow in numpy_path:
            problems.append(f"numpy imported from an unexpected location: {numpy_path}")
            break
    return problems


def _count(evidence: Mapping[str, Any], key: str, missing: int) -> int:
    """Read an integer counter, treating a missing/None value as ``missing``.

    Note: a legitimate ``0`` must not be replaced by the default, so ``or`` is not
    used here.
    """
    value = evidence.get(key)
    if value is None:
        return missing
    return int(value)


def evaluate_full_training_readiness(evidence: Mapping[str, Any]) -> bool:
    """Full-training readiness requires EVERY environment and smoke condition."""
    return bool(
        evidence.get("production_segmentation", False)
        and _count(evidence, "tokenizer_equivalence_examples", 0) >= 1
        and _count(evidence, "tokenizer_equivalence_failures", 1) == 0
        and _count(evidence, "unalignable_example_count", 1) == 0
        and evidence.get("dependency_restart_completed", False)
        and evidence.get("numpy_abi_preflight_passed", False)
        and evidence.get("train_loss_finite", False)
        and evidence.get("backward_completed", False)
        and evidence.get("optimizer_step_completed", False)
        and evidence.get("validation_completed", False)
        and evidence.get("checkpoint_saved", False)
        and evidence.get("checkpoint_reloaded", False)
    )


__all__ = [
    "INSTALL_AND_RESTART",
    "PROCEED",
    "FORBIDDEN_INSTALL_TOKENS",
    "PROTECTED_PACKAGES",
    "REQUIRED_MARKER_FIELDS",
    "DependencyContract",
    "MarkerFingerprint",
    "build_marker_fingerprint",
    "compute_contract_sha256",
    "marker_mismatches",
    "DependencyContractError",
    "build_install_command",
    "build_pip_constraints",
    "decide_bootstrap_action",
    "evaluate_full_training_readiness",
    "load_dependency_contract",
    "validate_abi_report",
    "validate_install_command",
]
