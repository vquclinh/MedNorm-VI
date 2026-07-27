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
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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


# --- dependency-health scoping (Audit 0024) -----------------------------------
#
# `pip check` audits the ENTIRE Colab image, including large preinstalled packages
# S1 never imports (Gradio, IPython/jedi, …). Treating any global `pip check`
# failure as a NumPy/Torch ABI failure blocked a runtime whose RandomState and
# AdamW checks both passed. Health is therefore scoped to S1's own dependency
# closure: a conflict blocks only when the COMPLAINING distribution is something
# S1 actually depends on.

_NAME_SEPARATORS = re.compile(r"[-_.]+")
_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_DISTRIBUTION_TOKEN = r"(?P<dependent>[A-Za-z0-9][A-Za-z0-9._-]*)\s+\S+"
_PIP_CHECK_MISSING_LINE = re.compile(
    rf"^{_DISTRIBUTION_TOKEN}\s+(?:requires|has requirement)\s+"
    r"(?P<requirement>.+),\s+which is not installed\.?\s*$"
)
_PIP_CHECK_INCOMPATIBLE_LINE = re.compile(
    rf"^{_DISTRIBUTION_TOKEN}\s+(?:requires|has requirement)\s+"
    r"(?P<requirement>.+),\s+but you have\s+"
    r"(?P<installed>[A-Za-z0-9][A-Za-z0-9._-]*)\s+.+?"
    r"(?:\s+which is incompatible)?\.?\s*$"
)


def normalize_distribution_name(name: str) -> str:
    """PEP 503 normalization (``huggingface_hub`` and ``huggingface-hub`` agree)."""
    return _NAME_SEPARATORS.sub("-", str(name).strip()).lower()


def parse_requirement_name(requirement: str) -> str:
    """Distribution name from a requirement string, ignoring the version spec."""
    match = _REQUIREMENT_NAME.match(str(requirement))
    return normalize_distribution_name(match.group(1)) if match else ""


def _is_extra_only(requirement: str) -> bool:
    """True for requirements gated behind an extra S1 does not install."""
    _, _, marker = str(requirement).partition(";")
    return "extra" in marker and "==" in marker


def compute_dependency_closure(
    roots: Iterable[str], requirements: Mapping[str, Sequence[str]],
) -> frozenset[str]:
    """Transitive closure of the distributions S1 depends on.

    ``requirements`` maps a normalized distribution name to its declared
    requirement strings (from ``importlib.metadata``). Requirements gated behind
    an extra are excluded — S1 installs no extras, so they are not in its closure.
    """
    closure: set[str] = set()
    pending = [normalize_distribution_name(root) for root in roots]
    while pending:
        name = pending.pop()
        if not name or name in closure:
            continue
        closure.add(name)
        for requirement in requirements.get(name, ()) or ():
            if _is_extra_only(requirement):
                continue
            dependency = parse_requirement_name(requirement)
            if dependency and dependency not in closure:
                pending.append(dependency)
    return frozenset(closure)


@dataclass(frozen=True, slots=True)
class DependencyConflict:
    """One `pip check` complaint, with the distribution that raised it."""

    dependent: str        # normalized name of the COMPLAINING distribution
    requirement: str      # normalized name of the distribution it is unhappy about
    kind: str             # "incompatible" | "missing" | "unparsed"
    message: str          # the original, untruncated `pip check` line

    def as_dict(self) -> dict[str, str]:
        return {
            "dependent": self.dependent, "requirement": self.requirement,
            "kind": self.kind, "message": self.message,
        }


def parse_pip_check_output(output: str) -> list[DependencyConflict]:
    """Parse every non-empty `pip check` line; unrecognized lines are kept."""
    conflicts: list[DependencyConflict] = []
    for raw in str(output or "").splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("no broken requirements"):
            continue
        missing_match = _PIP_CHECK_MISSING_LINE.match(line)
        incompatible_match = _PIP_CHECK_INCOMPATIBLE_LINE.match(line)
        match = missing_match or incompatible_match
        if match is None:
            conflicts.append(DependencyConflict("", "", "unparsed", line))
            continue
        conflicts.append(DependencyConflict(
            dependent=normalize_distribution_name(match.group("dependent")),
            requirement=parse_requirement_name(match.group("requirement")),
            kind="missing" if missing_match else "incompatible",
            message=line,
        ))
    return conflicts


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    """Closure-scoped verdict over the global `pip check` output."""

    blocking: tuple[DependencyConflict, ...]
    non_blocking: tuple[DependencyConflict, ...]
    closure: frozenset[str]
    pip_check_output: str
    pip_check_returncode: int

    @property
    def healthy(self) -> bool:
        """S1 is healthy when nothing inside ITS closure is broken."""
        return not self.blocking

    def as_dict(self) -> dict[str, Any]:
        return {
            "s1_dependency_healthy": self.healthy,
            "s1_dependency_closure": sorted(self.closure),
            "blocking_dependency_conflicts": [c.message for c in self.blocking],
            "non_blocking_dependency_conflicts": [c.message for c in self.non_blocking],
            "dependency_conflicts": [c.as_dict() for c in (*self.blocking, *self.non_blocking)],
            "pip_check_passed": self.pip_check_returncode == 0,
            "pip_check_returncode": self.pip_check_returncode,
            "pip_check_output": self.pip_check_output,   # complete, never truncated
        }


def classify_dependency_health(
    pip_check_output: str, closure: Iterable[str], pip_check_returncode: int = 0,
) -> DependencyHealth:
    """Split `pip check` complaints into S1-blocking and unrelated-but-recorded.

    A conflict blocks only when the **complaining** distribution is inside S1's
    dependency closure. ``gradio requires huggingface-hub<1.0`` is Gradio's
    problem: S1 never imports Gradio, and huggingface-hub must not be moved to
    satisfy it. ``transformers requires tokenizers…`` is S1's problem and blocks.

    A line that cannot be parsed blocks only if it mentions a closure member, so
    unfamiliar pip wording about unrelated packages can never fail the run.
    """
    normalized = frozenset(normalize_distribution_name(name) for name in closure)
    blocking: list[DependencyConflict] = []
    non_blocking: list[DependencyConflict] = []
    for conflict in parse_pip_check_output(pip_check_output):
        if conflict.kind == "unparsed":
            tokens = {normalize_distribution_name(t) for t in re.split(r"\s+", conflict.message)}
            relevant = bool(tokens & normalized)
        else:
            relevant = conflict.dependent in normalized
        (blocking if relevant else non_blocking).append(conflict)
    return DependencyHealth(
        blocking=tuple(blocking), non_blocking=tuple(non_blocking),
        closure=normalized, pip_check_output=str(pip_check_output or ""),
        pip_check_returncode=int(pip_check_returncode),
    )


@dataclass(frozen=True, slots=True)
class DependencyContract:
    contract_version: str
    inherited_from_runtime: tuple[str, ...]
    install_specifiers: tuple[str, ...]
    forbidden_operations: tuple[str, ...]
    marker_path: str
    expected_passes: int
    not_installed: tuple[str, ...] = field(default_factory=tuple)
    # (distribution, module) pairs S1 actually imports. These are the ROOTS of the
    # dependency closure that scopes `pip check`; everything else in the Colab
    # image is out of scope for S1 health.
    import_closure_roots: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # SHA-256 of the exact contract file bytes; changes whenever the tracked
    # contract changes in any way, forcing a fresh install+restart.
    contract_sha256: str = ""

    @property
    def closure_root_distributions(self) -> tuple[str, ...]:
        return tuple(normalize_distribution_name(d) for d, _ in self.import_closure_roots)

    @property
    def install_requirement_hash(self) -> str:
        """Deterministic hash of the normalized requirements + protected policy."""
        payload = {
            "install": sorted(self.install_specifiers),
            "protected": sorted(PROTECTED_PACKAGES),
            "inherited": sorted(self.inherited_from_runtime),
            "not_installed": sorted(self.not_installed),
            "import_closure": sorted(f"{d}:{m}" for d, m in self.import_closure_roots),
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
        import_closure_roots=_load_import_closure_roots(doc),
        contract_sha256=compute_contract_sha256(path),
    )


def _load_import_closure_roots(doc: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Read the governed (distribution, module) roots of the S1 import closure."""
    roots: list[tuple[str, str]] = []
    for entry in doc.get("s1_import_closure") or []:
        distribution = str(entry.get("distribution", "")).strip()
        module = str(entry.get("module", "")).strip()
        if not distribution or not module:
            raise DependencyContractError(
                f"s1_import_closure entry needs distribution and module: {entry}")
        roots.append((distribution, module))
    if not roots:
        raise DependencyContractError("dependency contract declares no s1_import_closure")
    return tuple(roots)


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
    # `pip check` is DIAGNOSTIC ONLY. Its global verdict covers the whole Colab
    # image; only conflicts inside S1's own dependency closure may block the run.
    if not report.get("s1_dependency_closure_verified", False):
        problems.append("S1 dependency closure was not verified")
    for module in report.get("s1_import_failures", ()) or ():
        problems.append(f"S1 dependency import failed: {module}")
    for message in report.get("blocking_dependency_conflicts", ()) or ():
        problems.append(f"S1 dependency conflict: {message}")
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
        and evidence.get("s1_dependency_closure_verified", False)
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
    "DependencyConflict",
    "DependencyContract",
    "DependencyHealth",
    "MarkerFingerprint",
    "classify_dependency_health",
    "compute_dependency_closure",
    "normalize_distribution_name",
    "parse_pip_check_output",
    "parse_requirement_name",
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
