"""S1 Colab dependency bootstrap contract tests (Audit 0023).

Pure logic only: no package installation, no network, and the local Python
environment is never mutated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mednorm_vi.training.colab_bootstrap import (
    INSTALL_AND_RESTART,
    PROCEED,
    PROTECTED_PACKAGES,
    REQUIRED_MARKER_FIELDS,
    DependencyContractError,
    build_install_command,
    build_marker_fingerprint,
    build_pip_constraints,
    classify_dependency_health,
    compute_contract_sha256,
    compute_dependency_closure,
    decide_bootstrap_action,
    evaluate_full_training_readiness,
    load_dependency_contract,
    marker_mismatches,
    normalize_distribution_name,
    parse_pip_check_output,
    parse_requirement_name,
    validate_abi_report,
    validate_install_command,
)

REPO = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO / "configs" / "training" / "s1_mention_colab_dependencies.yaml"
BASELINE = {"numpy": "2.0.2", "torch": "2.8.0+cu126"}
PYTHON_MM = "3.12"

# The exact conflicts a fresh Colab GPU runtime reported after the S1 install.
JEDI_LINE = "ipython 7.34.0 requires jedi, which is not installed."
GRADIO_LINE = ("gradio 6.20.0 has requirement huggingface-hub<2.0,>=1.2.0, "
               "but you have huggingface-hub 0.36.2.")
TRANSFORMERS_LINE = ("transformers 4.44.2 requires tokenizers<0.20,>=0.19, but you have "
                     "tokenizers 0.21.0 which is incompatible.")
MISSING_TOKENIZERS_LINE = "transformers 4.44.2 requires tokenizers, which is not installed."
S1_CLOSURE = frozenset({
    "numpy", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors",
    "sentencepiece", "accelerate", "py-vncorenlp", "pyyaml", "filelock", "requests",
})


@pytest.fixture(scope="module")
def contract():
    return load_dependency_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def fingerprint(contract):
    return build_marker_fingerprint(contract, PYTHON_MM, BASELINE)


def _exact_marker(fingerprint) -> dict:
    """The marker the notebook writes immediately before the restart."""
    return {"install_completed": True, **fingerprint.as_dict()}


# --- contract -----------------------------------------------------------------

def test_contract_loads_and_declares_inherited_stack(contract) -> None:
    assert contract.contract_version == "s1-colab-deps-v2"
    assert "numpy" in contract.inherited_from_runtime
    assert "torch" in contract.inherited_from_runtime
    assert contract.expected_passes == 2
    assert contract.marker_path.startswith("/content/")      # never on Drive


def test_contract_does_not_install_protected_packages(contract) -> None:
    for specifier in contract.install_specifiers:
        bare = specifier.split("==")[0].split(">=")[0].split("<")[0].strip()
        assert bare not in PROTECTED_PACKAGES, f"{bare} must be inherited, not installed"


def test_contract_excludes_pyarrow_from_s1(contract) -> None:
    assert "pyarrow" in contract.not_installed
    assert not any("pyarrow" in s for s in contract.install_specifiers)


def test_contract_rejects_malformed_documents(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("install: []\n", encoding="utf-8")
    with pytest.raises(DependencyContractError):
        load_dependency_contract(bad)


# --- marker fingerprint identity ----------------------------------------------

def test_contract_sha256_is_the_hash_of_the_exact_file_bytes(contract) -> None:
    expected = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert contract.contract_sha256 == expected
    assert compute_contract_sha256(CONTRACT_PATH) == expected


def test_contract_sha256_changes_when_the_contract_bytes_change(tmp_path: Path) -> None:
    copy = tmp_path / "contract.yaml"
    copy.write_bytes(CONTRACT_PATH.read_bytes())
    before = compute_contract_sha256(copy)
    copy.write_bytes(CONTRACT_PATH.read_bytes() + b"\n# drift\n")
    assert compute_contract_sha256(copy) != before


def test_install_requirement_hash_is_deterministic_and_order_independent(contract) -> None:
    from dataclasses import replace
    reordered = replace(
        contract, install_specifiers=tuple(reversed(contract.install_specifiers)))
    assert contract.install_requirement_hash == reordered.install_requirement_hash
    changed = replace(
        contract, install_specifiers=(*contract.install_specifiers, "extra-pkg==1.0"))
    assert changed.install_requirement_hash != contract.install_requirement_hash


def test_fingerprint_carries_every_required_marker_field(fingerprint) -> None:
    marker = _exact_marker(fingerprint)
    for required in REQUIRED_MARKER_FIELDS:
        assert required in marker, f"marker must persist {required}"
    assert fingerprint.protected_baseline_versions == BASELINE   # protected only


# --- two-pass decision / restart-loop guard -----------------------------------

def test_first_pass_installs_and_restarts(fingerprint) -> None:
    assert decide_bootstrap_action(None, fingerprint) == INSTALL_AND_RESTART


def test_exact_marker_match_proceeds(fingerprint) -> None:
    """PROCEED requires every field to match, not just the contract version."""
    marker = _exact_marker(fingerprint)
    assert marker_mismatches(marker, fingerprint) == []
    assert decide_bootstrap_action(marker, fingerprint) == PROCEED


def test_marker_prevents_restart_loop(fingerprint) -> None:
    """An exactly matching marker must never trigger another install/restart."""
    marker = _exact_marker(fingerprint)
    for _ in range(5):
        assert decide_bootstrap_action(marker, fingerprint) == PROCEED


def test_changed_contract_hash_forces_reinstall(fingerprint) -> None:
    marker = _exact_marker(fingerprint) | {"dependency_contract_sha256": "0" * 64}
    assert decide_bootstrap_action(marker, fingerprint) == INSTALL_AND_RESTART
    assert "dependency_contract_sha256 mismatch" in marker_mismatches(marker, fingerprint)


def test_changed_python_major_minor_forces_reinstall(fingerprint) -> None:
    marker = _exact_marker(fingerprint) | {"python_major_minor": "3.11"}
    assert decide_bootstrap_action(marker, fingerprint) == INSTALL_AND_RESTART
    assert "python_major_minor mismatch" in marker_mismatches(marker, fingerprint)


def test_changed_protected_baseline_forces_reinstall(fingerprint) -> None:
    """A runtime whose NumPy/Torch moved must not reuse the marker — this is the
    exact drift that produced the ``dtype size changed`` ABI failure."""
    drifted = dict(fingerprint.protected_baseline_versions) | {"numpy": "1.26.4"}
    marker = _exact_marker(fingerprint) | {"protected_baseline_versions": drifted}
    assert decide_bootstrap_action(marker, fingerprint) == INSTALL_AND_RESTART
    assert "protected_baseline_versions mismatch" in marker_mismatches(marker, fingerprint)


def test_changed_requirement_hash_forces_reinstall(fingerprint) -> None:
    marker = _exact_marker(fingerprint) | {"install_requirement_hash": "f" * 64}
    assert decide_bootstrap_action(marker, fingerprint) == INSTALL_AND_RESTART
    assert "install_requirement_hash mismatch" in marker_mismatches(marker, fingerprint)


@pytest.mark.parametrize("missing", REQUIRED_MARKER_FIELDS)
def test_incomplete_marker_forces_reinstall(fingerprint, missing) -> None:
    marker = _exact_marker(fingerprint)
    marker.pop(missing)
    assert decide_bootstrap_action(marker, fingerprint) == INSTALL_AND_RESTART
    assert f"missing field: {missing}" in marker_mismatches(marker, fingerprint)


def test_legacy_version_only_marker_is_never_accepted(contract, fingerprint) -> None:
    """The old marker proved only a version string — far too weak to skip install."""
    legacy = {"dependency_contract_version": contract.contract_version,
              "install_completed": True}
    assert decide_bootstrap_action(legacy, fingerprint) == INSTALL_AND_RESTART


def test_stale_or_incomplete_marker_reinstalls(fingerprint) -> None:
    stale = _exact_marker(fingerprint) | {"dependency_contract_version": "old"}
    unfinished = _exact_marker(fingerprint) | {"install_completed": False}
    assert decide_bootstrap_action(stale, fingerprint) == INSTALL_AND_RESTART
    assert decide_bootstrap_action(unfinished, fingerprint) == INSTALL_AND_RESTART
    assert decide_bootstrap_action({}, fingerprint) == INSTALL_AND_RESTART


# --- constraints and install command ------------------------------------------

def test_constraints_pin_detected_baseline_not_a_blind_numpy_pin() -> None:
    lines = build_pip_constraints(BASELINE)
    assert "numpy==2.0.2" in lines and "torch==2.8.0+cu126" in lines
    assert not any(line.startswith("numpy<") for line in lines)   # no blind numpy<2


def test_constraints_require_a_detected_baseline() -> None:
    with pytest.raises(DependencyContractError):
        build_pip_constraints({})


def test_install_command_is_consolidated_and_constrained(contract) -> None:
    command = build_install_command(contract, "/content/c.txt", "python3")
    validate_install_command(command)
    assert command.count("install") == 1                  # ONE transaction
    assert "--constraint" in command
    assert "--force-reinstall" not in command and "--upgrade" not in command
    for specifier in contract.install_specifiers:
        assert specifier in command


@pytest.mark.parametrize("bad", [
    ["pip", "install", "--force-reinstall", "--constraint", "c.txt", "numpy"],
    ["pip", "install", "--upgrade", "--constraint", "c.txt", "transformers"],
    ["pip", "install", "--constraint", "c.txt", "torch==2.0.0"],
    ["pip", "install", "transformers"],                    # missing constraints
])
def test_validate_install_command_rejects_violations(bad) -> None:
    with pytest.raises(DependencyContractError):
        validate_install_command(bad)


# --- ABI report ---------------------------------------------------------------

def _healthy() -> dict:
    return {
        "numpy_imported": True, "numpy_random_imported": True, "torch_imported": True,
        "adamw_constructed": True, "numpy_distribution_count": 1,
        "pip_check_passed": True, "numpy_path": "/usr/local/lib/python3.12/dist-packages/numpy",
        "s1_dependency_closure_verified": True, "s1_import_failures": [],
        "blocking_dependency_conflicts": [],
    }


def test_healthy_abi_report_has_no_problems() -> None:
    assert validate_abi_report(_healthy()) == []


@pytest.mark.parametrize("key,message", [
    ("numpy_imported", "numpy import failed"),
    ("numpy_random_imported", "numpy.random.RandomState import failed"),
    ("torch_imported", "torch import failed"),
    ("adamw_constructed", "torch.optim.AdamW construction failed"),
])
def test_abi_report_flags_each_failure(key, message) -> None:
    report = _healthy()
    report[key] = False
    assert message in validate_abi_report(report)


def test_abi_report_flags_duplicate_numpy() -> None:
    report = _healthy() | {"numpy_distribution_count": 2}
    assert "multiple numpy distributions installed" in validate_abi_report(report)


def test_abi_report_ignores_a_failed_global_pip_check() -> None:
    """A red global `pip check` caused by unrelated Colab packages must not fail S1."""
    report = _healthy() | {"pip_check_passed": False,
                           "non_blocking_dependency_conflicts": [GRADIO_LINE, JEDI_LINE]}
    assert validate_abi_report(report) == []


def test_abi_report_flags_blocking_closure_conflicts_and_import_failures() -> None:
    report = _healthy() | {"blocking_dependency_conflicts": [TRANSFORMERS_LINE],
                           "s1_import_failures": ["transformers: ImportError: boom"]}
    problems = validate_abi_report(report)
    assert any(TRANSFORMERS_LINE in p for p in problems)
    assert any("S1 dependency import failed: transformers" in p for p in problems)


def test_abi_report_fails_closed_when_the_closure_was_never_verified() -> None:
    report = _healthy()
    del report["s1_dependency_closure_verified"]
    assert "S1 dependency closure was not verified" in validate_abi_report(report)


@pytest.mark.parametrize("path", [
    "/content/drive/MyDrive/MedNorm-VI/numpy",
    "/content/MedNorm-VI/src/numpy",
])
def test_abi_report_flags_shadowed_numpy(path) -> None:
    report = _healthy()
    report["numpy_path"] = path
    assert any("unexpected location" in p for p in validate_abi_report(report))


# --- dependency closure and pip check scoping ---------------------------------

def test_contract_declares_the_s1_import_closure(contract) -> None:
    modules = {module for _, module in contract.import_closure_roots}
    assert {"numpy", "torch", "transformers", "tokenizers", "huggingface_hub",
            "sentencepiece", "py_vncorenlp", "yaml"} <= modules
    assert "huggingface-hub" in contract.closure_root_distributions   # PEP 503
    assert "gradio" not in contract.closure_root_distributions
    assert "jedi" not in contract.closure_root_distributions


def test_contract_requires_an_import_closure(tmp_path: Path) -> None:
    doc = CONTRACT_PATH.read_text(encoding="utf-8").replace("s1_import_closure:", "unused_key:")
    bad = tmp_path / "no_closure.yaml"
    bad.write_text(doc, encoding="utf-8")
    with pytest.raises(DependencyContractError):
        load_dependency_contract(bad)


@pytest.mark.parametrize("raw,expected", [
    ("huggingface_hub", "huggingface-hub"),
    ("HuggingFace.Hub", "huggingface-hub"),
    ("py_vncorenlp", "py-vncorenlp"),
])
def test_distribution_names_are_pep503_normalized(raw, expected) -> None:
    assert normalize_distribution_name(raw) == expected


@pytest.mark.parametrize("requirement,expected", [
    ("huggingface-hub<1.0,>=0.28.1", "huggingface-hub"),
    ("huggingface-hub<2.0,>=1.2.0", "huggingface-hub"),
    ("tokenizers>=0.19", "tokenizers"),
    ("requests[socks]>=2.0", "requests"),
    ('jedi>=0.16; extra == "test"', "jedi"),
])
def test_requirement_names_ignore_version_specs(requirement, expected) -> None:
    assert parse_requirement_name(requirement) == expected


def test_dependency_closure_is_transitive_and_excludes_extras() -> None:
    graph = {
        "transformers": ["tokenizers>=0.19", "huggingface-hub<1.0", 'pytest; extra == "dev"'],
        "huggingface-hub": ["requests", "filelock"],
        "tokenizers": ["huggingface-hub"],
        "gradio": ["huggingface-hub<1.0"],          # depends INTO the closure
        "requests": [], "filelock": [],
    }
    closure = compute_dependency_closure(["transformers"], graph)
    assert closure == {"transformers", "tokenizers", "huggingface-hub", "requests", "filelock"}
    assert "pytest" not in closure                  # extras are not installed by S1
    assert "gradio" not in closure                  # a reverse dependency is NOT in the closure


def test_dependency_closure_survives_cycles_and_unknown_distributions() -> None:
    graph = {"a": ["b"], "b": ["a", "missing-dist"]}
    assert compute_dependency_closure(["a"], graph) == {"a", "b", "missing-dist"}


def test_parse_pip_check_output_extracts_dependent_and_kind() -> None:
    conflicts = parse_pip_check_output(f"{JEDI_LINE}\n{GRADIO_LINE}\n")
    assert [(c.dependent, c.requirement, c.kind) for c in conflicts] == [
        ("ipython", "jedi", "missing"),
        ("gradio", "huggingface-hub", "incompatible"),
    ]
    assert conflicts[1].message == GRADIO_LINE      # original line retained verbatim


def test_observed_colab_conflicts_are_non_blocking() -> None:
    """The exact fresh-Colab output: IPython/jedi and Gradio/huggingface-hub.

    Neither IPython nor Gradio is imported by S1, so neither may fail the run --
    and huggingface_hub must NOT be upgraded to satisfy Gradio.
    """
    health = classify_dependency_health(
        f"{JEDI_LINE}\n{GRADIO_LINE}", S1_CLOSURE, pip_check_returncode=1)
    assert health.healthy is True
    assert health.blocking == ()
    assert [c.dependent for c in health.non_blocking] == ["ipython", "gradio"]
    assert health.as_dict()["pip_check_passed"] is False        # honestly recorded
    assert health.as_dict()["pip_check_output"] == f"{JEDI_LINE}\n{GRADIO_LINE}"


@pytest.mark.parametrize("line,dependent,requirement", [
    ("ipython 8.12.3 requires jedi>=0.16, which is not installed.", "ipython", "jedi"),
    ("ipython 7.34.0 has requirement jedi>=0.16, which is not installed.",
     "ipython", "jedi"),
    ("gradio 5.49.1 has requirement huggingface-hub<1.0,>=0.28.1, "
     "but you have huggingface-hub 1.0.1.", "gradio", "huggingface-hub"),
    ("gradio 6.20.0 requires huggingface-hub<2.0,>=1.2.0, "
     "but you have huggingface-hub 0.36.2 which is incompatible.",
     "gradio", "huggingface-hub"),
])
def test_equivalent_colab_conflict_variants_remain_non_blocking(
    line: str, dependent: str, requirement: str,
) -> None:
    health = classify_dependency_health(line, S1_CLOSURE, pip_check_returncode=1)
    assert health.healthy is True
    assert health.blocking == ()
    assert [(c.dependent, c.requirement) for c in health.non_blocking] == [
        (dependent, requirement)
    ]


@pytest.mark.parametrize("line,dependent", [
    (TRANSFORMERS_LINE, "transformers"),
    (MISSING_TOKENIZERS_LINE, "transformers"),
    ("torch 2.8.0 requires filelock, which is not installed.", "torch"),
    ("huggingface-hub 1.0.1 requires requests>=2.0, which is not installed.",
     "huggingface-hub"),
])
def test_conflicts_raised_inside_the_closure_block(line, dependent) -> None:
    health = classify_dependency_health(line, S1_CLOSURE, pip_check_returncode=1)
    assert health.healthy is False
    assert [c.dependent for c in health.blocking] == [dependent]


def test_blocking_and_non_blocking_conflicts_are_separated_together() -> None:
    output = f"{JEDI_LINE}\n{TRANSFORMERS_LINE}\n{GRADIO_LINE}"
    health = classify_dependency_health(output, S1_CLOSURE, pip_check_returncode=1)
    assert [c.message for c in health.blocking] == [TRANSFORMERS_LINE]
    assert [c.message for c in health.non_blocking] == [JEDI_LINE, GRADIO_LINE]
    payload = health.as_dict()
    assert payload["s1_dependency_healthy"] is False
    assert payload["pip_check_output"] == output                # complete diagnostics
    assert len(payload["dependency_conflicts"]) == 3


def test_clean_pip_check_is_healthy() -> None:
    for output in ("", "No broken requirements found.\n"):
        health = classify_dependency_health(output, S1_CLOSURE, pip_check_returncode=0)
        assert health.healthy is True and health.non_blocking == ()


def test_unparsed_lines_block_only_when_they_mention_the_closure() -> None:
    unrelated = classify_dependency_health(
        "gradio has some brand new pip wording", S1_CLOSURE, 1)
    relevant = classify_dependency_health(
        "transformers has some brand new pip wording", S1_CLOSURE, 1)
    assert unrelated.healthy is True
    assert [c.kind for c in unrelated.non_blocking] == ["unparsed"]
    assert relevant.healthy is False
    assert [c.kind for c in relevant.blocking] == ["unparsed"]


def test_huggingface_hub_is_never_a_remediation_target(contract) -> None:
    """Gradio's complaint must not turn into an install of huggingface_hub."""
    assert not any("huggingface" in s.lower() for s in contract.install_specifiers)
    health = classify_dependency_health(GRADIO_LINE, S1_CLOSURE, 1)
    assert health.healthy is True


# --- readiness formula --------------------------------------------------------

def _ready() -> dict:
    return {
        "production_segmentation": True, "tokenizer_equivalence_examples": 4,
        "tokenizer_equivalence_failures": 0, "unalignable_example_count": 0,
        "dependency_restart_completed": True, "numpy_abi_preflight_passed": True,
        "s1_dependency_closure_verified": True,
        "train_loss_finite": True, "backward_completed": True,
        "optimizer_step_completed": True, "validation_completed": True,
        "checkpoint_saved": True, "checkpoint_reloaded": True,
    }


def test_full_readiness_true_only_when_everything_holds() -> None:
    assert evaluate_full_training_readiness(_ready()) is True


@pytest.mark.parametrize("key", [
    "production_segmentation", "dependency_restart_completed",
    "numpy_abi_preflight_passed", "s1_dependency_closure_verified",
    "train_loss_finite", "backward_completed",
    "optimizer_step_completed", "validation_completed", "checkpoint_saved",
    "checkpoint_reloaded",
])
def test_readiness_false_when_any_condition_fails(key) -> None:
    evidence = _ready()
    evidence[key] = False
    assert evaluate_full_training_readiness(evidence) is False


def test_readiness_false_on_equivalence_or_alignment_problems() -> None:
    no_examples = _ready() | {"tokenizer_equivalence_examples": 0}
    failures = _ready() | {"tokenizer_equivalence_failures": 1}
    unalignable = _ready() | {"unalignable_example_count": 3}
    assert evaluate_full_training_readiness(no_examples) is False
    assert evaluate_full_training_readiness(failures) is False
    assert evaluate_full_training_readiness(unalignable) is False
