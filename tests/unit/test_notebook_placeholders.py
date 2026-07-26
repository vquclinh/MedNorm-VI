"""Static placeholder scan over every tracked notebook (Audit 0019).

Audit 0018's static tests passed while the VietMed notebook's `git clone` was
commented out. These checks target that failure mode directly: commented-out
required commands, `<...>` placeholders, and describe-only print cells.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks"
VIETMED = "MedNorm_Data_VietMed_Preprocess.ipynb"
S1_SMOKE = "MedNorm_S1_Mention_FirstRun_Smoke.ipynb"
S1_VALIDATION = "MedNorm_S1_Smoke_Artifact_Validation.ipynb"
S1_FULL = "MedNorm_S1_Mention_Full_Training.ipynb"
ALL_NOTEBOOKS = sorted(p.name for p in NB.glob("*.ipynb"))

# Notebooks honestly classified as DESIGN_DRAFT in the integrity report. They may
# still contain placeholders, but must never be described as executable/complete.
DESIGN_DRAFTS = {
    "MedNorm_S0_DomainAdaptation.ipynb",
    "MedNorm_S1_MentionExtraction.ipynb",
    "MedNorm_S2_Assertion.ipynb",
    "MedNorm_S3_Retrieval.ipynb",
    "MedNorm_S4_Reranker.ipynb",
    "MedNorm_S5_QwenLoRA.ipynb",
    "MedNorm_S6_Calibration.ipynb",
    "MedNorm_Full_Offline_Inference_and_Packaging.ipynb",
}

_PLACEHOLDER_TOKENS = (
    "<repo-url>", "<fill:", "REPLACE_ME", "IMPLEMENT_ME", "NotImplemented",
    "TODO", "FIXME", "your-repo", "YOUR_", "INSERT_",
)
_COMMENTED_REQUIRED = re.compile(r"^\s*#\s*!(pip|git)\b", re.M)
_ONCE_WIRED = re.compile(r"once .{0,40}wired", re.I)


def _code(name: str) -> str:
    doc = json.loads((NB / name).read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", []))
                     for c in doc["cells"] if c["cell_type"] == "code")


# --- strict rules for the VietMed notebook -----------------------------------

def test_vietmed_notebook_has_no_placeholder_tokens() -> None:
    code = _code(VIETMED)
    for token in _PLACEHOLDER_TOKENS:
        assert token not in code, f"VietMed notebook contains placeholder {token!r}"
    assert not _ONCE_WIRED.search(code)


def test_vietmed_notebook_has_no_commented_required_commands() -> None:
    code = _code(VIETMED)
    found = _COMMENTED_REQUIRED.findall(code)
    assert not found, f"VietMed notebook has commented-out required command(s): {found}"


def test_vietmed_notebook_actually_invokes_checkout_and_install() -> None:
    code = _code(VIETMED)
    # real checkout (helper call + an executed clone in the bootstrap branch)
    assert "checkout_repository(REPO_URL, REPO_DIR, REPO_REF" in code
    assert "subprocess.run(['git', 'clone'" in code
    # real pinned install actually executed (not commented)
    assert "'pip', 'install', '-q', PYARROW_PIN" in code
    # adapter import happens only after checkout verification
    verify_idx = code.index("assert Path(checkout.src_dir).is_dir()")
    import_idx = code.index("from mednorm_vi.data_engine import vietmed_ner as vm")
    assert verify_idx < import_idx, "adapter imported before checkout verification"


def test_vietmed_notebook_uses_real_repository_url() -> None:
    code = _code(VIETMED)
    assert "https://github.com/vquclinh/MedNorm-VI.git" in code
    assert "REPO_REF" in code and "RESOLVED_COMMIT = checkout.resolved_commit" in code


def test_vietmed_notebook_writes_and_verifies_artifacts() -> None:
    code = _code(VIETMED)
    assert "vm.write_artifacts(" in code                 # artifact creation
    assert "vm.load_vietmed_artifacts(" in code          # hash-verified reload
    assert "assert reloaded == examples" in code
    assert "assert summary['offset_invalid'] == 0" in code
    assert "assert summary['human_review_required'] == 0" in code
    assert "run_mode=RUN_MODE" in code
    assert "repo_commit=RESOLVED_COMMIT" in code         # provenance is the resolved SHA


def test_vietmed_notebook_success_prints_follow_assertions() -> None:
    """A cell that reports verification must actually assert or call something.

    Guards the failure mode where a cell only *prints* what should have happened.
    """
    doc = json.loads((NB / VIETMED).read_text(encoding="utf-8"))
    for i, cell in enumerate(doc["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "verified" not in src.lower():
            continue
        non_print = [ln for ln in src.splitlines()
                     if ln.strip() and not ln.strip().startswith(("print(", "#"))]
        assert non_print, f"cell {i} only prints; it performs no operation"
        assert ("assert " in src or "vm." in src or "checkout" in src), (
            f"cell {i} reports verification without an assertion or real call")


# --- strict rules for the S1 first-run smoke notebook -------------------------

def test_s1_smoke_notebook_has_no_placeholder_tokens() -> None:
    code = _code(S1_SMOKE)
    for token in _PLACEHOLDER_TOKENS:
        assert token not in code, f"S1 smoke notebook contains placeholder {token!r}"
    assert "smoke would run" not in code
    assert not _ONCE_WIRED.search(code)
    assert not re.search(r"\bpass\b", code)


def test_s1_smoke_notebook_has_no_commented_required_commands() -> None:
    code = _code(S1_SMOKE)
    found = _COMMENTED_REQUIRED.findall(code)
    assert not found, f"S1 smoke notebook has commented-out required command(s): {found}"


def test_s1_smoke_notebook_uses_required_colab_paths() -> None:
    code = _code(S1_SMOKE)
    assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/MedNorm-VI")' in code
    assert 'REPO_DIR = Path("/content/MedNorm-VI")' in code
    assert 'REPO_URL = "https://github.com/vquclinh/MedNorm-VI.git"' in code
    assert 'REPO_REF = "main"' in code
    # Audit 0026: the corrected rerun writes to its own versioned directory.
    assert ('DRIVE_ROOT / "artifacts" / '
            'f"s1_mention_first_run_smoke_{SMOKE_ARTIFACT_VERSION}"') in code
    assert 'MODEL_CACHE_DIR = DRIVE_ROOT / "model_cache" / "huggingface"' in code
    assert "CORPUS_DIR = (\n    DRIVE_ROOT" in code
    assert "REPO_DIR / \"data\"" not in code


def test_s1_smoke_notebook_orders_corpus_gate_before_model_acquisition() -> None:
    """Audit 0023 order: clone -> install/restart -> ABI -> corpus gate -> tokenizer -> model.

    (The old single `PIP_PACKAGES` cell was replaced by one consolidated,
    constraint-protected transaction driven by the tracked dependency contract.)
    """
    code = _code(S1_SMOKE)
    clone_idx = code.index('"git",\n    "clone"')
    install_idx = code.index("subprocess.run(install_command")
    abi_idx = code.index("import numpy as np")
    import_idx = code.index("from mednorm_vi.training.s1_mention_smoke import")
    verify_idx = code.index("corpus_report = verify_governed_corpus")
    tokenizer_idx = code.index("AutoTokenizer.from_pretrained")
    model_idx = code.index("AutoModel.from_pretrained")
    assert clone_idx < install_idx < abi_idx < import_idx < verify_idx
    assert verify_idx < tokenizer_idx < model_idx
    assert "assert IN_COLAB_BOOTSTRAP" in code
    assert "assert torch.cuda.is_available()" in code


def test_s1_smoke_notebook_supports_slow_phobert_tokenizer() -> None:
    """Audit 0022: ViHealthBERT-Word has no fast tokenizer; the notebook must not
    assert `is_fast` nor rely on fast-only offset APIs."""
    code = _code(S1_SMOKE)
    assert 'assert getattr(tokenizer, "is_fast", False)' not in code
    assert "use_fast=False" in code                     # loaded honestly as slow
    assert "use_fast=True" not in code
    assert "return_offsets_mapping" not in code         # fast-only API
    assert "word_ids(" not in code
    assert "token_to_chars" not in code and "char_to_token" not in code
    # the tracked alignment backend is what supplies character spans
    assert "encode_mention_example_slow(" in code
    assert "describe_backend(tokenizer)" in code
    assert "ALIGNMENT_BACKEND" in code
    assert 'tokenizer_report["tokenizer_is_fast"] is False' in code


def test_s1_smoke_notebook_runs_alignment_preflight_before_model_download() -> None:
    code = _code(S1_SMOKE)
    seg_idx = code.index("segment_example_text")
    preflight_idx = code.index("alignment_preflight = {")
    model_idx = code.index("AutoModel.from_pretrained")
    assert seg_idx < preflight_idx < model_idx
    # segmentation resources are verified/recorded, with a fail-fast assertion
    assert "word_segmenter_resource_hashes" in code
    assert "VnCoreNLP resources missing after acquisition" in code
    # unalignable examples are recorded as diagnostics, never silently mislabeled
    assert "except (AlignmentError, ValueError) as exc:" in code
    assert "alignment_diagnostic(" in code
    assert "summarize_alignment_diagnostics(alignment_diagnostics)" in code


def test_s1_smoke_notebook_covers_train_and_validation_in_one_preflight() -> None:
    """Audit 0026: validation alignment failures used to be silently dropped."""
    code = _code(S1_SMOKE)
    assert 'SMOKE_SPLITS = (("train", train_examples), ("validation", validation_examples))' in code
    assert "EXAMPLES_CONSIDERED = sum(len(rows) for _, rows in SMOKE_SPLITS)" in code
    # equivalence and alignment must share the same denominator
    assert '"tokenizer_equivalence_considered": EXAMPLES_CONSIDERED' in code
    assert '"examples_considered": EXAMPLES_CONSIDERED' in code
    assert '"tokenizer_equivalence_skipped_unmappable"' in code
    # every considered example is accounted for exactly once
    assert "RECONCILED = (" in code
    assert "assert RECONCILED" in code


def test_s1_smoke_notebook_separates_unexpected_failures_from_governed_exclusions() -> None:
    code = _code(S1_SMOKE)
    assert "load_governed_exclusions(" in code
    assert "governed_exclusion_diagnostic(" in code
    assert '"governed_exclusion_count"' in code
    assert '"unalignable_examples"' in code
    assert "BOUNDARY_MERGE_POLICY" in code


def test_s1_notebooks_record_only_privacy_safe_alignment_diagnostics() -> None:
    """No raw clinical text or verbatim example id may reach a manifest."""
    for notebook in (S1_SMOKE, S1_FULL):
        code = _code(notebook)
        assert "privacy_safe_example_id(" in code, notebook
        for leak in ('row["text"]}', '"raw_text"', '"entity_text"', 'exc)}"'):
            assert leak not in code, f"{notebook} may leak content via {leak!r}"
        # diagnostics are always built through the tracked, text-free helper
        assert "alignment_diagnostic(" in code, notebook


def _s1_index(needle: str) -> int:
    code = _code(S1_SMOKE)
    idx = code.find(needle)
    assert idx >= 0, f"S1 notebook missing required code: {needle!r}"
    return idx


def test_s1_smoke_notebook_never_imports_numpy_or_torch_before_abi_preflight() -> None:
    """Audit 0023: importing the scientific stack before the restart is what
    corrupted the NumPy C-ABI."""
    code = _code(S1_SMOKE)
    abi_idx = _s1_index("import numpy as np")
    head = code[:abi_idx]
    pattern = (r"^\s*(?:import|from)\s+"
               r"(numpy|torch|transformers|pandas|scipy|sklearn|pyarrow)\b")
    bad = re.findall(pattern, head, re.M)
    bad += re.findall(r'import_module\(["\'](torch|transformers|numpy)', head)
    assert not bad, f"scientific imports before the ABI preflight: {bad}"


def test_s1_smoke_notebook_installs_once_and_forces_one_restart() -> None:
    code = _code(S1_SMOKE)
    assert code.count("subprocess.run(install_command") == 1     # consolidated
    assert code.count("os.kill(os.getpid(), 9)") == 1            # exactly one restart
    assert "validate_install_command(install_command)" in code
    assert "build_pip_constraints(baseline_versions)" in code
    # the marker guards against a restart loop, and is bound to the full fingerprint
    assert "decide_bootstrap_action(bootstrap_marker, MARKER_FINGERPRINT)" in code
    assert "DEPENDENCY_RESTART_COMPLETED = BOOTSTRAP_ACTION == PROCEED" in code


def test_s1_smoke_notebook_marker_is_bound_to_the_full_fingerprint() -> None:
    """The marker must never be accepted on the contract version alone."""
    code = _code(S1_SMOKE)
    assert "build_marker_fingerprint(" in code
    assert "**MARKER_FINGERPRINT.as_dict()" in code                # persisted in full
    assert "marker_mismatches(bootstrap_marker, MARKER_FINGERPRINT)" in code
    # the fingerprint is re-validated inside the restarted kernel
    assert "assert not POST_RESTART_MARKER_MISMATCHES" in code
    assert _s1_index("build_marker_fingerprint(") < _s1_index("MARKER_FINGERPRINT)")


def test_s1_smoke_notebook_forbids_dangerous_pip_operations() -> None:
    code = _code(S1_SMOKE)
    assert "--force-reinstall" not in code
    assert "--upgrade" not in code
    assert "numpy<2" not in code                                  # no blind pin
    for protected in ("torch==", "torchvision==", "torchaudio=="):
        assert protected not in code, f"must not reinstall {protected}"


def test_s1_smoke_notebook_abi_preflight_precedes_all_acquisition() -> None:
    order = [
        _s1_index("subprocess.run(install_command"),
        _s1_index("os.kill(os.getpid(), 9)"),
        _s1_index("assert DEPENDENCY_RESTART_COMPLETED"),
        _s1_index("import numpy as np"),
        _s1_index("from numpy.random import RandomState"),
        _s1_index("torch.optim.AdamW([dummy_parameter]"),
        _s1_index("drive_module.mount"),
        _s1_index("verify_governed_corpus"),
        _s1_index("py_vncorenlp.VnCoreNLP"),
        _s1_index("AutoTokenizer.from_pretrained"),
        _s1_index("AutoModel.from_pretrained"),
    ]
    assert order == sorted(order), "ABI preflight must precede every acquisition step"


def test_s1_smoke_notebook_abi_preflight_is_fail_fast() -> None:
    code = _code(S1_SMOKE)
    assert "rng = RandomState(42)" in code
    assert "dummy_optimizer.zero_grad(set_to_none=True)" in code
    assert "abi_problems = validate_abi_report(abi_report)" in code
    assert "assert not abi_problems" in code                      # blocks later cells
    assert "NUMPY_ABI_PREFLIGHT_PASSED = True" in code


def test_s1_smoke_notebook_keeps_adamw_as_the_real_optimizer() -> None:
    code = _code(S1_SMOKE)
    assert "optimizer = torch.optim.AdamW(model.parameters()" in code


def test_s1_smoke_notebook_scopes_pip_check_without_remediation() -> None:
    code = _code(S1_SMOKE)
    assert "classify_dependency_health(" in code
    assert "DEPENDENCY_HEALTH.as_dict()" in code
    assert (
        '"blocking_dependency_conflicts": '
        "[c.message for c in DEPENDENCY_HEALTH.blocking]"
    ) in code
    assert (
        '"non_blocking_dependency_conflicts": '
        "[c.message for c in DEPENDENCY_HEALTH.non_blocking]"
    ) in code
    assert '"pip_check_stdout": pip_check.stdout' in code
    assert '"pip_check_stderr": pip_check.stderr' in code
    assert '"pip_check_output": pip_check_output' in code
    assert "[:2000]" not in code

    install_related_lines = []
    for line in code.splitlines():
        lower = line.lower()
        if line.lstrip().startswith("#"):
            continue
        if "install" in lower or "pip" in lower:
            install_related_lines.append(lower)
    assert not any("jedi" in line for line in install_related_lines)
    assert not any("gradio" in line for line in install_related_lines)
    assert not any("huggingface-hub" in line for line in install_related_lines)


def test_s1_smoke_notebook_manifest_records_environment_fields() -> None:
    code = _code(S1_SMOKE)
    for field in ("dependency_contract_version", "dependency_restart_completed",
                  "numpy_abi_preflight_passed", "python_version", "numpy_version",
                  "numpy_path", "numpy_mtrand_path", "torch_version", "torch_path",
                  "transformers_version", "tokenizers_version", "py_vncorenlp_version",
                  "pip_check_passed", "dependency_contract_sha256",
                  "install_requirement_hash", "python_major_minor",
                  "protected_baseline_versions", "bootstrap_action",
                  "bootstrap_marker_mismatches", "s1_dependency_closure",
                  "blocking_dependency_conflicts", "non_blocking_dependency_conflicts",
                  "pip_check_output", "s1_dependency_closure_verified"):
        assert field in code, f"manifest env field {field!r} missing"
    assert "evaluate_full_training_readiness({" in code


def test_s1_smoke_notebook_documents_two_pass_execution() -> None:
    doc = json.loads((NB / S1_SMOKE).read_text(encoding="utf-8"))
    markdown = "\n".join("".join(c.get("source", []))
                         for c in doc["cells"] if c["cell_type"] == "markdown")
    assert "PASS 1" in markdown and "PASS 2" in markdown
    assert "restart" in markdown.lower()


def test_s1_smoke_notebook_defaults_to_vncorenlp_segmenter() -> None:
    """Production S1 smoke requires VnCoreNLP; degraded mode is opt-in only."""
    code = _code(S1_SMOKE)
    assert 'os.environ.get("MEDNORM_SEGMENTER_MODE", "vncorenlp")' in code
    assert 'DEGRADED_FALLBACK = SEGMENTER_MODE == "whitespace_fallback"' in code
    # fail fast on missing/broken resources
    assert "VnCoreNLP resources missing after acquisition" in code
    assert "VnCoreNLP resource hashes are empty" in code
    # degraded mode warns prominently and blocks production classification
    assert "DEGRADED MODE" in code
    assert "cannot be classified as a successful production-path S1 smoke" in code
    assert "PRODUCTION_SEGMENTATION = (" in code
    assert '"degraded_fallback": segmenter_report["degraded_fallback"]' in code


def test_s1_smoke_notebook_checks_tokenizer_equivalence_before_model() -> None:
    code = _code(S1_SMOKE)
    eq_idx = code.index("verify_tokenizer_equivalence(")
    model_idx = code.index("AutoModel.from_pretrained")
    tok_idx = code.index("AutoTokenizer.from_pretrained")
    assert tok_idx < eq_idx < model_idx           # after tokenizer, before weights
    assert 'tokenizer_equivalence["tokenizer_equivalence_failures"] == 0' in code
    assert "tokenizer_equivalence_checked" in code
    assert "tokenizer_equivalence_examples" in code


def test_s1_smoke_notebook_records_truncation_and_readiness_fields() -> None:
    code = _code(S1_SMOKE)
    for field in ("fully_dropped_entity_count", "partially_truncated_entity_count",
                  "partial_truncation_policy", "full_training_readiness",
                  "segmenter_mode"):
        assert field in code, f"manifest field {field!r} missing"
    # readiness must depend on production segmentation and equivalence
    assert "PRODUCTION_SEGMENTATION" in code


def test_s1_smoke_notebook_is_strictly_bounded_and_smoke_only() -> None:
    code = _code(S1_SMOKE)
    assert "FULL_TRAINING_ENABLED = False" in code
    assert "CONFIRM_FULL_TRAINING = \"\"" in code
    assert "SMOKE_ONLY" in code
    assert "max_optimizer_steps" in code
    assert "limits.max_optimizer_steps" in code
    assert "limits.max_train_batches" in code
    assert "output.zip" not in code
    assert "organizer" not in code.lower()


def test_s1_smoke_notebook_writes_required_manifest_fields() -> None:
    code = _code(S1_SMOKE)
    for key in (
        '"runtime": runtime_report',
        '"resolved_commit": RESOLVED_COMMIT',
        '"corpus": corpus_report',
        '"actual_base_parameters": base_parameter_count_actual',
        '"trainable_parameter_count": trainable_parameter_count',
        '"loss_values": loss_values',
        '"validation_metrics": validation_metrics',
        '"checkpoint_sha256": checkpoint_sha256',
        '"smoke_only_not_full_training": True',
    ):
        assert key in code
    assert "training_manifest = {" in code
    assert "TRAINING_MANIFEST_PATH.write_text" in code


# --- honest classification for every other notebook --------------------------

@pytest.mark.parametrize("name", sorted(DESIGN_DRAFTS))
def test_design_draft_notebooks_are_documented_as_such(name: str) -> None:
    doc = (REPO / "docs" / "notebooks" / "notebook_execution_integrity.md").read_text(
        encoding="utf-8")
    assert name in doc, f"{name} missing from the notebook integrity report"


def test_integrity_report_does_not_claim_unverified_notebooks_are_verified() -> None:
    doc = (REPO / "docs" / "notebooks" / "notebook_execution_integrity.md").read_text(
        encoding="utf-8")
    # Only the VietMed notebook row may carry a smoke-verified status. Inventory rows
    # are the table lines that name a notebook file (the legend row names none).
    for line in doc.splitlines():
        if not line.strip().startswith("|") or ".ipynb" not in line:
            continue
        if "SYNTHETIC_SMOKE_VERIFIED" in line:
            assert VIETMED in line, f"non-VietMed notebook claims smoke-verified: {line}"
        if "REAL_DATA_EXECUTED" in line or "ARTIFACTS_VERIFIED" in line:
            assert VIETMED in line, f"unsupported real-data/artifact claim: {line}"


def test_all_notebooks_are_valid_json() -> None:
    for name in ALL_NOTEBOOKS:
        doc = json.loads((NB / name).read_text(encoding="utf-8"))
        assert doc["nbformat"] == 4 and doc["cells"]


# --- S1 smoke-artifact validation notebook (Audit 0025) -----------------------

def test_validation_notebook_reads_the_real_manifest_and_checkpoint() -> None:
    code = _code(S1_VALIDATION)
    assert "validate_smoke_artifact(" in code
    assert "load_smoke_manifest(" in code
    assert "EXPECTED_SMOKE_CHECKPOINT_SHA256" in code
    assert "outcome.smoke_validated" in code
    # It must fail loudly and enumerate every failed condition.
    assert "for failure in outcome.failures" in code
    assert "raise AssertionError" in code


def test_validation_notebook_treats_global_pip_check_as_a_diagnostic() -> None:
    code = _code(S1_VALIDATION)
    assert "non_blocking_dependency_conflicts" in code
    assert "pip_check_output" in code
    assert "does NOT invalidate this artifact" in code


def test_validation_notebook_never_trains_installs_or_restarts() -> None:
    code = _code(S1_VALIDATION)
    for forbidden in ("pip install", "os.kill", "loss.backward", "optimizer.step",
                      "from_pretrained", "output.zip"):
        assert forbidden not in code, f"validation notebook must not contain {forbidden!r}"


def test_validation_notebook_pins_the_revision_from_the_validated_manifest() -> None:
    code = _code(S1_VALIDATION)
    assert "pinned_revision_from_outcome(outcome)" in code
    assert _s1_index_in(S1_VALIDATION, "outcome.smoke_validated") < _s1_index_in(
        S1_VALIDATION, "pinned_revision_from_outcome(outcome)")


# --- S1 full-training notebook (Audit 0025) -----------------------------------

def test_full_training_notebook_has_an_explicit_guard_before_training() -> None:
    code = _code(S1_FULL)
    assert "CONFIRM_FULL_TRAINING" in code
    assert 'confirmation_phrase' in code
    assert "full training requires explicit confirmation" in code
    # The guard must precede model acquisition, the real optimizer, and the loop.
    # (The ABI preflight's dummy AdamW legitimately runs earlier.)
    guard = _s1_index_in(S1_FULL, "raise SystemExit(\"full training requires")
    for later in ("AutoModel.from_pretrained", "torch.optim.AdamW(parameter_groups)",
                  "scaled.backward()"):
        assert guard < _s1_index_in(S1_FULL, later), f"guard must precede {later}"


def test_full_training_notebook_requires_a_validated_smoke_artifact_first() -> None:
    code = _code(S1_FULL)
    assert "validate_smoke_artifact(" in code
    assert "smoke_outcome.smoke_validated" in code
    assert "full training is not authorized" in code
    assert _s1_index_in(S1_FULL, "validate_smoke_artifact(") < _s1_index_in(
        S1_FULL, "AutoModel.from_pretrained")


def test_full_training_notebook_uses_the_pinned_immutable_revision() -> None:
    code = _code(S1_FULL)
    assert "PINNED_MODEL_REVISION = pinned_revision_from_outcome(smoke_outcome)" in code
    # The pinned hash reaches the config, the tokenizer, and the backbone.
    assert code.count("revision=PINNED_MODEL_REVISION") == 3
    assert "pinned_revision=PINNED_MODEL_REVISION" in code
    assert 'revision="main"' not in code
    assert "resolved_model_revision == PINNED_MODEL_REVISION" in code


def test_full_training_notebook_never_initializes_from_the_smoke_checkpoint() -> None:
    code = _code(S1_FULL)
    assert "s1_mention_smoke_model.pt" not in code
    assert "validate_resume_checkpoint(payload, config)" in code
    assert "OUTPUT_DIR.resolve() != SMOKE_ARTIFACT_DIR.resolve()" in code


def test_full_training_notebook_writes_to_a_separate_output_directory() -> None:
    code = _code(S1_FULL)
    assert "s1_mention_full_training_v1" not in code            # comes from the config
    assert "full_training_output_paths(OUTPUT_DIR)" in code
    assert "latest_checkpoint" in code and "best_checkpoint" in code
    assert "OUTPUT_PATHS[\"training_manifest\"]" in code


def test_full_training_notebook_preserves_the_validated_bootstrap_and_gates() -> None:
    code = _code(S1_FULL)
    for preserved in (
        "decide_bootstrap_action(bootstrap_marker, MARKER_FINGERPRINT)",
        "os.kill(os.getpid(), 9)",
        "rng = RandomState(42)",
        "dummy_optimizer.zero_grad(set_to_none=True)",
        "abi_problems = validate_abi_report(abi_report)",
        "classify_dependency_health(",
        "verify_governed_corpus(",
        "verify_tokenizer_equivalence(",
        "encode_mention_example_slow(",
        "py_vncorenlp.VnCoreNLP",
        "use_fast=False",
    ):
        assert preserved in code, f"full-training notebook lost {preserved!r}"
    assert "assert PRODUCTION_SEGMENTATION" in code
    assert code.count("os.kill(os.getpid(), 9)") == 1           # exactly one restart


def test_full_training_notebook_has_a_real_training_loop() -> None:
    code = _code(S1_FULL)
    for required in (
        "torch.optim.AdamW(parameter_groups)",
        "get_linear_schedule_with_warmup",
        "clip_grad_norm_",
        "gradient_accumulation_steps",
        "torch.autocast",
        "scaled.backward()",
        "MentionMetrics()",
        "is_better_metric(",
        "build_full_training_manifest(",
    ):
        assert required in code, f"full-training loop is missing {required!r}"
    assert "torch.cuda.OutOfMemoryError" in code                 # OOM guidance
    assert "resumable" in code


def test_full_training_notebook_runs_no_inference_or_packaging() -> None:
    code = _code(S1_FULL)
    for forbidden in ("output.zip", "openai", "anthropic", "requests.post"):
        assert forbidden not in code


def _s1_index_in(notebook: str, needle: str) -> int:
    code = _code(notebook)
    assert needle in code, f"{needle!r} not found in {notebook}"
    return code.index(needle)


# --- artifact lifecycle wiring in the notebooks (Audit 0026) ------------------

def test_smoke_notebook_writes_to_a_versioned_directory_not_over_history() -> None:
    code = _code(S1_SMOKE)
    assert 'f"s1_mention_first_run_smoke_{SMOKE_ARTIFACT_VERSION}"' in code
    # every earlier artifact is protected, not just v1
    assert "for _historical in HISTORICAL_SMOKE_ARTIFACT_DIRS:" in code
    assert "OUTPUT_DIR.resolve() != _historical.resolve()" in code
    # the path is tracked in configuration, not only in the notebook
    assert "smoke_artifact_paths_from_config(smoke_config)" in code
    assert "smoke_artifact_paths.artifact_version == SMOKE_ARTIFACT_VERSION" in code


@pytest.mark.parametrize("notebook", [S1_VALIDATION, S1_FULL])
def test_notebooks_take_the_artifact_dir_and_expected_hash_at_runtime(notebook) -> None:
    code = _code(notebook)
    assert "MEDNORM_SMOKE_ARTIFACT_DIR" in code
    assert "MEDNORM_EXPECTED_SMOKE_CHECKPOINT_SHA256" in code
    # the expected hash defaults to EMPTY, so nothing is auto-accepted
    assert 'os.environ.get(\n    "MEDNORM_EXPECTED_SMOKE_CHECKPOINT_SHA256", "")' in code
    # the default artifact directory is v4, never a historical one
    assert 'f"s1_mention_first_run_smoke_{SMOKE_ARTIFACT_VERSION}"' in code
    assert 'MEDNORM_SMOKE_ARTIFACT_VERSION", "v4"' in code


def test_no_notebook_hardcodes_a_checkpoint_digest() -> None:
    """Accepting a rerun must never require editing a notebook's Python."""
    for notebook in (S1_SMOKE, S1_VALIDATION, S1_FULL):
        assert not re.search(r"[0-9a-f]{64}", _code(notebook)), notebook


def test_validation_notebook_prints_the_recomputed_hash_for_confirmation() -> None:
    code = _code(S1_VALIDATION)
    assert 'outcome.diagnostics["expected_checkpoint_sha256_supplied"]' in code
    assert "outcome.checkpoint_sha256" in code
    assert "No Python source needs to change." in code


def test_full_training_notebook_refuses_every_historical_artifact() -> None:
    code = _code(S1_FULL)
    assert "for _historical in HISTORICAL_SMOKE_ARTIFACT_DIRS:" in code
    assert "SMOKE_ARTIFACT_DIR.resolve() != _historical.resolve()" in code
    assert "must not authorize full training" in code
    # and it passes the validated directory into the training config
    assert "smoke_artifact_dir=SMOKE_ARTIFACT_DIR" in code
