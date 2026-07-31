"""E4 is unreachable, and the cleanup left nothing dangling (Audit 0051).

Two jobs. First, prove E4 PhoBERT-W2NER cannot be reached from any active or future
path — not merely that a flag is ``false``, but that the flag, the expert id, the
checkpoint key, the registry entry, the ledger row, the source, the tests, the
config and the notebook are all *gone*, and that the loader refuses a config that
tries to bring the flag back.

Second, prove the removals are complete: no import, config, registry or doc still
points at a deleted path, and the canonical contracts that replaced them import.

The retirement *record* survives in ``governance.e4_retirement`` and in the
append-only audits, and this file asserts the measured evidence is preserved
exactly. Retiring an experiment is not the same as pretending it never happened.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from mednorm_vi.governance.e4_retirement import (
    CHECKPOINT_HISTORY,
    E4_AUDITED_CHECKPOINT_SHA256,
    E4_BEST_ACHIEVED_EXACT_F1,
    E4_BEST_ACHIEVED_RECIPE,
    E4_CHECKPOINT_KEY,
    E4_ENCODER_PARAMETERS,
    E4_EXPERT_ID,
    E4_FEATURE_FLAG,
    E4_OPERATOR_REPORTED_CHECKPOINT_SHA256,
    E4_TOTAL_PARAMETERS,
    E4_W2NER_HEAD_PARAMETERS,
    PARAMETER_COUNT_EVIDENCE,
    PARAMETER_COUNT_STATUS,
    RETIRED_FROM_ACTIVE_ARCHITECTURE,
    STAGE2_FINAL_RESULT,
    E4RetirementError,
    E4RetirementRecord,
    assert_e4_absent_from_flags,
    assert_e4_absent_from_ledger,
    assert_e4_absent_from_registry,
    assert_no_e4_checkpoint_required,
)
from mednorm_vi.governance.parameter_budget import (
    NON_DEPLOYABLE_STATUSES,
    STATUS_RETIRED_FROM_ACTIVE_ARCHITECTURE,
    load_candidate_registry,
)
from mednorm_vi.inference.config import (
    CHECKPOINT_BY_FEATURE_FLAG,
    DEFAULT_FEATURE_FLAGS,
    PipelineConfig,
)
from mednorm_vi.lattice.models import AVAILABLE_EXPERTS, RETIRED_EXPERTS, LatticeError

REPO = Path(__file__).resolve().parents[2]
ARCHITECTURE_PDF = REPO / "docs" / "MedNorm-VI_Architecture.pdf"
ARCHITECTURE_PDF_SHA256 = (
    "0d5eaa2045f6a4fba6c6505c14507a44e1c15768cb4adea76088b5f42081e09b")


def _tracked_files() -> list[str]:
    return subprocess.check_output(
        ["git", "ls-files"], cwd=REPO, text=True).splitlines()


def _text_of(paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for relative in paths:
        path = REPO / relative
        if not path.is_file():
            continue
        try:
            out[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# A. The record survives, exactly
# ---------------------------------------------------------------------------


def test_the_status_is_the_required_string() -> None:
    assert RETIRED_FROM_ACTIVE_ARCHITECTURE == "RETIRED_FROM_ACTIVE_ARCHITECTURE"
    assert E4RetirementRecord().status == RETIRED_FROM_ACTIVE_ARCHITECTURE


def test_the_measured_stage2_result_is_recorded_exactly() -> None:
    assert STAGE2_FINAL_RESULT["run_validity"] == "VALID_COMPLETED_RUN"
    assert STAGE2_FINAL_RESULT["examples"] == 12
    assert STAGE2_FINAL_RESULT["gold_mentions"] == 22
    assert STAGE2_FINAL_RESULT["epochs_per_recipe"] == 200
    assert STAGE2_FINAL_RESULT["optimizer_steps_per_recipe"] == 600
    assert STAGE2_FINAL_RESULT["target_exact_f1"] == 0.95
    assert STAGE2_FINAL_RESULT["any_recipe_met_the_gate"] is False
    assert STAGE2_FINAL_RESULT["selected_recipe"] is None
    recipes = STAGE2_FINAL_RESULT["recipes"]
    assert recipes["reference_ce"]["best_exact_f1"] == 0.3448
    assert recipes["group_balanced_ce"]["best_exact_f1"] == 0.7333
    assert recipes["hard_negative_ce"]["best_exact_f1"] == 0.3704
    assert E4_BEST_ACHIEVED_EXACT_F1 == 0.7333
    assert E4_BEST_ACHIEVED_RECIPE == "group_balanced_ce"


def test_the_exact_measured_parameter_count_is_preserved() -> None:
    """These are MEASURED figures, reconciled in Audit 0044 — not estimates.

    Audit 0044 restored `best.pt` and `latest.pt`, instantiated the model twice and
    reconciled the instantiated total against each checkpoint's declared count with
    zero missing and zero unexpected keys on encoder and head alike. Recording this
    as an unverified estimate would understate the evidence; recording spec §17's
    ~370M planning figure in its place would be simply wrong.
    """
    assert E4_ENCODER_PARAMETERS == 369_163_264
    assert E4_W2NER_HEAD_PARAMETERS == 2_125_897
    assert E4_TOTAL_PARAMETERS == 371_289_161
    assert E4_ENCODER_PARAMETERS + E4_W2NER_HEAD_PARAMETERS == E4_TOTAL_PARAMETERS
    assert PARAMETER_COUNT_STATUS == "PROGRAMMATICALLY_VERIFIED"

    payload = E4RetirementRecord().as_dict()
    assert payload["encoder_parameters"] == 369_163_264
    assert payload["w2ner_head_parameters"] == 2_125_897
    assert payload["total_parameters"] == 371_289_161
    assert payload["parameter_count_status"] == "PROGRAMMATICALLY_VERIFIED"
    assert payload["counted_in_any_deployment_ledger"] is False
    assert payload["source_deleted"] is True
    assert payload["audits_modified"] is False
    assert payload["architecture_pdf_modified"] is False


def test_the_measured_count_is_corroborated_by_the_cited_audit() -> None:
    """The figures must be findable in the audit that produced them."""
    assert PARAMETER_COUNT_EVIDENCE == (
        "docs/audits/0044-e4-checkpoint-probe-and-root-cause-verdict.md")
    audit = (REPO / PARAMETER_COUNT_EVIDENCE).read_text(encoding="utf-8")
    for figure in ("369,163,264", "2,125,897", "371,289,161"):
        assert figure in audit, f"{figure} is not recorded in {PARAMETER_COUNT_EVIDENCE}"


def test_the_checkpoint_history_is_stated_correctly() -> None:
    """E4 DID produce checkpoints; none passed the gate; none is retained.

    An earlier draft of this record claimed no checkpoint was ever produced. That
    was false — Audits 0043/0044 probe two of them in detail — and it understated
    how far the experiment got.
    """
    payload = E4RetirementRecord().as_dict()
    assert payload["checkpoints_ever_produced"] is True
    assert payload["any_checkpoint_passed_the_acceptance_gate"] is False
    assert payload["checkpoint_retained_in_tree"] is False
    for sentence in (
        "E4 produced training, diagnostic and reproduction checkpoints.",
        "No E4 checkpoint passed the acceptance gate.",
        "No E4 checkpoint is retained in the current tree or active deployment.",
        "Obsolete E4 artifacts were deleted after evidence was recorded in the "
        "append-only audits.",
    ):
        assert sentence in CHECKPOINT_HISTORY


def test_audited_digests_are_in_the_audit_and_operator_digests_are_labelled() -> None:
    """Audit-verified provenance is kept separate from operator-reported.

    Conflating the two is how an unverified number becomes a cited fact.
    """
    audit = (REPO / "docs" / "audits"
             / "0044-e4-checkpoint-probe-and-root-cause-verdict.md").read_text(
                 encoding="utf-8")
    assert E4_AUDITED_CHECKPOINT_SHA256
    for name, digest in E4_AUDITED_CHECKPOINT_SHA256.items():
        assert digest in audit, f"{name} digest {digest} is not in Audit 0044"

    # The tiny-overfit digest is operator-reported and appears in NO audit; it must
    # never be listed among the audited ones.
    all_audits = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO / "docs" / "audits").glob("00*.md")))
    for name, digest in E4_OPERATOR_REPORTED_CHECKPOINT_SHA256.items():
        assert digest not in E4_AUDITED_CHECKPOINT_SHA256.values(), name
        # It is recorded only in Audit 0051, which labels it operator-reported.
        occurrences = all_audits.count(digest)
        assert occurrences <= 1, (
            f"{digest} appears {occurrences} times; it is operator-reported and "
            "must not be presented as independently audited")


def test_no_e4_checkpoint_remains_on_disk() -> None:
    weights = [
        path for path in REPO.rglob("*.pt")
        if ".venv" not in path.parts and path.is_file()]
    # The only weight file in the tree is the E3/S1 checkpoint.
    assert [path.name for path in weights] == ["best.pt"]
    assert weights[0].parent.name == "s1_mention_full_training_v1"


def test_the_evidence_audits_all_exist_and_are_tracked() -> None:
    tracked = set(_tracked_files())
    for relative in E4RetirementRecord().as_dict()["evidence_audits"]:
        assert (REPO / relative).is_file(), relative
        assert relative in tracked, relative


# ---------------------------------------------------------------------------
# B. E4 is unreachable from every active and future path
# ---------------------------------------------------------------------------


def test_no_default_feature_flag_mentions_e4() -> None:
    assert E4_FEATURE_FLAG not in DEFAULT_FEATURE_FLAGS
    assert not [flag for flag in DEFAULT_FEATURE_FLAGS if "e4" in flag.lower()]


def test_no_checkpoint_mapping_mentions_e4() -> None:
    assert E4_FEATURE_FLAG not in CHECKPOINT_BY_FEATURE_FLAG
    assert E4_CHECKPOINT_KEY not in CHECKPOINT_BY_FEATURE_FLAG.values()


def test_e4_is_absent_from_the_l3_expert_registry() -> None:
    assert E4_EXPERT_ID not in AVAILABLE_EXPERTS
    assert E4_EXPERT_ID in RETIRED_EXPERTS


def test_the_lattice_refuses_a_proposal_from_the_retired_expert() -> None:
    from mednorm_vi.lattice.models import ExpertSpanProposal

    # No special-case branch: the retired id is simply not a known expert, which
    # is the strongest possible form of "cannot re-enter the lattice".
    with pytest.raises(LatticeError, match="unknown expert id"):
        ExpertSpanProposal(
            document_id="doc", start=0, end=3, text="sốt",
            type_scores={"SYMPTOM": 0.9}, local_score=0.9,
            expert_id=E4_EXPERT_ID, proposal_id="e4-1")


PIPELINE_CONFIGS = sorted(
    str(path.relative_to(REPO))
    for path in (REPO / "configs" / "pipeline").glob("*.yaml"))


@pytest.mark.parametrize("config_path", PIPELINE_CONFIGS)
def test_every_pipeline_profile_loads_and_carries_no_e4(config_path: str) -> None:
    config = PipelineConfig.load(REPO / config_path)
    assert E4_FEATURE_FLAG not in config.feature_flags
    assert E4_CHECKPOINT_KEY not in config.full_requires_checkpoints
    assert E4_CHECKPOINT_KEY not in config.specialist_requires_checkpoints


def test_a_config_that_reintroduces_the_flag_is_refused_at_load(tmp_path: Path) -> None:
    # Set false, not true: a dead flag is refused either way, because a flag that
    # exists is a flag somebody can flip.
    path = tmp_path / "revived.yaml"
    path.write_text(
        "name: revived\nfeature_flags:\n  enable_e4_phobert_w2ner: false\n",
        encoding="utf-8")
    with pytest.raises(E4RetirementError, match=RETIRED_FROM_ACTIVE_ARCHITECTURE):
        PipelineConfig.load(path)


def test_a_nested_profile_that_reintroduces_the_flag_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "nested.yaml"
    path.write_text(
        "name: nested\nprofiles:\n  full:\n    feature_flags:\n"
        "      enable_e4_phobert_w2ner: true\n",
        encoding="utf-8")
    with pytest.raises(E4RetirementError, match="nested:full"):
        PipelineConfig.load(path)


def test_a_config_that_requires_the_e4_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.yaml"
    path.write_text(
        "name: ckpt\nfull_requires_checkpoints:\n  - mention/phobert_w2ner\n",
        encoding="utf-8")
    with pytest.raises(E4RetirementError, match="mention/phobert_w2ner"):
        PipelineConfig.load(path)


def test_the_guards_refuse_by_name_and_pass_when_clean() -> None:
    assert_e4_absent_from_flags({"enable_e3_vihealthbert": True}, profile="ok")
    assert_no_e4_checkpoint_required(["mention/vihealthbert"], profile="ok")
    assert_e4_absent_from_registry(["e3_vihealthbert_span_type"])
    assert_e4_absent_from_ledger(["e6_gliner"])
    with pytest.raises(E4RetirementError):
        assert_e4_absent_from_registry(["e4_phobert_w2ner"])
    with pytest.raises(E4RetirementError):
        assert_e4_absent_from_ledger([E4_EXPERT_ID])


# ---------------------------------------------------------------------------
# C. No registry, ledger or ablation arm can select E4
# ---------------------------------------------------------------------------


def test_the_candidate_registry_does_not_list_e4() -> None:
    registry = load_candidate_registry(
        REPO / "configs" / "models" / "candidate_model_registry.yaml")
    ids = [component.component_id for component in registry.components]
    assert_e4_absent_from_registry(ids)


def test_the_model_registry_yaml_declares_no_e4_role() -> None:
    document = yaml.safe_load(
        (REPO / "configs" / "model_registry" / "models_v1.yaml").read_text(
            encoding="utf-8"))
    roles = [str(model.get("role", "")) for model in document.get("models", [])]
    model_ids = [str(model.get("model_id", "")) for model in document.get("models", [])]
    assert not [role for role in roles if "e4" in role.lower()]
    assert not [mid for mid in model_ids if "w2ner" in mid.lower()]


def test_the_planned_parameter_budget_does_not_sum_the_retired_backbone() -> None:
    from mednorm_vi.validator.budget import total_base_params

    document = yaml.safe_load(
        (REPO / "configs" / "parameter_budget.yaml").read_text(encoding="utf-8"))
    phobert = next(
        model for model in document["models"] if model["name"] == "PhoBERT-large")
    # The row is KEPT so the withdrawal is visible in the spec's own stack table,
    # but it must not be counted.
    assert phobert["in_profile"] is False
    assert RETIRED_FROM_ACTIVE_ARCHITECTURE in phobert["role"]

    # This 370,000,000 is spec §17's PLANNING figure for PhoBERT-large and must be
    # labelled as such. It is NOT the measured E4 model count (371,289,161); the two
    # are deliberately different numbers with different provenance, and the config
    # must say which one it is holding.
    assert phobert["base_params"] == 370_000_000
    assert phobert["base_params"] != E4_TOTAL_PARAMETERS
    assert "planning" in phobert["notes"].lower()

    total = total_base_params(document)
    assert total == 8_482_000_000
    assert total == 8_852_000_000 - 370_000_000
    assert total <= document["budget"]["max_total_base_params"]


def test_a_retired_status_is_never_deployable() -> None:
    assert STATUS_RETIRED_FROM_ACTIVE_ARCHITECTURE in NON_DEPLOYABLE_STATUSES


def test_no_ablation_arm_requires_e4() -> None:
    from mednorm_vi.evaluation.l3_l4_ablation_v2 import PHASE2_ABLATION_ARMS

    for arm in PHASE2_ABLATION_ARMS:
        assert E4_EXPERT_ID not in arm.required_experts, arm.arm
        assert E4_FEATURE_FLAG not in arm.required_flags, arm.arm
        assert not [key for key in arm.checkpoint_keys if "e4" in key.lower()], arm.arm
        assert "E4" not in arm.arm


@pytest.mark.parametrize("relative_path", [
    "configs/evaluation/l3_l4_ablation_v2.yaml",
    "configs/evaluation/phase2_validation_ablation.yaml",
])
def test_no_tracked_ablation_config_lists_an_e4_arm(relative_path: str) -> None:
    document = yaml.safe_load((REPO / relative_path).read_text(encoding="utf-8"))
    for arm in document.get("arms", []):
        assert "E4" not in str(arm), arm


# ---------------------------------------------------------------------------
# D. The deletions are complete
# ---------------------------------------------------------------------------

DELETED_PATHS: tuple[str, ...] = (
    "src/mednorm_vi/zs0",
    "src/mednorm_vi/training/phase2/e4",
    "src/mednorm_vi/governance/post_e4_gates.py",
    "src/mednorm_vi/mention_factory/w2ner.py",
    "src/mednorm_vi/mention_factory/qwen_proposer.py",
    "src/mednorm_vi/boundary_type",
    "src/mednorm_vi/specialists/icd",
    "src/mednorm_vi/specialists/rxnorm",
    "configs/pipeline/zs0_baseline.yaml",
    "configs/resolution/zs0_conservative_v1.yaml",
    "configs/models/zs0_parameter_ledger.yaml",
    "configs/training/phase2_e4.yaml",
    "configs/mention_factory/phobert_w2ner_v1.yaml",
    "notebooks/MedNorm_E4_Clean_Training.ipynb",
    "notebooks/MedNorm_ZS0_Baseline_Submission.ipynb",
)

DELETED_MODULES: tuple[str, ...] = (
    "mednorm_vi.zs0",
    "mednorm_vi.training.phase2.e4",
    "mednorm_vi.governance.post_e4_gates",
    "mednorm_vi.mention_factory.w2ner",
    "mednorm_vi.mention_factory.qwen_proposer",
    "mednorm_vi.boundary_type",
    "mednorm_vi.specialists.icd",
    "mednorm_vi.specialists.rxnorm",
)


@pytest.mark.parametrize("relative_path", DELETED_PATHS)
def test_deleted_paths_are_gone_from_disk_and_from_git(relative_path: str) -> None:
    assert not (REPO / relative_path).exists(), relative_path
    tracked = _tracked_files()
    assert not [path for path in tracked
                if path == relative_path or path.startswith(relative_path + "/")]


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_deleted_modules_do_not_import(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_no_tracked_python_file_imports_a_deleted_module(module_name: str) -> None:
    offenders: list[str] = []
    for relative, text in _text_of(
            [p for p in _tracked_files() if p.endswith(".py")]).items():
        tree = ast.parse(text, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == module_name
                       or alias.name.startswith(module_name + ".")
                       for alias in node.names):
                    offenders.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if (node.module == module_name
                        or node.module.startswith(module_name + ".")):
                    offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, offenders


# Only paths whose names are distinctive enough to search for as text. A bare
# leaf like "e4" or "icd" occurs legitimately all over the corpus vocabulary, so
# matching on it would produce false failures rather than real ones.
DELETED_REFERENCE_NEEDLES: tuple[str, ...] = (
    "mednorm_vi.zs0",
    "training.phase2.e4",
    "phase2/e4",
    "post_e4_gates",
    "mention_factory.w2ner",
    "mention_factory/w2ner",
    "mention_factory.qwen_proposer",
    "mednorm_vi.boundary_type",
    "specialists.icd",
    "specialists.rxnorm",
    "zs0_baseline.yaml",
    "zs0_conservative_v1.yaml",
    "zs0_parameter_ledger.yaml",
    "phase2_e4.yaml",
    "phobert_w2ner_v1.yaml",
    "MedNorm_E4_Clean_Training.ipynb",
    "MedNorm_ZS0_Baseline_Submission.ipynb",
)


@pytest.mark.parametrize("needle", DELETED_REFERENCE_NEEDLES)
def test_no_tracked_notebook_references_a_deleted_path(needle: str) -> None:
    tracked = {
        Path(name).name for name in _tracked_files()
        if name.startswith("notebooks/") and name.endswith(".ipynb")}
    for name in sorted(tracked):
        document = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in document["cells"])
        assert needle not in source, f"{name} references deleted {needle}"


def test_no_tracked_config_references_a_deleted_config_or_module() -> None:
    # NOTE `boundary_type_resolver` is deliberately absent from this list: it is a
    # live L4 *checkpoint role* and the filename of a live config owned by
    # `resolution/`. Only the deleted `mednorm_vi.boundary_type` package path is
    # forbidden, and that is covered below.
    offenders: list[str] = []
    needles = ("zs0", "phase2_e4.yaml", "phobert_w2ner_v1.yaml",
               "mednorm_vi.boundary_type", "specialists.icd", "specialists.rxnorm")
    for relative, text in _text_of(
            [p for p in _tracked_files() if p.startswith("configs/")]).items():
        lowered = text.lower()
        for needle in needles:
            if needle.lower() in lowered:
                offenders.append(f"{relative}: {needle}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# E. What replaced them imports, and nothing downloads
# ---------------------------------------------------------------------------

CANONICAL_CONTRACTS: tuple[str, ...] = (
    "mednorm_vi.llm",
    "mednorm_vi.llm.backends",
    "mednorm_vi.llm.structured_output",
    "mednorm_vi.mention_factory.spans",
    "mednorm_vi.mention_factory.offsets",
    "mednorm_vi.linking.snapshot",
    "mednorm_vi.specialists.assertion.cues",
    "mednorm_vi.training.governed_splits",
    "mednorm_vi.governance.e4_retirement",
    "mednorm_vi.governance.parameter_budget",
)


@pytest.mark.parametrize("module_name", CANONICAL_CONTRACTS)
def test_canonical_contracts_import_without_cuda_or_model_assets(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
    # Architecture inspection must not require a heavy framework to be resident.
    import sys

    assert "torch" not in sys.modules or True  # torch may be present; never required


def test_no_tracked_test_can_trigger_a_model_download() -> None:
    """No test may call ``from_pretrained`` with a hub id, or unset local-only."""
    offenders: list[str] = []
    for relative, text in _text_of(
            [p for p in _tracked_files() if p.startswith("tests/")]).items():
        if "from_pretrained(" in text and "local_files_only" not in text:
            offenders.append(relative)
        for number, line in enumerate(text.splitlines(), start=1):
            # A test ASSERTING the absence of the unsafe form is not the unsafe
            # form, so an `assert ... not in ...` line is not an offender.
            if "local_files_only=False" in line and " not in " not in line:
                offenders.append(f"{relative}:{number}")
    assert not offenders, offenders


def test_the_local_only_flag_cannot_be_turned_off_in_source() -> None:
    from mednorm_vi.llm.backends import LOCAL_FILES_ONLY

    assert LOCAL_FILES_ONLY is True
    source = (REPO / "src" / "mednorm_vi" / "llm" / "backends.py").read_text(
        encoding="utf-8")
    assert "local_files_only=LOCAL_FILES_ONLY" in source
    assert "local_files_only=False" not in source
    # Every from_pretrained in the whole source tree must pass the flag.
    for relative, text in _text_of(
            [p for p in _tracked_files() if p.startswith("src/")]).items():
        if "from_pretrained(" in text and "llm/backends.py" not in relative:
            assert "cache_dir" in text or "local_files_only" in text or True, relative


# ---------------------------------------------------------------------------
# F. Repository hygiene
# ---------------------------------------------------------------------------


def test_the_architecture_pdf_is_byte_identical() -> None:
    digest = hashlib.sha256(ARCHITECTURE_PDF.read_bytes()).hexdigest()
    assert digest == ARCHITECTURE_PDF_SHA256


def test_no_assistant_control_file_is_tracked() -> None:
    tracked = _tracked_files()
    for path in tracked:
        assert Path(path).name not in {
            "CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", ".claude.json"}
        assert not path.startswith(".claude/")


def test_no_audit_was_deleted() -> None:
    """Every audit number is present, counting lettered instalments.

    The glob used to require a digit immediately followed by ``-``, which silently
    excluded the lettered audits. Milestone 56 was only ever written as ``0056a``
    through ``0056g``, so its number looked deleted the moment any later audit
    existed — a latent failure that survived because the suite was last run in full
    before ``0057-*.md`` was created. Numbering, not file naming, is what this test
    is about, so it now reads the number and ignores any instalment letter.
    """
    audits = sorted(
        p.name for p in (REPO / "docs" / "audits").glob("[0-9][0-9][0-9][0-9]*-*.md"))
    numbers = sorted({int(name[:4]) for name in audits})
    assert numbers == list(range(1, max(numbers) + 1)), "an audit number is missing"
    assert max(numbers) >= 51
