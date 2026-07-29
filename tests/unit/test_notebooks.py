"""Colab notebook structure + safety validation (Audit 0017). No notebook is run."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks"


def _tracked_notebooks() -> list[str]:
    """Notebook names git actually tracks.

    The inventory is deliberately taken from git rather than from a directory
    listing. An untracked `.ipynb` is not part of any reproducible workflow — it
    cannot be cloned, reviewed or pinned — so it must neither join the contract
    silently nor make the inventory assertion fail for an operator who happens to
    have a scratch copy on disk. `test_no_untracked_notebook_is_present` reports
    strays separately, which is where a stray belongs: in a named failure, not
    hidden inside a count.
    """
    out = subprocess.check_output(
        ["git", "ls-files", "notebooks"], cwd=REPO, text=True).splitlines()
    return sorted(Path(name).name for name in out if name.endswith(".ipynb"))
STAGE_NOTEBOOKS = [
    "MedNorm_S0_DomainAdaptation.ipynb",
    "MedNorm_S1_MentionExtraction.ipynb",
    "MedNorm_S2_Assertion.ipynb",
    "MedNorm_S3_Retrieval.ipynb",
    "MedNorm_S4_Reranker.ipynb",
    "MedNorm_S5_QwenLoRA.ipynb",
    "MedNorm_S6_Calibration.ipynb",
]
ALL_NOTEBOOKS = _tracked_notebooks()
_SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN)")
_PERSONAL = re.compile(r"/home/\w+|/Users/[A-Za-z0-9]+")


def _src(name: str) -> str:
    d = json.loads((NB / name).read_text(encoding="utf-8"))
    return "".join("".join(c.get("source", [])) for c in d["cells"])


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_notebook_valid_json_and_safe(name: str) -> None:
    d = json.loads((NB / name).read_text(encoding="utf-8"))
    assert d["nbformat"] == 4 and d["cells"]
    src = _src(name)
    assert not _SECRET.search(src), f"{name}: secret-like token"
    assert not _PERSONAL.search(src), f"{name}: personal path"
    assert "RUN_FULL_TRAINING = True" not in src, f"{name}: auto full training"
    # A single editable Drive root. Stage notebooks hard-code PROJECT_ROOT; the VietMed
    # notebook derives DRIVE_ROOT from an env-overridable default (Audit 0019) so the
    # behavioral test can redirect it. No other hard-coded user path is allowed.
    assert ("PROJECT_ROOT = '/content/drive/MyDrive/mednorm-vi'" in src
            or "'/content/drive/MyDrive/MedNorm-VI'" in src
            or '"/content/drive/MyDrive/MedNorm-VI"' in src
            or "packaging" in name.lower() or "Folds" in name)


REQUIRED_SECTIONS = [
    "Colab", "GPU", "Drive", "Pinned dependencies", "Git commit",
    "split hash", "revision", "Smoke mode", "Full mode OFF", "Resume",
    "Checkpoint manifest", "Return-to-repository",
]


@pytest.mark.parametrize("name", STAGE_NOTEBOOKS)
def test_stage_notebook_has_required_sections(name: str) -> None:
    src = _src(name)
    for section in REQUIRED_SECTIONS:
        assert section in src, f"{name}: missing required section {section!r}"
    # full training is gated behind an explicit confirmation
    assert "CONFIRM_FULL" in src and "SystemExit" in src


# Audit 0017 documents exactly this inventory (1 VietMed prep + 7 stage S0-S6
# + 1 S1 smoke + 1 retained inference/packaging = 10).
AUDIT_0017_INVENTORY = {
    "MedNorm_Data_VietMed_Preprocess.ipynb",
    "MedNorm_S0_DomainAdaptation.ipynb",
    "MedNorm_S1_MentionExtraction.ipynb",
    "MedNorm_S2_Assertion.ipynb",
    "MedNorm_S3_Retrieval.ipynb",
    "MedNorm_S4_Reranker.ipynb",
    "MedNorm_S5_QwenLoRA.ipynb",
    "MedNorm_S6_Calibration.ipynb",
    "MedNorm_S1_Mention_FirstRun_Smoke.ipynb",
    "MedNorm_Full_Offline_Inference_and_Packaging.ipynb",
}

# Notebooks added after Audit 0017. Audits are immutable, so each later notebook
# is documented in the audit that introduced it, not retrofitted into 0017.
LATER_NOTEBOOKS = {
    "MedNorm_S1_Smoke_Artifact_Validation.ipynb":
        "0025-s1-smoke-artifact-validation-and-full-training-prep.md",
    "MedNorm_S1_Mention_Full_Training.ipynb":
        "0025-s1-smoke-artifact-validation-and-full-training-prep.md",
    "MedNorm_S1_Mention_InternalTest_Evaluation.ipynb":
        "0031-s1-full-training-run-validation-and-internal-test-evaluation.md",
    "MedNorm_E5_XLMR_MRC_NER_Training.ipynb":
        "0035-multi-expert-mention-ensemble-and-learned-l4-v2.md",
    "MedNorm_L4_Learned_Resolver_v2_Training.ipynb":
        "0035-multi-expert-mention-ensemble-and-learned-l4-v2.md",
    "MedNorm_Phase2_Validation_Ablation.ipynb":
        "0036-phase2-training-readiness-and-artifact-validation.md",
}
EXPECTED_INVENTORY = AUDIT_0017_INVENTORY | set(LATER_NOTEBOOKS)


# Notebooks Audit 0051 deleted, with the audit that removed them. They must stay
# absent: a notebook whose only purpose is a retired expert or an abandoned
# baseline is a live entry point into work the project has stopped doing.
REMOVED_NOTEBOOKS = {
    "MedNorm_E4_Clean_Training.ipynb":
        "0051-repository-wide-architecture-review-and-cleanup.md",
    "MedNorm_ZS0_Baseline_Submission.ipynb":
        "0051-repository-wide-architecture-review-and-cleanup.md",
    # Untracked stray deleted by owner decision (Audit 0051 section 11.3). Listed
    # here so it cannot quietly return, tracked or untracked.
    "MedNorm_E4_PhoBERT_W2NER_Training.ipynb":
        "0051-repository-wide-architecture-review-and-cleanup.md",
}


def test_notebook_inventory_matches_expected() -> None:
    assert set(ALL_NOTEBOOKS) == EXPECTED_INVENTORY
    assert len(ALL_NOTEBOOKS) == 16
    audit_dir = REPO / "docs" / "audits"
    audit_0017 = (audit_dir / "0017-training-readiness-and-governed-corpus.md").read_text(
        encoding="utf-8")
    for name in AUDIT_0017_INVENTORY:
        assert name in audit_0017, f"Audit 0017 does not list notebook {name}"
    for name, audit_name in LATER_NOTEBOOKS.items():
        audit = (audit_dir / audit_name).read_text(encoding="utf-8")
        assert name in audit, f"{audit_name} does not list notebook {name}"


def test_no_untracked_notebook_is_present() -> None:
    """A notebook on disk but not in git is unreviewable and unreproducible.

    It cannot be cloned, pinned or reviewed, and because `notebooks/` is not
    git-ignored it is one `git add -A` away from entering the repository unread.
    There are **no exceptions**: Audit 0051 briefly carried a quarantine list for one
    such stray, the owner decided to delete it, and the list was removed with it.
    """
    on_disk = {path.name for path in NB.glob("*.ipynb")}
    stray = sorted(on_disk - set(ALL_NOTEBOOKS))
    assert not stray, (
        f"untracked notebook(s) present: {stray}. Either commit them with an audit "
        "or remove them; an untracked notebook is one `git add -A` away from "
        "entering the repository unreviewed.")


def test_removed_notebooks_stay_removed_and_are_documented() -> None:
    audit_dir = REPO / "docs" / "audits"
    for name, audit_name in REMOVED_NOTEBOOKS.items():
        assert not (NB / name).exists(), (
            f"{name} was deleted and must not return to disk")
        assert name not in _tracked_notebooks(), (
            f"{name} was deleted and must not be tracked again")
        assert name not in set(ALL_NOTEBOOKS)
        audit = (audit_dir / audit_name).read_text(encoding="utf-8")
        assert name in audit, f"{audit_name} does not record removing {name}"


def test_vietmed_preprocess_is_cpu_dataprep_not_training() -> None:
    src = _src("MedNorm_Data_VietMed_Preprocess.ipynb")
    assert "pyarrow==" in src                      # pinned parquet reader
    assert "RUN_FULL_TRAINING" not in src          # not a training notebook
    assert "audio" in src.lower()                  # explicitly handles/excludes audio
    assert "run_mode=RUN_MODE" in src              # future manifests persist execution mode


def test_s1_first_run_smoke_uses_governed_config_and_drive_artifacts() -> None:
    src = _src("MedNorm_S1_Mention_FirstRun_Smoke.ipynb")
    assert 'Path("/content/drive/MyDrive/MedNorm-VI")' in src
    assert 'Path("/content/MedNorm-VI")' in src
    assert "configs/training/s1_mention_first_run_smoke.yaml" in src
    assert "verify_governed_corpus(CORPUS_DIR, expected_corpus)" in src
    # Audit 0026: the corrected rerun writes to a versioned directory so the
    # historical v1 artifact (full_training_readiness: false) is never overwritten.
    assert ('OUTPUT_DIR = DRIVE_ROOT / "artifacts" / '
            'f"s1_mention_first_run_smoke_{SMOKE_ARTIFACT_VERSION}"') in src
    assert "HISTORICAL_SMOKE_ARTIFACT_DIRS = (" in src
    assert "MODEL_CACHE_DIR = DRIVE_ROOT" in src
    assert "AutoTokenizer.from_pretrained" in src
    assert "AutoModel.from_pretrained" in src
    assert "checkpoint_reload_succeeded" in src
    assert "SMOKE_ONLY" in src
    assert "output.zip" not in src


def test_s1_first_run_smoke_notebook_status_is_implemented_unexecuted() -> None:
    report = (REPO / "docs" / "notebooks" / "notebook_execution_integrity.md").read_text(
        encoding="utf-8")
    line = next(
        line for line in report.splitlines()
        if "MedNorm_S1_Mention_FirstRun_Smoke.ipynb" in line
    )
    assert "IMPLEMENTED_UNEXECUTED" in line
    assert "SYNTHETIC_SMOKE_VERIFIED" not in line
    assert "ARTIFACTS_VERIFIED" not in line
