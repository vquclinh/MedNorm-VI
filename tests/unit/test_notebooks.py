"""Colab notebook structure + safety validation (Audit 0017). No notebook is run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks"
STAGE_NOTEBOOKS = [
    "MedNorm_S0_DomainAdaptation.ipynb",
    "MedNorm_S1_MentionExtraction.ipynb",
    "MedNorm_S2_Assertion.ipynb",
    "MedNorm_S3_Retrieval.ipynb",
    "MedNorm_S4_Reranker.ipynb",
    "MedNorm_S5_QwenLoRA.ipynb",
    "MedNorm_S6_Calibration.ipynb",
]
ALL_NOTEBOOKS = sorted(p.name for p in NB.glob("*.ipynb"))
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
    # editable single Drive root (stage notebooks use PROJECT_ROOT; the VietMed
    # preprocessing notebook uses the clearer DRIVE_ROOT). No other hard-coded path.
    assert ("PROJECT_ROOT = '/content/drive/MyDrive/mednorm-vi'" in src
            or "DRIVE_ROOT = '/content/drive/MyDrive/MedNorm-VI'" in src
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


EXPECTED_INVENTORY = {
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


def test_notebook_inventory_matches_expected() -> None:
    # Audit 0017 documents exactly this inventory (1 VietMed prep + 7 stage S0-S6
    # + 1 S1 smoke + 1 retained inference/packaging = 10). Keep them in sync.
    assert set(ALL_NOTEBOOKS) == EXPECTED_INVENTORY
    assert len(ALL_NOTEBOOKS) == 10
    audit = (REPO / "docs" / "audits"
             / "0017-training-readiness-and-governed-corpus.md").read_text(encoding="utf-8")
    for name in EXPECTED_INVENTORY:
        assert name in audit, f"audit does not list notebook {name}"


def test_vietmed_preprocess_is_cpu_dataprep_not_training() -> None:
    src = _src("MedNorm_Data_VietMed_Preprocess.ipynb")
    assert "pyarrow==" in src                      # pinned parquet reader
    assert "RUN_FULL_TRAINING" not in src          # not a training notebook
    assert "audio" in src.lower()                  # explicitly handles/excludes audio
