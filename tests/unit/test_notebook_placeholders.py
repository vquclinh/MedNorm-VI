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
ALL_NOTEBOOKS = sorted(p.name for p in NB.glob("*.ipynb"))

# Notebooks honestly classified as DESIGN_DRAFT in
# docs/notebooks/notebook_execution_integrity.md. They may still contain
# placeholders, but must never be described as executable/complete.
DESIGN_DRAFTS = {
    "MedNorm_S0_DomainAdaptation.ipynb",
    "MedNorm_S1_MentionExtraction.ipynb",
    "MedNorm_S1_Mention_FirstRun_Smoke.ipynb",
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
            # No notebook may claim real-data completion in this milestone.
            assert "pending" in line.lower(), f"unsupported real-data claim: {line}"


def test_all_notebooks_are_valid_json() -> None:
    for name in ALL_NOTEBOOKS:
        doc = json.loads((NB / name).read_text(encoding="utf-8"))
        assert doc["nbformat"] == 4 and doc["cells"]
