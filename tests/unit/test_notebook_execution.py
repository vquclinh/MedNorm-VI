"""Behavioral execution of the tracked VietMed notebook (Audit 0019).

Static keyword checks previously passed while the notebook was unrunnable (its
`git clone` was commented out), so this test *actually executes* the notebook's
real code cells in a fresh namespace, in SYNTHETIC_SMOKE mode, against a local
``file://`` git repository. No network, no models, no real VietMed data.

`nbclient`/`nbformat` are not repository dependencies; this deterministic harness
executes the notebook's code cells directly (same sources, fresh globals), which
is the equivalent execution evidence required by the milestone.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "MedNorm_Data_VietMed_Preprocess.ipynb"
CHECKOUT_HELPER_SRC = REPO / "src"

EXPECTED_ARTIFACTS = (
    "canonical_examples/vietmed_ner_examples.jsonl",
    "manifests/vietmed_ner_preprocessing_manifest.json",
    "manifests/vietmed_ner_source_hashes.json",
    "quality/vietmed_ner_label_inventory.json",
    "quality/vietmed_ner_offset_validation.json",
    "quality/vietmed_ner_repair_exclusion_summary.json",
    "quality/vietmed_ner_split_summary.json",
)


def _code_cells(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in doc["cells"] if c["cell_type"] == "code"]


def _run_notebook(env: dict[str, str]) -> dict[str, object]:
    """Execute every code cell in a fresh namespace with the given env."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    namespace: dict[str, object] = {"__name__": "__mednorm_notebook__"}
    try:
        for idx, source in enumerate(_code_cells(NOTEBOOK), 1):
            try:
                exec(compile(source, f"<notebook-cell-{idx}>", "exec"), namespace)  # noqa: S102
            except Exception as exc:  # pragma: no cover - surfaced as test failure
                raise AssertionError(f"notebook cell {idx} failed: {exc}\n---\n{source}") from exc
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return namespace


@pytest.fixture(scope="module")
def smoke_run() -> dict[str, object]:
    """Execute the notebook once in SYNTHETIC_SMOKE mode against a file:// repo."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = {
            "MEDNORM_RUN_MODE": "SYNTHETIC_SMOKE",
            "MEDNORM_DRIVE_ROOT": str(tmp / "drive"),
            "MEDNORM_REPO_URL": f"file://{REPO}",
            "MEDNORM_REPO_REF": "HEAD",
            "MEDNORM_REPO_DIR": str(tmp / "checkout"),
            "MEDNORM_CHECKOUT_HELPER_SRC": str(CHECKOUT_HELPER_SRC),
        }
        ns = _run_notebook(env)
        artifact_dir = Path(str(ns["ARTIFACT_DIR"]))
        # Snapshot what the run produced before the temp dir disappears.
        produced = {
            rel: (artifact_dir / rel).read_text(encoding="utf-8")
            for rel in EXPECTED_ARTIFACTS if (artifact_dir / rel).is_file()
        }
        return {
            "namespace": ns,
            "produced": produced,
            "artifact_names": sorted(produced),
            "resolved_commit": ns.get("RESOLVED_COMMIT"),
            "manifest": ns.get("manifest"),
            "examples": ns.get("examples"),
        }


def test_notebook_executes_end_to_end_in_smoke_mode(smoke_run) -> None:
    # If any cell raised, the fixture would have failed. Adapter import must have worked.
    ns = smoke_run["namespace"]
    assert ns["RUN_MODE"] == "SYNTHETIC_SMOKE"
    assert ns["vm"] is not None                       # adapter imported from the checkout
    assert str(ns["vm"].ADAPTER_VERSION).startswith("vietmed-adapter")


def test_notebook_resolved_real_commit_sha(smoke_run) -> None:
    sha = str(smoke_run["resolved_commit"])
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    manifest = smoke_run["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["repo_commit"] == sha            # provenance is the resolved SHA
    assert manifest["repo_commit"] not in ("main", "HEAD", "")


def test_notebook_created_exact_artifact_tree(smoke_run) -> None:
    assert smoke_run["artifact_names"] == sorted(EXPECTED_ARTIFACTS)


def test_notebook_manifest_hash_matches_generated_jsonl(smoke_run) -> None:
    import hashlib
    produced = smoke_run["produced"]
    jsonl = produced["canonical_examples/vietmed_ner_examples.jsonl"]
    manifest = json.loads(produced["manifests/vietmed_ner_preprocessing_manifest.json"])
    assert manifest["examples_jsonl_sha256"] == hashlib.sha256(jsonl.encode()).hexdigest()
    assert manifest["offset_invalid"] == 0
    assert manifest["resolved_columns"] == {"word_col": "words", "tag_col": "labels"}


def test_notebook_output_contains_no_audio(smoke_run) -> None:
    jsonl = smoke_run["produced"]["canonical_examples/vietmed_ner_examples.jsonl"]
    assert "audio" not in jsonl
    manifest = json.loads(
        smoke_run["produced"]["manifests/vietmed_ner_preprocessing_manifest.json"])
    assert manifest["audio_excluded"] is True


def test_notebook_emitted_offset_valid_examples(smoke_run) -> None:
    jsonl = smoke_run["produced"]["canonical_examples/vietmed_ner_examples.jsonl"]
    rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
    assert rows, "smoke run produced no examples"
    for row in rows:
        for ent in row["entities"]:
            assert row["text"][ent["start"]:ent["end"]] == ent["text"]
    # the synthetic medication span must map approximately
    assert any(en["target_type"] == "MEDICATION" for r in rows for en in r["entities"])


def test_real_mode_fails_fast_without_parquet() -> None:
    """REAL mode must not silently fall back to synthetic when data is absent.

    Executes the notebook's REAL-mode source-discovery cell directly (the earlier
    cells install pyarrow, which must not run in the offline test sandbox), and
    asserts it raises rather than degrading to synthetic rows.
    """
    discovery = next(c for c in _code_cells(NOTEBOOK) if "missing Parquet dir" in c)
    with tempfile.TemporaryDirectory() as td:
        drive = Path(td) / "drive"
        ns = {
            "RUN_MODE": "REAL",
            "VIETMED_PARQUET_DIR": drive / "data" / "external" / "public_ner"
            / "vietmed_ner" / "data",
            "Path": Path,
        }
        exec(compile("import hashlib", "<setup>", "exec"), ns)  # noqa: S102
        with pytest.raises(AssertionError, match="missing Parquet dir"):
            exec(compile(discovery, "<real-discovery>", "exec"), ns)  # noqa: S102


def test_real_mode_never_falls_back_to_synthetic_rows() -> None:
    """The REAL branch must not contain any synthetic-row fallback."""
    cells = _code_cells(NOTEBOOK)
    rows_cell = next(c for c in cells if "rows = vm.read_vietmed_parquet" in c)
    real_branch = rows_cell.split("else:")[0]
    assert "B-DRUGCHEMICAL" not in real_branch          # synthetic rows only in the else branch
    assert "read_vietmed_parquet" in real_branch
