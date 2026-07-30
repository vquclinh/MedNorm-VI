"""Audit 0056g — VnCoreNLP conditional readiness and build-context hygiene.

These tests do not load E3, call torch.load, execute notebooks, build Docker or
run corpus inference. They exercise the optional configured VnCoreNLP path with
small temporary directories and mocked import probes only.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "configs" / "pipeline" / "full_v1.yaml"
CHECKPOINT = REPO / "checkpoint" / "s1_mention_full_training_v1" / "best.pt"


def _spec(name: str) -> importlib.machinery.ModuleSpec:
    return importlib.machinery.ModuleSpec(name, loader=None)


def _fake_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not loaded by readiness")
    return checkpoint


def _fake_vncore_dir(tmp_path: Path, *, with_jar: bool = True) -> Path:
    directory = tmp_path / "vncorenlp"
    directory.mkdir()
    if with_jar:
        (directory / "VnCoreNLP-1.2.jar").write_bytes(b"jar placeholder")
    return directory


def _install_vncore_probe_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jnius: bool = True,
    py_vncorenlp: bool = True,
    java: bool = True,
    py_error: BaseException | None = None,
    constructor_error: BaseException | None = None,
) -> None:
    from mednorm_vi.mention_factory.neural import runtime

    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def fake_find_spec(name: str, package: str | None = None) -> Any:
        if name in {"torch", "transformers"}:
            return _spec(name)
        if name == "jnius_config":
            return _spec(name) if jnius else None
        if name == "py_vncorenlp":
            return _spec(name) if py_vncorenlp else None
        return real_find_spec(name, package)

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "jnius_config":
            if not jnius:
                raise ModuleNotFoundError("No module named 'jnius_config'", name=name)
            return object()
        if name == "py_vncorenlp":
            if py_error is not None:
                raise py_error
            if not py_vncorenlp:
                raise ModuleNotFoundError("No module named 'py_vncorenlp'", name=name)

            class DummyPyVnCoreNLP:
                class VnCoreNLP:
                    def __init__(self, **_kwargs: Any) -> None:
                        if constructor_error is not None:
                            raise constructor_error

                    def word_segment(self, text: str) -> list[str]:
                        return [text]

            return DummyPyVnCoreNLP
        return real_import_module(name, package)

    monkeypatch.setattr(runtime.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/usr/bin/java" if java else None)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="real E3 checkpoint absent")
def test_default_profile_without_vncorenlp_path_remains_specialist_ready() -> None:
    from mednorm_vi.inference.config import READY, PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(PROFILE)
    assert not config.expert_settings.get("e3_vncorenlp_dir")

    report = evaluate_readiness(config, mode="specialist")
    assert report.status == READY, report.errors
    assert "enable_e3_vihealthbert" in report.runnable_experts
    assert not any("e3_vncorenlp_not_ready" in error for error in report.errors)


def test_configured_vncorenlp_directory_without_a_jar_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.experts.e3_vihealthbert import E3VietHealthBertExpert

    _install_vncore_probe_mocks(monkeypatch)
    expert = E3VietHealthBertExpert(
        checkpoint_path=str(_fake_checkpoint(tmp_path)),
        vncorenlp_dir=str(_fake_vncore_dir(tmp_path, with_jar=False)),
    )
    ready, reason, detail = expert.readiness()
    assert ready is False
    assert reason == "e3_vncorenlp_not_ready:missing_jar"
    assert detail.endswith("vncorenlp")


def test_configured_vncorenlp_with_fake_jar_but_no_jnius_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.experts.e3_vihealthbert import E3VietHealthBertExpert

    _install_vncore_probe_mocks(monkeypatch, jnius=False)
    expert = E3VietHealthBertExpert(
        checkpoint_path=str(_fake_checkpoint(tmp_path)),
        vncorenlp_dir=str(_fake_vncore_dir(tmp_path)),
    )
    ready, reason, detail = expert.readiness()
    assert ready is False
    assert reason == "e3_vncorenlp_not_ready:missing_jnius_config"
    assert detail == "jnius_config"


def test_configured_vncorenlp_missing_py_vncorenlp_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.experts.e3_vihealthbert import E3VietHealthBertExpert

    _install_vncore_probe_mocks(monkeypatch, py_vncorenlp=False)
    expert = E3VietHealthBertExpert(
        checkpoint_path=str(_fake_checkpoint(tmp_path)),
        vncorenlp_dir=str(_fake_vncore_dir(tmp_path)),
    )
    ready, reason, detail = expert.readiness()
    assert ready is False
    assert reason == "e3_vncorenlp_not_ready:missing_py_vncorenlp"
    assert detail == "py_vncorenlp"


def test_vncorenlp_dependency_import_failure_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.experts.e3_vihealthbert import E3VietHealthBertExpert

    _install_vncore_probe_mocks(monkeypatch, py_error=RuntimeError("JNI bootstrap failed"))
    expert = E3VietHealthBertExpert(
        checkpoint_path=str(_fake_checkpoint(tmp_path)),
        vncorenlp_dir=str(_fake_vncore_dir(tmp_path)),
    )
    ready, reason, detail = expert.readiness()
    assert ready is False
    assert reason == "e3_vncorenlp_not_ready:runtime_import_failed"
    assert "py_vncorenlp:RuntimeError" in detail


def test_vncorenlp_runtime_construction_failure_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.neural.runtime import build_segmenter

    _install_vncore_probe_mocks(
        monkeypatch,
        constructor_error=RuntimeError("JVM unavailable"),
    )
    with pytest.raises(
        RuntimeError,
        match="e3_vncorenlp_not_ready:runtime_construction_failed:RuntimeError",
    ):
        build_segmenter(_fake_vncore_dir(tmp_path))


def test_valid_mocked_vncorenlp_capability_can_be_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.experts.e3_vihealthbert import E3VietHealthBertExpert
    from mednorm_vi.mention_factory.neural.runtime import probe_vncorenlp_capability

    _install_vncore_probe_mocks(monkeypatch)
    directory = _fake_vncore_dir(tmp_path)
    capability = probe_vncorenlp_capability(directory)
    assert capability.ready is True

    expert = E3VietHealthBertExpert(
        checkpoint_path=str(_fake_checkpoint(tmp_path)),
        vncorenlp_dir=str(directory),
    )
    ready, reason, _detail = expert.readiness()
    assert ready is True
    assert reason == ""


def test_invalid_configured_vncorenlp_cannot_reach_e3_construction_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mednorm_vi.mention_factory.experts.e3_vihealthbert import E3VietHealthBertExpert
    from mednorm_vi.mention_factory.neural import runtime

    _install_vncore_probe_mocks(monkeypatch, jnius=False)
    load_calls: list[Any] = []

    def fail_if_model_load_starts(*args: Any, **kwargs: Any) -> Any:
        load_calls.append((args, kwargs))
        raise AssertionError("invalid VnCoreNLP must fail before E3 model construction")

    monkeypatch.setattr(runtime, "load_expert", fail_if_model_load_starts)
    expert = E3VietHealthBertExpert(
        checkpoint_path=str(_fake_checkpoint(tmp_path)),
        vncorenlp_dir=str(_fake_vncore_dir(tmp_path)),
    )
    with pytest.raises(RuntimeError, match="e3_vncorenlp_not_ready:missing_jnius_config"):
        expert.prepare()
    assert load_calls == []
    assert expert.loaded is False


def _dockerignore_patterns() -> tuple[str, ...]:
    patterns: list[str] = []
    for raw in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return tuple(patterns)


def _dockerignore_matches(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = path.strip("/")
    parts = normalized.split("/")
    for raw_pattern in patterns:
        pattern = raw_pattern.strip("/")
        if not pattern:
            continue
        if pattern.startswith("*.") and any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        if "/" not in pattern and any(part == pattern for part in parts):
            return True
        if normalized == pattern or normalized.startswith(f"{pattern}/"):
            return True
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def test_dockerignore_excludes_private_heavy_paths() -> None:
    patterns = _dockerignore_patterns()
    for path in (
        ".git/config",
        ".venv/bin/python",
        "checkpoint/s1_mention_full_training_v1/best.pt",
        "models/hf/hub/config.json",
        "data/governed/private.jsonl",
        "indices/rxnorm/index.json",
        "notebooks/MedNorm_S1.ipynb",
        "internal_test/1.txt",
        "output.zip",
        "tmp/smoke/output.zip",
        "artifact.safetensors",
    ):
        assert _dockerignore_matches(path, patterns), path


def test_dockerignore_preserves_required_build_inputs() -> None:
    patterns = _dockerignore_patterns()
    for path in (
        "Dockerfile",
        "requirements-image.lock",
        "pyproject.toml",
        "src/mednorm_vi/inference/cli.py",
        "configs/pipeline/full_v1.yaml",
        "schemas/organizer_output.schema.json",
    ):
        assert not _dockerignore_matches(path, patterns), path


def test_e4_retirement_remains_intact_in_executable_notebook_cells() -> None:
    forbidden = (
        "enable_e4_phobert_w2ner",
        "validate_e4_artifact",
        "E4_FULL_ARTIFACT",
        "mention/phobert_w2ner",
        "e4_phobert_w2ner",
        "phobert-base",
    )
    offenders: list[str] = []
    for notebook in (REPO / "notebooks").glob("*.ipynb"):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload.get("cells", ())):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", ()))
            executable = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith("#")
            )
            for token in forbidden:
                if token in executable:
                    offenders.append(f"{notebook.name}:cell{index}:{token}")
    assert offenders == []
