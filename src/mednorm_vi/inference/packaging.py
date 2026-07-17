"""Deterministic organizer output directory and ZIP packaging."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ..validator.submission import (
    SUBMISSION_DIR_NAME,
    validate_output_directory,
    validate_submission_zip,
)


def write_output_directory(payloads: dict[str, str], output_dir: str | Path) -> Path:
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name, payload in sorted(payloads.items(), key=lambda item: int(Path(item[0]).stem)):
        (root / name).write_text(payload, encoding="utf-8")
    return root


def package_output_zip(
    output_dir: str | Path, zip_path: str | Path, *, expected_count: int
) -> Path:
    root = Path(output_dir)
    validation = validate_output_directory(root, n=expected_count)
    if not validation.ok:
        codes = [i.code for i in validation.errors]
        raise ValueError(f"output directory failed validation: {codes}")
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.glob("*.json"), key=lambda p: int(p.stem)):
            zf.write(path, f"{SUBMISSION_DIR_NAME}/{path.name}")
    zip_validation = validate_submission_zip(target, n=expected_count)
    if not zip_validation.ok:
        raise ValueError(f"output zip failed validation: {[i.code for i in zip_validation.errors]}")
    return target


__all__ = ["package_output_zip", "write_output_directory"]
