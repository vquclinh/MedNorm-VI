"""Deterministic organizer output directory and ZIP packaging.

"Deterministic" was true of the *contents* and false of the *archive*. Until Audit 0054
``package_output_zip`` used ``ZipFile.write``, which copies each file's mtime into the
entry header, so two runs over identical inputs produced ZIPs with different SHA-256
digests. The run manifest caught it immediately: ``output_zip_sha256`` differed between
two back-to-back runs of the same document.

That matters beyond tidiness. Appendix A asks for a reproducible rebuild, and an archive
whose hash changes every second cannot be pinned, compared, or attested to. Entries are
now written with a **fixed timestamp and fixed permissions**, so the digest depends only
on the payloads and their names.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ..validator.submission import (
    SUBMISSION_DIR_NAME,
    validate_output_directory,
    validate_submission_zip,
)

# Fixed entry timestamp. 1980-01-01 00:00:00 is the earliest value the ZIP format can
# represent, and using a constant is what makes the archive byte-reproducible.
ZIP_FIXED_TIMESTAMP: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
# Fixed entry mode: regular file, rw-r--r--. Keeps the digest independent of the umask
# the run happened to inherit.
ZIP_FIXED_EXTERNAL_ATTR = (0o100644 & 0xFFFF) << 16


def write_output_directory(payloads: dict[str, str], output_dir: str | Path) -> Path:
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name, payload in sorted(
        payloads.items(), key=lambda item: int(Path(item[0]).stem)
    ):
        (root / name).write_text(payload, encoding="utf-8")
    return root


def package_output_zip(
    output_dir: str | Path, zip_path: str | Path, *, expected_count: int
) -> Path:
    """Package a validated ``output/`` directory into a byte-reproducible ZIP."""
    root = Path(output_dir)
    validation = validate_output_directory(root, n=expected_count)
    if not validation.ok:
        codes = [i.code for i in validation.errors]
        raise ValueError(f"output directory failed validation: {codes}")
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.glob("*.json"), key=lambda p: int(p.stem)):
            info = zipfile.ZipInfo(
                filename=f"{SUBMISSION_DIR_NAME}/{path.name}",
                date_time=ZIP_FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ZIP_FIXED_EXTERNAL_ATTR
            archive.writestr(info, path.read_bytes())
    zip_validation = validate_submission_zip(target, n=expected_count)
    if not zip_validation.ok:
        raise ValueError(
            f"output zip failed validation: {[i.code for i in zip_validation.errors]}")
    return target


__all__ = [
    "ZIP_FIXED_EXTERNAL_ATTR",
    "ZIP_FIXED_TIMESTAMP",
    "package_output_zip",
    "write_output_directory",
]
