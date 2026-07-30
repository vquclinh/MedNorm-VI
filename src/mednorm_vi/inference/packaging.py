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


class UnsafeOutputDirectory(RuntimeError):
    """Raised when the requested output directory must not be recursively deleted.

    ``write_output_directory`` starts by removing its destination, so the
    destination has to be something this workflow owns. Until Audit 0056a the
    removal was unconditional — ``if root.exists(): shutil.rmtree(root)`` with no
    guard at all — and the function is exported, so any caller passing any path
    destroyed it. The canonical runner bounded the damage by forcing the *name* to
    ``output``, but a pre-existing unrelated ``output/`` next to the requested ZIP
    was still deleted without warning, and a direct caller had no protection at all.
    """


#: A directory is recognisably ours when every file in it is a numbered prediction
#: JSON, i.e. what a previous run of this function left behind.
_PREDICTION_SUFFIX = ".json"


def _symlink_component(path: Path) -> Path | None:
    """Return the first path component that is a symlink, without resolving it."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            return component
    return None


def _is_previous_output_directory(root: Path) -> bool:
    """True when ``root`` contains only what a previous packaging run wrote."""
    entries = list(root.iterdir())
    if not entries:
        return True
    for entry in entries:
        if entry.is_dir():
            return False
        if entry.is_symlink():
            return False
        if entry.suffix != _PREDICTION_SUFFIX or not entry.stem.isdigit():
            return False
    return True


def assert_safe_output_directory(output_dir: str | Path) -> Path:
    """Refuse any destination this workflow does not own. Returns the input path.

    Refused: a filesystem root, a home directory, the repository root, a path that
    is a symlink or reached through one, a non-``output`` directory name, and any
    existing directory holding files a previous packaging run would not have
    written. Permitted: a directory named ``output`` that is absent, empty, or
    contains only numbered prediction JSON.
    """
    root = Path(output_dir)
    symlink = _symlink_component(root)
    if symlink is not None:
        raise UnsafeOutputDirectory(
            f"refusing to write to {root}: path component {symlink} is a symlink; "
            "removing through it would delete the target path"
        )
    resolved = root.resolve()
    if resolved.parent == resolved:
        raise UnsafeOutputDirectory(f"refusing to write to {resolved}: this is a filesystem root")
    if resolved == Path.home().resolve():
        raise UnsafeOutputDirectory(
            f"refusing to write to {resolved}: this is the user's home directory"
        )
    repository_root = Path(__file__).resolve().parents[3]
    if resolved == repository_root:
        raise UnsafeOutputDirectory(f"refusing to write to {resolved}: this is the repository root")
    if resolved.name != SUBMISSION_DIR_NAME:
        raise UnsafeOutputDirectory(
            f"refusing to write to {resolved}: the organizer output directory must "
            f"be named {SUBMISSION_DIR_NAME!r} (spec Appendix A). "
            "`write_output_directory` recursively removes its destination, so it "
            "only ever writes to a directory this workflow owns"
        )
    if resolved.exists():
        if not resolved.is_dir():
            raise UnsafeOutputDirectory(
                f"refusing to write to {resolved}: the destination exists and is not a directory"
            )
        if not _is_previous_output_directory(resolved):
            raise UnsafeOutputDirectory(
                f"refusing to overwrite {resolved}: it contains files this workflow "
                "did not write. Move or delete it deliberately — a packaging run "
                "must never destroy unrelated data"
            )
    return root


# Fixed entry timestamp. 1980-01-01 00:00:00 is the earliest value the ZIP format can
# represent, and using a constant is what makes the archive byte-reproducible.
ZIP_FIXED_TIMESTAMP: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
# Fixed entry mode: regular file, rw-r--r--. Keeps the digest independent of the umask
# the run happened to inherit.
ZIP_FIXED_EXTERNAL_ATTR = (0o100644 & 0xFFFF) << 16


def write_output_directory(payloads: dict[str, str], output_dir: str | Path) -> Path:
    """Write the organizer payloads into a fresh ``output/`` directory.

    The destination is checked before anything is removed
    (:func:`assert_safe_output_directory`); an unowned or dangerous path raises
    :class:`UnsafeOutputDirectory` and nothing is deleted.
    """
    root = assert_safe_output_directory(output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name, payload in sorted(payloads.items(), key=lambda item: int(Path(item[0]).stem)):
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
                filename=f"{SUBMISSION_DIR_NAME}/{path.name}", date_time=ZIP_FIXED_TIMESTAMP
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ZIP_FIXED_EXTERNAL_ATTR
            archive.writestr(info, path.read_bytes())
    zip_validation = validate_submission_zip(target, n=expected_count)
    if not zip_validation.ok:
        raise ValueError(f"output zip failed validation: {[i.code for i in zip_validation.errors]}")
    return target


__all__ = [
    "ZIP_FIXED_EXTERNAL_ATTR",
    "ZIP_FIXED_TIMESTAMP",
    "UnsafeOutputDirectory",
    "assert_safe_output_directory",
    "package_output_zip",
    "write_output_directory",
]
