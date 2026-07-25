"""Deterministic comparison + classification for an updated organizer-input drop.

Pure, offline helpers used when the organizer publishes a revised input dataset
(e.g. a new ``turn``/``vong`` archive). Compares two extracted input trees by
content hash and classifies the change so a human can decide whether to activate
the new dataset. No raw clinical text is returned — only paths, sizes, and
hashes.
"""

from __future__ import annotations

import hashlib
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Primary change categories (exactly one applies).
EXACTLY_IDENTICAL = "EXACTLY_IDENTICAL"
REPACKAGED_ONLY = "REPACKAGED_ONLY"
CONTENT_CHANGED = "CONTENT_CHANGED"
SCHEMA_OR_LAYOUT_CHANGED = "SCHEMA_OR_LAYOUT_CHANGED"
INVALID_OR_INCOMPLETE = "INVALID_OR_INCOMPLETE"


@dataclass(frozen=True, slots=True)
class SnapshotFingerprint:
    root: str
    file_count: int
    total_bytes: int
    tree_hash: str
    files: dict[str, str] = field(default_factory=dict)   # relpath -> sha256
    sizes: dict[str, int] = field(default_factory=dict)   # relpath -> bytes


@dataclass(frozen=True, slots=True)
class InputChangeReport:
    classification: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    modified: tuple[str, ...]
    byte_identical: int
    text_identical: int
    newline_or_bom_only: int
    genuinely_changed: int
    old_tree_hash: str
    new_tree_hash: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unsafe_zip_members(zip_path: str | Path) -> tuple[str, ...]:
    """Return archive members that are unsafe to extract: absolute paths,
    ``..`` traversal, backslash names, or symlinks. Empty tuple == safe."""
    unsafe: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if (name.startswith("/")
                    or (len(name) > 1 and name[1] == ":")
                    or ".." in Path(name).parts
                    or "\\" in name
                    or stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)):
                unsafe.append(name)
    return tuple(unsafe)


def _input_root(extract_dir: Path) -> Path:
    """The organizer input root inside an extraction (``input/`` if present)."""
    nested = extract_dir / "input"
    return nested if nested.is_dir() else extract_dir


@contextmanager
def temporary_input_extraction(zip_path: str | Path) -> Iterator[Path]:
    """Safely extract an organizer input ZIP into a temporary directory and yield
    its input root, cleaning up on exit. Refuses unsafe archives.

    Enables comparing an archived historical release without keeping it
    permanently extracted (the active input is the only persistent extraction).
    """
    unsafe = unsafe_zip_members(zip_path)
    if unsafe:
        raise ValueError(f"refusing to extract unsafe archive members: {unsafe}")
    tmp = Path(tempfile.mkdtemp(prefix="mednorm-input-verify-"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        yield _input_root(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fingerprint_archive(zip_path: str | Path) -> SnapshotFingerprint:
    """Fingerprint the input payload of a ZIP archive without a persistent extract."""
    with temporary_input_extraction(zip_path) as root:
        return fingerprint(root)


def fingerprint(root: str | Path) -> SnapshotFingerprint:
    """Deterministic fingerprint of an extracted input tree."""
    base = Path(root)
    files: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = p.relative_to(base).as_posix()
            files[rel] = _sha256(p.read_bytes())
            sizes[rel] = p.stat().st_size
    tree = hashlib.sha256(
        "".join(f"{k}\t{files[k]}\n" for k in sorted(files)).encode("utf-8")
    ).hexdigest()
    return SnapshotFingerprint(
        root=base.as_posix(), file_count=len(files), total_bytes=sum(sizes.values()),
        tree_hash=tree, files=files, sizes=sizes,
    )


def _text_class(a: bytes, b: bytes) -> str:
    """How two changed files differ: 'byte'|'text'|'newline_bom'|'content'."""
    if a == b:
        return "byte"
    try:
        ta = a.decode("utf-8")
        tb = b.decode("utf-8")
    except UnicodeDecodeError:
        return "content"
    if ta == tb:
        return "text"
    if ta.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n") == \
            tb.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n"):
        return "newline_bom"
    return "content"


def classify_input_change(old_root: str | Path, new_root: str | Path) -> InputChangeReport:
    """Compare an old vs new extracted input tree and classify the change.

    - EXACTLY_IDENTICAL: same relative paths and byte content.
    - SCHEMA_OR_LAYOUT_CHANGED: the set of relative paths differs (add/remove/rename).
    - REPACKAGED_ONLY: same paths+content up to encoding/newline/BOM differences.
    - CONTENT_CHANGED: at least one common file has genuinely different text.
    - INVALID_OR_INCOMPLETE: the new tree has no files.
    """
    old = fingerprint(old_root)
    new = fingerprint(new_root)
    ok, nk = set(old.files), set(new.files)
    added = tuple(sorted(nk - ok))
    removed = tuple(sorted(ok - nk))
    common = sorted(ok & nk)

    if new.file_count == 0:
        classification = INVALID_OR_INCOMPLETE
    elif old.tree_hash == new.tree_hash:
        classification = EXACTLY_IDENTICAL
    elif added or removed:
        classification = SCHEMA_OR_LAYOUT_CHANGED
    else:
        classification = None  # decided below from per-file text classes

    byte_identical = text_identical = newline_bom = genuinely = 0
    modified: list[str] = []
    old_base, new_base = Path(old_root), Path(new_root)
    for rel in common:
        if old.files[rel] == new.files[rel]:
            byte_identical += 1
            continue
        cls = _text_class((old_base / rel).read_bytes(), (new_base / rel).read_bytes())
        if cls == "text":
            text_identical += 1
        elif cls == "newline_bom":
            newline_bom += 1
        else:
            genuinely += 1
        modified.append(rel)

    if classification is None:
        classification = CONTENT_CHANGED if genuinely else REPACKAGED_ONLY

    return InputChangeReport(
        classification=classification,
        added=added, removed=removed, modified=tuple(modified),
        byte_identical=byte_identical, text_identical=text_identical,
        newline_or_bom_only=newline_bom, genuinely_changed=genuinely,
        old_tree_hash=old.tree_hash, new_tree_hash=new.tree_hash,
    )


__all__ = [
    "SnapshotFingerprint", "InputChangeReport", "fingerprint", "classify_input_change",
    "unsafe_zip_members", "temporary_input_extraction", "fingerprint_archive",
    "EXACTLY_IDENTICAL", "REPACKAGED_ONLY", "CONTENT_CHANGED",
    "SCHEMA_OR_LAYOUT_CHANGED", "INVALID_OR_INCOMPLETE",
]
