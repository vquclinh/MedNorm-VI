"""Content hashing for experiment outputs (no network, no heavy deps)."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_zip_hash(zip_path: str | Path) -> str:
    """Hash of the raw ZIP file bytes."""
    return sha256_file(zip_path)


def zip_prediction_hashes(zip_path: str | Path) -> dict[str, str]:
    """Map ``output/N.json`` entry name -> sha256 of its content (sorted)."""
    out: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".json"):
                out[name] = sha256_bytes(zf.read(name))
    return out


__all__ = ["sha256_file", "sha256_bytes", "output_zip_hash", "zip_prediction_hashes"]
