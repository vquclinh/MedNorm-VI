"""PDF inspection helpers for the official Vietnamese ICD-10 source documents."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PdfInspectionError(RuntimeError):
    """Raised when local PDF metadata/text tools cannot inspect a source PDF."""


@dataclass(frozen=True, slots=True)
class PdfInspection:
    path: str
    bytes: int
    md5: str
    sha256: str
    page_count: int
    pdf_signature: str
    mime_hint: str
    has_embedded_text: bool
    tool: str
    warnings: tuple[str, ...] = ()


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=False, text=True, capture_output=True)
    except OSError as exc:
        raise PdfInspectionError(f"cannot execute {args[0]!r}: {exc}") from exc


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - provenance hash, not security.
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _page_count(path: Path) -> int:
    if shutil.which("pdfinfo") is None:
        raise PdfInspectionError("pdfinfo is required for page-count validation")
    proc = _run(["pdfinfo", str(path)])
    if proc.returncode != 0:
        raise PdfInspectionError(proc.stderr.strip() or "pdfinfo failed")
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise PdfInspectionError("pdfinfo did not report a Pages field")


def _has_text(path: Path) -> bool:
    if shutil.which("pdftotext") is None:
        raise PdfInspectionError("pdftotext is required for embedded-text inspection")
    proc = _run(["pdftotext", "-layout", "-f", "1", "-l", "1", str(path), "-"])
    if proc.returncode != 0:
        raise PdfInspectionError(proc.stderr.strip() or "pdftotext failed")
    return bool(proc.stdout.strip())


def inspect_pdf(path: str | Path) -> PdfInspection:
    """Inspect a PDF without rewriting it or extracting full content."""
    p = Path(path)
    if not p.is_file():
        raise PdfInspectionError(f"PDF does not exist: {p}")
    sig = p.read_bytes()[:8].decode("latin-1", errors="replace")
    if not sig.startswith("%PDF-"):
        raise PdfInspectionError(f"not a PDF signature: {p}")
    md5, sha256 = _hashes(p)
    return PdfInspection(
        path=p.as_posix(),
        bytes=p.stat().st_size,
        md5=md5,
        sha256=sha256,
        page_count=_page_count(p),
        pdf_signature=sig.strip(),
        mime_hint="application/pdf",
        has_embedded_text=_has_text(p),
        tool="pdfinfo+pdftotext",
    )
