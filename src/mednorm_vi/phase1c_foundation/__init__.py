"""Phase 1C-A foundation: organizer policy, resource governance, KB forensics,
position policy, and the deterministic resolver — wired into one diagnostic CLI.

No network access, no external data downloaded, no final ICD/RxNorm prediction.
"""

from __future__ import annotations

from .doctor import DoctorPaths, DoctorReport, build_report, render_report

__all__ = ["DoctorPaths", "DoctorReport", "build_report", "render_report"]
