"""Round-2 upgrade comparison tooling."""

from .compare import ComparisonReport, compare_task_descriptors
from .input_update import (
    InputChangeReport,
    SnapshotFingerprint,
    classify_input_change,
    fingerprint,
    unsafe_zip_members,
)

__all__ = [
    "ComparisonReport",
    "compare_task_descriptors",
    "InputChangeReport",
    "SnapshotFingerprint",
    "classify_input_change",
    "fingerprint",
    "unsafe_zip_members",
]
