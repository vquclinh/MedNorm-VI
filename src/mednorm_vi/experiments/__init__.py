"""Local leaderboard experiment tracking (no network, no scraping).

Local gold/silver/synthetic scores are kept separate from the manually entered
leaderboard score, which never substitutes for detailed local error analysis.
"""

from __future__ import annotations

from .hashing import output_zip_hash, zip_prediction_hashes
from .leaderboard import attach_output, compare, record_leaderboard, record_local_score
from .models import ExperimentRecord, LeaderboardResult, LocalScore, LocalScoreKind
from .registry import ExperimentExistsError, ExperimentRegistry

__all__ = [
    "ExperimentRecord",
    "LocalScore",
    "LocalScoreKind",
    "LeaderboardResult",
    "ExperimentRegistry",
    "ExperimentExistsError",
    "attach_output",
    "record_local_score",
    "record_leaderboard",
    "compare",
    "output_zip_hash",
    "zip_prediction_hashes",
]
