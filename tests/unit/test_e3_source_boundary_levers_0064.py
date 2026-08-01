"""The two training levers introduced for Milestone 4E (Audit 0064).

Neither lever was accepted into the runtime - R5 passed the development seed and its
confirmation seed did not hold, so the Audit-0062 checkpoint stays active. The *code*
still ships, because the next milestone will reuse it, and an unaccepted result is not a
reason to leave a sampler and a loss mask untested.

Both functions are pure and Torch-free, which is what makes them testable without a GPU:

* `boundary_mask_from_labels` derives its mask from the supervision itself, so it cannot
  drift away from the labels it is weighting;
* `source_balanced_order` equalises source exposure **without discarding data** - a small
  source is oversampled rather than a large one truncated - and preserves epoch length so
  the optimizer-step schedule stays comparable to Audit 0062.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "train_e3_source_boundary_0064.py"


def _module():
    """Load the training script as a module; it is a script, not a package member."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("train_e3_source_boundary_0064", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pytest.importorskip("numpy")
MODULE = _module()


# --- boundary mask -------------------------------------------------------------


def test_the_boundary_mask_marks_only_the_first_and_last_token_of_a_run() -> None:
    labels = [[0, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 1], [0, 0]]
    mask = MODULE.boundary_mask_from_labels(labels)
    # Type 0: a three-token run at 1..3 -> only 1 and 3 are boundaries.
    assert list(mask[:, 0]) == [0, 1, 0, 1, 0, 0, 0]
    # Type 1: a single token is simultaneously the first and last of its run.
    assert list(mask[:, 1]) == [0, 0, 0, 0, 0, 1, 0]


def test_two_adjacent_runs_of_one_type_each_keep_their_own_boundaries() -> None:
    """A negative token between two runs must produce four boundary marks, not two."""
    labels = [[1], [1], [0], [1], [1]]
    assert list(MODULE.boundary_mask_from_labels(labels)[:, 0]) == [1, 1, 0, 1, 1]


def test_a_run_touching_the_sequence_edges_is_still_bounded() -> None:
    labels = [[1], [1], [1]]
    assert list(MODULE.boundary_mask_from_labels(labels)[:, 0]) == [1, 0, 1]


def test_an_all_negative_sequence_has_no_boundary_tokens() -> None:
    assert MODULE.boundary_mask_from_labels([[0], [0], [0]]).sum() == 0


def test_the_mask_never_marks_a_token_the_labels_call_negative() -> None:
    """The weight must not reach a token that carries no positive supervision."""
    import numpy

    rng = numpy.random.default_rng(0)
    labels = rng.integers(0, 2, size=(64, 5))
    mask = MODULE.boundary_mask_from_labels(labels.tolist())
    assert not ((mask == 1) & (labels == 0)).any()


def test_labels_must_be_two_dimensional() -> None:
    with pytest.raises(MODULE.RefinementError, match="tokens, types"):
        MODULE.boundary_mask_from_labels([1, 0, 1])


# --- source-balanced sampler ---------------------------------------------------


def _features(counts: dict[str, int]) -> list[dict[str, str]]:
    return [{"source_dataset": name} for name, n in counts.items() for _ in range(n)]


def test_the_sampler_equalises_sources_and_preserves_epoch_length() -> None:
    counts = {"A": 600, "B": 300, "C": 100}
    features = _features(counts)
    order = MODULE.source_balanced_order(features, random.Random(0))
    assert len(order) == len(features), "epoch length must not change"
    drawn = Counter(features[i]["source_dataset"] for i in order)
    assert max(drawn.values()) - min(drawn.values()) <= 1, drawn


def test_no_governed_example_is_discarded_from_the_largest_source() -> None:
    """Balancing oversamples the small sources; it never truncates the large one."""
    counts = {"A": 50, "B": 10}
    features = _features(counts)
    order = MODULE.source_balanced_order(features, random.Random(1))
    from_a = {i for i in order if features[i]["source_dataset"] == "A"}
    # A contributes 30 of the 60 draws, and every one of them is a distinct A example:
    # the queue is drained before it is refilled, so no A example is skipped in favour
    # of repeating another.
    assert len(from_a) == 30
    assert Counter(features[i]["source_dataset"] for i in order)["B"] == 30


def test_a_small_source_is_oversampled_rather_than_the_epoch_shrinking() -> None:
    features = _features({"A": 90, "B": 10})
    order = MODULE.source_balanced_order(features, random.Random(2))
    b_draws = [i for i in order if features[i]["source_dataset"] == "B"]
    assert len(b_draws) == 50
    assert len(set(b_draws)) == 10, "only 10 distinct B examples exist; they repeat"


def test_a_single_source_corpus_is_left_alone() -> None:
    features = _features({"only": 25})
    order = MODULE.source_balanced_order(features, random.Random(3))
    assert sorted(order) == list(range(25)), "one source means a plain permutation"


def test_the_sampler_is_deterministic_for_a_given_seed() -> None:
    features = _features({"A": 40, "B": 25, "C": 15})
    first = MODULE.source_balanced_order(features, random.Random(7))
    second = MODULE.source_balanced_order(features, random.Random(7))
    assert first == second


# --- the levers are off by default ---------------------------------------------


def test_the_contract_version_names_this_milestone() -> None:
    assert MODULE.CONTRACT_VERSION == "e3-source-boundary-refinement-0064-v1"


def test_the_active_runtime_did_not_adopt_either_lever() -> None:
    """Audit 0064 rejected every candidate; the scored 0062 checkpoint stays active."""
    import yaml

    profiles = yaml.safe_load(
        (REPO / "configs" / "models" / "e3_checkpoint_profiles.yaml").read_text(encoding="utf-8")
    )
    assert profiles["active"] == "boundary_refinement_0062"
    entry = profiles["profiles"]["boundary_refinement_0062"]
    assert entry["path"] == "checkpoint/e3_boundary_refinement_0062/best.pt"
    assert entry["sha256"] == "524ece1e7d190838cb8b1ce3b0a0f337bc5b8b7cc7cef70c4c3e0b0310adde3a"
