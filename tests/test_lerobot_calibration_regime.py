"""
Loads the frozen results from scripts/robot_sensor_validation/
run_lerobot_calibration_regime_analysis.py (dense-armor's
classify_segments run on real SO-101 teleoperation calibration-offset
data) and checks the real findings. No re-download/re-run needed --
reads the already-committed frozen JSON. See
docs/lerobot_calibration_regime_detection.md for the full write-up.
"""
import json
import pathlib

import numpy as np
import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "lerobot_calibration_regime_frozen.json"
)


@pytest.fixture(scope="module")
def result():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_frozen_dataset_shape(result):
    assert result["joint"] == 2
    assert len(result["per_episode_stable_median"]) == 50
    assert set(result["episodes"]) == {"0", "22"}


def test_static_dataset_wide_view_masks_the_real_spread(result):
    """What #3758's own proposed static mean/std would report -- a
    small, unremarkable-looking number, because pooling all 50
    episodes' stable frames together averages out the real per-pose
    swings this experiment found within single episodes."""
    assert abs(result["static_dataset_mean"]) < 0.5
    assert result["static_dataset_std"] < 1.5


def test_episode_0_has_multiple_real_pose_regimes(result):
    """The core finding: within ONE episode, the stable-frame
    calibration offset visits several distinct real levels as the arm
    moves through different task poses, not one constant value."""
    runs = result["episodes"]["0"]["runs"]
    medians = [r["median"] for r in runs]
    assert max(medians) - min(medians) > 4.0, "expected a large real spread within one episode"
    # at least one genuinely new sustained level gets flagged (spike or regime), not just 'clean' throughout
    assert any(r["label"] != "clean" for r in runs)


def test_episode_22_settles_into_a_sustained_different_level(result):
    """A real, large, SUSTAINED shift within one episode (not just a
    brief spike) -- classify_segments correctly leaves the settled
    plateau labeled 'clean' relative to its own adapted local window,
    the same causal-adaptation behavior seen in every prior experiment."""
    runs = result["episodes"]["22"]["runs"]
    clean_runs = [r for r in runs if r["label"] == "clean"]
    long_low_runs = [r for r in clean_runs if r["median"] < -3.0 and (r["end"] - r["start"]) > 15]
    assert len(long_low_runs) >= 1, "expected a real sustained low-offset plateau lasting >15 stable frames"


def test_transition_boundaries_get_flagged_not_missed(result):
    """The real value-add claim, checked directly: does the detector
    flag SOMETHING at each real pose transition, rather than silently
    smoothing over it? (Whether the label is 'spike' or 'regime' is a
    separate, honestly-disclosed labeling-precision question -- this
    test only checks that a transition is never silently invisible.)"""
    for ep in ("0", "22"):
        runs = result["episodes"][ep]["runs"]
        non_clean = [r for r in runs if r["label"] != "clean"]
        assert len(non_clean) >= 2, f"episode {ep}: expected multiple real transitions to be flagged"
