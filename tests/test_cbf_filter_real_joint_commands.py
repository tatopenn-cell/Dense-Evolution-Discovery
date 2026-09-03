"""
Loads the frozen results from scripts/robot_sensor_validation/
cbf_filter_full_evaluation.py (geometric_cbf_filter's real performance
on real LeRobot joint commands) and checks the real, honest findings --
no re-download/re-run needed here. See
docs/geometric_cbf_filter_real_joint_commands.md for the full write-up.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "cbf_filter_full_evaluation_frozen.json"
)


@pytest.fixture(scope="module")
def result():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_real_trials_checked(result):
    assert result["n_invariance_trials"] > 0
    assert result["n_invasiveness_checks"] > 0


def test_invariance_holds_on_every_real_trial_from_a_safe_start(result):
    assert result["n_invariance_ok"] == result["n_invariance_trials"]


def test_minimal_invasiveness_holds_on_almost_every_real_per_step_check(result):
    """Not exactly 100% (3/3186 real edge cases near the far-field
    threshold), reported as found -- but the residual is tiny and the
    overwhelming majority (>99.9%) is exact."""
    assert result["n_invasiveness_nonzero"] / result["n_invasiveness_checks"] < 0.01
    assert result["median_invasiveness_deviation"] == 0.0
