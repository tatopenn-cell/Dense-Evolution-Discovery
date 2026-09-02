"""
Loads the frozen results from scripts/cusum_detectability_theory/
validate_against_real_lidar.py (does detectability_report() correctly
predict a real, already-measured lidar detection case, not just
synthetic Monte Carlo) and checks the real findings. No re-run needed.
See docs/cusum_detectability_theory.md's update section.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "cusum_detectability_theory" / "real_lidar_arl_validation_frozen.json"
)


@pytest.fixture(scope="module")
def result():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_seven_real_independent_points_checked(result):
    assert result["n_total"] == 7


def test_all_real_points_have_a_valid_prediction_and_observation(result):
    for row in result["rows"]:
        assert row["predicted_arl"] > 0
        assert row["real_latency"] is not None


def test_theory_systematically_overestimates_real_detection_latency(result):
    """The real, honest finding: in every one of the 7 real, independent
    cases, real detection happened FASTER than the theory's predicted
    mean ARL -- a consistent directional bias, not scatter around the
    prediction. Re-verified directly against the frozen numbers."""
    assert result["n_below"] == result["n_total"]


def test_bias_direction_matches_experiment_44s_null_case_finding(result):
    """Coherent with, not a separate mystery from, Experiment 44's own
    finding that the real false-alarm ARL was lower than theory
    predicted -- both point to the same real cause (real lidar data is
    not well-approximated by the theory's iid-Gaussian assumption)."""
    ratios = [row["real_latency"] / row["predicted_arl"] for row in result["rows"]]
    assert all(r < 1.0 for r in ratios)
    assert max(ratios) < 0.9, "expected a real, substantial gap, not a marginal one"
