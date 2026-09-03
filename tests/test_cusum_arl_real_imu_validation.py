"""
Loads the frozen results from scripts/cusum_detectability_theory/
validate_against_real_imu.py -- a second, independent real physical
domain (accelerometer, UCI HAR) for detectability_report(), after the
real lidar check. No re-run needed. See docs/cusum_detectability_theory.md's
second-domain update section.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "cusum_detectability_theory" / "real_imu_arl_validation_frozen.json"
)


@pytest.fixture(scope="module")
def result():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_five_real_independent_points_checked_in_each_condition(result):
    assert len(result["check1_rows"]) == 5
    assert result["check2_n_total"] == 5


def test_extreme_snr_formula_predicts_sub_one_arl_on_real_quiet_baseline(result):
    """Real, honest finding: this real standing-accelerometer baseline is
    quiet enough (local MAD ~0.001-0.003g) that the original experiment's
    own real +3.0g injection sits above 1000 sigma of real local noise --
    the closed-form formula's raw prediction is below 1 sample in all 5
    real cases, which is not physically meaningful."""
    assert result["check1_n_sub_one"] == 5
    for row in result["check1_rows"]:
        assert row["predicted_arl_raw"] < 1.0
        assert row["real_latency"] == 1


def test_moderate_snr_gives_a_mixed_not_uniform_result_on_this_domain(result):
    """Unlike the real lidar check (7/7 real points, latency always below
    the predicted ARL), this real accelerometer domain at a moderate
    2-sigma injection gives a genuinely MIXED result -- 2/5 real points
    faster than predicted, 3/5 slower. Documented as found, not forced to
    match the lidar direction."""
    assert result["check2_n_below"] == 2
    assert result["check2_n_total"] == 5
