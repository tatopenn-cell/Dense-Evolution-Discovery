"""
Loads the frozen results from scripts/robot_sensor_validation/
cross_channel_mahalanobis_imu.py (real UCI HAR accel+gyro, a real
temporal-reordering fault, two closed-form cross-channel attempts) and
checks the real negative finding -- no re-download/re-run of the
dataset needed here, this only reads the already-committed frozen
JSON. See docs/cross_channel_mahalanobis_imu.md for the full write-up.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "cross_channel_mahalanobis_imu_frozen.json"
)


@pytest.fixture(scope="module")
def results():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_frozen_shape(results):
    assert results["n_samples"] == 1152
    assert results["fault_at"] == 700
    assert results["fault_width"] == 150


def test_point_wise_mahalanobis_missed_the_reordering_fault(results):
    # The real negative finding: a per-sample joint distance cannot see
    # a fault that only reorders otherwise-normal values in time.
    assert results["rate_cross"] == 0.0
    assert results["fp_cross"] < 0.1, "false-positive rate should stay low on unfaulted data"


def test_single_channel_detectors_also_barely_caught_it(results):
    # Neither single-channel detector was fooled into over-firing either
    # -- the fault genuinely is subtle to magnitude-based detection.
    assert results["rate_accel"] < 0.1
    assert results["rate_gyro"] < 0.1


def test_rolling_correlation_baseline_was_already_near_zero(results):
    # The real reason attempt 2 also came back negative: there was
    # little real cross-channel correlation structure to break in the
    # first place, for this specific (magnitude) channel pair.
    assert abs(results["corr_baseline_mean"]) < 0.1
    assert results["corr_baseline_std"] < 0.2
