"""
Loads the frozen results from scripts/robot_sensor_validation/
run_imu_validation.py (dense-armor's classify_segments/cusum_detector
run on real UCI HAR IMU accelerometer telemetry) and checks the real
findings -- no re-download/re-run of the dataset needed here, this only
reads the already-committed frozen JSON. See
docs/imu_sensor_validation.md for the full write-up.
"""
import json
import pathlib

import numpy as np
import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "imu_validation_frozen.json"
)


@pytest.fixture(scope="module")
def results():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_frozen_dataset_shape(results):
    assert set(results) == {"A_normal", "B_transient", "C_persistent", "D_legit_switch"}
    assert len(results["A_normal"]["x"]) == 1920
    assert len(results["D_legit_switch"]["x"]) == 1920 + 1152


def test_transient_glitch_is_caught(results):
    r = results["B_transient"]
    at, width = r["injected_at"], r["injected_width"]
    comb = np.array(r["comb"])
    assert np.all(comb[at:at + width]), "the injected 2-sample IMU spike must be flagged"


def test_persistent_shift_transition_is_caught(results):
    r = results["C_persistent"]
    at = r["injected_at"]
    comb = np.array(r["comb"])
    assert np.any(comb[at:at + 50]), "the sustained-offset transition must be flagged within 1s"


def test_baseline_false_positive_rate_is_real_not_zero(results):
    """Real IMU noise on a genuinely quiet real recording is not
    gaussian -- expect a nonzero but bounded false-positive rate, not
    the near-zero seen on synthetic gaussian noise."""
    comb = np.array(results["A_normal"]["comb"])
    fp_rate = float(np.mean(comb[50:]))
    assert 0.0 < fp_rate < 0.20, f"unexpected baseline FP rate: {fp_rate}"


def test_legit_activity_switch_triggers_heavy_flagging():
    """Real, honest finding: unlike the LLM-latency benchmark's D
    scenario (low false-reject), a real STANDING->WALKING transition
    flags MOST of the first second -- gait is a large, real, oscillatory
    change in acceleration magnitude, not a subtle shift. Re-verified
    directly here, not just asserted in the docs page."""
    with open(_DATA_PATH, encoding="utf-8") as f:
        results = json.load(f)
    r = results["D_legit_switch"]
    switch_at = r["switch_at"]
    comb = np.array(r["comb"])
    flag_rate = float(np.mean(comb[switch_at:switch_at + 50]))
    assert flag_rate > 0.3, f"expected heavy flagging at a real gait transition, got {flag_rate}"
