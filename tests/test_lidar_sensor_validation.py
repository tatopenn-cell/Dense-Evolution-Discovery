"""
Loads the frozen results from scripts/robot_sensor_validation/
run_lidar_validation.py (dense-armor's classify_segments/cusum_detector
run on real Sydney Urban Objects lidar telemetry) and checks the real
findings -- no re-download/re-run of the dataset needed here, this only
reads the already-committed frozen JSON. See
docs/lidar_sensor_validation.md for the full write-up.
"""
import json
import pathlib

import numpy as np
import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "lidar_validation_frozen.json"
)


@pytest.fixture(scope="module")
def results():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_frozen_dataset_shape(results):
    assert set(results) == {"A_normal", "B_transient", "C_persistent", "D_real_gap_transition"}
    assert len(results["A_normal"]["x"]) == 631


def test_transient_glitch_is_caught(results):
    r = results["B_transient"]
    at, width = r["injected_at"], r["injected_width"]
    comb = np.array(r["comb"])
    assert np.all(comb[at:at + width]), "the injected 2-object range spike must be flagged"


def test_baseline_false_positive_rate_is_real_not_zero(results):
    """Real lidar range noise across a mixed-object real driving session
    is not gaussian -- expect a nonzero but bounded false-positive rate."""
    comb = np.array(results["A_normal"]["comb"])
    fp_rate = float(np.mean(comb[30:]))
    assert 0.0 < fp_rate < 0.20, f"unexpected baseline FP rate: {fp_rate}"


def test_persistent_shift_detection_is_low_on_this_dataset():
    """Real, honest negative finding: unlike the IMU experiment's 50%
    detection of a sustained shift, this irregularly-sampled, mixed-
    object-class real lidar sequence has much higher natural range
    variance (cars vs pedestrians vs trees genuinely differ by many
    meters), so a fixed +10m persistent offset is far LESS
    distinguishable from real variability here. Re-verified directly,
    not just asserted in the docs page."""
    with open(_DATA_PATH, encoding="utf-8") as f:
        results = json.load(f)
    r = results["C_persistent"]
    at = r["injected_at"]
    comb = np.array(r["comb"])
    detect_rate = float(np.mean(comb[at:at + 30]))
    assert detect_rate < 0.30, f"expected low detection on this high-variance real sequence, got {detect_rate}"


def test_real_gap_transition_does_not_trigger_heavy_flagging(results):
    r = results["D_real_gap_transition"]
    idx = r["gap_at_index"]
    comb = np.array(r["comb"])
    flag_rate = float(np.mean(comb[idx:idx + 30]))
    assert flag_rate < 0.30, f"expected the real session pause to be mostly quiet, got {flag_rate}"
