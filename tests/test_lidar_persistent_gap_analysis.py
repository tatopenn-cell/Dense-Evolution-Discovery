"""
Loads the frozen results from scripts/robot_sensor_validation/
analyze_lidar_persistent_gap.py (the quantitative decomposition of
Experiment 42's 3.3% persistent-drift-detection finding) and checks
the real, reported numbers. No re-download/re-run needed -- reads the
already-committed frozen JSON.
"""
import json
import pathlib


_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "lidar_persistent_gap_analysis_frozen.json"
)


def _load():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_intra_class_variance_dominates_inter_class():
    """The real finding that corrected this project's own initial
    (wrong) hypothesis in Experiment 42's docs -- most of the real
    range variance is WITHIN a class (a pedestrian can be near or far),
    not between classes (car vs pedestrian vs tree)."""
    r = _load()
    assert r["intra_class_var"] > r["inter_class_var"]
    assert r["inter_class_var_fraction"] < 0.30


def test_raw_offset_to_mad_ratio_is_below_detection_threshold():
    """The direct, mechanical explanation for the 3.3% detection rate:
    the injected +10m offset sits at ~1.3 sigma of the real local
    causal-window noise, structurally below n_sigmas=3.0."""
    r = _load()
    assert r["ratio_raw"] < r["n_sigmas_threshold"]
    assert 1.0 < r["ratio_raw"] < 1.6


def test_class_normalization_helps_but_does_not_close_the_gap():
    """GPT's hypothesis (regime-stratified normalization) tested
    directly: real improvement, not a decisive fix."""
    r = _load()
    assert r["ratio_class_normalized"] > r["ratio_raw"]
    assert r["detect_rate_class_normalized"] >= r["detect_rate_raw"]
    assert r["ratio_class_normalized"] < r["n_sigmas_threshold"]
    assert r["detect_rate_class_normalized"] < 0.5 * r["imu_detect_rate"]


def test_imu_signal_to_noise_is_orders_of_magnitude_higher():
    r = _load()
    assert r["imu_ratio"] > 100 * r["ratio_raw"]
    assert r["imu_detect_rate"] > r["detect_rate_raw"]
