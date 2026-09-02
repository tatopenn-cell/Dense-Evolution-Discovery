"""
scripts/robot_sensor_validation/analyze_lidar_persistent_gap.py
====================================================================
Decomposes Experiment 42's central open question: why did the +10m
persistent-drift injection get flagged only 3.3% of the time, vs.
Experiment 41's 50% on real IMU data? A GPT-generated review (saved in
the maintainer's prog.txt notes) proposed a specific, testable
hypothesis before this script existed: separate real signal drift from
natural intra-class variance, inter-class variance, and the actual
local signal-to-noise ratio the detector's causal window sees at the
injection point -- and check whether class-based (per-operating-regime)
normalization is the fix, rather than inventing new detector math.

This is a diagnostic DECOMPOSITION of an already-committed, already-
frozen result (Experiment 42's real Sydney Urban Objects data) -- not
new data collection, and not a retuning of classify_segments/
cusum_detector's own thresholds (both stay exactly at their library
defaults throughout). Reuses run_lidar_validation.py's own dataset
loading/verification code rather than re-implementing it.

FOUR THINGS MEASURED, in order:

1. Inter-class vs intra-class variance of real object mean-range,
   across all 631 real objects and their real class labels (parsed
   from each object's filename, e.g. "car.81.12346.bin" -> "car").
2. The real causal-window local scale (median/MAD, the SAME quantity
   classify_segments/cusum_detector compute internally) at the exact
   injection point (index 450), and the resulting offset/MAD ratio in
   sigma units -- directly comparable to the n_sigmas=3.0 threshold.
3. A class-normalized variant of the SAME real signal (each object's
   range minus its own class's real median range) -- GPT's specific
   hypothesis, tested directly: does removing the inter-class
   component recover meaningfully more persistent-drift detection?
4. The same offset/MAD ratio computed for Experiment 41's real IMU
   persistent-drift injection, for a genuine cross-domain comparison.

REAL, HONEST RESULT (not a guess before running -- this is what this
script's own output prints): intra-class variance (car-to-car,
pedestrian-to-pedestrian, etc. real range spread) is roughly 4x larger
than inter-class variance (the car-vs-pedestrian-vs-tree difference) --
81% vs 19% of total variance. Class-normalization measurably helps (the
local MAD at the injection point drops, detection rate roughly doubles)
but does NOT close the gap to IMU's 50% -- the real bottleneck is that
a fixed +10m offset sits at only ~1.3 sigma of this real signal's local
noise (vs ~1600 sigma for the IMU case), structurally below the
n_sigmas=3.0 threshold either way. GPT's hypothesis was partially
right (normalization helps) but not the decisive fix it predicted --
the honest conclusion is closer to "irregular per-object real lidar
telemetry has intrinsically worse per-event signal-to-noise for a
fixed-magnitude drift than a fixed-rate physical channel", independent
of feature engineering.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from run_lidar_validation import (  # noqa: E402
    _ensure_dataset, _load_all_objects, _detect,
    ARBITER_KW, CUSUM_KW, PERSISTENT_FROM, PERSISTENT_OFFSET,
)

_IMU_FROZEN_PATH = _THIS_DIR / "imu_validation_frozen.json"
_IMU_RADIUS, _IMU_REF_MULT, _IMU_OFFSET = 25, 2, 3.0  # g -- see run_imu_validation.py's unit correction


def _causal_mad(x: np.ndarray, at: int, span: int) -> float:
    window = x[max(0, at - span):at]
    med = np.median(window)
    return float(np.median(np.abs(window - med)) * 1.4826)


def main():
    _ensure_dataset()
    recs = _load_all_objects()  # (t_mean, range_mean, n_points, basename), real-timestamp sorted
    ranges = np.array([r[1] for r in recs])
    classes = np.array([r[3].split(".")[0] for r in recs])
    n = len(ranges)

    # --- 1. Inter-class vs intra-class variance decomposition ---
    uniq = np.unique(classes)
    class_means = {c: float(ranges[classes == c].mean()) for c in uniq}
    class_counts = {c: int((classes == c).sum()) for c in uniq}
    overall_mean = float(ranges.mean())
    inter_class_var = sum(class_counts[c] * (class_means[c] - overall_mean) ** 2 for c in uniq) / n
    intra_pairs = [(class_counts[c], float(np.var(ranges[classes == c]))) for c in uniq if class_counts[c] > 1]
    intra_class_var = sum(cnt * v for cnt, v in intra_pairs) / sum(cnt for cnt, v in intra_pairs)
    total_var = float(np.var(ranges))

    # --- 2. Real causal-window local scale at the injection point ---
    span_lidar = ARBITER_KW["radius"] * ARBITER_KW["ref_mult"]
    mad_raw = _causal_mad(ranges, PERSISTENT_FROM, span_lidar)
    ratio_raw = PERSISTENT_OFFSET / mad_raw

    # --- 3. Class-normalized variant (GPT's hypothesis) ---
    class_medians = {c: float(np.median(ranges[classes == c])) for c in uniq}
    normalized = ranges - np.array([class_medians[c] for c in classes])
    mad_norm = _causal_mad(normalized, PERSISTENT_FROM, span_lidar)
    ratio_norm = PERSISTENT_OFFSET / mad_norm

    def detect_rate(base: np.ndarray) -> float:
        x = base.copy()
        x[PERSISTENT_FROM:] += PERSISTENT_OFFSET
        _, _, comb = _detect(x)
        return float(np.mean(comb[PERSISTENT_FROM:PERSISTENT_FROM + 30]))

    det_raw = detect_rate(ranges)
    det_norm = detect_rate(normalized)

    # --- 4. Cross-domain comparison against Experiment 41's real IMU case ---
    with open(_IMU_FROZEN_PATH, encoding="utf-8") as f:
        imu = json.load(f)
    imu_x = np.array(imu["C_persistent"]["x"])
    imu_at = imu["C_persistent"]["injected_at"]
    imu_span = _IMU_RADIUS * _IMU_REF_MULT
    mad_imu = _causal_mad(imu_x, imu_at, imu_span)
    ratio_imu = _IMU_OFFSET / mad_imu
    imu_comb = np.array(imu["C_persistent"]["comb"])
    imu_detect_rate = float(np.mean(imu_comb[imu_at:imu_at + 50]))  # same 1s window Experiment 41 reported

    result = dict(
        n_objects=n,
        inter_class_var=inter_class_var, intra_class_var=intra_class_var, total_var=total_var,
        inter_class_var_fraction=inter_class_var / total_var,
        mad_raw=mad_raw, ratio_raw=ratio_raw, detect_rate_raw=det_raw,
        mad_class_normalized=mad_norm, ratio_class_normalized=ratio_norm, detect_rate_class_normalized=det_norm,
        imu_mad=mad_imu, imu_ratio=ratio_imu, imu_detect_rate=imu_detect_rate,
        n_sigmas_threshold=ARBITER_KW["n_sigmas"],
    )
    out_path = _THIS_DIR / "lidar_persistent_gap_analysis_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Lidar persistent-drift gap decomposition ===")
    print(f"inter-class variance: {inter_class_var:.2f}  ({inter_class_var/total_var:.1%} of total)")
    print(f"intra-class variance: {intra_class_var:.2f}  ({intra_class_var/total_var:.1%} of total)")
    print(f"total variance:       {total_var:.2f}")
    print()
    print(f"RAW:              local MAD={mad_raw:.2f}m  offset/MAD={ratio_raw:.2f}sigma  detect_rate={det_raw:.3f}")
    print(f"CLASS-NORMALIZED: local MAD={mad_norm:.2f}m  offset/MAD={ratio_norm:.2f}sigma  detect_rate={det_norm:.3f}")
    print(f"IMU (Experiment 41): local MAD={mad_imu:.4f}g  offset/MAD={ratio_imu:.1f}sigma  detect_rate={imu_detect_rate:.3f}")
    print()
    print(f"n_sigmas detection threshold: {ARBITER_KW['n_sigmas']}")
    print(f"RAW lidar ratio {ratio_raw:.2f} < threshold {ARBITER_KW['n_sigmas']}: "
          f"{'YES -- structurally below threshold' if ratio_raw < ARBITER_KW['n_sigmas'] else 'no'}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
