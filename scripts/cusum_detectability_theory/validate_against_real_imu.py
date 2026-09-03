"""
scripts/cusum_detectability_theory/validate_against_real_imu.py
====================================================================
A second, independent real physical domain for `detectability_report()`
(arl_theory.py), after Experiment 44's real lidar check -- real
accelerometer data (UCI HAR, subject 17, STANDING, already committed at
scripts/robot_sensor_validation/imu_validation_frozen.json's A_normal
scenario, no new download). Same protocol as the lidar validation: real
local MAD from a causal pre-injection window, a real telemetry-layer
injection, `cusum_detector(reference="fixed")` sliced so its own fixed
reference literally IS that same local window.

CHECK 1 -- same real injection magnitude as the original IMU experiment's
own C_persistent scenario (+3.0g, see docs/imu_sensor_validation.md):
this real baseline is extremely quiet (local MAD ~0.001-0.003g standing
accelerometer magnitude), so a real +3.0g injection sits at >1000 sigma
of real local noise -- the closed-form formula's exponential term
underflows and `detection_arl` comes out below 1 sample. That is not a
code bug, it is what the Wald/Siegmund asymptotic approximation actually
predicts at extreme SNR, and it has no physical meaning: no detector can
flag in under one sample. HONEST FINDING, not previously surfaced by the
lidar check (whose real local noise was never this quiet): the formula
needs a floor at ARL=1 to stay physically meaningful, and predictions
below roughly ARL~2-3 should be read qualitatively ("near-instant"), not
taken as a precise number.

CHECK 2 -- to still get a meaningful, comparable-to-lidar ARL check (not
just a degenerate near-zero case), the injection is scaled to a fixed
2-sigma-of-real-local-noise offset at each point instead of a fixed real
magnitude. Preregistered before running (5 points, spacing 300 samples,
k=0.5, h=5.0, target 2.0 sigma, never adjusted after seeing results).

RESULT (real, not the hypothesis this docstring originally guessed):
2/5 real points had observed latency below the predicted mean ARL, 3/5
above -- a genuinely MIXED result, not the same one-directional "always
faster" bias the lidar check found (7/7). Documented as found: this
real accelerometer domain does not simply reproduce lidar's finding: a
second independent real domain, an honestly different real outcome, not
forced to match. Consistent with the extreme-SNR floor issue Check 1
found: this signal's real local noise is on a very different real scale
(much quieter) than the lidar range data, so the theory's known
iid-Gaussian mismatch does not necessarily bias in the same direction
across domains that are physically this different.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from arl_theory import detectability_report, two_sided_arl  # noqa: E402

sys.path.insert(0, r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor")
from dense_armor.utility.cusum import cusum_detector  # noqa: E402

_IMU_FROZEN = _THIS_DIR.parent / "robot_sensor_validation" / "imu_validation_frozen.json"
K, H = 0.5, 5.0
RADIUS, REF_MULT = 25, 2
SPAN = RADIUS * REF_MULT
LOCAL_WINDOW = 100
POINTS = [300, 600, 900, 1200, 1500]
ORIGINAL_OFFSET = 3.0  # g, same real magnitude as the original C_persistent injection
TARGET_SIGMA = 2.0


def _local_mad(x, pt):
    w = x[pt - LOCAL_WINDOW:pt]
    med = np.median(w)
    return float(np.median(np.abs(w - med)) * 1.4826)


def _observed_latency(x, pt, offset):
    seg = x[pt - LOCAL_WINDOW:].copy()
    seg[LOCAL_WINDOW:] += offset
    flagged, _ = cusum_detector(seg, radius=RADIUS, ref_mult=REF_MULT, k=K, h=H, reference="fixed")
    idx = np.where(flagged[LOCAL_WINDOW:])[0]
    return int(idx[0]) + 1 if len(idx) else None


def main():
    with open(_IMU_FROZEN, encoding="utf-8") as f:
        d = json.load(f)
    x = np.array(d["A_normal"]["x"])

    print("=== CHECK 1: real +3.0g injection (extreme-SNR floor check) ===")
    print("(uses the RAW two_sided_arl formula, unfloored, to demonstrate the finding that")
    print(" motivated detectability_report()'s own floor -- see arl_theory.py.)")
    check1_rows = []
    for pt in POINTS:
        mad = _local_mad(x, pt)
        mu_std = ORIGINAL_OFFSET / mad
        raw_arl = two_sided_arl(mu_std, K, H)
        latency = _observed_latency(x, pt, ORIGINAL_OFFSET)
        check1_rows.append(dict(point=pt, local_mad=mad, shift_in_sigma=mu_std,
                                 predicted_arl_raw=raw_arl, real_latency=latency))
        print(f"pt={pt:4d}  MAD={mad:.4f}  shift_sigma={mu_std:.1f}  "
              f"predicted_ARL_raw={raw_arl:.4f}  real_latency={latency}")
    sub_one = sum(1 for r in check1_rows if r["predicted_arl_raw"] < 1.0)
    print(f"{sub_one}/{len(check1_rows)} points: raw formula predicts ARL < 1 (physically meaningless)")

    print("\n=== CHECK 2: injection scaled to 2.0 sigma of real local noise ===")
    check2_rows = []
    for pt in POINTS:
        mad = _local_mad(x, pt)
        offset = TARGET_SIGMA * mad
        report = detectability_report(local_noise_scale=mad, k=K, h=H, candidate_shift=offset)
        latency = _observed_latency(x, pt, offset)
        check2_rows.append(dict(point=pt, local_mad=mad, shift_in_sigma=report["shift_in_sigma"],
                                 predicted_arl=report["detection_arl"], real_latency=latency))
        print(f"pt={pt:4d}  MAD={mad:.4f}  shift_sigma={report['shift_in_sigma']:.2f}  "
              f"predicted_ARL={report['detection_arl']:.2f}  real_latency={latency}")
    n_below = sum(1 for r in check2_rows if r["real_latency"] is not None and r["real_latency"] < r["predicted_arl"])
    print(f"\n{n_below}/{len(check2_rows)} real points: observed latency < predicted mean ARL")

    out_path = _THIS_DIR / "real_imu_arl_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(
            k=K, h=H, radius=RADIUS, ref_mult=REF_MULT,
            check1_original_offset=ORIGINAL_OFFSET, check1_rows=check1_rows, check1_n_sub_one=sub_one,
            check2_target_sigma=TARGET_SIGMA, check2_rows=check2_rows,
            check2_n_below=n_below, check2_n_total=len(check2_rows),
        ), f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
