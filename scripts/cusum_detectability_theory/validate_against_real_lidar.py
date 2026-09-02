"""
scripts/cusum_detectability_theory/validate_against_real_lidar.py
======================================================================
Does `detectability_report()` (arl_theory.py) correctly predict a REAL,
already-measured case, not just synthetic Monte Carlo? Reuses
Experiment 42's own committed, real Sydney Urban Objects lidar data
(scripts/robot_sensor_validation/lidar_validation_frozen.json's
A_normal scenario -- the real, unshifted range sequence) -- no new data
collected, no new formula added, per the explicit scoping for this
validation.

METHODOLOGICAL FIX made before trusting any number here: a first
attempt compared `detectability_report()`'s prediction (built from a
real LOCAL causal window's MAD right before an injection point) against
`cusum_detector(reference="fixed")` run on the FULL 631-point array --
but `reference="fixed"` always locks its reference to the array's own
FIRST `span` samples, not a window near wherever the injection is. That
first attempt was comparing two DIFFERENT real reference windows (the
session's start vs. the local pre-injection neighborhood) -- an
apples-to-oranges mismatch, caught before drawing any conclusion. Fixed
by slicing the real array so `x_sub[:span]` literally IS the same real
local window the noise-scale estimate came from.

PROTOCOL: 7 real, independent points spread through the real 631-object
driving session (spacing 80 objects, avoiding edge effects), each with
its own real local MAD (computed from real data, no injection) and its
own real telemetry-layer +10m injection (same convention as Experiment
42's own C_persistent scenario) -- giving 7 genuinely different real
noise regimes to check the prediction against, not just n=1.

HONEST RESULT: in ALL 7 cases, the real observed detection latency was
LOWER than detectability_report()'s predicted mean ARL -- a consistent,
one-directional bias, not scatter. This is the SAME direction as
Experiment 44's own null-case finding (the real false-alarm ARL was
also lower than theory predicted) -- a coherent explanation, not two
separate mysteries: real lidar range data across mixed real object
classes is not well-approximated by the theory's iid-Gaussian
assumption (heavier tails / more real local variability), so real
threshold crossings -- in both directions, false alarms and true
detections -- happen faster than the idealized model predicts. No new
formula is proposed to correct this (per explicit scope for this
validation) -- documented as a real, disclosed limit of applying this
classical theory to real, non-Gaussian sensor data.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from arl_theory import detectability_report  # noqa: E402

sys.path.insert(0, r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor")
from dense_armor.utility.cusum import cusum_detector  # noqa: E402

_LIDAR_FROZEN = (
    _THIS_DIR.parent / "robot_sensor_validation" / "lidar_validation_frozen.json"
)
SPAN = 30  # matches Experiment 42's radius=15, ref_mult=2
SHIFT = 10.0  # matches Experiment 42's C_persistent injection magnitude
K, H = 0.5, 5.0
CANDIDATE_STEP = 80


def main():
    with open(_LIDAR_FROZEN, encoding="utf-8") as f:
        d = json.load(f)
    x_clean = np.array(d["A_normal"]["x"])
    n = len(x_clean)

    candidates = list(range(SPAN + 10, n - 50, CANDIDATE_STEP))
    rows = []
    for pt in candidates:
        window = x_clean[pt - SPAN:pt]
        med = np.median(window)
        mad = float(np.median(np.abs(window - med)) * 1.4826)
        if mad < 1e-6:
            continue

        x_sub = x_clean[pt - SPAN:].copy()
        x_sub[SPAN:] += SHIFT

        flagged, _ = cusum_detector(x_sub, radius=SPAN, ref_mult=1, k=K, h=H, reference="fixed")
        idx = np.where(flagged)[0]
        idx_after = idx[idx >= SPAN]
        real_latency = int(idx_after[0]) - SPAN if len(idx_after) > 0 else None

        report = detectability_report(local_noise_scale=mad, k=K, h=H, candidate_shift=SHIFT)
        rows.append(dict(
            point=pt, local_mad=mad, shift_in_sigma=report["shift_in_sigma"],
            predicted_arl=report["detection_arl"], real_latency=real_latency,
        ))
        print(f"pt={pt:4d}  local_MAD={mad:6.2f}  shift_in_sigma={report['shift_in_sigma']:5.2f}  "
              f"predicted_ARL={report['detection_arl']:6.2f}  real_latency={real_latency}")

    n_below = sum(1 for r in rows if r["real_latency"] is not None and r["real_latency"] < r["predicted_arl"])
    print(f"\n{n_below}/{len(rows)} real points: observed latency < predicted mean ARL")

    out_path = _THIS_DIR / "real_lidar_arl_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(span=SPAN, shift=SHIFT, k=K, h=H, rows=rows, n_below=n_below, n_total=len(rows)), f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
