"""
scripts/robot_sensor_validation/run_imu_validation.py
========================================================
First validation of Dense-Armor's runtime detectors on REAL sensor
telemetry from a real physical device -- not quantum-chemistry
simulation output (the arbiter's prior "real data" validation, see
project memory project_dense_armor_arbiter_physics_validation.md), and
not LLM agent latency (this repo's own Experiment 40/Benchmark v2).

DATA: UCI HAR ("Human Activity Recognition Using Smartphones"), Anguita
et al. 2013, UCI ML Repository -- real 3-axis accelerometer + gyroscope
readings from a Samsung Galaxy S II worn on the waist by 30 real
volunteers, sampled at a real, fixed 50 Hz. Genuine IMU sensor data, the
same sensing modality (accelerometer/gyroscope) real robots use for
proprioception/attitude estimation -- not synthetic, not a robot per se
(a person, not a robot, wore the sensor), but the closest practical
real, public, small-enough-to-download-in-session dataset for this
sensing modality. A real lidar validation is a separate, larger
undertaking (public lidar datasets are GB-scale and mostly require
registration) and is explicitly NOT attempted here.

RECONSTRUCTING A CONTINUOUS SIGNAL -- two real mistakes caught and
fixed before trusting any of this, not assumed correct on the first try:

  1. A first attempt assumed every window sharing a (subject, activity)
     pair was one continuous recording. WRONG: subject 1's STANDING
     windows turned out to be THREE separate, non-adjacent recording
     bouts. Fixed by requiring genuinely contiguous raw-file indices.

  2. A second attempt assumed contiguous raw-file indices (i, i+1, i+2,
     ...) implied a true continuous recording. STILL WRONG in one case:
     subject 21's indices 4010-4056 (all labeled SITTING, contiguous
     indices) had an exact overlap identity (window[i][64:] ==
     window[i+1][:64], real sensor samples, not statistics) at every
     boundary except ONE (index 4032->4033, diff=0.081) -- a real
     recording seam sitting inside what looked like one contiguous run.
     A systematic search (checking the EXACT overlap identity at every
     single boundary in the dataset, not just index adjacency) further
     found that NO run in this dataset ever crosses an activity-label
     change without also breaking exact overlap -- i.e. in this real
     collection protocol, changing activity always means a real
     recording pause, never a seamless in-stream transition. That is a
     genuine property of how this dataset was collected, not an
     assumption.

  Consequence: conditions A/B/C below use ONE run verified to hold
  exact overlap identity at every single boundary (subject 17, activity
  STANDING, indices 3237-3265, 29 windows). Condition D concatenates two
  SEPARATE real, internally-verified segments from the same subject's
  real session (STANDING then WALKING) -- not claimed to be a
  zero-gap seamless transition (this dataset has none), the same honest
  framing Benchmark v2's own "D" condition used for a separately
  constructed real LLM trajectory rather than a literal mid-conversation
  switch.

SIGNAL: total acceleration magnitude, sqrt(x^2+y^2+z^2), from the
dataset's total_acc_{x,y,z} channels (raw accelerometer including
gravity) -- a single scalar channel, same shape as the latency signal
Benchmark v2 used, but now a real physical quantity from a real sensor.

PREREGISTERED PROTOCOL, declared before this was run and never adjusted
after seeing results:

  A. Normal    -- subject 17, real STANDING segment (indices 3237-3265,
                  verified exact-overlap at every boundary -- a real
                  quiet baseline, gravity only, minimal real motion).
  B. Transient -- SAME real STANDING base signal, a short 2-sample spike
                  injected at the telemetry layer (a documented, exact
                  fault: simulates a real, known IMU failure mode -- a
                  bit-flip/glitch sample -- not a change to the real
                  sensor recording itself).
  C. Persistent-- SAME real STANDING base signal, a sustained +3.0g
                  offset added from a fixed point onward -- simulates
                  accelerometer bias/calibration drift, a real, common
                  MEMS-IMU failure mode (e.g. thermal bias drift).
                  UNIT CORRECTION (found while cross-comparing against
                  Experiment 42's lidar signal-to-noise numbers): this
                  was originally documented as "m/s^2", which was wrong
                  -- UCI HAR's raw total_acc channel is in units of g
                  (~9.8 m/s^2), confirmed empirically from the real
                  baseline value at rest (~1.03, unmistakably 1g, not
                  1 m/s^2). The detection numbers below were always
                  computed correctly on the real data; only the printed
                  physical unit label was wrong, corrected here.
  D. Legit change -- REAL DATA, NO INJECTION: subject 17's real STANDING
                  segment (3237-3265) followed by subject 17's real
                  WALKING segment (3348-3364, a separate, internally
                  verified real recording bout from the same subject's
                  session) -- a genuine behavioral transition, the real
                  analogue of Benchmark v2's "legitimate task switch"
                  condition.

DETECTOR: classify_segments + cusum_detector, TWO-SIDED (unlike the
LLM-latency work, one_sided_upper_filter is NOT applied here --
deliberate, not an oversight: for latency, only an increase is ever
meaningful; for acceleration magnitude, an unusually LOW reading is
also a real fault signature (e.g. a sensor freeze/flatline, or genuine
free-fall), so discarding "flagged because low" would discard real
fault coverage this signal actually needs).

radius=25, ref_mult=2 (span=50 samples = 1.0s at 50Hz) declared here
before running, chosen from the real sampling rate, not tuned against
results. cusum k=0.5, h=5.0 -- unchanged library defaults, as used
throughout every prior benchmark in this repo.
"""
import json
import pathlib
import urllib.request
import zipfile

import numpy as np

from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_DATA_ROOT = _THIS_DIR / "data"
_DATA_DIR = _DATA_ROOT / "UCI HAR Dataset" / "train"
_DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def _ensure_dataset():
    """Downloads and extracts UCI HAR (61MB) into the gitignored data/
    folder if not already present -- real network I/O, only runs once."""
    if _DATA_DIR.exists():
        return
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    outer_zip = _DATA_ROOT / "har.zip"
    print(f"Downloading {_DATASET_URL} ...")
    urllib.request.urlretrieve(_DATASET_URL, outer_zip)
    with zipfile.ZipFile(outer_zip) as zf:
        zf.extractall(_DATA_ROOT)
    inner_zip = _DATA_ROOT / "UCI HAR Dataset.zip"
    with zipfile.ZipFile(inner_zip) as zf:
        zf.extractall(_DATA_ROOT)
    assert _DATA_DIR.exists(), f"extraction did not produce the expected {_DATA_DIR}"

ARBITER_KW = dict(radius=25, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
CUSUM_KW = dict(radius=25, ref_mult=2, k=0.5, h=5.0)

SUBJECT = 17
# Real, verified-exact-overlap-at-every-boundary index ranges (inclusive)
# in train/y_train.txt -- two SEPARATE real recording bouts, not one
# continuous session (see module docstring).
STANDING_RANGE = (3237, 3265)  # activity label 5
WALKING_RANGE = (3348, 3364)   # activity label 1
FS_HZ = 50
WINDOW_LEN = 128
STRIDE = 64  # 50% overlap

TRANSIENT_AT = 800
TRANSIENT_WIDTH = 2
TRANSIENT_MAG = 15.0  # g (see module docstring's unit correction note)
PERSISTENT_FROM = 1500
PERSISTENT_OFFSET = 3.0  # g sustained offset (see module docstring's unit correction note)


def _load_channel(name: str) -> np.ndarray:
    return np.loadtxt(_DATA_DIR / "Inertial Signals" / f"{name}_train.txt")


def _reconstruct_continuous(channel: np.ndarray, idx_start: int, idx_end: int) -> np.ndarray:
    """Deoverlap over an EXPLICIT index range [idx_start, idx_end]
    (inclusive) -- verified exactly (max abs diff over the FULL range,
    every boundary) that window[i][64:] == window[i+1][:64] before
    trusting this reconstruction; raises loudly if not, rather than
    silently concatenating across a real recording seam."""
    windows = channel[idx_start:idx_end + 1]
    n = len(windows)
    max_diff = 0.0
    for i in range(n - 1):
        max_diff = max(max_diff, float(np.max(np.abs(windows[i][64:] - windows[i + 1][:64]))))
    if max_diff > 1e-9:
        raise RuntimeError(f"overlap assumption violated for range [{idx_start},{idx_end}], "
                            f"max_diff={max_diff} -- windows not contiguous")
    pieces = [windows[i][:64] for i in range(n - 1)] + [windows[-1]]
    return np.concatenate(pieces)


def _verify_subject_and_labels(idx_start: int, idx_end: int, expected_subject: int, expected_activity: int):
    subj = np.loadtxt(_DATA_DIR / "subject_train.txt").astype(int)
    y = np.loadtxt(_DATA_DIR / "y_train.txt").astype(int)
    assert np.all(subj[idx_start:idx_end + 1] == expected_subject), "subject mismatch in declared range"
    assert np.all(y[idx_start:idx_end + 1] == expected_activity), "activity mismatch in declared range"


def _magnitude(idx_start: int, idx_end: int, expected_subject: int, expected_activity: int) -> np.ndarray:
    _verify_subject_and_labels(idx_start, idx_end, expected_subject, expected_activity)
    x = _reconstruct_continuous(_load_channel("total_acc_x"), idx_start, idx_end)
    yc = _reconstruct_continuous(_load_channel("total_acc_y"), idx_start, idx_end)
    z = _reconstruct_continuous(_load_channel("total_acc_z"), idx_start, idx_end)
    return np.sqrt(x ** 2 + yc ** 2 + z ** 2)


def _detect(x: np.ndarray):
    labels, _, _ = classify_segments(x, **ARBITER_KW)
    flags_da = labels != "clean"
    flags_cs, _ = cusum_detector(x, **CUSUM_KW)
    return flags_da, flags_cs, (flags_da | flags_cs)


def main():
    _ensure_dataset()
    standing = _magnitude(*STANDING_RANGE, SUBJECT, 5)
    walking = _magnitude(*WALKING_RANGE, SUBJECT, 1)
    print(f"Reconstructed STANDING: {len(standing)} samples ({len(standing)/FS_HZ:.1f}s)")
    print(f"Reconstructed WALKING:  {len(walking)} samples ({len(walking)/FS_HZ:.1f}s)")
    assert len(standing) > PERSISTENT_FROM + 200, "STANDING segment too short for the declared protocol"

    results = {}

    # --- A: Normal ---
    x_a = standing.copy()
    da, cs, comb = _detect(x_a)
    results["A_normal"] = dict(x=x_a.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist())

    # --- B: Transient (telemetry-layer injection on the SAME real base) ---
    x_b = standing.copy()
    x_b[TRANSIENT_AT:TRANSIENT_AT + TRANSIENT_WIDTH] += TRANSIENT_MAG
    da, cs, comb = _detect(x_b)
    results["B_transient"] = dict(
        x=x_b.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist(),
        injected_at=TRANSIENT_AT, injected_width=TRANSIENT_WIDTH,
    )

    # --- C: Persistent (telemetry-layer injection on the SAME real base) ---
    x_c = standing.copy()
    x_c[PERSISTENT_FROM:] += PERSISTENT_OFFSET
    da, cs, comb = _detect(x_c)
    results["C_persistent"] = dict(
        x=x_c.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist(),
        injected_at=PERSISTENT_FROM,
    )

    # --- D: Legit change (REAL activity transition, no injection) ---
    x_d = np.concatenate([standing, walking])
    da, cs, comb = _detect(x_d)
    results["D_legit_switch"] = dict(
        x=x_d.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist(),
        switch_at=len(standing),
    )

    out_path = _THIS_DIR / "imu_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\nWrote {out_path}")

    print("\n=== Summary ===")
    fp_a = float(np.mean(np.array(results["A_normal"]["comb"])[50:]))
    print(f"[A] Normal baseline FP rate (skip first 1s warmup): {fp_a:.4f}")

    b_region = slice(TRANSIENT_AT, TRANSIENT_AT + TRANSIENT_WIDTH)
    det_b = float(np.mean(np.array(results["B_transient"]["comb"])[b_region]))
    print(f"[B] Transient injected-region detection rate: {det_b:.3f}")

    c_region = slice(PERSISTENT_FROM, PERSISTENT_FROM + 50)  # first 1s after the shift
    det_c = float(np.mean(np.array(results["C_persistent"]["comb"])[c_region]))
    print(f"[C] Persistent shift, first 1s post-transition detection rate: {det_c:.3f}")

    switch_at = len(standing)
    d_region = slice(switch_at, switch_at + 50)  # first 1s of real walking
    flagged_d = float(np.mean(np.array(results["D_legit_switch"]["comb"])[d_region]))
    print(f"[D] Legit switch, first 1s of real walking flag rate: {flagged_d:.3f}")


if __name__ == "__main__":
    main()
