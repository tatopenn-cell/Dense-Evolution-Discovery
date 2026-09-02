"""
scripts/robot_sensor_validation/run_lidar_validation.py
===========================================================
Second sensor-modality validation of Dense-Armor's runtime detectors on
real hardware (after Experiment 41's IMU work) -- this time a real
rotating lidar, the sensing modality actually requested.

DATA: Sydney Urban Objects Dataset (Quadros, Underwood, Douillard 2013,
University of Sydney ACFR) -- 631 real objects (cars, pedestrians,
trees, signs, ...) individually segmented from a real Velodyne HDL-64E
LIDAR mounted on a vehicle driven through Sydney CBD on 2011-11-04. Not
KITTI/nuScenes (GB-scale, mostly registration-gated) -- chosen because
it is real, public, no login wall, and small (79.5MB).

FORMAT VERIFICATION, done before trusting anything: this dataset ships
TWO different binary layouts under one archive -- `objects/*.bin` (a
documented 9-field struct: t,intensity,id,x,y,z,azimuth,range,pid) and
`scans/*.bin` (raw, undocumented-in-detail Velodyne packet format, "8
byte timestamp + 1206 byte packet" per the README). A first attempt
parsed `scans/*.bin` with the OBJECTS dtype -- produced impossible
timestamps (values near int64 min/max in the same file). Caught by
sanity-checking the decoded timestamp against a real calendar date
instead of trusting the parse. The `objects/*.bin` files, parsed with
the documented dtype, decode to a real, exact date (2011-11-04) that
matches the dataset's own separately-provided CSV export of the same
object to the microsecond -- cross-verified, not assumed -- so this
script uses ONLY `objects/*.bin`, never the raw `scans/*.bin` packets
(reconstructing real values from raw Velodyne packets is a separate,
larger undertaking, deliberately not attempted here rather than risk a
silently-wrong parse).

RECONSTRUCTING A REAL TEMPORAL SEQUENCE: unlike the IMU experiment
(one fixed-rate continuous channel), lidar objects arrive at irregular
real intervals (median gap 1.5ms -- many objects detected in the same
360-degree sweep -- up to a real 175s gap between some consecutive
detections). Verified this is ONE continuous real session (2011-11-04
01:30:15 to 01:51:23, 21 real minutes, no gap over 175s, so no separate
multi-day recording bouts the way the IMU dataset had). Signal: mean
range (meters) of each object's points, one scalar per object, ordered
by that object's real mean timestamp -- an irregularly-sampled real
index sequence, same treatment as Benchmark v2's LLM-latency sequence
(ordered by real event, not wall-clock time).

PREREGISTERED PROTOCOL, declared before this was run and never adjusted
after seeing results:

  A. Normal    -- all 631 real objects' mean range, in real chronological
                  order -- a real lidar's actual mixed-object operating
                  regime (cars, pedestrians, trees, signs at varying real
                  distances), not an artificially quiet baseline.
  B. Transient -- SAME real range sequence, a short 2-object spike
                  injected at the telemetry layer (+50m, simulating a
                  real, known lidar fault: a spurious long-range return,
                  e.g. a specular reflection or multi-path echo).
  C. Persistent-- SAME real range sequence, a sustained +10m offset
                  added from a fixed point onward -- simulates a real
                  range-calibration bias drift.
  D. Real natural transition -- REAL DATA, NO INJECTION: split at the
                  dataset's own largest real time gap (~175s, found by
                  direct inspection, not assumed) -- whatever the
                  detector does across a genuine real pause in the
                  session is reported as-is, not framed as a specific
                  expected behavioral change (unlike Experiment 41's
                  STANDING->WALKING, there is no single clean "activity"
                  label change to point to here).

DETECTOR: classify_segments + cusum_detector, TWO-SIDED -- for lidar
range, both directions are real fault signatures (unusually LOW: a
false near-return / occlusion artifact; unusually HIGH: signal loss /
specular reflection), same reasoning as Experiment 41's acceleration
signal, so one_sided_upper_filter is deliberately not applied here
either.

radius=15, ref_mult=2 (a 30-object causal window) declared here before
running -- chosen for the real ~631-object sequence length, not tuned
against results. cusum k=0.5, h=5.0 -- unchanged library defaults.
"""
import glob
import json
import os
import pathlib
import tarfile
import urllib.request

import numpy as np

from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_DATA_ROOT = _THIS_DIR / "lidar_data"
_DATA_DIR = _DATA_ROOT / "sydney-urban-objects-dataset"
_OBJECTS_DIR = _DATA_DIR / "objects"
_DATASET_URL = "https://www.acfr.usyd.edu.au/papers/data/sydney-urban-objects-dataset.tar.gz"


def _ensure_dataset():
    """Downloads and extracts the Sydney Urban Objects dataset (79.5MB)
    into the gitignored lidar_data/ folder if not already present."""
    if _DATA_DIR.exists():
        return
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = _DATA_ROOT / "sydney.tar.gz"
    print(f"Downloading {_DATASET_URL} ...")
    urllib.request.urlretrieve(_DATASET_URL, archive)
    with tarfile.open(archive) as tf:
        tf.extractall(_DATA_ROOT)
    assert _DATA_DIR.exists(), f"extraction did not produce the expected {_DATA_DIR}"

_DTYPE_NAMES = ["t", "intensity", "id", "x", "y", "z", "azimuth", "range", "pid"]
_DTYPE_FORMATS = ["int64", "uint8", "uint8", "float32", "float32", "float32", "float32", "float32", "int32"]
_BIN_DTYPE = np.dtype(dict(names=_DTYPE_NAMES, formats=_DTYPE_FORMATS))

ARBITER_KW = dict(radius=15, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
CUSUM_KW = dict(radius=15, ref_mult=2, k=0.5, h=5.0)

TRANSIENT_AT = 300
TRANSIENT_WIDTH = 2
TRANSIENT_MAG = 50.0  # meters
PERSISTENT_FROM = 450
PERSISTENT_OFFSET = 10.0  # meters


def _load_all_objects():
    files = sorted(glob.glob(str(_OBJECTS_DIR / "*.bin")))
    recs = []
    for f in files:
        d = np.fromfile(f, _BIN_DTYPE)
        if len(d) == 0:
            continue
        t_mean = float(d["t"].mean())
        range_mean = float(d["range"].mean())
        recs.append((t_mean, range_mean, len(d), os.path.basename(f)))
    recs.sort(key=lambda r: r[0])
    return recs


def _detect(x: np.ndarray):
    labels, _, _ = classify_segments(x, **ARBITER_KW)
    flags_da = labels != "clean"
    flags_cs, _ = cusum_detector(x, **CUSUM_KW)
    return flags_da, flags_cs, (flags_da | flags_cs)


def main():
    _ensure_dataset()
    recs = _load_all_objects()
    print(f"Loaded {len(recs)} real objects")
    t = np.array([r[0] for r in recs])
    ranges = np.array([r[1] for r in recs])
    gaps_s = np.diff(t) / 1e6
    max_gap_idx = int(np.argmax(gaps_s))
    print(f"Max real time gap: {gaps_s[max_gap_idx]:.1f}s at object index {max_gap_idx}")
    assert len(ranges) > PERSISTENT_FROM + 100, "not enough real objects for the declared protocol"

    results = {}

    # --- A: Normal (real mixed-object range sequence, no injection) ---
    x_a = ranges.copy()
    da, cs, comb = _detect(x_a)
    results["A_normal"] = dict(x=x_a.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist())

    # --- B: Transient (telemetry-layer injection on the SAME real base) ---
    x_b = ranges.copy()
    x_b[TRANSIENT_AT:TRANSIENT_AT + TRANSIENT_WIDTH] += TRANSIENT_MAG
    da, cs, comb = _detect(x_b)
    results["B_transient"] = dict(
        x=x_b.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist(),
        injected_at=TRANSIENT_AT, injected_width=TRANSIENT_WIDTH,
    )

    # --- C: Persistent (telemetry-layer injection on the SAME real base) ---
    x_c = ranges.copy()
    x_c[PERSISTENT_FROM:] += PERSISTENT_OFFSET
    da, cs, comb = _detect(x_c)
    results["C_persistent"] = dict(
        x=x_c.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist(),
        injected_at=PERSISTENT_FROM,
    )

    # --- D: Real natural transition (no injection) ---
    x_d = ranges.copy()
    da, cs, comb = _detect(x_d)
    results["D_real_gap_transition"] = dict(
        x=x_d.tolist(), da=da.tolist(), cs=cs.tolist(), comb=comb.tolist(),
        gap_at_index=max_gap_idx, gap_seconds=float(gaps_s[max_gap_idx]),
    )

    out_path = _THIS_DIR / "lidar_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\nWrote {out_path}")

    print("\n=== Summary ===")
    fp_a = float(np.mean(np.array(results["A_normal"]["comb"])[30:]))
    print(f"[A] Normal baseline FP rate (skip first 30-object warmup): {fp_a:.4f}")

    b_region = slice(TRANSIENT_AT, TRANSIENT_AT + TRANSIENT_WIDTH)
    det_b = float(np.mean(np.array(results["B_transient"]["comb"])[b_region]))
    print(f"[B] Transient injected-region detection rate: {det_b:.3f}")

    c_region = slice(PERSISTENT_FROM, PERSISTENT_FROM + 30)
    det_c = float(np.mean(np.array(results["C_persistent"]["comb"])[c_region]))
    print(f"[C] Persistent shift, first 30 objects post-transition detection rate: {det_c:.3f}")

    d_region = slice(max_gap_idx, max_gap_idx + 30)
    flagged_d = float(np.mean(np.array(results["D_real_gap_transition"]["comb"])[d_region]))
    print(f"[D] Real gap transition, next 30 objects flag rate: {flagged_d:.3f}")


if __name__ == "__main__":
    main()
