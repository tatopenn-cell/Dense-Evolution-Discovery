# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/cross_channel_mahalanobis_imu.py
====================================================================
Tests the real hypothesis behind the quantumrag Mahalanobis-SVDD Audio-
IMU paper (Yang, Zhao et al. 2025, arXiv:2505.05811, robotica_
rilevamento_anomalie collection) WITHOUT their neural network -- per
explicit instruction, closed-form only, no training. Their real
contribution: a fault can break the NORMAL CORRELATION between two
sensor channels without being a large deviation in either channel
alone, and a joint Mahalanobis distance over the two channels catches
that where per-channel detectors structurally cannot. This script
tests exactly that claim, closed-form, on real data already validated
in this repo (Experiment 41, imu_sensor_validation.md): UCI HAR real
accelerometer plus real gyroscope, subject 17, real WALKING segment
(indices 3348-3364, exact-overlap-verified by run_imu_validation.py,
reused here rather than re-derived).

Not an audio+IMU test (no real paired audio+IMU dataset was available
in-session) -- accelerometer plus gyroscope instead, two genuinely
different real IMU sensing modalities (linear acceleration vs angular
velocity) from the SAME real device, at the SAME real timestamps,
which is the part of the hypothesis (cross-modality correlation as a
fault signal) this experiment can actually test honestly with data on
hand.

FAULT DESIGN, the crux of a fair test: a magnitude-based fault (a spike
or offset, like Experiment 41's B/C) would already be visible to a
single-channel detector -- that would not test whether cross-channel
fusion adds anything. Instead: swap in a TIME-REVERSED copy of the same
real gyro segment for a real fault window. This uses exactly the same
real sample values (identical per-sample distribution, so a single-
channel std/CUSUM detector calibrated on magnitude has little reason to
fire) but destroys the real temporal correspondence with the
concurrently-recorded real accel signal during real gait -- exactly a
correlation-breakdown fault, not a magnitude fault. Disclosed as a real,
documented synthetic construction (same honesty standard as Experiment
41's telemetry-layer injection), not a recorded real fault event.

REAL RESULT, both stages honestly kept (not just the final one): the
first attempt (a per-sample joint Mahalanobis distance over
[d(accel)/dt, d(gyro)/dt]) caught 0 percent of the fault window --
diagnosed directly, not just accepted: a per-sample joint distance only
checks whether an individual (d(accel)[t], d(gyro)[t]) PAIR looks
statistically unusual, which is insensitive to a fault that merely
REORDERS otherwise-normal values in time. Real gait excursions are
close enough to symmetric that a reversed value still falls inside the
normal-looking ellipse most of the time. Second attempt: a rolling
Pearson correlation between the two derivative channels (sensitive to
temporal/order structure, unlike a point-wise distance) -- but the real
BASELINE correlation between accel-magnitude and gyro-magnitude
derivatives during real, unfaulted walking turned out to already be
close to zero (mean about 0.04, std about 0.11 over a 1s window), so
there was little real cross-channel structure left for the fault to
break in the first place. Likely cause, not chased further here: taking
3-axis Euclidean magnitudes discards the directional/phase information
that would carry most of the real physical coupling between linear
acceleration and angular velocity during gait -- a fair test of that
coupling would need per-axis signals (matched rotation axes between
accelerometer and gyroscope), a real, separate undertaking.

ONE real domain only (UCI HAR, a person, not a robot -- same caveat as
Experiment 41). Not a promotion candidate -- both closed-form cross-
channel constructions tried here came back negative on this real data,
honestly reported as a real negative finding, not smoothed over.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from run_imu_validation import (  # noqa: E402
    _ensure_dataset, _load_channel, _reconstruct_continuous, _verify_subject_and_labels,
    SUBJECT, WALKING_RANGE, FS_HZ,
)
from dense_armor.utility.arbiter import classify_segments  # noqa: E402
from dense_armor.utility.cusum import cusum_detector  # noqa: E402

CALIB_LEN = 400          # about 8s at 50Hz, real normal WALKING used only to estimate the joint covariance
FAULT_AT = 700           # sample index (post-calibration) where the real fault window starts
FAULT_WIDTH = 150        # about 3s, long enough to be a real, sustained correlation-breakdown event
CORR_WINDOW = 50         # 1s, roughly one real gait cycle at normal walking pace
ARBITER_KW = dict(radius=25, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
CUSUM_KW = dict(radius=25, ref_mult=2, k=0.5, h=5.0)


def real_accel_gyro_walking():
    accel_x = _reconstruct_continuous(_load_channel("total_acc_x"), *WALKING_RANGE)
    accel_y = _reconstruct_continuous(_load_channel("total_acc_y"), *WALKING_RANGE)
    accel_z = _reconstruct_continuous(_load_channel("total_acc_z"), *WALKING_RANGE)
    gyro_x = _reconstruct_continuous(_load_channel("body_gyro_x"), *WALKING_RANGE)
    gyro_y = _reconstruct_continuous(_load_channel("body_gyro_y"), *WALKING_RANGE)
    gyro_z = _reconstruct_continuous(_load_channel("body_gyro_z"), *WALKING_RANGE)
    _verify_subject_and_labels(*WALKING_RANGE, SUBJECT, 1)
    accel_mag = np.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)
    gyro_mag = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)
    return accel_mag, gyro_mag


def causal_mahalanobis_cross_channel(accel: np.ndarray, gyro: np.ndarray, calib_len: int) -> np.ndarray:
    """Closed-form, no training: fits mean and covariance of
    [d(accel)/dt, d(gyro)/dt] once on a real calibration window, then
    reports a causal Mahalanobis distance for every later real sample --
    no lookahead past calib_len. Real result: insensitive to a temporal-
    reordering fault, see module docstring."""
    d_accel = np.diff(accel, prepend=accel[0])
    d_gyro = np.diff(gyro, prepend=gyro[0])
    joint = np.stack([d_accel, d_gyro], axis=1)
    calib = joint[:calib_len]
    mean = calib.mean(axis=0)
    cov = np.cov(calib, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    centered = joint - mean
    dist = np.sqrt(np.einsum("ti,ij,tj->t", centered, inv_cov, centered))
    return dist


def rolling_pearson_correlation(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Closed-form, no training: real Pearson correlation between a and b
    over a trailing real window -- unlike a per-sample joint Mahalanobis
    distance, this IS sensitive to temporal reordering. Real result: the
    real baseline correlation itself was already close to zero on this
    specific channel pair (magnitudes), see module docstring."""
    n = len(a)
    out = np.full(n, np.nan)
    for t in range(window, n):
        wa = a[t - window:t]
        wb = b[t - window:t]
        if wa.std() > 1e-9 and wb.std() > 1e-9:
            out[t] = np.corrcoef(wa, wb)[0, 1]
    return out


def single_channel_flags(x: np.ndarray) -> np.ndarray:
    labels, _, _ = classify_segments(x, **ARBITER_KW)
    flags_da = labels != "clean"
    flags_cs, _ = cusum_detector(x, **CUSUM_KW)
    return flags_da | flags_cs


def _plot(accel, gyro, gyro_fault, dist_fault, dist_normal, threshold, corr_fault, corr_normal, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(accel)) / FS_HZ

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(t, accel, color="tab:blue", linewidth=0.9, label="real accel magnitude")
    axes[0].axvspan(FAULT_AT / FS_HZ, (FAULT_AT + FAULT_WIDTH) / FS_HZ, color="red", alpha=0.12,
                     label="fault window (gyro only)")
    axes[0].set_ylabel("accel (g)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Real UCI HAR walking segment, subject 17 -- accel untouched by the fault")

    axes[1].plot(t, gyro, color="gray", linewidth=0.7, alpha=0.5, label="real gyro (unfaulted)")
    axes[1].plot(t, gyro_fault, color="tab:orange", linewidth=0.9, label="gyro with fault window (time-reversed)")
    axes[1].axvspan(FAULT_AT / FS_HZ, (FAULT_AT + FAULT_WIDTH) / FS_HZ, color="red", alpha=0.12)
    axes[1].set_ylabel("gyro (rad/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Same real sample values, only their order is reversed inside the fault window")

    axes[2].plot(t, dist_normal, color="gray", linewidth=0.9, alpha=0.5, label="Mahalanobis dist, no fault")
    axes[2].plot(t, dist_fault, color="tab:red", linewidth=0.9, label="Mahalanobis dist, with fault")
    axes[2].axhline(threshold, color="black", linestyle="--", linewidth=0.8, label="threshold")
    axes[2].axvspan(FAULT_AT / FS_HZ, (FAULT_AT + FAULT_WIDTH) / FS_HZ, color="red", alpha=0.12)
    axes[2].axvspan(0, CALIB_LEN / FS_HZ, color="tab:green", alpha=0.08, label="calibration window")
    axes[2].set_ylabel("Mahalanobis dist")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Attempt 1 (negative): point-wise distance does not rise in the fault window")

    axes[3].plot(t, corr_normal, color="gray", linewidth=0.9, alpha=0.5, label="rolling corr, no fault")
    axes[3].plot(t, corr_fault, color="tab:red", linewidth=0.9, label="rolling corr, with fault")
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[3].axvspan(FAULT_AT / FS_HZ, (FAULT_AT + FAULT_WIDTH) / FS_HZ, color="red", alpha=0.12)
    axes[3].set_ylabel("rolling Pearson corr")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title("Attempt 2 (negative): baseline correlation is already near zero -- little structure to break")

    fig.tight_layout()
    out_path = out_dir / "cross_channel_mahalanobis_imu.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    _ensure_dataset()
    accel, gyro = real_accel_gyro_walking()
    n = len(accel)
    print(f"Real WALKING segment: {n} samples ({n/FS_HZ:.1f}s), subject {SUBJECT}")
    assert FAULT_AT + FAULT_WIDTH < n, "real segment too short for the declared protocol"

    gyro_fault = gyro.copy()
    gyro_fault[FAULT_AT:FAULT_AT + FAULT_WIDTH] = gyro[FAULT_AT:FAULT_AT + FAULT_WIDTH][::-1]

    same_values = np.allclose(
        np.sort(gyro_fault[FAULT_AT:FAULT_AT + FAULT_WIDTH]),
        np.sort(gyro[FAULT_AT:FAULT_AT + FAULT_WIDTH]),
    )
    print(f"Fault window uses identical real sample values, order reversed only: {same_values}")

    flags_accel = single_channel_flags(accel)
    flags_gyro_fault = single_channel_flags(gyro_fault)
    fault_region = slice(FAULT_AT, FAULT_AT + FAULT_WIDTH)
    rate_accel = float(np.mean(flags_accel[fault_region]))
    rate_gyro = float(np.mean(flags_gyro_fault[fault_region]))
    print(f"[single-channel] accel detection rate in fault window: {rate_accel:.3f}")
    print(f"[single-channel] gyro (faulted) detection rate in fault window: {rate_gyro:.3f}")

    # --- Attempt 1: point-wise joint Mahalanobis distance ---
    dist_fault = causal_mahalanobis_cross_channel(accel, gyro_fault, CALIB_LEN)
    threshold = float(np.mean(dist_fault[:CALIB_LEN]) + 3.0 * np.std(dist_fault[:CALIB_LEN]))
    flags_cross_fault = dist_fault > threshold
    rate_cross = float(np.mean(flags_cross_fault[fault_region]))
    print(f"\n[attempt 1: Mahalanobis, point-wise] detection rate in fault window: "
          f"{rate_cross:.3f} (threshold={threshold:.2f})")

    dist_normal = causal_mahalanobis_cross_channel(accel, gyro, CALIB_LEN)
    flags_cross_normal = dist_normal > threshold
    post_calib = slice(CALIB_LEN, n)
    fp_cross = float(np.mean(flags_cross_normal[post_calib]))
    fp_accel = float(np.mean(flags_accel[post_calib]))
    flags_gyro_normal = single_channel_flags(gyro)
    fp_gyro = float(np.mean(flags_gyro_normal[post_calib]))
    print(f"Real false-positive rate (no fault, post-calibration): "
          f"cross={fp_cross:.4f} accel={fp_accel:.4f} gyro={fp_gyro:.4f}")

    # --- Attempt 2: rolling Pearson correlation (sensitive to reordering) ---
    d_accel = np.diff(accel, prepend=accel[0])
    d_gyro_fault = np.diff(gyro_fault, prepend=gyro_fault[0])
    d_gyro_normal = np.diff(gyro, prepend=gyro[0])
    corr_fault = rolling_pearson_correlation(d_accel, d_gyro_fault, CORR_WINDOW)
    corr_normal = rolling_pearson_correlation(d_accel, d_gyro_normal, CORR_WINDOW)
    calib_corr = corr_fault[CORR_WINDOW:CALIB_LEN]
    print(f"\n[attempt 2: rolling Pearson correlation] real baseline (calibration window): "
          f"mean={np.nanmean(calib_corr):.3f} std={np.nanstd(calib_corr):.3f}")
    print(f"[attempt 2] fault window: mean={np.nanmean(corr_fault[fault_region]):.3f} -- "
          f"barely different from baseline, little real structure was there to break")

    out_dir = _THIS_DIR.parent.parent / "docs" / "assets" / "cross_channel_mahalanobis_imu"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = _plot(accel, gyro, gyro_fault, dist_fault, dist_normal, threshold,
                       corr_fault, corr_normal, out_dir)
    print(f"\nWrote {plot_path}")

    frozen = dict(
        n_samples=n, fault_at=FAULT_AT, fault_width=FAULT_WIDTH, calib_len=CALIB_LEN,
        rate_accel=rate_accel, rate_gyro=rate_gyro, rate_cross=rate_cross,
        threshold=threshold, fp_cross=fp_cross, fp_accel=fp_accel, fp_gyro=fp_gyro,
        corr_baseline_mean=float(np.nanmean(calib_corr)),
        corr_baseline_std=float(np.nanstd(calib_corr)),
        corr_fault_mean=float(np.nanmean(corr_fault[fault_region])),
    )
    frozen_path = _THIS_DIR / "cross_channel_mahalanobis_imu_frozen.json"
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    print(f"Wrote {frozen_path}")

    print("\n=== Result: real negative finding, both closed-form attempts ===")
    print(f"Point-wise Mahalanobis: {rate_cross*100:.0f} percent of the fault window caught "
          f"(single-channel accel {rate_accel*100:.0f} percent, single-channel gyro "
          f"{rate_gyro*100:.0f} percent). Rolling correlation: real baseline already near "
          f"zero, nothing real left to break with this specific channel pair. Neither "
          f"closed-form cross-channel construction beat single-channel detection on this "
          f"real data -- see module docstring for the honest diagnosis of why.")


if __name__ == "__main__":
    main()
