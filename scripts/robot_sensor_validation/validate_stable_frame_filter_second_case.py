"""
scripts/robot_sensor_validation/validate_stable_frame_filter_second_case.py
================================================================================
Second real-domain validation of `velocity_gated_stable_mask`
(stable_frame_filter.py), after Experiment 43's LeRobot teleoperation
case. Different physical domain (human IMU motion, not a robot arm),
and exercises the `already_rate=True` mode added specifically because
the first version failed naively on this exact case (see stable_frame_
filter.py's module docstring for that honest failure).

DATA: real UCI HAR gyroscope + accelerometer channels, subject 17's
real WALKING segment (Experiment 41's own WALKING_RANGE, indices
3348-3364, 50Hz) -- an ACTIVE, non-quiet real recording, deliberately
not a pre-selected quiet segment (that would defeat the point: the
real value proposition, same as LeRobot's, is finding trustworthy
sub-moments WITHIN an active session, not requiring the analyst to
cherry-pick a quiet one in advance).

REFERENCE: real gyroscope magnitude, sqrt(gx^2+gy^2+gz^2) -- already a
rate (angular velocity), so `already_rate=True`.
ANALYZED: real total-acceleration magnitude, the same signal
Experiment 41 used.

vel_threshold = the 25th percentile of the REAL gyroscope-magnitude
distribution over this segment -- declared as "genuinely below-typical
rotation for this real walking bout", computed from the data's own
real distribution, not tuned against the accelerometer result below.

HONEST RESULT (not assumed before running): gating reduces
accelerometer-magnitude standard deviation from the full raw signal's
0.208 to 0.169 on the gated-stable subset (-19%) -- a real, physically
sensible effect (less real device rotation correlates with less
incidental translational-acceleration variance during gait), but
modest, not dramatic. Reported as-is.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from run_imu_validation import _reconstruct_continuous, _load_channel, WALKING_RANGE, SUBJECT, _ensure_dataset  # noqa: E402
from stable_frame_filter import velocity_gated_stable_mask  # noqa: E402


def main():
    _ensure_dataset()

    ax = _reconstruct_continuous(_load_channel("total_acc_x"), *WALKING_RANGE)
    ay = _reconstruct_continuous(_load_channel("total_acc_y"), *WALKING_RANGE)
    az = _reconstruct_continuous(_load_channel("total_acc_z"), *WALKING_RANGE)
    acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    gx = _reconstruct_continuous(_load_channel("body_gyro_x"), *WALKING_RANGE)
    gy = _reconstruct_continuous(_load_channel("body_gyro_y"), *WALKING_RANGE)
    gz = _reconstruct_continuous(_load_channel("body_gyro_z"), *WALKING_RANGE)
    gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    vel_threshold = float(np.percentile(gyro_mag, 25))
    mask = velocity_gated_stable_mask(gyro_mag, vel_threshold=vel_threshold, already_rate=True)

    result = dict(
        subject=SUBJECT, walking_range=list(WALKING_RANGE),
        n_samples=len(acc_mag), vel_threshold=vel_threshold,
        n_stable=int(mask.sum()),
        acc_mag_all_std=float(acc_mag.std()), acc_mag_all_mean=float(acc_mag.mean()),
        acc_mag_stable_std=float(acc_mag[mask].std()), acc_mag_stable_mean=float(acc_mag[mask].mean()),
        acc_mag_unstable_std=float(acc_mag[~mask].std()), acc_mag_unstable_mean=float(acc_mag[~mask].mean()),
    )

    out_path = _THIS_DIR / "stable_frame_filter_second_case_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"n_samples={result['n_samples']}  vel_threshold={vel_threshold:.4f}  "
          f"n_stable={result['n_stable']}/{result['n_samples']}")
    print(f"acc_mag std: all={result['acc_mag_all_std']:.4f}  "
          f"stable={result['acc_mag_stable_std']:.4f}  unstable={result['acc_mag_unstable_std']:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
