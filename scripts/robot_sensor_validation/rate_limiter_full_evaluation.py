# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/rate_limiter_full_evaluation.py
=====================================================================
Tests causal_rate_limited_follower.py's causal velocity+acceleration
rate limiter -- the mechanism tried after healing_filter's causal
rewrite proved a structural dead end (Experiment 52: 1.7% win rate vs
a trivial moving median, since spike-vs-real-change is causally
unclassifiable at the instant it happens).

This mechanism sidesteps classification entirely: instead of trying to
tell WHETHER a deviation is real, it bounds HOW FAST the applied
command can physically change, regardless of cause -- the same
principle real robot controllers already use (torque/velocity/
acceleration limits), grounded in Berscheid & Kroger (2021),
"Jerk-limited Real-time Trajectory Generation with Arbitrary Target
States", Robotics: Science and Systems XVII, arXiv:2105.04830 (fetched
and verified directly: causal by construction -- uses only the current
kinematic state, no lookahead -- validated on 1e9 real trajectories,
~20us real compute per DoF; indexed in quantumrag's new
robotica_generazione_traiettoria collection). This module implements a
deliberately SIMPLER special case -- a causal double-integrator
velocity+acceleration limiter, not Ruckig's full time-optimal
jerk-limited synthesis.

PROTOCOL: same real spike injection as Experiment 52 (5% density, 5x
real local std), all 6 real LeRobot joints x 20 real seeds (120 real
trials), max_vel/max_accel derived from each real joint's OWN 99th-
percentile velocity/acceleration on the clean signal (not hand-picked).
TWO metrics, not one: RMSE vs real clean signal (as before), AND max
real instantaneous jump in the applied output -- the actual physical-
safety metric RMSE alone doesn't capture.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from causal_rate_limited_follower import causal_rate_limited_follower  # noqa: E402


def load_all_joints():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    data_root = _THIS_DIR / "lerobot_data"
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    return np.stack(sub["action"].values)  # (303, 6)


def moving_median_baseline(x, radius=2):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        out[i] = np.median(x[lo:hi])
    return out


def inject_spikes(x_clean, seed):
    rng = np.random.default_rng(seed)
    n = len(x_clean)
    real_std = np.std(x_clean)
    spike_mag = 5.0 * real_std
    n_spikes = max(1, n // 20)
    spike_idx = rng.choice(np.arange(10, n - 10), size=n_spikes, replace=False)
    x_spiked = x_clean.copy()
    x_spiked[spike_idx] += rng.choice([-1, 1], size=n_spikes) * spike_mag
    return x_spiked


def real_limits_from_clean(x_clean, dt=1.0):
    vel = np.diff(x_clean) / dt
    accel = np.diff(vel) / dt
    return float(np.percentile(np.abs(vel), 99)), float(np.percentile(np.abs(accel), 99))


def main():
    action = load_all_joints()
    n_joints = action.shape[1]
    seeds = range(20)

    rmse_rl, rmse_med = [], []
    jump_rl, jump_med, jump_raw = [], [], []
    wins_rmse = wins_jump = total = 0

    for j in range(n_joints):
        x_clean = action[:, j].astype(np.float64)
        max_vel, max_accel = real_limits_from_clean(x_clean)
        for seed in seeds:
            x_spiked = inject_spikes(x_clean, seed=seed * 100 + j)
            rl = causal_rate_limited_follower(x_spiked, max_vel=max_vel, max_accel=max_accel)
            med = moving_median_baseline(x_spiked, radius=2)

            r_rl = float(np.sqrt(np.mean((rl - x_clean) ** 2)))
            r_med = float(np.sqrt(np.mean((med - x_clean) ** 2)))
            j_rl = float(np.max(np.abs(np.diff(rl))))
            j_med = float(np.max(np.abs(np.diff(med))))
            j_raw = float(np.max(np.abs(np.diff(x_spiked))))

            rmse_rl.append(r_rl); rmse_med.append(r_med)
            jump_rl.append(j_rl); jump_med.append(j_med); jump_raw.append(j_raw)
            wins_rmse += int(r_rl < r_med)
            wins_jump += int(j_rl < j_med)
            total += 1

    result = dict(
        n_trials=total,
        wins_rmse=wins_rmse, mean_rmse_rate_limited=float(np.mean(rmse_rl)), mean_rmse_moving_median=float(np.mean(rmse_med)),
        wins_jump=wins_jump, mean_jump_raw=float(np.mean(jump_raw)),
        mean_jump_rate_limited=float(np.mean(jump_rl)), mean_jump_moving_median=float(np.mean(jump_med)),
    )
    print(f"n={total} real trials (6 joints x 20 seeds)")
    print(f"RMSE: rate_limited wins {wins_rmse}/{total} ({wins_rmse/total*100:.1f}%) vs moving median")
    print(f"  mean RMSE: rate_limited={result['mean_rmse_rate_limited']:.4f}  moving_median={result['mean_rmse_moving_median']:.4f}")
    print(f"Max jump: rate_limited wins {wins_jump}/{total} ({wins_jump/total*100:.1f}%) vs moving median")
    print(f"  mean max jump: raw={result['mean_jump_raw']:.2f}  rate_limited={result['mean_jump_rate_limited']:.2f}  moving_median={result['mean_jump_moving_median']:.2f}")

    out_path = _THIS_DIR / "rate_limiter_full_evaluation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
