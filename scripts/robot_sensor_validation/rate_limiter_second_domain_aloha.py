# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/rate_limiter_second_domain_aloha.py
=========================================================================
Second, independent real physical domain for causal_rate_limited_follower,
after LeRobot SO-101 (single 6-DOF arm, 30Hz). This domain: real ALOHA
(lerobot/aloha_static_coffee) -- a genuinely different real robot
(bimanual, 14-DOF action, real 50Hz control rate, different actuator
class) -- not just a different episode of the same robot. Same real
protocol as the SO-101 evaluation: real spike injection (5% density, 5x
real local std), real per-joint limits from each joint's own 99th-
percentile velocity/acceleration on the clean signal, scored on both
RMSE-vs-clean and max real instantaneous jump.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from causal_rate_limited_follower import causal_rate_limited_follower  # noqa: E402
from rate_limiter_full_evaluation import (  # noqa: E402
    moving_median_baseline, inject_spikes, real_limits_from_clean,
)


def load_aloha_episode_0():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    data_root = _THIS_DIR / "lerobot_data_aloha"
    parquet_path = hf_hub_download(
        repo_id="lerobot/aloha_static_coffee", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    return np.stack(sub["action"].values)  # (n, 14)


def main():
    action = load_aloha_episode_0()
    n_joints = action.shape[1]
    seeds = range(20)
    print(f"Real ALOHA episode 0: {action.shape[0]} frames, {n_joints} real DoF")

    rmse_rl, rmse_med = [], []
    jump_rl, jump_med, jump_raw = [], [], []
    wins_rmse = wins_jump = total = 0

    for j in range(n_joints):
        x_clean = action[:, j].astype(np.float64)
        max_vel, max_accel = real_limits_from_clean(x_clean)
        if max_vel < 1e-9 or max_accel < 1e-9:
            continue  # a real DoF that never moves in this episode -- skip, not a real test case
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
        domain="aloha_static_coffee", n_dof_used=n_joints, n_trials=total,
        wins_rmse=wins_rmse, mean_rmse_rate_limited=float(np.mean(rmse_rl)), mean_rmse_moving_median=float(np.mean(rmse_med)),
        wins_jump=wins_jump, mean_jump_raw=float(np.mean(jump_raw)),
        mean_jump_rate_limited=float(np.mean(jump_rl)), mean_jump_moving_median=float(np.mean(jump_med)),
    )
    print(f"n={total} real trials")
    print(f"RMSE: rate_limited wins {wins_rmse}/{total} ({wins_rmse/total*100:.1f}%) vs moving median")
    print(f"  mean RMSE: rate_limited={result['mean_rmse_rate_limited']:.4f}  moving_median={result['mean_rmse_moving_median']:.4f}")
    print(f"Max jump: rate_limited wins {wins_jump}/{total} ({wins_jump/total*100:.1f}%) vs moving median")
    print(f"  mean max jump: raw={result['mean_jump_raw']:.2f}  rate_limited={result['mean_jump_rate_limited']:.2f}  moving_median={result['mean_jump_moving_median']:.2f}")

    out_path = _THIS_DIR / "rate_limiter_second_domain_aloha_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
