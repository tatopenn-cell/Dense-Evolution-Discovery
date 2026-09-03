# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/cbf_filter_second_domain_aloha.py
========================================================================
Second, independent real physical domain for geometric_cbf_filter, after
LeRobot SO-101 (single 6-DoF arm, 30Hz). Same real ALOHA domain
(lerobot/aloha_static_coffee, bimanual, 14-DoF action, real 50Hz) already
used for rate_limited_follower's own second-domain check -- a genuinely
different real robot, not just a different episode of the same one. Same
protocol as the SO-101 CBF evaluation: obstacles placed IN each real
joint's own path, invariance checked from a real safe start, minimal
invasiveness measured per-step on u.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from geometric_cbf_filter import cbf_filtered_trajectory  # noqa: E402


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
    print(f"Real ALOHA episode 0: {action.shape[0]} frames, {n_joints} real DoF")

    n_invariance_trials = n_invariance_ok = 0
    deviations = []

    for j in range(n_joints):
        x = action[:, j].astype(np.float64)
        real_range = x.max() - x.min()
        if real_range < 1e-6:
            continue
        for frac in (0.25, 0.5, 0.75):
            obstacle = float(x.min() + frac * real_range)
            safe_dist = real_range * 0.05
            h_raw = (x - obstacle) ** 2 - safe_dist ** 2
            if not np.any(h_raw < 0) or h_raw[0] < 0:
                continue
            n_invariance_trials += 1

            filtered = cbf_filtered_trajectory(x, obstacle=obstacle, safe_dist=safe_dist, alpha_gain=2.0, n_substeps=20)
            h_filtered = (filtered - obstacle) ** 2 - safe_dist ** 2
            ok = not np.any(h_filtered < -1e-6)
            n_invariance_ok += int(ok)
            if not ok:
                print(f"  VIOLATION joint={j} frac={frac}: min h_filtered={h_filtered.min():.6f}")

            for i in range(1, len(x)):
                if abs(filtered[i - 1] - obstacle) > 3 * safe_dist:
                    u_des = x[i] - x[i - 1]
                    u_safe = filtered[i] - filtered[i - 1]
                    deviations.append(abs(u_safe - u_des))

    deviations = np.array(deviations)
    result = dict(
        domain="aloha_static_coffee", n_dof=n_joints,
        n_invariance_trials=n_invariance_trials, n_invariance_ok=n_invariance_ok,
        n_invasiveness_checks=int(len(deviations)),
        n_invasiveness_nonzero=int(np.sum(deviations > 1e-6)),
        max_invasiveness_deviation=float(deviations.max()) if len(deviations) else 0.0,
        median_invasiveness_deviation=float(np.median(deviations)) if len(deviations) else 0.0,
    )
    print(f"Invariance: {n_invariance_ok}/{n_invariance_trials} real (joint, obstacle) trials never violate the real safe set")
    print(f"Minimal invasiveness: {result['n_invasiveness_nonzero']}/{result['n_invasiveness_checks']} real per-step checks nonzero, "
          f"max={result['max_invasiveness_deviation']:.6f}, median={result['median_invasiveness_deviation']:.6f}")

    out_path = _THIS_DIR / "cbf_filter_second_domain_aloha_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
