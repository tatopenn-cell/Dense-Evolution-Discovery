# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/cbf_filter_full_evaluation.py
==================================================================
Rigor pass for geometric_cbf_filter.py's CBF-QP safety filter on real
LeRobot joint commands (all 6 SO-101 joints), with obstacles placed IN
each real trajectory's own path (so the constraint is genuinely tested).

TWO properties checked, both required for a real safety filter:
1. INVARIANCE: from a real, SAFE starting condition (h(x0) >= 0 --
   the CBF theory's own stated guarantee is conditional on this, not a
   claim to retroactively fix an already-unsafe start), the filtered
   trajectory's h(x) never goes negative, even though the raw real
   trajectory does enter the forbidden zone (verified as a precondition).
2. MINIMAL INVASIVENESS: measured PER-STEP on the control input u (not
   cumulative position, which can legitimately stay offset for a while
   after any real correction event in a stateful causal integrator --
   the same property causal_rate_limited_follower has) -- when the
   current real state is far from the obstacle, u_safe must equal
   u_des exactly.

A real, honest numerical finding along the way: the CBF's forward-
invariance guarantee is a CONTINUOUS-time result. A single large
discrete Euler step (real robot commands can jump substantially between
samples) can overshoot past the barrier even though the instantaneous
constraint was satisfied at the step's start -- confirmed directly
(n_substeps=1 let the real safe set be violated, min h=-0.48;
n_substeps>=5 fully restored the guarantee on the same real data).
geometric_cbf_filter.py defaults to 20 substeps per real sample.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from geometric_cbf_filter import cbf_filtered_trajectory  # noqa: E402


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
    return np.stack(sub["action"].values)


def main():
    action = load_all_joints()
    n_joints = action.shape[1]

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

            for i in range(1, len(x)):
                if abs(filtered[i - 1] - obstacle) > 3 * safe_dist:
                    u_des = x[i] - x[i - 1]
                    u_safe = filtered[i] - filtered[i - 1]
                    deviations.append(abs(u_safe - u_des))

    deviations = np.array(deviations)
    result = dict(
        n_invariance_trials=n_invariance_trials, n_invariance_ok=n_invariance_ok,
        n_invasiveness_checks=int(len(deviations)),
        n_invasiveness_nonzero=int(np.sum(deviations > 1e-6)),
        max_invasiveness_deviation=float(deviations.max()) if len(deviations) else 0.0,
        median_invasiveness_deviation=float(np.median(deviations)) if len(deviations) else 0.0,
    )
    print(f"Invariance: {n_invariance_ok}/{n_invariance_trials} real (joint, obstacle) trials never violate the real safe set")
    print(f"Minimal invasiveness: {result['n_invasiveness_nonzero']}/{result['n_invasiveness_checks']} real per-step checks nonzero, "
          f"max={result['max_invasiveness_deviation']:.6f}, median={result['median_invasiveness_deviation']:.6f}")

    out_path = _THIS_DIR / "cbf_filter_full_evaluation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
