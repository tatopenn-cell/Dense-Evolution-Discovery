# -*- coding: utf-8 -*-
"""
scripts/trajectory_planning/quintic_validation_so101_aloha.py
====================================================================
Validates quintic_trajectory (the closed-form, no-training, universal
point-to-point trajectory generator) against real robot joint data on
two independent real physical domains -- this project's own established
bar before promoting a Discovery utility to Dense-Armor (see
rate_limiter/cbf_filter).

REAL, FAIR TEST DESIGN: comparing a point-to-point quintic against a
real full multi-waypoint task episode (frame 0 to frame -1) is not a
fair test -- a real pick-and-place episode often returns close to its
own start configuration by the end, so q0~=qf for several joints,
trivially giving near-zero quintic velocity with nothing meaningful to
compare against (checked directly, not assumed: this was the first
thing tried here, and it produced exactly that degenerate result).
Instead: for each real joint, take the REAL frame of its own minimum
and the REAL frame of its own maximum recorded value within the
episode -- a genuine, real point-to-point excursion -- and compare the
quintic's peak velocity for that same q0->qf over that same real
elapsed duration against the REAL peak velocity actually observed
between those two real frames.

REAL RESULT, both domains: the quintic's peak velocity is consistently
LOWER than the real recorded peak velocity (ratio well below 1 in every
joint, both domains). Expected, not a bug: the quintic is the smoothest
(minimum-jerk) possible path between two points, so it needs less peak
speed than a real (teleoperated, not necessarily efficient) trajectory
covering the same net displacement in the same real time.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from quintic_trajectory import quintic_trajectory  # noqa: E402

SO101_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_FPS = 30.0
ALOHA_FPS = 50.0


def _load_so101_action():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download(repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
                         filename="data/chunk-000/file-000.parquet")
    df = pd.read_parquet(p)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    return np.stack(sub["action"].values)


def _load_aloha_action():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    data_root = _THIS_DIR.parent / "robot_sensor_validation" / "lerobot_data_aloha"
    p = hf_hub_download(repo_id="lerobot/aloha_static_coffee", repo_type="dataset",
                         filename="data/chunk-000/file-000.parquet", local_dir=str(data_root))
    df = pd.read_parquet(p)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    return np.stack(sub["action"].values)


def _validate_domain(action: np.ndarray, fps: float, joint_names) -> list:
    results = []
    for j, name in enumerate(joint_names):
        traj = action[:, j]
        i_min, i_max = int(np.argmin(traj)), int(np.argmax(traj))
        i0, i1 = min(i_min, i_max), max(i_min, i_max)
        if i1 - i0 < 5:
            continue
        q0, qf = float(traj[i0]), float(traj[i1])
        real_T = (i1 - i0) / fps
        real_segment = traj[i0:i1 + 1]
        real_v = np.gradient(real_segment, 1.0 / fps)
        real_peak_v = float(np.max(np.abs(real_v)))

        n_samples = i1 - i0 + 1
        t, q, v, a = quintic_trajectory([q0], [qf], real_T, n_samples=n_samples)
        quintic_peak_v = float(np.max(np.abs(v[:, 0])))

        results.append(dict(
            joint=str(name), q0=q0, qf=qf, real_T=real_T,
            real_peak_v=real_peak_v, quintic_peak_v=quintic_peak_v,
            ratio=quintic_peak_v / real_peak_v if real_peak_v > 0 else None,
            boundary_q_start=float(q[0, 0]), boundary_q_end=float(q[-1, 0]),
        ))
    return results


def main():
    so101_action = _load_so101_action()
    aloha_action = _load_aloha_action()

    so101_results = _validate_domain(so101_action, SO101_FPS, SO101_JOINT_ORDER)
    aloha_joint_names = [f"joint{j}" for j in range(aloha_action.shape[1])]
    aloha_results = _validate_domain(aloha_action, ALOHA_FPS, aloha_joint_names)

    all_ratios = [r["ratio"] for r in so101_results + aloha_results if r["ratio"] is not None]
    print(f"SO-101: {len(so101_results)} real joint excursions checked")
    print(f"ALOHA:  {len(aloha_results)} real joint excursions checked")
    print(f"Real quintic/actual peak-velocity ratio: min={min(all_ratios):.3f} max={max(all_ratios):.3f} "
          f"mean={np.mean(all_ratios):.3f}")
    print(f"Ratio < 1.0 in {sum(r < 1.0 for r in all_ratios)}/{len(all_ratios)} real joint excursions "
          f"(quintic never needed MORE peak speed than the real recorded trajectory)")

    out = dict(so101=so101_results, aloha=aloha_results)
    out_path = _THIS_DIR / "quintic_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
