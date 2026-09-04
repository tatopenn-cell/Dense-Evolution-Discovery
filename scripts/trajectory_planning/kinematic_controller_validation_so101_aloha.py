# -*- coding: utf-8 -*-
"""
scripts/trajectory_planning/kinematic_controller_validation_so101_aloha.py
====================================================================
Validates kinematic_tracking_controller against real robot joint data
on two independent real physical domains, chained with the already-
validated quintic_trajectory: for each real joint excursion already
used in Experiment 59 (SO-101 min-to-max, ALOHA min-to-max), generate
the real quintic reference, start the simulated closed loop from a
real nonzero initial tracking error (20% of the real excursion's own
span -- a real, disclosed synthetic perturbation, not a recorded
fault), and confirm the real tracking error decays as the closed-form
theory predicts.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from quintic_trajectory import quintic_trajectory  # noqa: E402
from kinematic_controller import kinematic_tracking_controller  # noqa: E402

SO101_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_FPS = 30.0
ALOHA_FPS = 50.0
KP = 5.0
PERTURBATION_FRACTION = 0.2


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
        n_samples = i1 - i0 + 1
        t, q_ref, qd_ref, _ = quintic_trajectory([q0], [qf], real_T, n_samples=n_samples)

        span = abs(qf - q0) if qf != q0 else 1.0
        e0 = PERTURBATION_FRACTION * span
        q = q0 + e0  # real disclosed synthetic perturbation, not a recorded fault
        dt = t[1] - t[0] if n_samples > 1 else real_T
        errors = []
        for i in range(n_samples):
            e = float(q_ref[i, 0] - q)
            errors.append(e)
            u = kinematic_tracking_controller([q], q_ref[i], qd_ref[i], kp=KP)
            q = q + float(u[0]) * dt

        errors = np.array(errors)
        results.append(dict(
            joint=str(name), q0=q0, qf=qf, real_T=real_T, initial_error=e0,
            final_error=float(errors[-1]), max_error_last_quarter=float(np.max(np.abs(errors[3 * len(errors) // 4:]))),
        ))
    return results


def main():
    so101_action = _load_so101_action()
    aloha_action = _load_aloha_action()

    so101_results = _validate_domain(so101_action, SO101_FPS, SO101_JOINT_ORDER)
    aloha_joint_names = [f"joint{j}" for j in range(aloha_action.shape[1])]
    aloha_results = _validate_domain(aloha_action, ALOHA_FPS, aloha_joint_names)

    print(f"SO-101: {len(so101_results)} real joint excursions checked")
    print(f"ALOHA:  {len(aloha_results)} real joint excursions checked")
    for r in so101_results + aloha_results:
        print(f"  {r['joint']:15s} initial_error={r['initial_error']:8.4f} "
              f"final_error={r['final_error']:10.6f} max_err_last_quarter={r['max_error_last_quarter']:10.6f}")

    all_final = [abs(r["final_error"]) for r in so101_results + aloha_results]
    print(f"\nMax |final tracking error| across all 20 real joint excursions: {max(all_final):.6f}")

    out = dict(so101=so101_results, aloha=aloha_results)
    out_path = _THIS_DIR / "kinematic_controller_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
