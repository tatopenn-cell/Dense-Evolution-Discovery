# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/validate_streaming_deviation.py
=================================================================
The real correctness bar for StreamingDeviationDetector: feeding a real
series one point at a time through it must reproduce classify_segments'
own `deviante` array (the per-point deviation flag, before the
spike/regime run-length logic) bit-for-bit -- not "looks similar".
Tested on real data already cached in this repo (LeRobot arm, Dense-
Armor's own real agent telemetry), not synthetic-only.
"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor")

from streaming_deviation import StreamingDeviationDetector

RADIUS, REF_MULT, N_SIGMAS = 5, 2, 3.0


def _batch_deviante(x: np.ndarray) -> np.ndarray:
    """Reimplements classify_segments' OWN deviante computation directly
    (arbiter.py lines 113-132) rather than importing classify_segments
    itself, so this validation checks the exact per-point logic being
    ported, not the full function's post-processing behavior."""
    n = x.size
    deviante = np.zeros(n, dtype=bool)
    span = RADIUS * REF_MULT
    for i in range(n):
        lo = max(0, i - span)
        w = x[lo:i]
        if w.size < 4:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        scala = 1.4826 * mad
        scarto = abs(x[i] - med)
        if scala < 1e-9:
            deviante[i] = scarto > 1e-9
        else:
            deviante[i] = (scarto / scala) > N_SIGMAS
    return deviante


def _streaming_deviante(x: np.ndarray) -> np.ndarray:
    det = StreamingDeviationDetector(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=N_SIGMAS)
    return np.array([det.update(float(v)) for v in x], dtype=bool)


def check(name: str, x: np.ndarray):
    batch = _batch_deviante(x)
    streaming = _streaming_deviante(x)
    exact_match = np.array_equal(batch, streaming)
    n_mismatch = int(np.sum(batch != streaming))
    print(f"{name}: n={len(x)}  exact_match={exact_match}  mismatches={n_mismatch}")
    if not exact_match:
        idx = np.where(batch != streaming)[0][:5]
        print(f"  first mismatches at indices: {idx.tolist()}")
    return exact_match


def main():
    all_ok = True

    # 1. real LeRobot arm data (joint 2 diff, Experiment 43's own signal)
    from huggingface_hub import hf_hub_download
    data_root = pathlib.Path(__file__).resolve().parent.parent / "robot_sensor_validation" / "lerobot_data"
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    for ep in [0, 5, 22, 37]:
        sub = df[df.episode_index == ep].sort_values("frame_index")
        action = np.stack(sub["action"].values)
        state = np.stack(sub["observation.state"].values)
        diff = (action - state)[:, 2]
        all_ok &= check(f"LeRobot episode {ep} (joint 2 diff)", diff)

    # 2. real Dense-Armor agent telemetry (latency_s, all 4 scenarios)
    import json
    from collections import defaultdict
    agent_path = pathlib.Path(r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor\test\agent_v2\telemetry_v2_frozen.jsonl")
    records = [json.loads(l) for l in open(agent_path, encoding="utf-8")]
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r["scenario"]].append(r)
    for scen in by_scenario:
        by_scenario[scen].sort(key=lambda r: r["step_id"])
        x = np.array([r["latency_s"] for r in by_scenario[scen]])
        all_ok &= check(f"Agent telemetry {scen}", x)

    # 3. pure synthetic noise + injected outliers, for a clean edge case
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    x[[50, 51, 150]] = [30.0, -30.0, 25.0]
    all_ok &= check("Synthetic noise + 3 injected outliers", x)

    print(f"\nALL EXACT MATCH: {all_ok}")


if __name__ == "__main__":
    main()
