# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/realtime_lerobot_streaming.py
================================================================
First real-time (not batch-replay) check of MultiChannelStreamingDeviationDetector
against real LeRobot data. Prior validation (validate_multichannel.py) confirmed
CORRECTNESS (bit-exact match against a hand-written per-channel loop) and cited an
isolated ~18.6kHz throughput figure from streaming.py's own docstring benchmark --
neither of those actually measured this detector's real per-call latency against
this real robot's real recorded frame rate, or checked it doesn't fall behind over
a sustained run. This experiment does both, on the same real dataset/episode
already used to validate correctness.

DATA: lerobot/svla_so101_pickplace, episode 0, real `action` column (6 real
joints), already cached from prior experiments -- no new download. The dataset's
own `timestamp` column gives the REAL recorded frame rate: 30.0 Hz (dt=33.33ms),
verified directly, not assumed.
"""
import pathlib
import time

import numpy as np
import pandas as pd

from dense_armor.utility.streaming import MultiChannelStreamingDeviationDetector

DET_KW = dict(radius=5, ref_mult=2, n_sigmas=3.0)  # matches validate_multichannel.py


def load_episode_0():
    from huggingface_hub import hf_hub_download
    data_root = pathlib.Path(__file__).resolve().parent.parent / "robot_sensor_validation" / "lerobot_data"
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    action = np.stack(sub["action"].values)  # (n, 6)
    ts = sub["timestamp"].to_numpy()
    return action, ts


def check_1_per_call_latency(action, real_dt):
    n, c = action.shape
    det = MultiChannelStreamingDeviationDetector(n_channels=c, **DET_KW)
    latencies_s = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        det.update(action[i])
        latencies_s[i] = time.perf_counter() - t0

    median_us = np.median(latencies_s) * 1e6
    std_us = np.std(latencies_s) * 1e6
    max_us = np.max(latencies_s) * 1e6
    real_budget_us = real_dt * 1e6
    n_over_budget = int(np.sum(latencies_s > real_dt))

    print("=== CHECK 1: real per-call latency vs. real 30Hz frame budget ===")
    print(f"n={n} real frames, real budget per frame = {real_budget_us:.0f}us (30Hz)")
    print(f"latency: median={median_us:.1f}us  std={std_us:.1f}us  max={max_us:.1f}us")
    print(f"headroom (median): {real_budget_us / median_us:.0f}x")
    print(f"frames where update() alone exceeded the real 33.3ms budget: {n_over_budget}/{n}")
    return dict(n=int(n), median_us=float(median_us), std_us=float(std_us), max_us=float(max_us),
                real_budget_us=float(real_budget_us), n_over_budget=int(n_over_budget))


def check_2_sustained_realtime_playback(action, ts):
    """Simulates genuinely consuming the stream at its real recorded rate (sleep
    to the next real timestamp before each call, exactly like a live subscriber
    would) and checks whether processing latency ever causes real, measured
    cumulative drift over a sustained run -- not just a single-call number."""
    n, c = action.shape
    det = MultiChannelStreamingDeviationDetector(n_channels=c, **DET_KW)

    t_start = time.perf_counter()
    max_drift_s = 0.0
    for i in range(n):
        target = t_start + ts[i]
        now = time.perf_counter()
        if target > now:
            time.sleep(float(target - now))
        det.update(action[i])
        drift = time.perf_counter() - target
        max_drift_s = max(max_drift_s, drift)
    total_wall_s = time.perf_counter() - t_start
    real_duration_s = ts[-1] - ts[0]

    print("\n=== CHECK 2: sustained real-time playback, real drift ===")
    print(f"real recorded duration: {real_duration_s:.2f}s, wall-clock consumed: {total_wall_s:.2f}s")
    print(f"max single-frame drift (processing pushing behind the real target time): {max_drift_s * 1e3:.2f}ms")
    return dict(real_duration_s=float(real_duration_s), wall_s=float(total_wall_s), max_drift_ms=float(max_drift_s * 1e3))


def main():
    import json
    action, ts = load_episode_0()
    real_dt = float(np.median(np.diff(ts)))
    print(f"Real episode 0: {action.shape[0]} frames, {action.shape[1]} joints, real dt={real_dt*1000:.2f}ms")
    r1 = check_1_per_call_latency(action, real_dt)
    r2 = check_2_sustained_realtime_playback(action, ts)

    out_path = pathlib.Path(__file__).resolve().parent / "realtime_lerobot_streaming_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(n_frames=int(action.shape[0]), n_joints=int(action.shape[1]), real_dt_ms=float(real_dt * 1000),
                        check1=r1, check2=r2), f, indent=2)
    print("Wrote " + str(out_path))
    return r1, r2


if __name__ == "__main__":
    main()
