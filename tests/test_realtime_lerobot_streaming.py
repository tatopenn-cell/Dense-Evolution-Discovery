"""
Real, executable test for scripts/dense_armor_streaming/realtime_lerobot_streaming.py --
runs the actual real-time check against real LeRobot data (network I/O, uses the
already-cached parquet, no new download) and asserts the real invariants that
matter: it keeps up with the real 30Hz budget and doesn't drift.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "dense_armor_streaming"))

from realtime_lerobot_streaming import load_episode_0, check_1_per_call_latency, check_2_sustained_realtime_playback
import numpy as np


def test_real_per_call_latency_never_exceeds_real_30hz_budget():
    action, ts = load_episode_0()
    real_dt = float(np.median(np.diff(ts)))
    r1 = check_1_per_call_latency(action, real_dt)
    assert r1["n_over_budget"] == 0
    assert r1["median_us"] < r1["real_budget_us"] / 10  # real, comfortable headroom


def test_sustained_realtime_playback_does_not_drift_past_budget():
    action, ts = load_episode_0()
    r2 = check_2_sustained_realtime_playback(action, ts)
    real_dt_ms = float(np.median(np.diff(ts))) * 1000
    assert r2["max_drift_ms"] < real_dt_ms
