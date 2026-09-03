# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/validate_cross_channel_correlation.py
=========================================================================
Validates cross_channel_correlation_detector on real SO-101 arm data
(the same lerobot/svla_so101_pickplace dataset already cached from
Experiment 43, no new download) via a controlled fault injection --
same methodology as this project's other real-data benchmarks (inject
a known event into real background, check false-alarm rate on
unmodified real data AND detection on the injected fault).

FAULT MODEL: a "stuck joint" -- one real joint's trajectory frozen at
its own value from the start of the injection window, while the other
5 joints continue their REAL recorded motion. This is a real, physically
meaningful fault class (a jammed servo, a disconnected encoder reporting
stale values) -- not an arbitrary corruption.

Parameters (radius=10, ref_mult=3, drop_threshold=0.4) declared here
BEFORE looking at results, matching this project's own preregistration
discipline for every other benchmark in this repo.
"""
import sys
import pathlib

import numpy as np
import pandas as pd

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from cross_channel_correlation import cross_channel_correlation_detector

_DATA_ROOT = _THIS_DIR / "lerobot_data"
_REPO_ID = "lerobot/svla_so101_pickplace"

RADIUS, REF_MULT, DROP_THRESHOLD = 10, 3, 0.4
INJECT_AT, INJECT_LEN = 100, 15  # a 15-frame stuck-joint window, mid-episode
STUCK_JOINT = 2  # arbitrary, not the joint found special in Experiment 43


def _ensure_dataset():
    from huggingface_hub import hf_hub_download
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=_REPO_ID, repo_type="dataset",
        filename="data/chunk-000/file-000.parquet",
        local_dir=str(_DATA_ROOT),
    )


def main():
    parquet_path = _ensure_dataset()
    df = pd.read_parquet(parquet_path)

    # --- false-alarm rate on real, unmodified episodes ---
    n_episodes_tested = 0
    n_flag_points_total = 0
    n_points_total = 0
    for ep in sorted(df.episode_index.unique()):
        sub = df[df.episode_index == ep].sort_values("frame_index")
        action = np.stack(sub["action"].values)  # (n_frames, 6)
        if action.shape[0] < RADIUS * REF_MULT + 10:
            continue
        n_episodes_tested += 1
        flags = cross_channel_correlation_detector(action, radius=RADIUS, ref_mult=REF_MULT, drop_threshold=DROP_THRESHOLD)
        n_flag_points_total += int(flags.any(axis=1).sum())
        n_points_total += action.shape[0]

    print(f"False-alarm check: {n_episodes_tested} real episodes, {n_points_total} total frames")
    print(f"  points with >=1 channel flagged: {n_flag_points_total} ({100*n_flag_points_total/n_points_total:.2f}%)")

    # --- fault injection: stuck joint in a real episode ---
    ep0 = df[df.episode_index == 0].sort_values("frame_index")
    action0 = np.stack(ep0["action"].values).copy()
    injected = action0.copy()
    stuck_value = injected[INJECT_AT, STUCK_JOINT]
    injected[INJECT_AT:INJECT_AT + INJECT_LEN, STUCK_JOINT] = stuck_value

    flags_clean = cross_channel_correlation_detector(action0, radius=RADIUS, ref_mult=REF_MULT, drop_threshold=DROP_THRESHOLD)
    flags_injected = cross_channel_correlation_detector(injected, radius=RADIUS, ref_mult=REF_MULT, drop_threshold=DROP_THRESHOLD)

    print(f"\nFault injection: episode 0, joint {STUCK_JOINT} stuck at frames [{INJECT_AT}:{INJECT_AT+INJECT_LEN}]")
    print(f"  clean episode, this window: any flag on joint {STUCK_JOINT}? {flags_clean[INJECT_AT:INJECT_AT+INJECT_LEN, STUCK_JOINT].any()}")
    print(f"  injected episode, this window: any flag on joint {STUCK_JOINT}? {flags_injected[INJECT_AT:INJECT_AT+INJECT_LEN, STUCK_JOINT].any()}")
    window_flags = flags_injected[INJECT_AT:INJECT_AT+INJECT_LEN, STUCK_JOINT]
    print(f"  detection rate within injected window: {window_flags.mean():.3f} ({window_flags.sum()}/{INJECT_LEN} frames)")
    first_flag = np.where(window_flags)[0]
    print(f"  first flag at offset: {first_flag[0] if len(first_flag) else 'never'}")


if __name__ == "__main__":
    main()
