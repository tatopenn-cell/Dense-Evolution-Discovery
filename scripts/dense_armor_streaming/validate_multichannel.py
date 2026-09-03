# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/validate_multichannel.py
==========================================================
Correctness bar: classify_segments_multichannel and
MultiChannelStreamingDeviationDetector must reproduce, column by
column, exactly what a hand-written per-channel loop (the same pattern
used in every prior experiment in this repo) already gives -- on real
multi-channel data, not synthetic-only.
"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor")

from dense_armor.utility.arbiter import classify_segments
from multichannel import classify_segments_multichannel, MultiChannelStreamingDeviationDetector
from streaming_deviation import StreamingDeviationDetector

ARBITER_KW = dict(radius=5, ref_mult=2, n_sigmas=3.0, spike_run_max=2)


def check_batch(name: str, X: np.ndarray) -> bool:
    n, c = X.shape
    etichette_mc, deviazione_mc, incertezza_mc = classify_segments_multichannel(X, classify_segments, **ARBITER_KW)

    all_ok = True
    for j in range(c):
        e, d, u = classify_segments(X[:, j], **ARBITER_KW)
        labels_match = np.array_equal(etichette_mc[:, j], e)
        dev_match = np.allclose(deviazione_mc[:, j], d)
        unc_match = np.allclose(incertezza_mc[:, j], u)
        ok = labels_match and dev_match and unc_match
        all_ok &= ok
        if not ok:
            print(f"  MISMATCH channel {j}: labels={labels_match} dev={dev_match} unc={unc_match}")
    print(f"{name} (batch, n={n}, c={c}): all_channels_match={all_ok}")
    return all_ok


def check_streaming(name: str, X: np.ndarray) -> bool:
    n, c = X.shape
    mc_det = MultiChannelStreamingDeviationDetector(n_channels=c, **{k: v for k, v in ARBITER_KW.items() if k in ("radius", "ref_mult", "n_sigmas")})
    mc_flags = np.array([mc_det.update(X[i]) for i in range(n)])

    solo_dets = [StreamingDeviationDetector(radius=ARBITER_KW["radius"], ref_mult=ARBITER_KW["ref_mult"], n_sigmas=ARBITER_KW["n_sigmas"]) for _ in range(c)]
    solo_flags = np.array([[solo_dets[j].update(float(X[i, j])) for j in range(c)] for i in range(n)])

    match = np.array_equal(mc_flags, solo_flags)
    print(f"{name} (streaming, n={n}, c={c}): all_channels_match={match}")
    return match


def main():
    from huggingface_hub import hf_hub_download
    data_root = pathlib.Path(__file__).resolve().parent.parent / "robot_sensor_validation" / "lerobot_data"
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)

    all_ok = True
    for ep in [0, 22]:
        sub = df[df.episode_index == ep].sort_values("frame_index")
        action = np.stack(sub["action"].values)  # (n, 6) -- all 6 real joints, no manual loop
        all_ok &= check_batch(f"LeRobot episode {ep} (all 6 joints)", action)
        all_ok &= check_streaming(f"LeRobot episode {ep} (all 6 joints)", action)

    print(f"\nALL EXACT MATCH: {all_ok}")


if __name__ == "__main__":
    main()
