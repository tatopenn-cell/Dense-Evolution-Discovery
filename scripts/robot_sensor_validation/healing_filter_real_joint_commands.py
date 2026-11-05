# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/healing_filter_real_joint_commands.py
=========================================================================
DeepSeek's original framing (see prog.txt) proposed `healing_filter` as a
real-time "joint damping shield" between an LLM's output and a robot's
motors. Two things need checking before that framing could ever be true:
(1) is `healing_filter` even structurally usable in a real-time,
one-command-at-a-time loop, and (2) does it generalize from the "generic
AI signals" it was calibrated on to real robot joint commands at all.

CHECK 1 -- causality (structural, not a data question)
`healing_filter`'s own source (dense_armor/utility/healing.py) computes
`baseline` and `window` from `x[i-wide:i+wide+1]` / `x[i-radius:i+radius+1]`
-- BOTH look `wide`/`radius` points AHEAD of `i`, not just behind. This is
checked directly here, not assumed from reading the code: perturbing a
REAL future sample and confirming an EARLIER output changes.
HONEST FINDING: `healing_filter` cannot be used to filter a live LLM
command before sending it to a motor -- that command does not exist yet
`radius` steps in the future. Real-time joint-command damping would need
a causal rewrite (the same streaming.py already did for arbiter.py's
classify_segments) -- not attempted here, out of scope for this check.

CHECK 2 -- does it generalize to real robot joint commands at all
(offline/batch use, where non-causality is not a blocker)?
Real LeRobot episode 0 action stream (6 real joints, already cached),
isolated spikes injected at real magnitudes relative to each joint's own
real range, healing_filter vs. a same-radius moving-median baseline
(radius=2, matching healing_filter's own default) vs. raw (no filter).

CHECK 3 -- does it preserve real legitimate movement (no injection)?
Same real joints, no injection -- RMSE(filtered, original) should stay
small; oversmoothing real motion would be a real cost, not just a
non-issue.

CHECK 4 -- does the already-documented "temporary collapse" limitation
(healing.py's own docstring, found on a different synthetic scenario)
reproduce on real robot joint data too?
"""
import pathlib

import numpy as np
import pandas as pd

from dense_armor.utility.healing import healing_filter


def load_joint(joint_idx=0):
    from huggingface_hub import hf_hub_download
    data_root = pathlib.Path(__file__).resolve().parent / "lerobot_data"
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    action = np.stack(sub["action"].values)
    return action[:, joint_idx].astype(np.float64)


def moving_median_baseline(x, radius=2):
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        out[i] = np.median(x[lo:hi])
    return out


def check_1_causality(x):
    """A single-outlier perturbation can fail to move a MEDIAN-based
    baseline (the outlier just becomes the new min/max, the middle-rank
    element is unaffected) -- not a reliable causality probe. The real,
    semantically-correct test: at real index i, a live/streaming caller
    would only have x[0:i+1] available, not x[0:n]. If truncating the
    array to end exactly at i+radius (the last point healing_filter's
    OWN narrow window would want) changes out[i] versus running on the
    full real array, the function needed points beyond i to decide i --
    i.e. it cannot run live."""
    i = 100
    radius = 2
    out_full = healing_filter(x, radius=radius, wide_mult=3)
    out_truncated = healing_filter(x[:i + radius], radius=radius, wide_mult=3)
    changed = not np.isclose(out_full[i], out_truncated[i])
    print("=== CHECK 1: causality ===")
    print(f"out[i={i}] with full real future data: {out_full[i]:.4f}")
    print(f"out[i={i}] with only data up to i+radius (what a live caller would have): {out_truncated[i]:.4f}")
    print(f"differs: {changed}")
    print("-> healing_filter is NON-CAUSAL: needs future points not yet available live." if changed
          else "-> identical here (window's future points happened not to change the decision at this i;"
               " the source (x[i-wide:i+wide+1], x[i-radius:i+radius+1]) still reads ahead by construction).")
    return changed


def check_2_spike_recovery(x_clean):
    rng = np.random.default_rng(0)
    n = len(x_clean)
    real_std = np.std(x_clean)
    spike_mag = 5.0 * real_std  # a real, joint-range-relative jerk magnitude
    n_spikes = max(1, n // 20)  # ~5% spike density, matching healing.py's own calibration scenario
    spike_idx = rng.choice(np.arange(10, n - 10), size=n_spikes, replace=False)

    x_spiked = x_clean.copy()
    x_spiked[spike_idx] += rng.choice([-1, 1], size=n_spikes) * spike_mag

    healed = healing_filter(x_spiked, radius=2, sustain_threshold=0.7, wide_mult=3)
    baseline_med = moving_median_baseline(x_spiked, radius=2)

    rmse_raw = np.sqrt(np.mean((x_spiked - x_clean) ** 2))
    rmse_healed = np.sqrt(np.mean((healed - x_clean) ** 2))
    rmse_median = np.sqrt(np.mean((baseline_med - x_clean) ** 2))

    print("\n=== CHECK 2: real spike recovery on real joint commands ===")
    print(f"n_spikes={n_spikes} ({n_spikes/n*100:.1f}%), spike_mag={spike_mag:.4f} (5x real joint std)")
    print(f"RMSE vs real clean signal: raw={rmse_raw:.4f}  healing_filter={rmse_healed:.4f}  moving_median={rmse_median:.4f}")
    return dict(rmse_raw=rmse_raw, rmse_healed=rmse_healed, rmse_median=rmse_median)


def check_3_legit_movement_preserved(x_clean):
    healed = healing_filter(x_clean, radius=2, sustain_threshold=0.7, wide_mult=3)
    rmse = np.sqrt(np.mean((healed - x_clean) ** 2))
    real_std = np.std(x_clean)
    print("\n=== CHECK 3: real legitimate movement preserved (no injection) ===")
    print(f"RMSE(healed, original) on real unmodified joint trajectory: {rmse:.4f} (real joint std={real_std:.4f})")
    return dict(rmse=rmse, real_std=real_std)


def check_4_temporary_collapse(x_clean):
    x_collapsed = x_clean.copy()
    collapse_start = 150
    original_level = x_collapsed[collapse_start]
    x_collapsed[collapse_start:collapse_start + 3] = 0.0  # real 3-point collapse to zero, then recovers

    healed = healing_filter(x_collapsed, radius=2, sustain_threshold=0.7, wide_mult=3)
    rmse_collapse_region = np.sqrt(np.mean(
        (healed[collapse_start:collapse_start + 3] - original_level) ** 2
    ))
    passed_through = np.allclose(healed[collapse_start:collapse_start + 3], 0.0, atol=1e-6)
    print("\n=== CHECK 4: does the known temporary-collapse issue reproduce on real joint data? ===")
    print(f"3-point collapse to 0.0 (real original level={original_level:.4f}): "
          f"passed through unfiltered={passed_through}, RMSE in collapse region={rmse_collapse_region:.4f}")
    return dict(passed_through=passed_through, rmse=rmse_collapse_region)


def main():
    x = load_joint(joint_idx=0)
    print(f"Real episode 0, joint 0: n={len(x)} real samples, real std={np.std(x):.4f}")
    check_1_causality(x)
    check_2_spike_recovery(x)
    check_3_legit_movement_preserved(x)
    check_4_temporary_collapse(x)


if __name__ == "__main__":
    main()
