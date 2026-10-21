# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/arl_check.py
================================================
Check 5 of the verification protocol of Experiment 45: the average run
length (ARL) of the CUSUM under NO change, measured and compared with the
closed-form theory, for the textbook threshold h=5 and the calibrated h=20.

Three numbers are put side by side for the same k=0.5:
  - theory: scripts/cusum_detectability_theory/arl_theory.two_sided_arl
    (Reynolds 1975 with Siegmund's boundary correction), true mean and
    sigma known exactly;
  - simulation with the true mean and sigma known (a pure random walk);
  - simulation of the prototype cusum_river.CUSUM, whose baseline is
    estimated on the first `warmup_period` points.

The last table runs the evaluation harness of the experiment
(benchmark_harness.evaluate) on 1000 + 1000 streams, which is the number
of trials the false-alarm rates of the page rest on.

Run from this folder:  python arl_check.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cusum_detectability_theory"))

from arl_theory import one_sided_arl, two_sided_arl  # noqa: E402
from benchmark_harness import evaluate  # noqa: E402
from cusum_river import CUSUM  # noqa: E402

K = 0.5
HORIZON = 1000


def simulate_known(h: float, n_trials: int, cap: int, seed: int) -> np.ndarray:
    """First-alarm index of the two-sided CUSUM with the true mean 0 and
    sigma 1 known (z = x). `cap` means 'no alarm within cap samples'."""
    rng = np.random.default_rng(seed)
    sp = np.zeros(n_trials)
    sn = np.zeros(n_trials)
    first = np.full(n_trials, cap, dtype=np.int64)
    alive = np.ones(n_trials, dtype=bool)
    for t in range(1, cap + 1):
        z = rng.standard_normal(n_trials)
        sp = np.maximum(0.0, sp + z - K)
        sn = np.minimum(0.0, sn + z + K)
        hit = alive & ((sp > h) | (sn < -h))
        first[hit] = t
        alive &= ~hit
        if not alive.any():
            break
    return first


def simulate_prototype(h: float, n_trials: int, cap: int, seed: int, warmup: int = 30) -> np.ndarray:
    """First-alarm index of cusum_river.CUSUM (baseline estimated on the
    first `warmup` points) on a stable N(0,1) stream."""
    out = np.full(n_trials, cap, dtype=np.int64)
    for i in range(n_trials):
        rng = random.Random(seed + i)
        det = CUSUM(warmup_period=warmup, k=K, h=h)
        for t in range(1, cap + 1):
            det.update(rng.gauss(0.0, 1.0))
            if det.drift_detected:
                out[i] = t
                break
    return out


def summarise(first: np.ndarray) -> dict:
    return {
        "alarm_within_horizon": float(np.mean(first <= HORIZON)),
        "median_first_alarm": float(np.median(first)),
        "mean_first_alarm_capped": float(np.mean(first)),
    }


def main() -> None:
    print("Two-sided CUSUM on a stable N(0,1) stream, k=0.5")
    print("Theory: two_sided_arl(mu, k, h), true mean and sigma known\n")
    print(f"{'h':<6}{'ARL, no change':>16}{'ARL, shift 0.5 sigma':>22}{'P(alarm within 1000), theory':>30}")
    for h in (5.0, 10.0, 20.0):
        a0 = two_sided_arl(0.0, K, h)
        a05 = two_sided_arl(0.5, K, h)
        p1000 = 1.0 - np.exp(-HORIZON / a0)
        print(f"{h:<6}{a0:>16.1f}{a05:>22.1f}{p1000:>30.3f}")
    z = one_sided_arl(0.0, 5.0)
    print(f"\nOne branch with zero drift (the shift of 0.5 sigma), h=5: {z:.2f} = (h + 1.166)^2; "
          f"two branches with zero drift: {z / 2.0:.2f}")

    cap = 6000
    print(f"\nSimulation, first alarm on a stable stream (cap {cap} samples)")
    rows = [
        ("h=5, mean and sigma known", summarise(simulate_known(5.0, 4000, cap, seed=1))),
        ("h=5, baseline estimated on 30 points", summarise(simulate_prototype(5.0, 400, cap, seed=1000))),
    ]
    print(f"{'case':<40}{'P(alarm<=1000)':>16}{'median first alarm':>20}{'mean (capped)':>15}")
    for name, s in rows:
        print(f"{name:<40}{s['alarm_within_horizon']:>16.3f}{s['median_first_alarm']:>20.1f}"
              f"{s['mean_first_alarm_capped']:>15.1f}")

    print("\nThe evaluation harness (benchmark_harness.evaluate): 1000 stable streams of 1000 samples and")
    print("1000 streams with a shift of 1 sigma at sample 500. False-alarm rate = stable streams with at least one flag.")
    print(f"{'h':<6}{'seed':<6}{'false-alarm rate':>18}{'missed-detection rate':>24}{'F1, every flag counted':>25}")
    for h in (5.0, 10.0, 20.0):
        for seed in (42, 43):
            r = evaluate(name=f"h={h}", detector_factory=lambda h=h: CUSUM(warmup_period=30, k=K, h=h),
                         rng_seed=seed, n_trials=1000, n_points=1000, change_at=500,
                         mean_before=0.0, mean_after=1.0, std=1.0)
            print(f"{h:<6}{seed:<6}{r.false_alarm_rate:>18.3f}{r.missed_rate:>24.3f}{r.f1():>25.3f}")


if __name__ == "__main__":
    main()
