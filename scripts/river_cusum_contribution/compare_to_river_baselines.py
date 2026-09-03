# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/compare_to_river_baselines.py
==================================================================
Honest comparison: the CUSUM prototype (cusum_river.py) against river's
own ADWIN, KSWIN, and PageHinkley -- each with ITS OWN library default
parameters (no hand-tuning any baseline to make CUSUM look better), on
identical synthetic streams with a known change point, across several
shift sizes. Metrics match benchmark_harness.py: false-alarm rate on
purely stable data, missed-detection rate, detection delay, F1.

This is a synthetic, honest-but-limited validation (matching river issue
#1914's own "Month 1-2: evaluation harness + synthetic benchmarks"
step) -- not a claim to reproduce the Mozilla 174-series benchmark cited
in the issue, which would need that real dataset.
"""
import json

import river.drift

from benchmark_harness import evaluate
from cusum_river import CUSUM

SHIFTS = [0.5, 1.0, 1.5, 2.0]
N_TRIALS = 200
N_POINTS = 1000
CHANGE_AT = 500


def detector_factories():
    return {
        "CUSUM (this prototype)": lambda: CUSUM(warmup_period=30, k=0.5, h=20.0),
        "ADWIN (river default)": lambda: river.drift.ADWIN(),
        "KSWIN (river default)": lambda: river.drift.KSWIN(seed=0),
        "PageHinkley (river default)": lambda: river.drift.PageHinkley(),
    }


def main():
    results = {}
    for shift in SHIFTS:
        results[str(shift)] = {}
        for name, factory in detector_factories().items():
            res = evaluate(
                name=name, detector_factory=factory, rng_seed=123,
                n_trials=N_TRIALS, n_points=N_POINTS, change_at=CHANGE_AT,
                mean_before=0.0, mean_after=shift, std=1.0,
            )
            results[str(shift)][name] = {
                "false_alarm_rate": res.false_alarm_rate,
                "missed_rate": res.missed_rate,
                "mean_delay": res.mean_delay,
                "median_delay": res.median_delay,
                "f1": res.f1(),
            }
            print(f"shift={shift:.1f}  {name:32s}  "
                  f"false_alarm_rate={res.false_alarm_rate:.3f}  "
                  f"missed_rate={res.missed_rate:.3f}  "
                  f"mean_delay={res.mean_delay}  "
                  f"f1={res.f1():.3f}")
        print()

    with open("compare_to_river_baselines_frozen.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
