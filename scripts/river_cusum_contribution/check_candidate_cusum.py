# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/check_candidate_cusum.py
============================================================
The verification of Experiment 45 applied to ANY streaming CUSUM that
follows river's DriftDetector interface (update(x), drift_detected), for
example a candidate pull request. It does not copy or store the candidate:
the file is loaded from a path.

For each threshold h it prints, on purely stable N(0,1) streams and on
streams with a shift of 1 sigma at sample 500 (benchmark_harness.evaluate):
  - the false-alarm rate: stable streams of 1000 samples with at least one flag;
  - the missed-detection rate and the F1 with every spurious flag counted;
and, for the same h, the number of flags on 10 stable streams of 2000 samples
when the detector keeps running after each flag.

Usage, from this folder:

    python check_candidate_cusum.py FILE CLASS "kw=value,kw={h}" [--thresholds 5,10,20] [--trials 1000] [--seed 42]

Example (this folder's own prototype):

    python check_candidate_cusum.py cusum_river.py CUSUM "warmup_period=30,k=0.5,h={h}"
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark_harness import evaluate  # noqa: E402


def load_class(path: str, name: str):
    spec = importlib.util.spec_from_file_location("candidate_cusum", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_cusum"] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def parse_kwargs(template: str, h: float) -> dict:
    out = {}
    for item in template.split(","):
        key, value = item.split("=", 1)
        value = value.strip().replace("{h}", repr(h))
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        out[key.strip()] = value
    return out


def flags_on_stable(factory, n_streams: int, n_points: int, seed: int) -> int:
    rng = random.Random(seed)
    total = 0
    for _ in range(n_streams):
        det = factory()
        for _ in range(n_points):
            det.update(rng.gauss(0.0, 1.0))
            if det.drift_detected:
                total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("cls")
    ap.add_argument("kwargs", help='constructor arguments, "{h}" is replaced by the threshold')
    ap.add_argument("--thresholds", default="5,10,20")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cls = load_class(a.file, a.cls)
    print(f"{a.file}:{a.cls}({a.kwargs}), {a.trials} + {a.trials} streams of 1000 samples, seed {a.seed}")
    print(f"{'h':<6}{'false-alarm rate':>18}{'missed-detection rate':>24}{'F1, every flag counted':>25}{'flags, 10 x 2000 stable':>25}")
    for h in [float(x) for x in a.thresholds.split(",")]:
        def factory(h=h):
            return cls(**parse_kwargs(a.kwargs, h))

        r = evaluate(name=f"h={h}", detector_factory=factory, rng_seed=a.seed, n_trials=a.trials,
                     n_points=1000, change_at=500, mean_before=0.0, mean_after=1.0, std=1.0)
        flags = flags_on_stable(factory, 10, 2000, seed=7)
        print(f"{h:<6g}{r.false_alarm_rate:>18.3f}{r.missed_rate:>24.3f}{r.f1():>25.3f}{flags:>25d}")


if __name__ == "__main__":
    main()
