# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/validate_ewma_on_real_agent_telemetry.py
=============================================================================
Adds the batch EWMA detector (ewma_batch.py) as a fifth arm to Dense-
Armor's own frozen real-agent benchmark (test/test_benchmark_v2_agent_
runtime.py, real Qwen2 1.8B tool-use telemetry via Ollama). Parameters
(lam=0.2, L=5.0) are the same ones already empirically calibrated
against synthetic data for river (Experiment 45) -- used here AS-IS,
not retuned after looking at this real data, matching this project's
own preregistration discipline for every other benchmark.
"""
import json
import pathlib
from collections import defaultdict

import numpy as np

from ewma_batch import ewma_detector

_DENSE_ARMOR = pathlib.Path(r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor")
_DATA_PATH = _DENSE_ARMOR / "test" / "agent_v2" / "telemetry_v2_frozen.jsonl"

import sys
sys.path.insert(0, str(_DENSE_ARMOR))
from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

ARBITER_KW = dict(radius=5, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
CUSUM_KW = dict(radius=5, ref_mult=2, k=0.5, h=5.0)
EWMA_KW = dict(radius=5, ref_mult=2, lam=0.2, L=5.0)


def _load_records():
    records = []
    with open(_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    by_scenario = defaultdict(list)
    for rec in records:
        by_scenario[rec["scenario"]].append(rec)
    for scen in by_scenario:
        by_scenario[scen].sort(key=lambda r: r["step_id"])
    return by_scenario


def _ground_truth_mask(records, label):
    return np.array([r["ground_truth"] == label for r in records])


def _rate_in_mask(flags, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(flags[mask]))


def _first_detection_latency(flags, mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return float("nan")
    start = idx[0]
    for i in range(start, len(flags)):
        if flags[i]:
            return float(i - start)
    return float("nan")


def main():
    by_scenario = _load_records()
    results = {}
    for scen, records in by_scenario.items():
        x = np.array([r["latency_s"] for r in records])
        flags_da, _, _ = classify_segments(x, **ARBITER_KW)
        flags_da = flags_da != "clean"
        flags_cs, _ = cusum_detector(x, **CUSUM_KW)
        flags_ewma, _ = ewma_detector(x, **EWMA_KW)
        results[scen] = dict(x=x, records=records, da=flags_da, cs=flags_cs, ewma=flags_ewma)

    print("\n[A] Normal -- false-positive rate")
    r = results["A_normal"]
    for name, flags in (("dense_armor", r["da"]), ("cusum", r["cs"]), ("ewma", r["ewma"])):
        fp = float(np.mean(flags[10:]))
        print(f"    {name:12s}  FP={fp:.3f}")

    print("\n[B] Transient (steps 25-26 latency x8) -- detection + latency")
    r = results["B_transient"]
    mask = _ground_truth_mask(r["records"], "transient_injected")
    for name, flags in (("dense_armor", r["da"]), ("cusum", r["cs"]), ("ewma", r["ewma"])):
        det = _rate_in_mask(flags, mask)
        lat = _first_detection_latency(flags, mask)
        print(f"    {name:12s}  detect={det:.3f}  latency={lat}")

    print("\n[C] Persistent (steps 25+ latency +2.0s) -- detection + latency")
    r = results["C_persistent"]
    mask = _ground_truth_mask(r["records"], "persistent_shift")
    for name, flags in (("dense_armor", r["da"]), ("cusum", r["cs"]), ("ewma", r["ewma"])):
        det = _rate_in_mask(flags, mask)
        lat = _first_detection_latency(flags, mask)
        print(f"    {name:12s}  detect={det:.3f}  latency={lat}")

    print("\n[D] Legitimate task switch -- false-reject rate (want LOW)")
    r = results["D_legit_switch"]
    mask = _ground_truth_mask(r["records"], "legit_switch")
    for name, flags in (("dense_armor", r["da"]), ("cusum", r["cs"]), ("ewma", r["ewma"])):
        fr = _rate_in_mask(flags, mask)
        print(f"    {name:12s}  false_reject={fr:.3f}")


if __name__ == "__main__":
    main()
