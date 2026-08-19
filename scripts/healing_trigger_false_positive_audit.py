"""Experiment 27: the shipped Phi-Trigger fires on ~90% of ordinary noisy
data -- a MAD-adaptive replacement fixes it without losing recall.

Origin: while validating the Colab's "Stratonovich projection" healing
claim (Experiment 26), the production Phi-Trigger's own replacement rate
looked suspiciously high (~85-90% of ALL steps replaced, not just the
deliberately corrupted ones) regardless of which correction method was
used. This experiment isolates and fixes that -- a real, separate bug
from Experiment 26's original question.

Method: characterize dense_evolution.mitigation.healing.evaluate_phi_trigger
(the mechanism ia_utils.vector_healing.enhanced_dense_healing_hybrid uses
by default to decide "genuine motion, keep as-is" vs "noise, replace with
local median") on both false-positive rate (fires on ordinary noisy,
uncorrupted data) and true-positive rate (correctly flags genuinely
corrupted rows), across the same 4 corruption scenarios and 40 seeds used
in Experiment 26, using the exact non-cascading loop structure of the
shipped function (baseline windows always read the original
sanitized-but-uncorrected sequence, never the healed output).

Finding: the shipped trigger has an ~89.6% false-positive rate on clean
data -- confirmed as a real, previously undocumented-in-numbers instance
of a caveat already noted (but not measured) in the function's own
docstring. A NaN/Inf-aware, MAD-adaptive local-deviation trigger (inspired
by, but not derived from, a passage in dinamicsHEal.txt's source material
-- see the module docstring of ia_utils.vector_healing for why the cited
paper, arXiv:1510.05279, contains no anomaly-detection math to derive this
from) cuts the false-positive rate to ~9-13% while matching or *exceeding*
the shipped trigger's recall on every corruption type tested. Promoted to
dense-evolution as an opt-in `trigger_mode='adaptive'` parameter on
`enhanced_dense_healing_hybrid` (default stays 'phi', since
ia_utils.adversarial_vector_attack's gradient-based red-teaming
specifically targets the differentiable phi mechanism).

    python scripts/healing_trigger_false_positive_audit.py
"""
import pathlib

import numpy as np
import pandas as pd

from dense_evolution.mitigation.healing import (
    calculate_phi_ab, calculate_vettore_dinamico, evaluate_phi_trigger, GLOBAL_CONSTANTS,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

N_STEPS = 50
DIM = 128
N_SEEDS = 60


def clean_trajectory(seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, N_STEPS)
    trend = np.sin(t)[:, None] * np.ones((1, DIM))
    return rng.normal(loc=0.0, scale=0.1, size=(N_STEPS, DIM)) + trend


def corrupt(clean, scenario, seed):
    rng = np.random.default_rng(seed + 500_000)
    corrupted = np.copy(clean)
    if scenario == "single_spike":
        idx = N_STEPS // 2
        corrupted[idx] += rng.normal(loc=8.0, scale=1.0, size=DIM)
        indices = [idx]
    elif scenario == "nan_string":
        idx = N_STEPS // 2
        corrupted[idx:idx + 3] = np.nan
        indices = [idx, idx + 1, idx + 2]
    elif scenario == "scattered_outliers":
        idxs = rng.choice(np.arange(5, N_STEPS - 5), size=4, replace=False)
        for idx in idxs:
            corrupted[idx] += rng.normal(loc=6.0, scale=1.5, size=DIM)
        indices = sorted(int(i) for i in idxs)
    elif scenario == "combined":
        spike_idx = N_STEPS // 3
        nan_idx = 2 * N_STEPS // 3
        corrupted[spike_idx] += rng.normal(loc=10.0, scale=1.0, size=DIM)
        corrupted[nan_idx:nan_idx + 3] = np.nan
        indices = [spike_idx, nan_idx, nan_idx + 1, nan_idx + 2]
    else:
        raise ValueError(scenario)
    return corrupted, [i for i in indices if i >= 2]


def _sanitize(vettori):
    processed = np.copy(vettori)
    processed[np.isinf(processed)] = np.nan
    all_nan_cols = np.all(np.isnan(processed), axis=0)
    safe_for_mean = np.where(all_nan_cols, 0.0, processed)
    col_means = np.nanmean(safe_for_mean, axis=0)
    return np.where(np.isnan(processed), col_means, processed)


def phi_trigger_replaces(vettori, radius_baseline=None):
    """Non-cascading, matching the exact structure of
    enhanced_dense_healing_hybrid's loop: baseline windows always read the
    sanitized-but-uncorrected sequence, never the healed output."""
    raw = np.asarray(vettori, dtype=float)
    n, hidden_dim = raw.shape
    processed = _sanitize(raw)
    adaptive_radius = (min(20, max(3, n // 3)) if n >= 3 else 0) if radius_baseline is None else radius_baseline
    replaces = []
    for i in range(2, n):
        lo = max(0, i - adaptive_radius)
        baseline_mean = np.mean(processed[lo:i], axis=0)
        state_B = processed[i]
        ipg_raw = processed[i - 1] - processed[i - 2]
        norm_ipg = np.linalg.norm(ipg_raw)
        ipg_vector = ipg_raw / norm_ipg if norm_ipg > 1e-9 else ipg_raw
        phi_ab = float(calculate_phi_ab(baseline_mean, state_B, ipg_vector))
        e_a, e_b = float(np.linalg.norm(baseline_mean)), float(np.linalg.norm(state_B))
        v_dinamic = float(calculate_vettore_dinamico(e_a, e_b, phi_ab))
        trig, _, _ = evaluate_phi_trigger(v_dinamic)
        is_dynamic = float(trig) > GLOBAL_CONSTANTS["NON_STATIC_THRESHOLD_A"]
        replaces.append(not is_dynamic)
    return np.array(replaces)


def adaptive_trigger_replaces(vettori, radius_baseline=None, sigma_mult=3.5, floor=0.25):
    """The MAD-adaptive, NaN/Inf-aware design promoted to dense-evolution's
    trigger_mode='adaptive' -- reproduced here (not imported) so this
    script stands as an independent verification of the shipped behavior."""
    raw = np.asarray(vettori, dtype=float)
    n, hidden_dim = raw.shape
    raw_nan_or_inf = np.isnan(raw).any(axis=1) | np.isinf(raw).any(axis=1)
    processed = _sanitize(raw)
    adaptive_radius = (min(20, max(3, n // 3)) if n >= 3 else 0) if radius_baseline is None else radius_baseline

    replaces = []
    deviation_history = []
    for i in range(2, n):
        lo = max(0, i - adaptive_radius)
        baseline_mean = np.mean(processed[lo:i], axis=0)
        current_deviation = np.linalg.norm(processed[i] - baseline_mean) / np.sqrt(hidden_dim)

        recent = deviation_history[-adaptive_radius:] if adaptive_radius > 0 else []
        if len(recent) > 1:
            recent_arr = np.array(recent)
            local_median = np.median(recent_arr)
            local_spread = 1.4826 * np.median(np.abs(recent_arr - local_median))
        else:
            local_median, local_spread = 0.1, 0.05
        adaptive_threshold = max(local_median + sigma_mult * local_spread, floor)

        is_replace = (current_deviation >= adaptive_threshold) or raw_nan_or_inf[i]
        replaces.append(is_replace)
        deviation_history.append(current_deviation)
    return np.array(replaces)


SCENARIOS = ("single_spike", "nan_string", "scattered_outliers", "combined")


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            clean = clean_trajectory(seed)
            corrupted, indices = corrupt(clean, scenario, seed)
            idx_arr = np.array(indices) - 2

            phi_fp = phi_trigger_replaces(clean).mean()
            adaptive_fp = adaptive_trigger_replaces(clean).mean()
            phi_hits = phi_trigger_replaces(corrupted)[idx_arr].sum()
            adaptive_hits = adaptive_trigger_replaces(corrupted)[idx_arr].sum()

            rows.append({
                "scenario": scenario, "seed": seed,
                "phi_false_positive_rate": phi_fp, "adaptive_false_positive_rate": adaptive_fp,
                "phi_hits": int(phi_hits), "adaptive_hits": int(adaptive_hits), "n_corrupted": len(idx_arr),
            })

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "healing_trigger_false_positive_audit.csv", index=False)

    print("=== FALSE-POSITIVE RATE (clean, uncorrupted data) & TRUE-POSITIVE RATE (per corrupted index) ===\n")
    summary_rows = []
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario]
        phi_fp_mean = sub["phi_false_positive_rate"].mean()
        adaptive_fp_mean = sub["adaptive_false_positive_rate"].mean()
        phi_tp = sub["phi_hits"].sum() / sub["n_corrupted"].sum()
        adaptive_tp = sub["adaptive_hits"].sum() / sub["n_corrupted"].sum()
        print(f"[{scenario}]")
        print(f"  false-positive rate: phi={phi_fp_mean:.1%}  adaptive={adaptive_fp_mean:.1%}")
        print(f"  true-positive rate:  phi={phi_tp:.1%}  adaptive={adaptive_tp:.1%}\n")
        summary_rows.append({
            "scenario": scenario, "phi_fp": phi_fp_mean, "adaptive_fp": adaptive_fp_mean,
            "phi_tp": phi_tp, "adaptive_tp": adaptive_tp,
        })
    pd.DataFrame(summary_rows).to_csv(_DATA_DIR / "healing_trigger_false_positive_audit_summary.csv", index=False)
