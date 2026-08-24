"""Stratonovich-inspired healing vs. the shipped median fallback: does
replacing the correction step change anything, once tested honestly?

Origin: following Hu & Sverak's "Regularity of a stochastically perturbed
Euler-Arnold equation" (arXiv:1510.05279), a variant of
`ia_utils.vector_healing.enhanced_dense_healing_hybrid` was proposed,
swapping the median fallback for a "Stratonovich projection" (local-mean
baseline + a drift term along the recent finite-difference velocity
direction, damped by a friction coefficient nu). The single-seed original
test (seed=42, one corruption pattern) reported a large win: cosine phase
alignment -0.16 -> +0.98. That single anecdote is not evidence -- this
script is the actual controlled test.

IMPORTANT CAVEAT ON THE PHYSICS CLAIM: nothing here computes an actual
Lie-group metric tensor or a rigorous Stratonovich stochastic integral --
the "geodesic projection" is a heuristic (local mean + finite-difference
velocity direction, linearly damped). The paper is real and its formalism
is real; this script does not implement it, it implements a hand-rolled
approximation of it. Treat the arXiv citation as inspiration
for the functional form, not as a proof this is "correct".

Method (controlled ablation, not a rewrite of the whole algorithm): the
real `enhanced_dense_healing_hybrid` couples two independent decisions --
(1) a Phi-Trigger (`dense_evolution.mitigation.healing.evaluate_phi_trigger`)
deciding step-by-step whether a point looks "dynamic" (keep raw) or
"static" (replace), and (2) *what* to replace it with (median of the local
window, in the shipped code). This script holds (1) fixed -- the exact
production Phi-Trigger, unmodified -- and swaps only (2): median vs.
Stratonovich-style projection. That isolates the actual claim ("the
replacement is better", not "the trigger is better") instead of
conflating the two, which the single-seed original test did not do cleanly.

4 corruption scenarios (single spike, NaN string, scattered multi-spike,
combined spike+NaN) x 40 seeds each = 160 trials, scored against the
known-clean ideal trajectory each corruption was applied to (trend + IID
Gaussian noise, matching the original synthetic setup) on two metrics:
L2 reconstruction error and flattened cosine phase alignment. A paired
Wilcoxon signed-rank test on both reports whether any observed edge
survives multi-seed testing, not just seed=42.

    python scripts/stratonovich_vector_healing.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from dense_evolution.mitigation.healing import (
    calculate_phi_ab,
    calculate_vettore_dinamico,
    evaluate_phi_trigger,
    GLOBAL_CONSTANTS,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_STEPS = 50
DIM = 128
N_SEEDS = 40
NU = 0.05  # friction coefficient, same value the original proposal used


def _clean_trajectory(seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, N_STEPS)
    trend = np.sin(t)[:, None] * np.ones((1, DIM))
    return rng.normal(loc=0.0, scale=0.1, size=(N_STEPS, DIM)) + trend


def _corrupt(clean, scenario, seed):
    """Returns (corrupted, corrupt_indices) -- the latter is the ground-truth
    set of indices that were actually tampered with, used by the forced
    (trigger-bypassed) test below."""
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


def _median_correction(window, prev1, prev2, state_A, nu):
    return np.median(window, axis=0)


def _stratonovich_correction(window, prev1, prev2, state_A, nu):
    """Local-mean baseline + damped drift along the recent finite-difference
    velocity direction -- a hand-rolled approximation of a
    Stratonovich-style projection. Not a rigorous SDE solve; see module
    docstring."""
    norm_A = np.linalg.norm(state_A)
    ipg_raw = prev1 - prev2
    norm_ipg = np.linalg.norm(ipg_raw)
    ipg_vector = ipg_raw / norm_ipg if norm_ipg > 1e-9 else ipg_raw
    q_force = ipg_vector * norm_A
    drift_step = q_force - nu * state_A
    return state_A + drift_step * 0.5


def online_healing(vettori, correction_fn, radius_baseline=None, nu=NU):
    """Reimplements `enhanced_dense_healing_hybrid`'s loop with the exact
    production Phi-Trigger, but a pluggable correction step -- see module
    docstring for why this (not calling the shipped function directly) is
    the correct ablation for testing the original claim."""
    vettori = np.asarray(vettori, dtype=float)
    n, hidden_dim = vettori.shape
    processed = np.copy(vettori)
    processed[np.isinf(processed)] = np.nan
    all_nan_cols = np.all(np.isnan(processed), axis=0)
    safe_for_mean = np.where(all_nan_cols, 0.0, processed)
    col_means = np.nanmean(safe_for_mean, axis=0)
    processed = np.where(np.isnan(processed), col_means, processed)
    out = np.copy(processed)

    adaptive_radius = (min(20, max(3, n // 3)) if n >= 3 else 0) if radius_baseline is None else radius_baseline
    replaced = np.zeros(n, dtype=bool)

    for i in range(2, n):
        lo = max(0, i - adaptive_radius)
        window = processed[lo:i]
        state_A = np.mean(window, axis=0)
        state_B = processed[i]
        ipg_raw = processed[i - 1] - processed[i - 2]
        norm_ipg = np.linalg.norm(ipg_raw)
        ipg_vector = ipg_raw / norm_ipg if norm_ipg > 1e-9 else ipg_raw

        phi_ab = float(calculate_phi_ab(state_A, state_B, ipg_vector))
        e_a, e_b = float(np.linalg.norm(state_A)), float(np.linalg.norm(state_B))
        v_dinamic = float(calculate_vettore_dinamico(e_a, e_b, phi_ab))
        trigger, _, _ = evaluate_phi_trigger(v_dinamic)
        fires = float(trigger) > GLOBAL_CONSTANTS["NON_STATIC_THRESHOLD_A"]

        if fires:
            healed = processed[i]
        else:
            healed = correction_fn(window, processed[i - 1], processed[i - 2], state_A, nu)
            replaced[i] = True

        out[i] = healed
        processed[i] = healed
    return out, replaced


def l2_error(healed, ideal):
    return float(np.linalg.norm(healed - ideal))


def cosine_alignment(healed, ideal):
    h, d = healed.ravel(), ideal.ravel()
    denom = np.linalg.norm(h) * np.linalg.norm(d)
    return float(np.real(np.vdot(h, d)) / denom) if denom > 1e-12 else 0.0


SCENARIOS = ("single_spike", "nan_string", "scattered_outliers", "combined")


def forced_healing(vettori, correction_fn, corrupt_indices, radius_baseline=None, nu=NU):
    """Same sequential-window machinery as online_healing, but correction
    fires exactly at the known-corrupted indices, bypassing the Phi-Trigger
    entirely -- this is what the original controlled test
    (healing_mediana_only / healing_stratonovich_only) actually did."""
    vettori = np.asarray(vettori, dtype=float)
    n, hidden_dim = vettori.shape
    processed = np.copy(vettori)
    processed[np.isinf(processed)] = np.nan
    all_nan_cols = np.all(np.isnan(processed), axis=0)
    safe_for_mean = np.where(all_nan_cols, 0.0, processed)
    col_means = np.nanmean(safe_for_mean, axis=0)
    processed = np.where(np.isnan(processed), col_means, processed)
    out = np.copy(processed)

    adaptive_radius = (min(20, max(3, n // 3)) if n >= 3 else 0) if radius_baseline is None else radius_baseline
    corrupt_set = set(corrupt_indices)

    for i in range(2, n):
        if i in corrupt_set:
            lo = max(0, i - adaptive_radius)
            window = processed[lo:i]
            state_A = np.mean(window, axis=0)
            healed = correction_fn(window, processed[i - 1], processed[i - 2], state_A, nu)
        else:
            healed = processed[i]
        out[i] = healed
        processed[i] = healed
    return out


def run_all():
    rows = []
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            clean = _clean_trajectory(seed)
            corrupted, corrupt_indices = _corrupt(clean, scenario, seed)

            healed_median, replaced_median = online_healing(corrupted, _median_correction)
            healed_strato, replaced_strato = online_healing(corrupted, _stratonovich_correction)

            forced_median = forced_healing(corrupted, _median_correction, corrupt_indices)
            forced_strato = forced_healing(corrupted, _stratonovich_correction, corrupt_indices)
            idx_arr = np.array(corrupt_indices, dtype=int)

            rows.append({
                "scenario": scenario, "seed": seed,
                "n_replaced_median": int(replaced_median.sum()),
                "n_replaced_strato": int(replaced_strato.sum()),
                "l2_median": l2_error(healed_median, clean),
                "l2_strato": l2_error(healed_strato, clean),
                "cos_median": cosine_alignment(healed_median, clean),
                "cos_strato": cosine_alignment(healed_strato, clean),
                "any_nan_median": bool(np.isnan(healed_median).any()),
                "any_nan_strato": bool(np.isnan(healed_strato).any()),
                # forced test: restricted to exactly the corrupted rows,
                # bypassing the Phi-Trigger -- matches the original setup
                "l2_forced_median": l2_error(forced_median[idx_arr], clean[idx_arr]),
                "l2_forced_strato": l2_error(forced_strato[idx_arr], clean[idx_arr]),
                "cos_forced_median": cosine_alignment(forced_median[idx_arr], clean[idx_arr]),
                "cos_forced_strato": cosine_alignment(forced_strato[idx_arr], clean[idx_arr]),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    df = run_all()
    df.to_csv(_DATA_DIR / "stratonovich_vector_healing.csv", index=False)

    assert not df["any_nan_median"].any(), "median-correction path leaked NaN into output"
    assert not df["any_nan_strato"].any(), "stratonovich-correction path leaked NaN into output"

    print("=== Stratonovich projection vs. shipped median fallback ===")
    print("(same production Phi-Trigger for both -- only the replacement value differs)\n")

    summary_rows = []
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario]
        l2_diff = sub["l2_strato"] - sub["l2_median"]  # negative = strato better
        cos_diff = sub["cos_strato"] - sub["cos_median"]  # positive = strato better

        l2_win_rate = float((l2_diff < 0).mean())
        cos_win_rate = float((cos_diff > 0).mean())
        l2_w, l2_p = stats.wilcoxon(sub["l2_strato"], sub["l2_median"])
        cos_w, cos_p = stats.wilcoxon(sub["cos_strato"], sub["cos_median"])

        f_l2_win_rate = float((sub["l2_forced_strato"] < sub["l2_forced_median"]).mean())
        f_cos_win_rate = float((sub["cos_forced_strato"] > sub["cos_forced_median"]).mean())
        f_l2_w, f_l2_p = stats.wilcoxon(sub["l2_forced_strato"], sub["l2_forced_median"])
        f_cos_w, f_cos_p = stats.wilcoxon(sub["cos_forced_strato"], sub["cos_forced_median"])

        print(f"[{scenario}] n={len(sub)} points replaced/traj (real trigger): "
              f"median={sub['n_replaced_median'].mean():.1f} strato={sub['n_replaced_strato'].mean():.1f}")
        print(f"  [real trigger]   L2 error : median={sub['l2_median'].mean():.4f} strato={sub['l2_strato'].mean():.4f} "
              f"strato-win-rate={l2_win_rate:.0%} wilcoxon p={l2_p:.4f}")
        print(f"  [real trigger]   cos align: median={sub['cos_median'].mean():.4f} strato={sub['cos_strato'].mean():.4f} "
              f"strato-win-rate={cos_win_rate:.0%} wilcoxon p={cos_p:.4f}")
        print(f"  [forced, on-target only] L2 error : median={sub['l2_forced_median'].mean():.4f} "
              f"strato={sub['l2_forced_strato'].mean():.4f} strato-win-rate={f_l2_win_rate:.0%} wilcoxon p={f_l2_p:.4f}")
        print(f"  [forced, on-target only] cos align: median={sub['cos_forced_median'].mean():.4f} "
              f"strato={sub['cos_forced_strato'].mean():.4f} strato-win-rate={f_cos_win_rate:.0%} wilcoxon p={f_cos_p:.4f}\n")

        summary_rows.append({
            "scenario": scenario,
            "l2_median_mean": sub["l2_median"].mean(), "l2_strato_mean": sub["l2_strato"].mean(),
            "l2_strato_win_rate": l2_win_rate, "l2_wilcoxon_p": l2_p,
            "cos_median_mean": sub["cos_median"].mean(), "cos_strato_mean": sub["cos_strato"].mean(),
            "cos_strato_win_rate": cos_win_rate, "cos_wilcoxon_p": cos_p,
            "l2_forced_median_mean": sub["l2_forced_median"].mean(), "l2_forced_strato_mean": sub["l2_forced_strato"].mean(),
            "l2_forced_strato_win_rate": f_l2_win_rate, "l2_forced_wilcoxon_p": f_l2_p,
            "cos_forced_median_mean": sub["cos_forced_median"].mean(), "cos_forced_strato_mean": sub["cos_forced_strato"].mean(),
            "cos_forced_strato_win_rate": f_cos_win_rate, "cos_forced_wilcoxon_p": f_cos_p,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(_DATA_DIR / "stratonovich_vector_healing_summary.csv", index=False)
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.35

    axes[0].bar(x - width / 2, summary["l2_median_mean"], width, label="median (shipped)", color="#888888")
    axes[0].bar(x + width / 2, summary["l2_strato_mean"], width, label="Stratonovich (candidate)", color="#00e5ff")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SCENARIOS, rotation=20, ha="right")
    axes[0].set_ylabel("mean L2 error vs. clean trajectory (lower = better)")
    axes[0].set_title("Reconstruction error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(x - width / 2, summary["cos_median_mean"], width, label="median (shipped)", color="#888888")
    axes[1].bar(x + width / 2, summary["cos_strato_mean"], width, label="Stratonovich (candidate)", color="#00e5ff")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SCENARIOS, rotation=20, ha="right")
    axes[1].set_ylabel("mean cosine phase alignment vs. clean trajectory (higher = better)")
    axes[1].set_title("Phase alignment")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Stratonovich-projection correction vs. median correction ({N_SEEDS} seeds/scenario, same Phi-Trigger)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "stratonovich_vector_healing.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'stratonovich_vector_healing.png'}")
