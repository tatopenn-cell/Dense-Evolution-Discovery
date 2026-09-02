"""
scripts/cusum_detectability_theory/validate_arl_theory.py
==============================================================
Monte Carlo validation of arl_theory.py's closed-form ARL formulas
against two things:
  1. A DIRECT simulation of the idealized model (a pure random walk
     with exactly the assumed mean/variance) -- isolates "is the
     formula itself correct" from "does the real implementation match
     the idealized model".
  2. dense-armor's REAL utility.cusum.cusum_detector (reference=
     "fixed") -- isolates "does the real library code behave like the
     idealized theory predicts".

All seeds/trial counts/parameter grids declared here before running,
never adjusted after seeing results.
"""
import json
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from arl_theory import one_sided_arl, two_sided_arl  # noqa: E402

sys.path.insert(0, r"C:\Users\Admin\Desktop\Fullwork\Dense-Armor")
from dense_armor.utility.cusum import cusum_detector  # noqa: E402

K, H = 0.5, 5.0
N_TRIALS_IDEAL = 3000
N_TRIALS_REAL = 1000
SEED = 42


def _simulate_ideal_one_sided(k, h, mu, n_trials, max_steps, seed):
    """Direct simulation of S[i]=max(0,S[i-1]+(X[i]-k)), X[i]~N(mu,1)
    iid, signal when S[i] > h -- the literal process the theory
    describes, not dense-armor's windowed/median-MAD implementation."""
    rng = np.random.default_rng(seed)
    run_lengths = []
    for _ in range(n_trials):
        s = 0.0
        for t in range(1, max_steps + 1):
            x = rng.normal(mu, 1.0)
            s = max(0.0, s + (x - k))
            if s > h:
                run_lengths.append(t)
                break
    return float(np.mean(run_lengths)) if run_lengths else float("nan"), len(run_lengths)


def _real_cusum_arl(k, h, mu, n_trials, max_steps, seed, span):
    """dense_armor.utility.cusum.cusum_detector, reference='fixed'.
    Warmup segment drawn at the NULL level (mean 0) so the fixed
    reference locks onto the true in-control state, matching the
    theory's 'known baseline, change occurs after' assumption -- the
    true shift mu is applied only to samples AFTER the warmup."""
    rng = np.random.default_rng(seed)
    run_lengths = []
    for _ in range(n_trials):
        warmup = rng.normal(0.0, 1.0, span)
        after = rng.normal(mu, 1.0, max_steps)
        x = np.concatenate([warmup, after])
        flagged, _ = cusum_detector(x, radius=span, ref_mult=1, k=k, h=h, reference="fixed")
        idx = np.where(flagged)[0]
        idx = idx[idx >= span]
        if len(idx) > 0:
            run_lengths.append(int(idx[0]) - span)
    return float(np.mean(run_lengths)) if run_lengths else float("nan"), len(run_lengths)


def main():
    result = {"k": K, "h": H, "ideal_vs_theory": [], "real_cusum_vs_theory": [], "span_sensitivity": []}

    print("=== 1. Idealized simulation vs theory (one-sided, k=0.5, h=5.0) ===")
    for mu in (0.5, 1.0, 1.5):
        delta = mu - K
        theory_corr = one_sided_arl(delta, H, corrected=True)
        theory_uncorr = one_sided_arl(delta, H, corrected=False)
        emp, n = _simulate_ideal_one_sided(K, H, mu, N_TRIALS_IDEAL, max_steps=5000, seed=SEED)
        rel_err = abs(emp - theory_corr) / theory_corr * 100
        row = dict(mu=mu, delta=delta, theory_corrected=theory_corr, theory_uncorrected=theory_uncorr,
                   empirical=emp, n=n, rel_err_pct=rel_err)
        result["ideal_vs_theory"].append(row)
        print(f"  mu={mu}  theory_corr={theory_corr:8.2f}  theory_uncorr={theory_uncorr:8.2f}  "
              f"empirical={emp:8.2f}  rel_err={rel_err:5.2f}%")

    print("\n=== 2. Real dense_armor.cusum_detector (reference='fixed', span=40) vs theory (two-sided) ===")
    for mu in (0.0, 0.5, 1.0, 1.5):
        theory = two_sided_arl(mu, K, H)
        emp, n_flagged = _real_cusum_arl(K, H, mu, N_TRIALS_REAL, max_steps=int(theory * 4) + 200, seed=SEED, span=40)
        rel_err = abs(emp - theory) / theory * 100
        row = dict(mu=mu, theory=theory, empirical=emp, n_flagged=n_flagged, rel_err_pct=rel_err)
        result["real_cusum_vs_theory"].append(row)
        print(f"  mu={mu}  theory={theory:8.2f}  real={emp:8.2f}  rel_err={rel_err:5.2f}%  (flagged {n_flagged}/{N_TRIALS_REAL})")

    print("\n=== 3. False-alarm ARL vs reference-window span (mu=0.0) ===")
    theory0 = two_sided_arl(0.0, K, H)
    for span in (40, 100, 300, 1000):
        emp, n_flagged = _real_cusum_arl(K, H, 0.0, 500, max_steps=int(theory0 * 3) + 200, seed=SEED + 1, span=span)
        rel_err = abs(emp - theory0) / theory0 * 100
        row = dict(span=span, theory=theory0, empirical=emp, rel_err_pct=rel_err)
        result["span_sensitivity"].append(row)
        print(f"  span={span:5d}  theory={theory0:8.2f}  real={emp:8.2f}  rel_err={rel_err:5.2f}%")

    out_path = _THIS_DIR / "arl_theory_validation_frozen.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
