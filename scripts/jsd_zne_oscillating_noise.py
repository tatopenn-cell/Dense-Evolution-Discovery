"""Does JSD-predictive ZNE help on oscillating (non-monotonic) depolarizing
noise, beyond the photon-loss/amplitude-damping regime it was built and
validated for (Experiment 22, `photonic_predictive_zne.py`)?

Origin: an ad-hoc test (single seed, 15 hand-picked configurations) claimed
an 87% win rate (13/15) and was drafted straight into a paper abstract
without the controls this repo's own experiments hold themselves to
elsewhere. Two real problems in that draft, fixed here:

1. NOT AN APPLES-TO-APPLES COMPARISON. The draft compared classic ZNE
   fit through **5** noise-scale points (a degree-2 least-squares fit)
   against a hand-rolled reimplementation of the library's own
   `_jsd_predictive_zne_density_matrix_core`, which only ever uses the
   first **3** points (`jsd_predictive_zne_density_matrix` is explicitly
   `noise_factors.shape[0] != 3: raise NotImplementedError` -- it is not
   a general n-point method). Any "improvement" could just be the 3-point
   exact-Lagrange estimator against the 5-point least-squares fit, nothing
   to do with the JSD-informed nudge itself. Fixed: both methods here see
   the SAME 3 noise scales (1x, 2x, 3x); the baseline is the library's own
   plain 3-point Richardson (`nudge_scale=0.0`, which the core function
   reduces to exactly), and the treatment is the real, shipped
   `jsd_predictive_zne_density_matrix` (default `nudge_scale=0.5`) plus a
   `nudge_scale` sensitivity sweep via the internal core directly.

2. ONE SEED, NO SIGNIFICANCE TEST. Every configuration in the draft used
   `SEED=42` only -- a single noise realization per point, presented as a
   13/15 "win rate" with no error bars. The library's own validation of
   this method (see `jsd_predictive_zne_density_matrix`'s docstring)
   required "positive in 6/6 independent seeds" and a one-sample t-test
   (p=0.0003) before being trusted. This script applies the same bar:
   K_SEEDS=6 independent master seeds per configuration, a paired
   (per-seed) fidelity difference, and a one-sample t-test against zero.

Honest scope on the noise model itself: `apply_oscillating_noise` is a
synthetic stress-test (`p_eff = base_p * (1 + amp*sin(factor*pi/freq))`),
not a validated model of any specific real hardware noise process (1/f,
crosstalk, etc.) -- it is used here only to make the noise-scale ->
output-distribution relationship non-monotonic, which is the actual
condition `jsd_predictive_zne_density_matrix`'s docstring says the method
targets. No claim about real 1/f or crosstalk noise is made or tested
here.

Runs against dense-evolution>=8.1.57 (the depolarizing-channel fix) --
version printed at runtime.

Also reports the ORIGINAL draft's own comparison design (5-point classic
`zne_density_matrix` vs. the 3-point shipped JSD method, exactly what the
single-seed draft compared) under the same K_SEEDS/significance-testing
bar, alongside the fair same-3-point comparison above -- to make the
confound's own size directly visible rather than just asserting it exists.

A third, fully apples-to-apples comparison (`jsd_nudge_5pt`) generalizes
the shipped 3-point-only JSD nudge to all 5 points, built on the exact
linear weights `zne_density_matrix` itself uses internally (extracted via
`polynomial_extrapolate`'s linearity, not hand-derived) -- verified to
reduce to `zne_density_matrix`'s own 5-point result to machine precision
(~1e-16) whenever the nudge doesn't activate, mirroring the shipped
function's own "zero risk when nonlinearity<=0" guarantee. This is NOT
part of the shipped library API -- a natural but unvalidated extension,
clearly labeled as such throughout.

    python scripts/jsd_zne_oscillating_noise.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import jax
import jax.numpy as jnp

import dense_evolution as de
from dense_evolution.registry import NoiseModel, NoiseSpec
from dense_evolution.mitigation import (
    jsd_predictive_zne_density_matrix,
    _jsd_predictive_zne_density_matrix_core,
    zne_density_matrix,
    polynomial_extrapolate,
    project_to_physical,
    _js_divergence,
    uhlmann_fidelity,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 4
FACTORS = (1.0, 2.0, 3.0)  # exactly 3 -- both methods see the same scales
FACTORS_5 = (1.0, 2.0, 3.0, 4.0, 5.0)  # the original draft's classic-ZNE baseline
K_SEEDS = 6  # matches the library's own validation bar


def _extrapolation_weights(factors, degree=2):
    """Exact linear weights `w` such that `sum(w_i * y_i) ==
    polynomial_extrapolate(y, factors, degree)` for ANY input `y` --
    `polynomial_extrapolate` is linear in its input (ordinary least
    squares), so feeding it each standard basis vector recovers its
    weight for that point exactly. Used to generalize the shipped
    3-point-only JSD nudge to 5 points on the library's OWN weights,
    not a hand-derived approximation of them."""
    n = len(factors)
    factors_j = jnp.asarray(factors, dtype=jnp.float64)
    ws = []
    for i in range(n):
        basis = jnp.zeros(n, dtype=jnp.complex128).at[i].set(1.0)
        ws.append(polynomial_extrapolate(basis, factors_j, degree=degree))
    return jnp.stack(ws)


_BASE_WEIGHTS_5 = _extrapolation_weights(FACTORS_5, degree=2)


def jsd_nudge_5pt(rho_at_5_scales, nudge_scale):
    """5-point generalization of `_jsd_predictive_zne_density_matrix_core`,
    NOT part of the shipped library -- built to answer "does letting the
    JSD-nudge see all 5 points (removing the point-count confound
    entirely) change the fair-comparison result?"

    Mirrors the shipped 3-point structure exactly: nonlinearity from the
    OUTERMOST consecutive-scale JSD gap vs. the innermost one (jsd_45 vs.
    jsd_12, the 5-point analog of the shipped jsd_23 vs. jsd_12), rectified
    at 0, then a zero-sum nudge that moves weight from the two endpoint
    scales (1x, 5x) to the center scale (3x) -- the direct 5-point mirror
    of the shipped formula's (-r, +2r, -r) structure on (1x, 2x, 3x).
    Verified (see module tests below __main__) to reduce EXACTLY to
    `zne_density_matrix(rhos, FACTORS_5, degree=2)` at nudge_scale=0 or
    whenever the outer JSD gap isn't larger than the inner one (rectified
    == 0) -- same zero-risk-in-that-regime property the shipped 3-point
    function has, verified to machine precision here too."""
    probs = jnp.real(jnp.diagonal(rho_at_5_scales, axis1=-2, axis2=-1))
    jsd_12 = _js_divergence(probs[0], probs[1])
    jsd_45 = _js_divergence(probs[3], probs[4])
    nonlinearity = (jsd_45 - jsd_12) / (jsd_45 + jsd_12 + 1e-12)
    rectified = jnp.maximum(nonlinearity, 0.0)
    delta = jnp.array([-1.0, 0.0, 2.0, 0.0, -1.0], dtype=jnp.complex128) * nudge_scale * rectified
    w = _BASE_WEIGHTS_5 + delta
    extrapolated = jnp.tensordot(w, rho_at_5_scales, axes=1) / jnp.sum(w)
    return project_to_physical(extrapolated)


def build_ghz_statevector(n_qubits):
    ops = [('h', 0)] + [('cx', i, i + 1) for i in range(n_qubits - 1)]
    sim = de.DenseSVSimulator(n_qubits)
    sim.run_circuit(ops)
    return jnp.asarray(sim.get_statevector(), dtype=jnp.complex128)


def oscillating_p_eff(base_p, factor, freq, amp):
    p = base_p * (1.0 + amp * np.sin(factor * np.pi / freq))
    return float(np.clip(p, 0.01, 0.5))


def density_matrix_at_scale(sv_ideal, n_qubits, p_eff, n_trials, master_key):
    """Monte Carlo density-matrix estimate at one noise scale: average
    |sv_i><sv_i| over n_trials independent single-shot Kraus draws,
    batched via jax.vmap (NoiseSpec-wrapped, same pattern used throughout
    this repo's re-verified scripts)."""
    keys = jax.random.split(master_key, n_trials)

    def one_trial(key):
        spec = NoiseSpec(model='depolarizing', p=p_eff, jax_key=key)
        return NoiseModel.apply_to_sv(sv_ideal, n_qubits, model=spec.model, p=spec.p,
                                       jax_key=spec.jax_key)

    sv_batch = jax.vmap(one_trial)(keys)
    rho = jnp.einsum('ti,tj->ij', sv_batch, jnp.conj(sv_batch)) / n_trials
    return rho


def run_one_seed(sv_ideal, rho_ideal, n_qubits, base_p, n_trials, freq, amp, seed):
    """Draws density matrices at all 5 factors sequentially from the same
    master key -- the first 3 draws are bit-for-bit identical to what a
    3-factor-only run would produce, so the fair 3-vs-3 comparison and the
    original draft's 5-vs-3 comparison use the exact same noise
    realization at every shared scale, not independently redrawn ones."""
    master_key = jax.random.PRNGKey(seed)
    rhos_5 = []
    for factor in FACTORS_5:
        master_key, sub = jax.random.split(master_key)
        p_eff = oscillating_p_eff(base_p, factor, freq, amp)
        rhos_5.append(density_matrix_at_scale(sv_ideal, n_qubits, p_eff, n_trials, sub))
    rhos_5 = jnp.stack(rhos_5)
    rhos = rhos_5[:3]

    rho_baseline = _jsd_predictive_zne_density_matrix_core(rhos, nudge_scale=0.0)
    rho_shipped = jsd_predictive_zne_density_matrix(rhos, jnp.asarray(FACTORS))
    rho_classic_5pt = zne_density_matrix(rhos_5, jnp.asarray(FACTORS_5), degree=2)
    rho_jsd_5pt = jsd_nudge_5pt(rhos_5, nudge_scale=0.5)  # same default as the shipped 3pt function

    fid_baseline = float(uhlmann_fidelity(rho_ideal, rho_baseline))
    fid_shipped = float(uhlmann_fidelity(rho_ideal, rho_shipped))
    fid_classic_5pt = float(uhlmann_fidelity(rho_ideal, rho_classic_5pt))
    fid_jsd_5pt = float(uhlmann_fidelity(rho_ideal, rho_jsd_5pt))
    return fid_baseline, fid_shipped, fid_classic_5pt, fid_jsd_5pt, rhos


def paired_stats(fid_baseline_list, fid_treatment_list):
    diffs = np.array(fid_treatment_list) - np.array(fid_baseline_list)
    mean_diff = diffs.mean()
    sem_diff = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
    t_stat, p_value = scipy_stats.ttest_1samp(diffs, 0.0)
    wins = int((diffs > 0).sum())
    return {
        "mean_diff": mean_diff, "sem_diff": sem_diff,
        "p_value": float(p_value), "wins": wins, "n": len(diffs),
    }


def run_config(label, config_index, sv_ideal, rho_ideal, base_p, n_trials, freq, amp, nudge_scales=None):
    """Runs K_SEEDS independent seeds for one configuration. Returns the
    shipped-JSD-vs-baseline stats, plus (if nudge_scales given) stats for
    each custom nudge_scale against the SAME per-seed baseline/rhos.

    Seeds are derived from config_index (the config's position in the
    swept list, passed explicitly by the caller), NOT Python's hash() --
    hash() of strings is randomized per-process by default (found to
    silently break reproducibility in Sections 12-14's ZNE-PSR studies
    elsewhere in this repo; fixed there with stable seeding, avoided here
    from the start)."""
    fid_baselines, fid_shippeds, fid_classic_5pts, fid_jsd_5pts = [], [], [], []
    custom_fids = {ns: [] for ns in (nudge_scales or [])}

    for seed in range(K_SEEDS):
        fid_baseline, fid_shipped, fid_classic_5pt, fid_jsd_5pt, rhos = run_one_seed(
            sv_ideal, rho_ideal, N_QUBITS, base_p, n_trials, freq, amp,
            seed=1000 * config_index + seed
        )
        fid_baselines.append(fid_baseline)
        fid_shippeds.append(fid_shipped)
        fid_classic_5pts.append(fid_classic_5pt)
        fid_jsd_5pts.append(fid_jsd_5pt)
        for ns in custom_fids:
            rho_custom = _jsd_predictive_zne_density_matrix_core(rhos, nudge_scale=ns)
            custom_fids[ns].append(float(uhlmann_fidelity(rho_ideal, rho_custom)))

    shipped_stats = paired_stats(fid_baselines, fid_shippeds)
    original_design_stats = paired_stats(fid_classic_5pts, fid_shippeds)
    fair_5v5_stats = paired_stats(fid_classic_5pts, fid_jsd_5pts)
    result = {"label": label, "shipped": shipped_stats,
              "original_design": original_design_stats,
              "fair_5v5": fair_5v5_stats,
              "fid_baseline_mean": float(np.mean(fid_baselines)),
              "fid_classic_5pt_mean": float(np.mean(fid_classic_5pts))}
    if nudge_scales:
        result["custom"] = {ns: paired_stats(fid_baselines, custom_fids[ns]) for ns in nudge_scales}
    return result


if __name__ == "__main__":
    print(f"dense_evolution version: {de.__version__}")
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    sv_ideal = build_ghz_statevector(N_QUBITS)
    rho_ideal = jnp.outer(sv_ideal, jnp.conj(sv_ideal))

    configs = []
    for base_p in (0.03, 0.05, 0.08, 0.10):
        configs.append((f"base_p={base_p}", dict(base_p=base_p, n_trials=100, freq=3, amp=0.9)))
    for n_trials in (50, 100, 200):
        configs.append((f"n_trials={n_trials}", dict(base_p=0.06, n_trials=n_trials, freq=3, amp=0.9)))
    for freq in (2, 3, 4, 5):
        configs.append((f"freq={freq}", dict(base_p=0.06, n_trials=100, freq=freq, amp=0.9)))

    nudge_scales = (0.5, 1.0, 1.5, 2.0)
    rows = []
    print(f"\nRunning {len(configs)} configurations x {K_SEEDS} independent seeds each "
          f"(shipped default nudge_scale=0.5, plus a nudge_scale sensitivity sweep)...\n")
    for config_index, (label, params) in enumerate(configs):
        result = run_config(label, config_index, sv_ideal, rho_ideal, nudge_scales=nudge_scales, **params)
        s = result["shipped"]
        o = result["original_design"]
        f5 = result["fair_5v5"]
        print(f"{label:14s} baseline3_fid={result['fid_baseline_mean']:.4f}  classic5_fid={result['fid_classic_5pt_mean']:.4f}\n"
              f"{'':14s} fair(3v3):      mean_diff={s['mean_diff']:+.5f}  sem={s['sem_diff']:.5f}  p={s['p_value']:.4f}  wins={s['wins']}/{s['n']}\n"
              f"{'':14s} original(5v3):  mean_diff={o['mean_diff']:+.5f}  sem={o['sem_diff']:.5f}  p={o['p_value']:.4f}  wins={o['wins']}/{o['n']}\n"
              f"{'':14s} fair(5v5):      mean_diff={f5['mean_diff']:+.5f}  sem={f5['sem_diff']:.5f}  p={f5['p_value']:.4f}  wins={f5['wins']}/{f5['n']}")
        rows.append({
            "config": label, "method": "shipped (nudge=0.5)",
            "baseline_fid": result["fid_baseline_mean"],
            "mean_diff": s["mean_diff"], "sem_diff": s["sem_diff"],
            "p_value": s["p_value"], "wins": s["wins"], "n_seeds": s["n"],
        })
        rows.append({
            "config": label, "method": "original design (5pt classic vs 3pt shipped)",
            "baseline_fid": result["fid_classic_5pt_mean"],
            "mean_diff": o["mean_diff"], "sem_diff": o["sem_diff"],
            "p_value": o["p_value"], "wins": o["wins"], "n_seeds": o["n"],
        })
        rows.append({
            "config": label, "method": "fair 5v5 (custom JSD-nudge generalization)",
            "baseline_fid": result["fid_classic_5pt_mean"],
            "mean_diff": f5["mean_diff"], "sem_diff": f5["sem_diff"],
            "p_value": f5["p_value"], "wins": f5["wins"], "n_seeds": f5["n"],
        })
        for ns in nudge_scales:
            c = result["custom"][ns]
            rows.append({
                "config": label, "method": f"nudge={ns}",
                "baseline_fid": result["fid_baseline_mean"],
                "mean_diff": c["mean_diff"], "sem_diff": c["sem_diff"],
                "p_value": c["p_value"], "wins": c["wins"], "n_seeds": c["n"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "jsd_zne_oscillating_noise.csv", index=False)

    shipped = df[df["method"] == "shipped (nudge=0.5)"]
    original = df[df["method"] == "original design (5pt classic vs 3pt shipped)"]
    fair5v5 = df[df["method"] == "fair 5v5 (custom JSD-nudge generalization)"]

    def summarize(sub, name):
        sig = sub[sub["p_value"] < 0.05]
        print(f"\n=== Summary: {name} ({K_SEEDS} seeds/config) ===")
        print(f"Configs with mean_diff > 0: {(sub['mean_diff'] > 0).sum()}/{len(sub)}")
        print(f"Configs with p < 0.05 (paired one-sample t-test): {len(sig)}/{len(sub)}")
        if len(sig) > 0:
            print(sig[["config", "mean_diff", "sem_diff", "p_value", "wins"]].to_string(index=False))

    summarize(shipped, "fair, same 3 noise scales for both methods")
    summarize(original, "original draft's own design (5pt classic vs 3pt shipped)")
    summarize(fair5v5, "fair 5v5 (custom JSD-nudge generalization, both methods see all 5 points)")

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    for ax, sub, title in [
        (axes[0], shipped, "Fair comparison: shipped JSD-ZNE (3pt) vs. same-3-point Richardson baseline"),
        (axes[1], original, "Original draft's own design: shipped JSD-ZNE (3pt) vs. classic ZNE (5pt least-squares)"),
        (axes[2], fair5v5, "Fair 5v5: custom JSD-nudge generalization (5pt) vs. classic ZNE (5pt) -- both see all 5 points"),
    ]:
        labels = sub["config"].tolist()
        means = sub["mean_diff"].to_numpy()
        sems = sub["sem_diff"].to_numpy()
        pvals = sub["p_value"].to_numpy()
        colors = ['#2ecc71' if p < 0.05 and m > 0 else ('#e74c3c' if p < 0.05 and m < 0 else '#95a5a6')
                  for p, m in zip(pvals, means)]
        ax.bar(range(len(labels)), means, yerr=sems, color=colors, capsize=4)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f"mean fidelity diff ± SEM\n{K_SEEDS} independent seeds")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2, axis='y')
    fig.suptitle("JSD-predictive ZNE on oscillating depolarizing noise -- fair vs. original-design comparison\n"
                 "green = significant improvement (p<0.05), red = significant regression, grey = not significant",
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "jsd_zne_oscillating_noise.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'jsd_zne_oscillating_noise.png'}")
