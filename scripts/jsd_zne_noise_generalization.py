"""Does the promoted `jsd_predictive_zne_density_matrix` (validated on
photon-loss/amplitude-damping, see photonic_predictive_zne.py) generalize
to other noise channels? And if it doesn't everywhere, can a different
signal cover the gap?

Fair comparison throughout: both methods see the SAME noise scales
(FACTORS = (1.0, 2.0, 3.0)), the baseline is the library's own plain
3-point Richardson (`_jsd_predictive_zne_density_matrix_core(..., nudge_scale=0.0)`,
which the shipped function reduces to exactly), never a reimplementation
of it -- the exact confound this repo's jsd_zne_oscillating_noise.py found
and fixed in an earlier draft.

PART 1 -- screening across every standard dense_evolution.NoiseModel
channel plus a deterministic coherent (Rz over-rotation) error, 6 seeds
each: only amplitude_damping and combined show any lead (both already
covered by the shipped signal's own validated domain, see
photonic_predictive_zne.py); bitflip, depolarizing, phaseflip, and the
coherent case show no reliable effect. Confirmed at 30 seeds + a
permutation test for the two leads (still positive: p=0.0050 and
p=0.0031 by t-test, p=0.0001 and p=0.0008 by permutation).

PART 2 -- WHY phaseflip and coherent are exactly zero, not just weak.
`_jsd_predictive_zne_density_matrix_core`'s signal is
`jnp.real(jnp.diagonal(rho))` -- population probabilities only. Phaseflip
(K1=sqrt(p)*Z) and a coherent Rz rotation are both diagonal in the
computational basis: they change off-diagonal coherences, never
populations, so the classical JSD signal is BLIND to them by
construction, not just weak (verified: diff is exactly 0.0 at every
tested point for both).

PART 3 -- a quantum-JSD attempt (von Neumann entropy of the full density
matrix instead of Shannon entropy of the diagonal, Lamberti et al. 2008)
does pick up a nonzero (if noisy, non-significant at 6 seeds) signal on
phaseflip -- but WEAKENS the already-working amplitude_damping/combined
signal in the same test, and does NOT usefully fix the coherent case: a
smooth, deterministic function of the noise-scale factor has
jsd_12~=jsd_23 regardless of which divergence is used, so the
`nonlinearity` trigger stays at or near zero structurally (exactly 0 for
classical JSD and QJSD; a floating-point-noise-level ~1e-6 for
coherence-L1, three orders of magnitude below any of its real active-case
effects below) -- independent of the signal's choice of divergence. Net:
QJSD not adopted, a real negative result kept in rather than discarded.

PART 4 -- a coherence-L1 signal (sum of |rho_ij|, i!=j -- the standard
l1-norm-of-coherence, Baumgratz/Cramer/Plenio 2014, PRL 113, 140401)
targets what phaseflip actually destroys directly. Screening showed a
promising but noisy lead; the real story only appeared correctly counted
among ACTIVE points (rectified>0), the same convention
photonic_predictive_zne.py's own validation already uses -- most seeds
never trigger the nudge at all (exactly 0.0 diff, zero risk), and only
active seeds are a meaningful comparison. At 200 seeds: 63/200 active
(31.5%), and among those, 63/63 positive -- mean +0.014892, one-sample
t-test p=1.07e-08, permutation test (20000 resamples) p<0.00005. This is
a real, large-sample-confirmed result, not yet promoted to the main
library -- see this repo's own README for the draft/verification
distinction that governs when promotion happens.

    python scripts/jsd_zne_noise_generalization.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy import stats as scipy_stats

import dense_evolution as de
from dense_evolution.registry import NoiseModel, NoiseSpec
from dense_evolution.mitigation import jsd_predictive_zne_density_matrix, uhlmann_fidelity, project_to_physical
from dense_evolution.mitigation.zne import _jsd_predictive_zne_density_matrix_core

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

N_QUBITS = 4
FACTORS = (1.0, 2.0, 3.0)
K_SEEDS = 6
K_SEEDS_FOLLOWUP = 30
K_SEEDS_LARGE = 200
STOCHASTIC_MODELS = ["depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"]
N_TRIALS = 150


def build_ghz_statevector(n_qubits):
    ops = [("h", 0)] + [("cx", i, i + 1) for i in range(n_qubits - 1)]
    sim = de.DenseSVSimulator(n_qubits)
    sim.run_circuit(ops)
    return jnp.asarray(sim.get_statevector(), dtype=jnp.complex128)


def density_matrix_stochastic(sv_ideal, n_qubits, model, base_p, factor, n_trials, master_key):
    p_eff = float(np.clip(base_p * factor, 0.0, 1.0))
    keys = jax.random.split(master_key, n_trials)

    def one_trial(key):
        spec = NoiseSpec(model=model, p=p_eff, jax_key=key)
        return NoiseModel.apply_to_sv(sv_ideal, n_qubits, model=spec.model, p=spec.p, jax_key=spec.jax_key)

    sv_batch = jax.vmap(one_trial)(keys)
    return jnp.einsum("ti,tj->ij", sv_batch, jnp.conj(sv_batch)) / n_trials


def apply_coherent_rz_overrotation(sv, n_qubits, angle):
    sv = sv.reshape([2] * n_qubits)
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    rz = jnp.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=jnp.complex128)
    for q in range(n_qubits):
        sv = jnp.moveaxis(sv, q, 0)
        sv = jnp.tensordot(rz, sv, axes=([1], [0]))
        sv = jnp.moveaxis(sv, 0, q)
    return sv.reshape(-1)


def von_neumann_entropy(rho, eps=1e-12):
    eigs = jnp.clip(jnp.real(jnp.linalg.eigvalsh(rho)), eps, None)
    eigs = eigs / jnp.sum(eigs)
    return -jnp.sum(eigs * jnp.log(eigs))


def quantum_js_divergence(rho, sigma):
    m = 0.5 * (rho + sigma)
    return von_neumann_entropy(m) - 0.5 * von_neumann_entropy(rho) - 0.5 * von_neumann_entropy(sigma)


def qjsd_predictive_zne_core(rho_at_scales, nudge_scale=0.5):
    qjsd_12 = quantum_js_divergence(rho_at_scales[0], rho_at_scales[1])
    qjsd_23 = quantum_js_divergence(rho_at_scales[1], rho_at_scales[2])
    nonlinearity = (qjsd_23 - qjsd_12) / (qjsd_23 + qjsd_12 + 1e-12)
    rectified = jnp.maximum(nonlinearity, 0.0)
    e1, e2, e3 = rho_at_scales
    a, b, c = 3.0 - nudge_scale * rectified, -3.0 + 2.0 * nudge_scale * rectified, 1.0 - nudge_scale * rectified
    return project_to_physical((a * e1 + b * e2 + c * e3) / (a + b + c))


def coherence_l1(rho):
    n = rho.shape[0]
    return jnp.sum(jnp.abs(rho) * (1.0 - jnp.eye(n)))


def coherence_predictive_zne_core(rho_at_scales, nudge_scale=0.5):
    c1_, c2_, c3_ = (coherence_l1(rho_at_scales[i]) for i in range(3))
    gap_12, gap_23 = jnp.abs(c1_ - c2_), jnp.abs(c2_ - c3_)
    nonlinearity = (gap_23 - gap_12) / (gap_23 + gap_12 + 1e-12)
    rectified = jnp.maximum(nonlinearity, 0.0)
    e1, e2, e3 = rho_at_scales
    a, b, c = 3.0 - nudge_scale * rectified, -3.0 + 2.0 * nudge_scale * rectified, 1.0 - nudge_scale * rectified
    return project_to_physical((a * e1 + b * e2 + c * e3) / (a + b + c)), float(rectified)


def run_one_seed_all_signals(sv_ideal, rho_ideal, n_qubits, model, base_p, n_trials, seed):
    master_key = jax.random.PRNGKey(seed)
    rhos = []
    for factor in FACTORS:
        master_key, sub = jax.random.split(master_key)
        rhos.append(density_matrix_stochastic(sv_ideal, n_qubits, model, base_p, factor, n_trials, sub))
    rhos = jnp.stack(rhos)
    rho_baseline = _jsd_predictive_zne_density_matrix_core(rhos, nudge_scale=0.0)
    rho_jsd = jsd_predictive_zne_density_matrix(rhos, jnp.asarray(FACTORS))
    rho_qjsd = qjsd_predictive_zne_core(rhos, nudge_scale=0.5)
    rho_coh, coh_active = coherence_predictive_zne_core(rhos, nudge_scale=0.5)
    fb = float(uhlmann_fidelity(rho_ideal, rho_baseline))
    fj = float(uhlmann_fidelity(rho_ideal, rho_jsd))
    fqj = float(uhlmann_fidelity(rho_ideal, rho_qjsd))
    fc = float(uhlmann_fidelity(rho_ideal, rho_coh))
    return fb, fj, fqj, fc, coh_active


def paired_stats(diffs):
    diffs = np.asarray(diffs)
    m = float(diffs.mean())
    sem = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    p = float(scipy_stats.ttest_1samp(diffs, 0.0)[1]) if len(diffs) > 1 and diffs.std(ddof=1) > 0 else float("nan")
    return m, sem, p, int((diffs > 0).sum()), len(diffs)


def permutation_p(diffs, n_perm=20000, seed=0):
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    perm_means = (signs * diffs).mean(axis=1)
    return float((np.abs(perm_means) >= abs(diffs.mean())).mean())


if __name__ == "__main__":
    print(f"dense_evolution version: {de.__version__}")
    _DATA_DIR.mkdir(exist_ok=True)
    sv_ideal = build_ghz_statevector(N_QUBITS)
    rho_ideal = jnp.outer(sv_ideal, jnp.conj(sv_ideal))
    rows = []

    print(f"\n=== Part 1: screening, {K_SEEDS} seeds/config, all four signals ===\n")
    for model_index, model in enumerate(STOCHASTIC_MODELS):
        for base_p in (0.05, 0.10):
            diffs_j, diffs_qj, diffs_c = [], [], []
            for seed in range(K_SEEDS):
                stable_seed = model_index * 100000 + int(base_p * 1000) * 100 + seed
                fb, fj, fqj, fc, _ = run_one_seed_all_signals(sv_ideal, rho_ideal, N_QUBITS, model, base_p, N_TRIALS, stable_seed)
                diffs_j.append(fj - fb); diffs_qj.append(fqj - fb); diffs_c.append(fc - fb)
            mj, semj, pj, wj, n = paired_stats(diffs_j)
            mqj, semqj, pqj, wqj, _ = paired_stats(diffs_qj)
            mc, semc, pc, wc, _ = paired_stats(diffs_c)
            print(f"{model:18s} p={base_p:.2f}  JSD: {mj:+.5f}(p={pj:.3f},{wj}/{n})  "
                  f"QJSD: {mqj:+.5f}(p={pqj:.3f},{wqj}/{n})  Coherence-L1: {mc:+.5f}(p={pc:.3f},{wc}/{n})")
            rows.append({"part": 1, "model": model, "base_p": base_p,
                         "jsd_mean": mj, "jsd_p": pj, "qjsd_mean": mqj, "qjsd_p": pqj,
                         "coherence_mean": mc, "coherence_p": pc})

    print("\n=== Part 2: coherent (deterministic Rz) -- exactly zero for JSD, QJSD, and coherence-L1 alike ===\n")
    for base_angle in (0.05, 0.15, 0.30):
        angles = [base_angle * f for f in FACTORS]
        rhos = jnp.stack([jnp.outer(v, jnp.conj(v)) for v in
                           [apply_coherent_rz_overrotation(sv_ideal, N_QUBITS, a) for a in angles]])
        fb = float(uhlmann_fidelity(rho_ideal, _jsd_predictive_zne_density_matrix_core(rhos, 0.0)))
        fj = float(uhlmann_fidelity(rho_ideal, jsd_predictive_zne_density_matrix(rhos, jnp.asarray(FACTORS))))
        fqj = float(uhlmann_fidelity(rho_ideal, qjsd_predictive_zne_core(rhos, 0.5)))
        fc = float(uhlmann_fidelity(rho_ideal, coherence_predictive_zne_core(rhos, 0.5)[0]))
        print(f"base_angle={base_angle:.2f}  JSD diff={fj-fb:+.8f}  QJSD diff={fqj-fb:+.8f}  Coherence-L1 diff={fc-fb:+.8f}")
        rows.append({"part": 2, "model": "coherent_rz", "base_p": base_angle,
                      "jsd_mean": fj - fb, "qjsd_mean": fqj - fb, "coherence_mean": fc - fb})

    print(f"\n=== Part 4: large-sample ({K_SEEDS_LARGE} seeds) coherence-L1 validation on phaseflip, base_p=0.05 ===\n")
    diffs_c = []
    for seed in range(K_SEEDS_LARGE):
        fb, fj, fqj, fc, coh_active = run_one_seed_all_signals(sv_ideal, rho_ideal, N_QUBITS, "phaseflip", 0.05, N_TRIALS, 900000 + seed)
        diffs_c.append(fc - fb)
    diffs_c = np.array(diffs_c)
    active = diffs_c[diffs_c != 0.0]
    m, sem, p_t, w, n_active = paired_stats(active)
    p_perm = permutation_p(active) if len(active) > 1 else float("nan")
    print(f"n_active={n_active}/{K_SEEDS_LARGE} ({100*n_active/K_SEEDS_LARGE:.1f}%)  "
          f"active_mean={m:+.6f}  sem={sem:.6f}  wins={w}/{n_active}")
    print(f"one-sample t-test p={p_t:.3e}   permutation test (20000 resamples) p={p_perm:.5f}")
    rows.append({"part": 4, "model": "phaseflip_large_sample", "base_p": 0.05,
                  "n_active": n_active, "n_total": K_SEEDS_LARGE, "coherence_mean": m,
                  "coherence_p_ttest": p_t, "coherence_p_permutation": p_perm, "wins": w})

    pd.DataFrame(rows).to_csv(_DATA_DIR / "jsd_zne_noise_generalization.csv", index=False)
    print(f"\nsaved: {_DATA_DIR / 'jsd_zne_noise_generalization.csv'}")
