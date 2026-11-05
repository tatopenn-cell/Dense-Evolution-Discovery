"""Re-verification of wormhole_syk_teleportation.py's Experiments 17 and 19
(does term-order sensitivity interact with physical noise, and does that
correlation hold across noise levels?) against dense-evolution>=8.1.60 --
i.e. AFTER the v8.1.57 fix to NoiseModel.apply_to_sv's depolarizing
channel. Same motivation and method as wormhole_noise_scan_reverified.py
(Experiment 9's re-verification); see that script's docstring for the
full bug history.

Experiment 17 (commit db58731, 2026-08-08) and Experiment 19 (commit
075b96b, 2026-08-09) both ran before v8.1.57 (2026-08-11), so their
published correlation (r=+0.340, p=0.0158 at n=50, noise_p=0.01) was
never checked against the corrected noise model.

Same JAX-vmap-batched approach as the Experiment 9 re-verification:
NoiseSpec-wrapped NoiseModel.apply_to_sv(jax_key=...) chained between
three separately-compiled Trotter phases via _compile_and_run_circuit_jit
(vmap-safe per its own docstring). mutual_information/partial_trace
aren't vmap-traceable, so that reduction runs eagerly on the returned
statevector batch, same as the Experiment 9 script. vmaps over the
n_trials axis per (seed, term-order, mu-sign); seeds are looped in
Python (JIT compiles once per distinct op-array SHAPE, which is the same
across seeds since K_TERMS is fixed, so this still only pays the
compilation cost once).

Verified before trusting any noisy number here: the noiseless delta for
seed 61's original term order at the paper-default point matches
_run_trotter_ordered's eager reference bit-for-bit.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import jax
import jax.numpy as jnp

import dense_evolution
from dense_evolution import mutual_information
from dense_evolution.trotter import trotter_evolve_ops
from dense_evolution.registry import NoiseModel, NoiseSpec
from dense_evolution.gates import GATE_IDS
from dense_evolution.compiler import QuantumTranspiler, _compile_and_run_circuit_jit

from dashboard_core.wormhole import _protocol_layout, _initial_state_ops
from wormhole_syk_teleportation import (
    N_MAJORANA, K_TERMS, J, _DATA_DIR, _IMAGES_DIR,
    find_multiple_seeds, _run_trotter_ordered,
)

T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60
N_STEPS_EVOLUTION, N_STEPS_COUPLING = 8, 16


def _compile_ops(ops):
    """Same compilation DenseSVSimulator.run_circuit_jit does internally
    -- see wormhole_noise_scan_reverified.py's identical helper."""
    target = QuantumTranspiler.transpile(ops)
    rows = []
    for cmd in target:
        name = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
        g_id = float(GATE_IDS[name])
        args = cmd[1:]
        if name in ('rx', 'ry', 'rz', 'p', 'u1', 'phase'):
            rows.append([g_id, float(args[0]), 0.0, float(args[1]) if len(args) > 1 else 0.0])
        elif name in ('cp', 'crz', 'cphase'):
            rows.append([g_id, float(args[0]), float(args[1]) if len(args) > 1 else 0.0,
                         float(args[2]) if len(args) > 2 else 0.0])
        elif name in ('cx', 'cz', 'swap', 'cy'):
            rows.append([g_id, float(args[0]), float(args[1]) if len(args) > 1 else 0.0, 0.0])
        else:
            rows.append([g_id, float(args[0]) if args else 0.0, 0.0, 0.0])
    return jnp.array(rows, dtype=jnp.float64)


def _apply_noise(sv, n_full, noise_p, key):
    spec = NoiseSpec(model='depolarizing', p=noise_p, jax_key=key)
    return NoiseModel.apply_to_sv(sv, n_full, model=spec.model, p=spec.p,
                                   jax_key=spec.jax_key,
                                   qubits=list(spec.qubits) if spec.qubits else None)


def _build_seed_pipeline(seed):
    """One seed's full setup: ops for both term orderings (original,
    reversed), both mu signs, ready to vmap over a trials axis."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    terms_reversed = list(reversed(terms_full))
    sv0 = jnp.zeros(2 ** n_full, dtype=jnp.complex128).at[0].set(1.0)

    ops_init = _compile_ops(_initial_state_ops(n_side, L, R, P, Q, True))
    ops_couple_pos = _compile_ops(trotter_evolve_ops(v_terms, +MU_PAPER, N_STEPS_COUPLING))
    ops_couple_neg = _compile_ops(trotter_evolve_ops(v_terms, -MU_PAPER, N_STEPS_COUPLING))

    orderings = {}
    for label, terms in (("original", terms_full), ("reversed", terms_reversed)):
        orderings[label] = (
            _compile_ops(trotter_evolve_ops(terms, T0_PAPER, N_STEPS_EVOLUTION)),
            _compile_ops(trotter_evolve_ops(terms, T1_PAPER, N_STEPS_EVOLUTION)),
        )

    def run_noisy_sv(key, ops_evolve1, ops_couple, ops_evolve2, noise_p):
        sv = _compile_and_run_circuit_jit(sv0, ops_init)
        sv = _compile_and_run_circuit_jit(sv, ops_evolve1)
        if noise_p > 0:
            key, sub = jax.random.split(key)
            sv = _apply_noise(sv, n_full, noise_p, sub)
        sv = _compile_and_run_circuit_jit(sv, ops_couple)
        if noise_p > 0:
            key, sub = jax.random.split(key)
            sv = _apply_noise(sv, n_full, noise_p, sub)
        sv = _compile_and_run_circuit_jit(sv, ops_evolve2)
        if noise_p > 0:
            key, sub = jax.random.split(key)
            sv = _apply_noise(sv, n_full, noise_p, sub)
        return sv

    return run_noisy_sv, orderings, ops_couple_pos, ops_couple_neg, n_full, P, R, terms_full


def mean_delta_vmap(seed, label, noise_p, n_trials, master_key):
    """Mean/std of delta = I_neg - I_pos over n_trials, for one seed and
    one term ordering, common-random-numbers between +mu/-mu (same key
    for both, matching the original script's re-seeded-rng convention)."""
    run_noisy_sv, orderings, ops_pos, ops_neg, n_full, P, R = _build_seed_pipeline(seed)[:7]
    ops_evolve1, ops_evolve2 = orderings[label]

    run_pos = jax.vmap(lambda k, p: run_noisy_sv(k, ops_evolve1, ops_pos, ops_evolve2, p), in_axes=(0, None))
    run_neg = jax.vmap(lambda k, p: run_noisy_sv(k, ops_evolve1, ops_neg, ops_evolve2, p), in_axes=(0, None))

    keys = jax.random.split(master_key, n_trials)
    sv_pos_batch = np.asarray(run_pos(keys, float(noise_p)))
    sv_neg_batch = np.asarray(run_neg(keys, float(noise_p)))
    i_pos = np.array([mutual_information(sv, n_full, [P], [R[0]]) for sv in sv_pos_batch])
    i_neg = np.array([mutual_information(sv, n_full, [P], [R[0]]) for sv in sv_neg_batch])
    deltas = i_neg - i_pos
    return float(deltas.mean()), float(deltas.std())


def run_term_order_noise_interaction_vmap(seeds, noise_p, n_trials, master_seed=0):
    master_key = jax.random.PRNGKey(master_seed)
    rows = []
    for seed in seeds:
        master_key, sub = jax.random.split(master_key)
        delta_mean_orig, delta_std_orig = mean_delta_vmap(seed, "original", noise_p, n_trials, sub)
        master_key, sub2 = jax.random.split(master_key)
        delta_mean_rev, delta_std_rev = mean_delta_vmap(seed, "reversed", noise_p, n_trials, sub2)
        rows.append({
            "seed": seed, "delta_mean_original_noisy": delta_mean_orig,
            "delta_std_original_noisy": delta_std_orig, "delta_mean_reversed_noisy": delta_mean_rev,
            "delta_std_reversed_noisy": delta_std_rev,
            "noisy_order_sensitivity": abs(delta_mean_rev - delta_mean_orig),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"dense_evolution version: {dense_evolution.__version__}")

    print("\n--- Noiseless correctness check (vmap path vs. eager path, seed 61, original order) ---")
    seeds_check = find_multiple_seeds(n_instances=1, n_candidates=200)
    seed0 = seeds_check[0] if seeds_check else 61

    run_noisy_sv, orderings, ops_pos, ops_neg, n_full, P, R, terms_full = _build_seed_pipeline(seed0)
    ops_evolve1, ops_evolve2 = orderings["original"]
    vmap_sv = jax.vmap(lambda k: run_noisy_sv(k, ops_evolve1, ops_pos, ops_evolve2, 0.0))(jax.random.split(jax.random.PRNGKey(0), 1))
    i_pos_vmap = mutual_information(np.asarray(vmap_sv[0]), n_full, [P], [R[0]])

    n_side, n_full2, L, R2, P2, Q, terms_full2, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed0)
    i_pos_eager = _run_trotter_ordered(terms_full2, v_terms, n_side, n_full2, L, R2, P2, Q,
                                        +MU_PAPER, T0_PAPER, T1_PAPER, N_STEPS_EVOLUTION, N_STEPS_COUPLING)
    match = abs(i_pos_vmap - i_pos_eager) < 1e-9
    print(f"  vmap: {i_pos_vmap:+.10f}  eager: {i_pos_eager:+.10f}  MATCH: {match}")
    if not match:
        raise SystemExit("Noiseless mismatch -- stopping, do not trust noisy output.")

    print("\n--- Experiment 17 re-verification: n=50 seeds, noise_p=0.01 ---")
    seeds50 = find_multiple_seeds(n_instances=50, n_candidates=50000)
    df17 = run_term_order_noise_interaction_vmap(seeds50, noise_p=0.01, n_trials=6, master_seed=17)
    df17.to_csv(_DATA_DIR / "wormhole_term_order_noise_interaction_reverified_v8160.csv", index=False)
    r17 = scipy_stats.pearsonr(df17["noisy_order_sensitivity"], df17["delta_mean_original_noisy"])
    print(f"  n={len(df17)}: r={r17.statistic:+.4f} (p={r17.pvalue:.4f})")

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df17["delta_mean_original_noisy"]]
    ax.errorbar(df17["noisy_order_sensitivity"], df17["delta_mean_original_noisy"],
                yerr=df17["delta_std_original_noisy"], fmt='none', ecolor='#444444', zorder=1)
    ax.scatter(df17["noisy_order_sensitivity"], df17["delta_mean_original_noisy"], c=colors, s=80, zorder=5)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("noisy order sensitivity |delta_reversed - delta_original| (p=0.01)", color='#888888')
    ax.set_ylabel("delta, original term order (noisy, mean of trials)", color='#888888')
    ax.set_title(f"Re-verified vs. corrected noise model (v8.1.60): Experiment 17 (n={len(df17)})\n"
                 f"r={r17.statistic:+.3f}, p={r17.pvalue:.4f}", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_term_order_noise_interaction_reverified_v8160.png", dpi=300)
    plt.close(fig)

    print("\n--- Experiment 19 re-verification: n=20 seeds, noise_p in (0.005, 0.01, 0.02) ---")
    seeds20 = find_multiple_seeds(n_instances=20, n_candidates=120000)
    rows19 = []
    for noise_p in (0.005, 0.01, 0.02):
        df_level = run_term_order_noise_interaction_vmap(seeds20, noise_p=noise_p, n_trials=6, master_seed=19)
        r_level = scipy_stats.pearsonr(df_level["noisy_order_sensitivity"], df_level["delta_mean_original_noisy"])
        n_wrong = int((df_level["delta_mean_original_noisy"] < 0).sum())
        rows19.append({"noise_p": noise_p, "n": len(df_level), "pearson_r": r_level.statistic,
                        "p_value": r_level.pvalue, "n_wrong_signed": n_wrong})
        print(f"  noise_p={noise_p}: r={r_level.statistic:+.4f}, p={r_level.pvalue:.4f}, "
              f"{n_wrong}/{len(df_level)} wrong-signed")

    df19 = pd.DataFrame(rows19)
    df19.to_csv(_DATA_DIR / "wormhole_noise_level_scan_reverified_v8160.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(df19["noise_p"], df19["pearson_r"], marker='o', color='#00FFFF', linewidth=1.5)
    ax1.axhline(0, color='#666666', linestyle=':')
    ax1.set_xlabel("noise_p", color='#888888')
    ax1.set_ylabel("Pearson r", color='#888888')
    ax1.set_title("Correlation strength vs. noise level", color='#CCCCCC')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.plot(df19["noise_p"], df19["p_value"], marker='o', color='#FF007F', linewidth=1.5)
    ax2.axhline(0.05, color='#FFAA00', linestyle='--', label='p=0.05')
    ax2.set_xlabel("noise_p", color='#888888')
    ax2.set_ylabel("p-value", color='#888888')
    ax2.set_title("Significance vs. noise level", color='#CCCCCC')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    fig.suptitle(f"Re-verified vs. corrected noise model (v8.1.60): Experiment 19 (n={len(seeds20)})",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_noise_level_scan_reverified_v8160.png", dpi=300)
    plt.close(fig)

    print("\nSaved data/wormhole_term_order_noise_interaction_reverified_v8160.csv, "
          "data/wormhole_noise_level_scan_reverified_v8160.csv, and matching images/*.png")
