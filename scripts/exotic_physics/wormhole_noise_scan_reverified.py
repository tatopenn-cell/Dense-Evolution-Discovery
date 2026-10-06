"""Re-verification of wormhole_syk_teleportation.py's Experiment 9 (does the
sign-dependent teleportation signal survive realistic depolarizing noise?)
against dense-evolution>=8.1.60 -- i.e. AFTER the v8.1.57 fix to
NoiseModel.apply_to_sv's depolarizing channel (it used to draw 2**(n-1)
independent fire/no-fire decisions per qubit per shot, one per amplitude
pair, instead of ONE decision per qubit per shot applied globally; see
registry.py's own docstring for the full bug history). The original
Experiment 9 (commit 823bdb9, 2026-08-07) ran two days BEFORE that fix
(v8.1.57 tagged 2026-08-11), so its published claim -- "the signal crosses
zero between p=0.01 and p=0.02" -- was never checked against the corrected
noise model. This script does that check.

Built as a JAX-vmap-batched replacement for the original eager, Python
for-loop-over-trials implementation, using the same pure primitives
run_circuit_jit uses internally (_compile_and_run_circuit_jit, explicitly
documented as vmap-safe) chained with three NoiseSpec-wrapped
NoiseModel.apply_to_sv(jax_key=...) calls (one per protocol phase) --
NoiseSpec is this library's public "Differentiable Noise" utility
(a JAX PyTree wrapping model/p/jax_key so noise composes natively with
jit/vmap/grad), used here instead of passing raw jax_key= by hand.
mutual_information/partial_trace are NOT jax.vmap-traceable (they call
np.asarray() internally), so the vmap batches only the expensive
Trotter-evolution + noise-injection part; the final mutual-information
reduction runs eagerly afterward on the returned batch of statevectors.

Verified before trusting any noisy number here: the noiseless (noise_p=0,
fully deterministic) path reproduces the original eager
run_trotter_noise_scan's noiseless delta bit-for-bit (+0.0172792189,
diff < 1e-9) -- confirms the whole init/evolve/couple/evolve pipeline is
correct independent of anything noise-related. Also spot-verified that
NoiseModel.apply_to_sv's jax_key path varies correctly with different keys
at every tested p in isolation.

Real result, not a bug: at n_trials=6 (the original script's budget,
sized for the old, much-more-aggressive buggy noise model), several low-p
points showed exactly zero trial-to-trial variance -- not a software bug,
but the correct, expected behavior of a properly single-shot-per-qubit
Kraus channel: at p=0.01 across 3 injection points x 10 qubits x 6 trials
(180 independent Bernoulli(p) draws), P(zero draws fire at all) = 0.99^180
~= 16%, a real, non-negligible chance every one of 6 trials draws no error
at all. n_trials=500 below resolves this cleanly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import dense_evolution
from dense_evolution import mutual_information
from dense_evolution.trotter import trotter_evolve_ops
from dense_evolution.registry import NoiseModel, NoiseSpec
from dense_evolution.gates import GATE_IDS
from dense_evolution.compiler import QuantumTranspiler, _compile_and_run_circuit_jit

from dashboard_core.wormhole import _protocol_layout, _initial_state_ops
from wormhole_syk_teleportation import find_seed, N_MAJORANA, K_TERMS, J, _DATA_DIR, _IMAGES_DIR


def _compile_ops(ops):
    """Same op-list -> [g_id, q1, q2, param] compilation DenseSVSimulator.
    run_circuit_jit does internally, reused here so the noise-injection
    seam sits between three separately-compiled jax arrays instead of
    inside one uninterrupted call."""
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


def build_pipeline(seed, t0, mu, t1, n_steps_evolution=8, n_steps_coupling=16):
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)

    ops_init = _compile_ops(_initial_state_ops(n_side, L, R, P, Q, True))
    ops_evolve1 = _compile_ops(trotter_evolve_ops(terms_full, t0, n_steps_evolution))
    ops_evolve2 = _compile_ops(trotter_evolve_ops(terms_full, t1, n_steps_evolution))
    ops_couple_pos = _compile_ops(trotter_evolve_ops(v_terms, +mu, n_steps_coupling))
    ops_couple_neg = _compile_ops(trotter_evolve_ops(v_terms, -mu, n_steps_coupling))

    sv0 = jnp.zeros(2 ** n_full, dtype=jnp.complex128).at[0].set(1.0)

    def _apply_noise(sv, n_full, noise_p, key):
        spec = NoiseSpec(model='depolarizing', p=noise_p, jax_key=key)
        return NoiseModel.apply_to_sv(sv, n_full, model=spec.model, p=spec.p,
                                       jax_key=spec.jax_key,
                                       qubits=list(spec.qubits) if spec.qubits else None)

    def run_noisy_pure_sv(key, ops_couple, noise_p):
        """Returns the final statevector (not mutual_information -- see
        module docstring for why the reduction step stays outside vmap)."""
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

    return run_noisy_pure_sv, ops_couple_pos, ops_couple_neg, n_full, P, R


def run_noise_scan_vmap(seed, t0, mu, t1, noise_levels, n_trials, master_seed=0):
    run_noisy_pure_sv, ops_pos, ops_neg, n_full, P, R = build_pipeline(seed, t0, mu, t1)
    run_pos = jax.vmap(lambda k, p: run_noisy_pure_sv(k, ops_pos, p), in_axes=(0, None))
    run_neg = jax.vmap(lambda k, p: run_noisy_pure_sv(k, ops_neg, p), in_axes=(0, None))

    master_key = jax.random.PRNGKey(master_seed)
    rows = []
    for noise_p in noise_levels:
        master_key, sub = jax.random.split(master_key)
        keys = jax.random.split(sub, n_trials)
        sv_pos_batch = np.asarray(run_pos(keys, float(noise_p)))
        sv_neg_batch = np.asarray(run_neg(keys, float(noise_p)))
        i_pos = np.array([mutual_information(sv, n_full, [P], [R[0]]) for sv in sv_pos_batch])
        i_neg = np.array([mutual_information(sv, n_full, [P], [R[0]]) for sv in sv_neg_batch])
        deltas = i_neg - i_pos
        rows.append({"noise_p": float(noise_p), "delta_mean": float(deltas.mean()),
                     "delta_std": float(deltas.std()), "sem": float(deltas.std() / np.sqrt(n_trials))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"dense_evolution version: {dense_evolution.__version__}")
    seed = find_seed()

    noise_levels = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
    df = run_noise_scan_vmap(seed, t0=0.70, mu=17.0, t1=0.36,
                              noise_levels=noise_levels, n_trials=500)
    print(df.to_string(index=False))

    crossing = df[df["delta_mean"] < 0]
    first_negative_p = crossing["noise_p"].min() if not crossing.empty else None
    print(f"\ncrosses zero: {'at p=' + str(first_negative_p) if first_negative_p is not None else 'NOWHERE in the scanned range'}")

    df.to_csv(_DATA_DIR / "wormhole_trotter_noise_scan_reverified_v8160.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(df["noise_p"], df["delta_mean"], yerr=df["sem"], fmt='o-',
                color='#00FFFF', ecolor='#FF007F', capsize=4, markersize=6)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("depolarizing noise probability p", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title(f"Re-verified vs. corrected noise model (v8.1.60, post v8.1.57 fix)\n"
                 f"(seed={seed}, t0=0.70, mu=17.0, t1=0.36, n=500 trials/point)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_trotter_noise_scan_reverified_v8160.png", dpi=300)
    plt.close(fig)
    print("\nSaved data/wormhole_trotter_noise_scan_reverified_v8160.csv "
          "and images/wormhole_trotter_noise_scan_reverified_v8160.png")
