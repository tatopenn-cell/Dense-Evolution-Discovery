# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# Investigates a real discrepancy: colab_gpu_mps_benchmark_v2.py measured
# run_circuit_jit's warm time at 8.89s on GPU for N=50/STEPS=5 -- SLOWER
# than the pre-promotion "shipped" baseline measured earlier this session
# (6.60s, same circuit), even though the isolated bucketed-runner
# comparison (colab_bucketed_svd_gpu_benchmark.py) found the bucketed
# dispatch itself 2.74x FASTER than the old fixed-size SVD in the same
# conditions. Something doesn't add up, and it needs to be found, not
# guessed at.
#
# Two real differences between that isolated comparison and the real
# run_circuit_jit call:
#   1. run_circuit_jit's own bookkeeping loop (mps.py, after the scan)
#      converts the stacked per-step diagnostics to numpy and iterates
#      over every 2-qubit gate in pure Python -- host-side work the
#      isolated _bucketed_runner benchmark never measured at all.
#   2. The real per-branch SVD step also computes trunc_err and
#      entanglement entropy every 2-qubit gate -- extra math the isolated
#      benchmark's simplified reimplementation skipped.
#
# This script times run_circuit_jit's own internal pieces separately:
# op compilation (host, pure Python), the compiled kernel call itself
# (sim._mps_runner -- same private attribute run_circuit_jit calls,
# accessed directly here only to add a timer around it), and the
# bookkeeping loop, to see which piece the 8.89s actually went into.
#
# Result (see mps_bucketed_svd_gpu_timing_followup.md): bookkeeping was
# negligible (62ms) -- the compiled kernel call itself took 8.6s. That
# ruled out hypothesis 1, leading to colab_gpu_diagnostics_cost_check.py.
PRECISION = "complex128"  # or "complex64"

import time
import jax

if PRECISION == "complex128":
    jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

import dense_evolution as de
from dense_evolution.backends.mps import _compile_mps_ops, _pad_gamma, _pad_lambda, _real_dtype_for
import jax.numpy as jnp

print("dense_evolution version:", de.__version__)

N, J, G, DT, STEPS, MAX_BOND = 50, 1.0, 1.0, 0.1, 5, 64
USE_FLOAT32 = PRECISION == "complex64"
CUTOFF = 1e-12 if PRECISION == "complex128" else 1e-6


def trotter_ops(n, dt, steps, J, g):
    theta_zz, theta_x = -2.0 * dt * J, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [("cx", i, i + 1), ("rz", i + 1, theta_zz), ("cx", i, i + 1)]
        for i in range(n):
            ops.append(("rx", i, theta_x))
    return ops


ops = trotter_ops(N, DT, STEPS, J, G)


def timed_run(sim, ops, label):
    t0 = time.perf_counter()
    compiled_rows = _compile_mps_ops(ops, sim.n)
    dtype = sim.gammas[0].dtype
    lambda_dtype = sim.lambdas[0].dtype
    ops_dtype = _real_dtype_for(dtype)
    ops_array = jnp.array(compiled_rows, dtype=ops_dtype) if compiled_rows else jnp.zeros((0, 5), dtype=ops_dtype)
    if sim._mps_runner is None:
        from dense_evolution.backends.mps import _build_mps_runner
        sim._mps_runner = _build_mps_runner(sim.n, sim.chi, sim.eps, sim.jsd_budget)
    gammas_padded = jnp.stack([_pad_gamma(g, sim.chi).astype(dtype) for g in sim.gammas])
    lambdas_padded = jnp.stack([_pad_lambda(l, sim.chi).astype(lambda_dtype) for l in sim.lambdas])
    real_chi_initial = jnp.asarray(sim._real_chi, dtype=jnp.int32)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_gammas, final_lambdas, final_real_chi, diag = sim._mps_runner(
        gammas_padded, lambdas_padded, real_chi_initial, ops_array)
    jax.block_until_ready((final_gammas, final_lambdas, final_real_chi, diag))
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim.gammas = [final_gammas[i] for i in range(sim.n)]
    sim.lambdas = [final_lambdas[i] for i in range(sim.n + 1)]
    sim._real_chi = final_real_chi.__array__() if hasattr(final_real_chi, "__array__") else final_real_chi
    import numpy as np
    if compiled_rows:
        chi_history, jsd_history, trunc_err_history, entropy_history = (
            np.asarray(diag[0]), np.asarray(diag[1]), np.asarray(diag[2]), np.asarray(diag[3]))
        g_ids = np.asarray([row[0] for row in compiled_rows])
        q1_ids = np.asarray([int(row[1]) for row in compiled_rows])
        is_2q_mask = g_ids >= 20
        for i in np.nonzero(is_2q_mask)[0]:
            chi_new = int(chi_history[i])
            jsd_val = float(jsd_history[i])
            sim._bond_history.append(chi_new)
            sim.jsd_per_bond.append(jsd_val)
            sim.truncation_errors.append(float(trunc_err_history[i]))
    t_bookkeeping = time.perf_counter() - t0

    print(f"{label}: prep={t_prep*1000:.1f}ms kernel={t_kernel*1000:.1f}ms "
          f"bookkeeping={t_bookkeeping*1000:.1f}ms total={( t_prep+t_kernel+t_bookkeeping)*1000:.1f}ms")
    return t_prep, t_kernel, t_bookkeeping


sim_cold = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
timed_run(sim_cold, ops, "cold")

sim_warm = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
timed_run(sim_warm, ops, "warm")

print("\nFor comparison, the full run_circuit_jit() call (includes the exact same three pieces):")
sim_full = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
sim_full.run_circuit_jit(ops)  # warms the runner cache
sim_full2 = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
t0 = time.perf_counter()
sim_full2.run_circuit_jit(ops)
print(f"run_circuit_jit warm: {(time.perf_counter()-t0)*1000:.1f}ms max_bond_used={sim_full2.max_bond_used()}")
