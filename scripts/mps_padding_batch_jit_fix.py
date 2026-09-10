"""
Third real bug found while re-verifying the second fix (jit-cached gate-
matrix constructors) on real Kaggle GPU: the second fix eliminated the
recompile penalty in _fuse_compiled_rows (confirmed correct -- two
separate fuse_gates=True calls now measure the same, 2.196s vs 2.223s),
but the end-to-end real API speedup stayed flat at 1.37-1.39x, barely
moved from the pre-fix 1.36x. Direct cProfile on the real, local
run_circuit_jit (both fuse_gates=True and fuse_gates=False) found the
actual dominant cost: _pad_gamma/_pad_lambda are called once per gamma/
lambda tensor (50-51 separate eager calls for N=50), each doing its own
jnp.zeros + .at[].set() -- 861-874 individual JAX primitive dispatches
measured, 52-55% of total run_circuit_jit time on both paths equally
(0.132s/0.254s default, 0.29s/0.524s fused). Because this overhead is
shared equally by both paths, it dilutes the real speedup ratio toward
1x -- exactly why the real API's 1.37x is so much lower than this
project's own kernel-only reimplementation ratio of 1.99x.

Fix: wrap the whole per-call padding+stacking loop (all gammas, all
lambdas) inside ONE module-level @jax.jit closure instead of doing 50+
separate un-jitted eager calls. JAX traces a pytree of ragged-shaped
arrays fine (each shape is static per site for a fixed circuit
structure), so this compiles the entire padding pipeline into a single
XLA program dispatched as ONE call, not 50+ separate ones -- same
"batch the dispatch, don't eliminate the math" principle as the first
fix, applied to a different function.
"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dense_evolution.backends.mps import MPSSimulator, _pad_gamma, _pad_lambda


@partial(jax.jit, static_argnums=(1, 2))
def _pad_all_gammas_fast(gammas, max_bond, dtype):
    return jnp.stack([_pad_gamma(g, max_bond).astype(dtype) for g in gammas])


@partial(jax.jit, static_argnums=(1, 2))
def _pad_all_lambdas_fast(lambdas, max_bond, dtype):
    return jnp.stack([_pad_lambda(l, max_bond).astype(dtype) for l in lambdas])


def _tfim_ops(n, dt, steps, J, g):
    theta_zz, theta_x = -2.0 * dt * J, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [("cx", i, i + 1), ("rz", i + 1, theta_zz), ("cx", i, i + 1)]
        for i in range(n):
            ops.append(("rx", i, theta_x))
    return ops


def correctness_check():
    sim = MPSSimulator(6, max_bond=16, svd_cutoff=1e-12)
    sim.run_circuit_jit(_tfim_ops(6, 0.1, 3, 1.0, 1.0))
    dtype = sim.gammas[0].dtype
    lambda_dtype = sim.lambdas[0].dtype

    old = jnp.stack([_pad_gamma(g, sim.chi).astype(dtype) for g in sim.gammas])
    new = _pad_all_gammas_fast(tuple(sim.gammas), sim.chi, dtype)
    np.testing.assert_allclose(np.asarray(old), np.asarray(new), atol=0)

    old_l = jnp.stack([_pad_lambda(l, sim.chi).astype(lambda_dtype) for l in sim.lambdas])
    new_l = _pad_all_lambdas_fast(tuple(sim.lambdas), sim.chi, lambda_dtype)
    np.testing.assert_allclose(np.asarray(old_l), np.asarray(new_l), atol=0)
    print("correctness: batched-jit padding produces identical results to the shipped per-tensor loop")


def repeated_call_timing():
    """Same discipline as the gate-matrix fix: several SEPARATE fresh-
    instance calls, since that's what the real benchmark script does
    and what a real user calling run_circuit_jit repeatedly experiences."""
    n = 50
    ops = _tfim_ops(n, 0.1, 5, 1.0, 1.0)

    print("\n--- shipped per-tensor padding loop, 3 separate fresh instances ---")
    for call_i in range(3):
        sim = MPSSimulator(n, max_bond=64, svd_cutoff=1e-12)
        sim.run_circuit_jit(ops)
        dtype = sim.gammas[0].dtype
        lambda_dtype = sim.lambdas[0].dtype
        t0 = time.perf_counter()
        jnp.stack([_pad_gamma(g, sim.chi).astype(dtype) for g in sim.gammas])
        jnp.stack([_pad_lambda(l, sim.chi).astype(lambda_dtype) for l in sim.lambdas])
        print(f"  call {call_i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")

    print("\n--- batched-jit padding, 3 separate fresh instances ---")
    for call_i in range(3):
        sim = MPSSimulator(n, max_bond=64, svd_cutoff=1e-12)
        sim.run_circuit_jit(ops)
        dtype = sim.gammas[0].dtype
        lambda_dtype = sim.lambdas[0].dtype
        t0 = time.perf_counter()
        _pad_all_gammas_fast(tuple(sim.gammas), sim.chi, dtype)
        _pad_all_lambdas_fast(tuple(sim.lambdas), sim.chi, lambda_dtype)
        print(f"  call {call_i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")


if __name__ == "__main__":
    correctness_check()
    repeated_call_timing()
