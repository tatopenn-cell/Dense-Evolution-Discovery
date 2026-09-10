"""
Second real bug found by testing the literal, promoted run_circuit_jit(ops,
fuse_gates=True) on GPU (Kaggle T4): the real speedup from gate blocking
alone measured only 1.36x, far short of the 1.99x measured on Dense-
Evolution-Discovery's own standalone reimplementation. Root-caused by
profiling _fuse_compiled_rows directly (cProfile, not guessed): _mps_1q_matrix/
_mps_2q_matrix are plain functions containing jax.lax.switch, never wrapped
in @jax.jit -- every EAGER call to them pays a real XLA compile cost, even
on a cache hit for the SAME (g_id, param, transpose_flag) combination
across DIFFERENT run_circuit_jit(fuse_gates=True) calls, because the
matrix_cache inside _fuse_compiled_rows is scoped to one call, and JAX
itself has no persistent compiled-executable cache for an undecorated
eager function.

Confirmed directly: a real N=50 TFIM circuit (985 rows, only 3 distinct
gate/parameter combinations) spent 661ms inside _fuse_compiled_rows alone,
on top of another 307ms padding gammas/lambdas -- about 1 second of pure
host-side overhead paid on EVERY call, present only on the fuse_gates=True
path (the default path just does one vectorized jnp.array(...) conversion,
no per-row Python loop calling eager JAX functions).

Fix: wrap the two gate-matrix constructors in module-level @jax.jit
closures (dtype marked static, since it changes the actual computation).
This gives them JAX's own persistent compiled-executable cache, keyed by
input shape/dtype, valid for the lifetime of the process -- not just one
_fuse_compiled_rows call. Confirmed directly: the identical 985-row pass
that took 660ms drops to 0.76ms once the (very first, one-time) warmup
compile has happened -- an ~870x reduction, and it is now genuinely a
one-time-per-process cost, not a one-time-per-call cost.
"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dense_evolution.backends.mps import (
    MPSSimulator,
    _compile_mps_ops,
    _mps_1q_matrix,
    _mps_2q_matrix,
)


@partial(jax.jit, static_argnums=(2,))
def _jit_mps_1q_matrix(g_id, param, dtype):
    return _mps_1q_matrix(g_id, param, dtype)


@partial(jax.jit, static_argnums=(2,))
def _jit_mps_2q_matrix(g_id, param, dtype):
    return _mps_2q_matrix(g_id, param, dtype)


def _embed_1q_matrix(mat1, qubit, pair):
    eye2 = np.eye(2, dtype=mat1.dtype)
    a, _ = pair
    return np.kron(mat1, eye2) if qubit == a else np.kron(eye2, mat1)


def fuse_compiled_rows_fast(rows, dtype):
    """Same fusion logic as _fuse_compiled_rows, using the module-level
    jitted gate constructors instead of the plain (recompiles-every-call)
    ones."""
    matrix_cache = {}

    def row_matrix(row):
        g_id, q1, q2, param, transpose_flag = row
        key = (g_id, param, transpose_flag)
        cached = matrix_cache.get(key)
        if cached is not None:
            return cached, (int(q1), int(q2)) if g_id >= 20 else (int(q1),)
        g_id_arr = jnp.asarray(g_id).astype(jnp.int32)
        if g_id >= 20:
            mat4 = np.asarray(_jit_mps_2q_matrix(g_id_arr, jnp.asarray(param), dtype))
            if transpose_flag > 0.5:
                mat4 = np.transpose(mat4, (1, 0, 3, 2))
            mat = mat4.reshape(4, 4)
        else:
            mat = np.asarray(_jit_mps_1q_matrix(g_id_arr, jnp.asarray(param), dtype))
        matrix_cache[key] = mat
        return mat, (int(q1), int(q2)) if g_id >= 20 else (int(q1),)

    fused = []
    i = 0
    n = len(rows)
    while i < n:
        mat, qubits = row_matrix(rows[i])
        pair = qubits if len(qubits) == 2 else None
        active_qubit = qubits[0]
        j = i + 1
        while j < n:
            nmat, nqubits = row_matrix(rows[j])
            n_is_2q = len(nqubits) == 2
            if pair is None:
                if not n_is_2q:
                    if nqubits[0] != active_qubit:
                        break
                    mat = nmat @ mat
                    j += 1
                    continue
                if active_qubit not in nqubits:
                    break
                pair = nqubits
                mat = _embed_1q_matrix(mat, active_qubit, pair)
                mat = nmat @ mat
                j += 1
                continue
            if n_is_2q:
                if nqubits != pair:
                    break
            else:
                if nqubits[0] not in pair:
                    break
                nmat = _embed_1q_matrix(nmat, nqubits[0], pair)
            mat = nmat @ mat
            j += 1
        if pair is not None:
            fused.append(('2q', pair[0], pair[1], mat.reshape(2, 2, 2, 2)))
        else:
            fused.append(('1q', active_qubit, mat))
        i = j
    return fused


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
    ops = _tfim_ops(6, 0.1, 3, 1.0, 1.0)
    rows = _compile_mps_ops(ops, 6)
    dtype = jnp.complex128

    from dense_evolution.backends.mps import _fuse_compiled_rows
    fused_old = _fuse_compiled_rows(rows, dtype)
    fused_new = fuse_compiled_rows_fast(rows, dtype)

    assert len(fused_old) == len(fused_new)
    for a, b in zip(fused_old, fused_new):
        assert a[0] == b[0] and a[1] == b[1]
        if a[0] == '2q':
            assert a[2] == b[2]
            np.testing.assert_allclose(a[3], b[3], atol=1e-12)
        else:
            np.testing.assert_allclose(a[2], b[2], atol=1e-12)
    print("correctness: fuse_compiled_rows_fast produces identical results to the shipped _fuse_compiled_rows")


def repeated_call_timing():
    """Simulates real usage: several SEPARATE run_circuit_jit(fuse_gates=True)
    calls across the process lifetime (not just one), which is exactly the
    scenario the shipped version pays repeated compile cost on -- the fix
    should show its benefit growing with the number of calls, not just on
    a single call."""
    ops = _tfim_ops(50, 0.1, 5, 1.0, 1.0)
    dtype = jnp.complex128

    from dense_evolution.backends.mps import _fuse_compiled_rows

    print("\n--- shipped _fuse_compiled_rows across 3 separate calls ---")
    for call_i in range(3):
        rows = _compile_mps_ops(ops, 50)
        t0 = time.perf_counter()
        _fuse_compiled_rows(rows, dtype)
        print(f"  call {call_i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")

    print("\n--- fuse_compiled_rows_fast (jit-cached constructors) across 3 separate calls ---")
    for call_i in range(3):
        rows = _compile_mps_ops(ops, 50)
        t0 = time.perf_counter()
        fuse_compiled_rows_fast(rows, dtype)
        print(f"  call {call_i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")


if __name__ == "__main__":
    correctness_check()
    repeated_call_timing()
