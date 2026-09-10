# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install -q cuquantum-python-cu12
#
# Cell 2: the code below.
#
# Companion to colab_gpu_mps_benchmark_v2.py -- same N=50 TFIM Trotter
# circuit, same PRECISION toggle, run against cuQuantum's own NetworkState
# MPS implementation instead of Dense-Evolution's. N=50 is chosen
# specifically to stay below N>=64, where cuQuantum's NetworkState MPS
# mode has a confirmed, open, unfixed integer-overflow bug (NVIDIA/
# cuQuantum GitHub issue #225) -- so this is a fair, unpatched,
# same-N/same-precision comparison in both directions, not a workaround
# for that bug.
#
# Result (T4, complex128): warm=1.14s, z0=0.5754 -- matches
# Dense-Evolution's own z0 exactly. See mps_bucketed_svd_gpu_timing_followup.md:
# this made cuQuantum look ~7.8x faster than Dense-Evolution 8.1.76's
# real run_circuit_jit (8.89s warm) on the same circuit -- worse than the
# pre-optimization comparison (~1.7-2.9x) -- which triggered the
# unresolved timing investigation in that doc, not yet root-caused.
PRECISION = "complex128"  # or "complex64"

import time
import numpy as np
import cupy as cp
from cuquantum.tensornet.experimental import NetworkState, NetworkOperator

DTYPE = "complex128" if PRECISION == "complex128" else "complex64"
NP_DTYPE = np.complex128 if PRECISION == "complex128" else np.complex64

N, J, G, DT, STEPS, MAX_BOND = 50, 1.0, 1.0, 0.1, 5, 64
print(f"PRECISION={PRECISION} N={N} max_bond={MAX_BOND}")

CUTOFF = 1e-12 if PRECISION == "complex128" else 1e-6


def cx_tensor():
    m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=NP_DTYPE)
    return m.reshape(2, 2, 2, 2)


def rz_tensor(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=NP_DTYPE)


def rx_tensor(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=NP_DTYPE)


def trotter_ops(n, dt, steps, J, g):
    theta_zz, theta_x = -2.0 * dt * J, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [("cx", (i, i + 1), cx_tensor()),
                    ("rz", (i + 1,), rz_tensor(theta_zz)),
                    ("cx", (i, i + 1), cx_tensor())]
        for i in range(n):
            ops.append(("rx", (i,), rx_tensor(theta_x)))
    return ops


ops = trotter_ops(N, DT, STEPS, J, G)
print(f"n_gates={len(ops)}")

state_mode_extents = (2,) * N
state = NetworkState(state_mode_extents, dtype=DTYPE,
                      config={"max_extent": MAX_BOND, "abs_cutoff": CUTOFF})

t0 = time.perf_counter()
for name, modes, tensor in ops:
    state.apply_tensor_operator(modes, cp.asarray(tensor), unitary=True)

# apply_tensor_operator is LAZY (builds a symbolic graph) -- the actual
# contraction only happens when a compute_* call is made, so the timer
# must include this call too or the "apply" loop above times ~nothing.
z_op = NetworkOperator(state_mode_extents, dtype=DTYPE)
z_op.append_product(1.0, [(0,)], [cp.asarray(np.array([[1, 0], [0, -1]], dtype=NP_DTYPE))])
z0_val, norm = state.compute_expectation(z_op, return_norm=True)
cp.cuda.Stream.null.synchronize()
t_cold = time.perf_counter() - t0
z0 = float((z0_val / norm).real)

state2 = NetworkState(state_mode_extents, dtype=DTYPE,
                       config={"max_extent": MAX_BOND, "abs_cutoff": CUTOFF})
t0 = time.perf_counter()
for name, modes, tensor in ops:
    state2.apply_tensor_operator(modes, cp.asarray(tensor), unitary=True)
z0_val2, norm2 = state2.compute_expectation(z_op, return_norm=True)
cp.cuda.Stream.null.synchronize()
t_warm = time.perf_counter() - t0

print(f"\ncuQuantum NetworkState MPS, {PRECISION}:")
print(f"  cold (incl. graph build + first contraction): {t_cold:.2f}s")
print(f"  warm (fresh state, same shapes): {t_warm:.2f}s")
print(f"  z0={z0:.4f}")

state.free()
state2.free()
