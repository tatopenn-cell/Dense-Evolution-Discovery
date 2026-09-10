# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# Second attempt at a correct "warm" GPU measurement for run_circuit_jit.
# v2 (fresh instance for "warm") was wrong: self._mps_runner is a
# per-instance, lazily-built @jax.jit closure -- a fresh instance means a
# fresh, uncompiled closure, so "warm" paid a full fresh compile every
# time, never showing real post-compile speed. This version calls
# run_circuit_jit TWICE on the SAME instance instead, to reuse the
# already-built closure. Result: also wrong, for a different reason --
# the second call's circuit runs on top of the FIRST call's already-
# evolved (already entangled) state, not a fresh |0...0> state, so it's
# not the same computation twice, it's twice the circuit depth with
# growing entanglement (10.2s -> 12.2s -> 42.2s across repeated calls,
# consistent with bond dimension climbing toward max_bond as entanglement
# keeps accumulating, not a bug). See colab_gpu_mps_benchmark_v4_
# shared_compiled_runner.py for the version that actually got this right.
PRECISION = "complex128"  # or "complex64"

import time
import jax

if PRECISION == "complex128":
    jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

import dense_evolution as de
print("dense_evolution version:", de.__version__)

N, J, G, DT, STEPS, MAX_BOND = 50, 1.0, 1.0, 0.1, 5, 64
USE_FLOAT32 = PRECISION == "complex64"
CUTOFF = 1e-12 if PRECISION == "complex128" else 1e-6
print(f"PRECISION={PRECISION} N={N} max_bond={MAX_BOND}")


theta_zz = -2.0 * DT * J
theta_x = -2.0 * DT * G
lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];"]
for _ in range(STEPS):
    for i in range(N - 1):
        lines.append(f"cx q[{i}],q[{i+1}];")
        lines.append(f"rz({theta_zz}) q[{i+1}];")
        lines.append(f"cx q[{i}],q[{i+1}];")
    for i in range(N):
        lines.append(f"rx({theta_x}) q[{i}];")
circuit = de.QASMParser().parse("\n".join(lines))
ops = circuit.to_tuples()
print(f"n_gates={len(ops)}")

sim = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)

t0 = time.perf_counter()
sim.run_circuit_jit(ops)
t_call1 = time.perf_counter() - t0
print(f"call 1 (compiles self._mps_runner): {t_call1:.3f}s")

t0 = time.perf_counter()
sim.run_circuit_jit(ops)
t_call2 = time.perf_counter() - t0
print(f"call 2 (SAME instance, self._mps_runner already built -- real warm): {t_call2:.3f}s")

t0 = time.perf_counter()
sim.run_circuit_jit(ops)
t_call3 = time.perf_counter() - t0
print(f"call 3 (same again, confirms stability): {t_call3:.3f}s")

print(f"\nspeedup call1/call2: {t_call1/t_call2:.2f}x (how much of call 1 was compile, not execution)")
