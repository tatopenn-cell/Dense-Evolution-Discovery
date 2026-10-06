# Run on Google Colab with a GPU runtime selected.
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# The circuit: a Trotterized Transverse Field Ising Model (TFIM) chain,
# same as colab_gpu_mps_benchmark_v2.py -- see that script's own comment
# for the plain-terms explanation. Built here as real OPENQASM 2.0 text
# parsed by dense_evolution's own QASMParser, exactly what a user would
# write themselves.
#
# This is the THIRD attempt at a correct "warm" GPU timing for
# run_circuit_jit, and the one that finally got it right:
#   v2 (fresh instance for "warm"): self._mps_runner is a per-instance,
#      lazily-built @jax.jit closure -- a fresh instance means a fresh,
#      uncompiled closure, so "warm" paid a full fresh compile every
#      time, never showing real post-compile speed.
#   v3 (same instance called twice): reuses the compiled closure, but
#      the second call's circuit runs on top of the FIRST call's already-
#      evolved (already entangled) state, not a fresh |0...0> state --
#      not the same computation twice, genuinely more work as
#      entanglement keeps growing.
#   This version: a FRESH |0...0> instance for each timing (so
#      entanglement starts from the same place both times), with the
#      ALREADY-COMPILED closure manually shared onto it (bypassing
#      self._mps_runner's own lazy "build if None" via direct attribute
#      assignment) -- a fresh state AND a pre-compiled kernel together,
#      isolating real post-compile execution time for the first time in
#      this investigation.
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

sim1 = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
t0 = time.perf_counter()
sim1.run_circuit_jit(ops)
t_cold = time.perf_counter() - t0
print(f"cold (fresh instance, compiles self._mps_runner): {t_cold:.3f}s max_bond_used={sim1.max_bond_used()}")

sim2 = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
sim2._mps_runner = sim1._mps_runner  # share the already-compiled closure, fresh |0...0> state
t0 = time.perf_counter()
sim2.run_circuit_jit(ops)
t_warm = time.perf_counter() - t0
print(f"warm (fresh |0...0> instance, REUSED compiled runner): {t_warm:.3f}s max_bond_used={sim2.max_bond_used()}")

sim3 = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
sim3._mps_runner = sim1._mps_runner
t0 = time.perf_counter()
sim3.run_circuit_jit(ops)
t_warm2 = time.perf_counter() - t0
print(f"warm again (confirms stability): {t_warm2:.3f}s max_bond_used={sim3.max_bond_used()}")

print(f"\nspeedup cold/warm: {t_cold/t_warm:.2f}x")
