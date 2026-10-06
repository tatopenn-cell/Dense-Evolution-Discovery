# Run on Google Colab with a GPU runtime selected.
#
# IMPORTANT: Runtime > Restart session FIRST (a plain restart is enough,
# not Factory reset) -- must be a session where dense_evolution has never
# been imported yet, otherwise Python reuses the already-imported module
# regardless of what pip installs afterward.
#
# Then paste this WHOLE file into a single fresh cell and run it once.
#
# The circuit: the same Trotterized TFIM chain used throughout this
# investigation -- see colab_gpu_mps_benchmark_v2.py's comment for the
# plain-terms explanation. Built as real OPENQASM 2.0 text via
# dense_evolution's own QASMParser.
#
# This is the final, correctly-controlled comparison: the OLD shipped
# run_circuit_jit (8.1.75, fixed max_bond-sized SVD, pre-optimization)
# measured with the SAME "fresh |0...0> instance + manually shared,
# already-compiled self._mps_runner" methodology that correctly isolated
# the NEW bucketed version's real post-compile GPU speed
# (colab_gpu_mps_benchmark_v4_shared_compiled_runner.py). Real result:
# 8.1.75 warm=3.746s/3.820s vs 8.1.76 warm=2.730s/2.737s -- a genuine,
# if modest, ~1.37-1.40x GPU speedup (see mps_bucketed_svd_gpu_timing_
# followup.md for the full story of how this number was arrived at).
!pip install --no-deps --force-reinstall --no-cache-dir -q dense-evolution==8.1.75

PRECISION = "complex128"  # or "complex64"

import time
import jax

if PRECISION == "complex128":
    jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

import dense_evolution as de
print("dense_evolution version:", de.__version__)
assert de.__version__ == "8.1.75", f"expected 8.1.75 (pre-optimization baseline), got {de.__version__}"

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

print(f"\n8.1.75 (old, shipped) warm={t_warm:.3f}s -- compare against 8.1.76's warm=2.730s/2.737s")
