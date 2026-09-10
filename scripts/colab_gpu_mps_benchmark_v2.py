# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# What this circuit is, in plain terms: a Trotterized simulation of the
# Transverse Field Ising Model (TFIM) -- a chain of N qubits, each
# entangled with its neighbor (cx), given a small phase kick (rz) tuned
# by the interaction strength J, then a small rotation (rx) tuned by the
# transverse field g. Repeating this "layer" STEPS times approximates
# continuous time evolution under the TFIM Hamiltonian. Same circuit
# family used throughout this repo's MPS benchmarking (dense_evolution_
# mps_benchmark.ipynb).
#
# Built as real OPENQASM 2.0 text, parsed by dense_evolution's own
# QASMParser -- not hand-written gate tuples -- so this is exactly what
# a user would write to run this circuit themselves via run_circuit_jit.
PRECISION = "complex128"  # or "complex64"

import time
import numpy as np
import jax

if PRECISION == "complex128":
    jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
if jax.default_backend() != "gpu":
    print("\nWARNING: not running on GPU -- check Runtime > Change runtime type > T4 GPU, "
          "then Runtime > Restart session, before trusting any timing below.\n")

import dense_evolution as de
print("dense_evolution version:", de.__version__)
assert de.__version__ == "8.1.76", f"expected 8.1.76, got {de.__version__} -- pip install -q dense-evolution==8.1.76"

N, J, G, DT, STEPS, MAX_BOND = 50, 1.0, 1.0, 0.1, 5, 64
USE_FLOAT32 = PRECISION == "complex64"
CUTOFF = 1e-12 if PRECISION == "complex128" else 1e-6
print(f"PRECISION={PRECISION} use_float32={USE_FLOAT32} svd_cutoff={CUTOFF} max_bond={MAX_BOND}")

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
print(f"N={N} n_gates={len(ops)}")

sim = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
t0 = time.perf_counter()
sim.run_circuit_jit(ops)
t_cold = time.perf_counter() - t0

sim2 = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=CUTOFF, use_float32=USE_FLOAT32)
t0 = time.perf_counter()
sim2.run_circuit_jit(ops)
t_warm = time.perf_counter() - t0

gamma0 = np.asarray(sim2.gammas[0])
lambda0 = np.asarray(sim2.lambdas[1])
A = gamma0[0] * lambda0[None, :]
rho0 = A @ A.conj().T
rho0 = rho0 / np.trace(rho0).real
z0 = float(np.real(np.trace(rho0 @ np.array([[1.0, 0.0], [0.0, -1.0]]))))

print(f"\nDense-Evolution 8.1.76 MPS on {jax.default_backend()}, {PRECISION}:")
print(f"  cold (incl. compile): {t_cold:.2f}s")
print(f"  warm (fresh sim, same shapes, cache reused): {t_warm:.2f}s")
print(f"  memory={sim2.memory_mb():.2f}MB max_bond_used={sim2.max_bond_used()} "
      f"truncation_error={sim2.total_truncation_error():.2e} z0={z0:.4f}")
