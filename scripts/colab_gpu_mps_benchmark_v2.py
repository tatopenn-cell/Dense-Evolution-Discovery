# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# Re-run of this session's N=50 GPU benchmark, now pinned to 8.1.76 --
# the release that promotes the bucketed-SVD dispatch into
# run_circuit_jit itself (dense_evolution/backends/mps.py). No script
# changes needed: run_circuit_jit's public signature/behavior is
# unchanged, only its internal SVD sizing. This is the "does the real
# release beat cuQuantum now" follow-up to the earlier same-N comparison
# (which found cuQuantum ~1.7-2.9x faster than the pre-optimization
# Dense-Evolution MPS backend).
#
# Companion script: colab_cuquantum_mps_benchmark_v2.py -- same N=50,
# same PRECISION toggle, run separately (cuQuantum is a different
# environment/install, not combined into one script).
#
# See mps_bucketed_svd_gpu_timing_followup.md for what this actually
# found: a real, unresolved discrepancy between this script's warm time
# (8.89s) and the isolated bucketed-runner benchmark that validated the
# optimization (2.41s, same circuit) -- not yet root-caused.
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
