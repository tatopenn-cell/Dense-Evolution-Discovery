"""
First timing signal for the bucketed-SVD MPS optimization (correctness
already verified end-to-end in mps_bucketed_svd_circuit_integration.py:
fidelity 0.999999999998+ against the reference eager simulator, same
final max_bond_used, on complex128 and complex64).

CPU-only, informal (this sandbox has no GPU) -- the real comparison that
matters is on GPU via Colab, same as every other benchmark this session.
This is just a same-machine, same-circuit sanity check that the bucketed
approach is actually faster before spending a Colab round-trip on it.

Same N=50 TFIM Trotter circuit used throughout this session's GPU
benchmarking (colab_gpu_mps_benchmark.py / colab_cuquantum_mps_benchmark.py),
so the shapes/entanglement growth pattern match what was already measured
there.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import dense_evolution as de
from mps_bucketed_svd_circuit_integration import _bucketed_runner, _tfim_gate_list, _to_ops_rows

N, DT, STEPS, J, G, MAX_BOND = 50, 0.1, 5, 1.0, 1.0, 64


def trotter_ops_de(n, dt, steps, j, g):
    theta_zz, theta_x = -2.0 * dt * j, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [("cx", i, i + 1), ("rz", i + 1, theta_zz), ("cx", i, i + 1)]
        for i in range(n):
            ops.append(("rx", i, theta_x))
    return ops


def main():
    """_bucketed_runner is a plain Python function -- called eagerly,
    every invocation retraces jax.lax.scan from scratch (no compiled
    program persists across separate calls without an outer jax.jit).
    Wrapping it once (run_bucketed_jit, below) is the fix -- the first
    call compiles (real "cold"), the second reuses that exact compiled
    program (real "warm"), same cold/warm distinction run_circuit_jit
    already gets for free from its own cached self._mps_runner.

    N=50 is too large for contract_to_statevector() (2^50 amplitudes) --
    z0_from_gamma_lambda instead compares the single-qubit reduced
    density matrix on qubit 0, the same z0 = <Z> metric used throughout
    this session's GPU benchmarking, to check that the max_bond_used
    (8 vs 6) difference between the shipped and bucketed paths doesn't
    come with a real physical disagreement, not just eyeballing the
    bond numbers."""
    print(f"N={N} max_bond={MAX_BOND} STEPS={STEPS}")

    ops_de = trotter_ops_de(N, DT, STEPS, J, G)
    sim = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=1e-12, use_float32=False)
    t0 = time.perf_counter()
    sim.run_circuit_jit(ops_de)
    t_shipped_cold = time.perf_counter() - t0

    sim2 = de.MPSSimulator(N, max_bond=MAX_BOND, svd_cutoff=1e-12, use_float32=False)
    t0 = time.perf_counter()
    sim2.run_circuit_jit(ops_de)
    t_shipped_warm = time.perf_counter() - t0
    print(f"shipped run_circuit_jit (always max_bond={MAX_BOND}): cold={t_shipped_cold:.2f}s warm={t_shipped_warm:.2f}s "
          f"max_bond_used={sim2.max_bond_used()}")

    gate_list = _tfim_gate_list(N, DT, STEPS, J, G)
    ops_jnp = jnp.asarray(_to_ops_rows(gate_list), dtype=jnp.float64)

    run_bucketed_jit = jax.jit(lambda ops: _bucketed_runner(N, MAX_BOND, 1e-12, 1e-5, jnp.complex128, ops))

    t0 = time.perf_counter()
    gammas_f, lambdas_f, real_chi_f = run_bucketed_jit(ops_jnp)
    jax.block_until_ready((gammas_f, lambdas_f, real_chi_f))
    t_bucketed_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    gammas_f, lambdas_f, real_chi_f = run_bucketed_jit(ops_jnp)
    jax.block_until_ready((gammas_f, lambdas_f, real_chi_f))
    t_bucketed_warm = time.perf_counter() - t0
    print(f"bucketed runner: cold={t_bucketed_cold:.2f}s warm={t_bucketed_warm:.2f}s "
          f"max_bond_used={int(np.max(real_chi_f))}")

    print(f"\nspeedup (warm): {t_shipped_warm / t_bucketed_warm:.2f}x")

    def z0_from_gamma_lambda(gamma0, lambda1):
        A = gamma0[0] * lambda1[None, :]
        rho0 = A @ A.conj().T
        rho0 = rho0 / jnp.trace(rho0).real
        return float(jnp.real(jnp.trace(rho0 @ jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=rho0.dtype))))

    z0_shipped = z0_from_gamma_lambda(sim2.gammas[0], sim2.lambdas[1])
    chi0 = int(real_chi_f[1])
    z0_bucketed = z0_from_gamma_lambda(gammas_f[0, :1, :, :chi0], lambdas_f[1, :chi0])
    print(f"\nz0: shipped={z0_shipped:.6f} bucketed={z0_bucketed:.6f} diff={abs(z0_shipped - z0_bucketed):.2e}")


if __name__ == "__main__":
    main()
