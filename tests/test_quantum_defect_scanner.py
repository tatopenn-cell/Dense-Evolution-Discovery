"""
run_parametric_batch_jit positional-slot contract — regression guard
----------------------------------------------------------------------
Found while auditing the repo against dense-evolution 8.1.21: every
rotation gate (rx/ry/rz/p) in base_circuit consumes a parameter_batch
column in order of appearance, even when given a literal float instead
of a string placeholder — the literal is silently discarded. This bit
quantum_defect_scanner.py, which assumed RY(pi/4) literals stayed fixed
while only the RZ("batch_param_q") gates varied; the RZ columns actually
landed on the RY slots and ran out of bounds (JAX silently clips).

1 test, mirrors quantum_defect_scanner.py's exact circuit shape at a
small N_Q, cross-checked against an independent single-circuit reference
via run_circuit_jit_beast_mode (ground truth, not the same code path).
"""

import numpy as np
import jax
import jax.numpy as jnp
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 4
RY_ANGLE = np.pi / 4
ATOL = 1e-10


def _build_base_ops():
    ops = [['ry', q, float(RY_ANGLE)] for q in range(N_Q)]
    ops += [['rz', q, f"batch_param_{q}"] for q in range(N_Q)]
    ops += [['cx', q, q + 1] for q in range(N_Q - 1)]
    return ops


def _reference_statevector(rz_angles: np.ndarray) -> np.ndarray:
    """Independent ground truth: single-circuit run with literal RZ angles."""
    ops = [['ry', q, float(RY_ANGLE)] for q in range(N_Q)]
    ops += [['rz', q, float(rz_angles[q])] for q in range(N_Q)]
    ops += [['cx', q, q + 1] for q in range(N_Q - 1)]
    sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ops)
    return np.asarray(sim.get_statevector())


def test_batch_positional_slots_ry_fixed_rz_varying():
    """Every row of parameter_batch must fill ALL 2*N_Q rotation slots
    (N_Q constant RY columns + N_Q varying RZ columns) — not just the
    ones written as string placeholders in base_circuit."""
    sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
    base_ops = _build_base_ops()

    grid = np.zeros((N_Q, 2 * N_Q), dtype=np.float64)
    grid[:, :N_Q] = RY_ANGLE
    for q in range(N_Q):
        grid[q, N_Q + q] = 0.5

    batch = sim.run_parametric_batch_jit(base_ops, jnp.array(grid, dtype=jnp.float64))
    assert batch.shape == (N_Q, 1 << N_Q)

    for row in range(N_Q):
        rz_angles = np.zeros(N_Q)
        rz_angles[row] = 0.5
        expected = _reference_statevector(rz_angles)
        got = np.asarray(batch[row])
        assert np.max(np.abs(got - expected)) < ATOL, (
            f"row {row}: batch output diverges from independent single-circuit reference"
        )
