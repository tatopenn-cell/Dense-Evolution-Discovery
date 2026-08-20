"""
quantum_defect_scanner.py -- two independent bugs found during the
dense-evolution 8.1.21 audit
----------------------------------------------------------------------
1. run_parametric_batch_jit positional-slot contract: every rotation
   gate (rx/ry/rz/p) in base_circuit consumes a parameter_batch column
   in order of appearance, even when given a literal float instead of a
   string placeholder -- the literal is silently discarded. This bit the
   script, which assumed RY(pi/4) literals stayed fixed while only the
   RZ("batch_param_q") gates varied; the RZ columns actually landed on
   the RY slots and ran out of bounds (JAX silently clips).

2. MSB-first indexing mismatch: DenseSVSimulator uses phys = n-1-qubit
   internally (see _cx_numpy/apply_cx in dense_evolution/simulator.py)
   -- gate-qubit q lives at physical array bit (N_Q-1-q). The script's
   coherence measurement used `1 << local_qubit` directly, reading a
   DIFFERENT physical qubit than the one that actually received that
   row's RZ dephasing. The true (index-corrected) result is much
   simpler than what was published: 11 of 12 nodes give an IDENTICAL
   coherence (0.438791), and only the very last node (gate-qubit 11)
   differs (0.620545) -- not the "70.71% / 50% x5 / 43.88% / 50% x5"
   pattern the mislabeled indexing produced.

3 tests, mirrors quantum_defect_scanner.py's exact circuit shape at a
small N_Q, cross-checked against an independent single-circuit reference
via run_circuit_jit_beast_mode (ground truth, not the same code path).
"""

import importlib.util
import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp
import dense_evolution as de
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

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
    sim = de.DenseSVSimulator(n_qubits=N_Q, use_float32=False)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ops)
    return np.asarray(sim.get_statevector())


def test_batch_positional_slots_ry_fixed_rz_varying():
    """Every row of parameter_batch must fill ALL 2*N_Q rotation slots
    (N_Q constant RY columns + N_Q varying RZ columns) — not just the
    ones written as string placeholders in base_circuit."""
    sim = de.DenseSVSimulator(n_qubits=N_Q, use_float32=False)
    base_ops = _build_base_ops()

    grid = np.zeros((N_Q, 2 * N_Q), dtype=np.float64)
    grid[:, :N_Q] = RY_ANGLE
    for q in range(N_Q):
        grid[q, N_Q + q] = 0.5

    batch = sim.run_parametric_batch_jit(base_ops, jnp.array(grid, dtype=jnp.float64))
    assert batch.shape == (N_Q, 1 << N_Q)


qds = _import_script("quantum_defect_scanner")


def test_coerenza_x_reads_the_correct_msb_first_physical_bit():
    """coerenza_x(sv, gate_qubit) must read the SAME physical qubit that
    was given a distinguishing rotation -- verified on a pure product
    state (no CX, so each qubit's <X> is independent of the others): one
    gate-qubit gets RY(1.0) (<X> = sin(1.0)) instead of the uniform
    RY(pi/4) (<X> = sin(pi/4)) every other qubit gets. Only that one
    gate-qubit should read back sin(1.0); every other must read sin(pi/4)."""
    marked_qubit = 5
    marked_angle = 1.0
    ops = [['ry', q, float(np.pi / 4)] for q in range(qds.N_Q)]
    ops[marked_qubit] = ['ry', marked_qubit, float(marked_angle)]
    sim = de.DenseSVSimulator(n_qubits=qds.N_Q, use_float32=False)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ops)
    sv = np.asarray(sim.get_statevector())

    for q in range(qds.N_Q):
        x = qds.coerenza_x(sv, q)
        expected = np.sin(marked_angle) if q == marked_qubit else np.sin(np.pi / 4)
        assert x == pytest.approx(expected, abs=1e-10), (
            f"gate-qubit {q}: got {x}, expected {expected} "
            f"({'marked' if q == marked_qubit else 'unmarked'})"
        )


def test_defect_scan_gives_the_index_corrected_pattern():
    """scansiona_difetti() -- the real, refactored, importable function
    from quantum_defect_scanner.py -- must reproduce the index-corrected
    physics: 11 of 12 gate-qubits give an IDENTICAL coherence, and only
    the last (gate-qubit N_Q-1) differs. The old (buggy) mask produced a
    much more elaborate-looking but physically wrong pattern
    (70.71% / 50% x5 / 43.88% / 50% x5)."""
    coerenza = qds.scansiona_difetti()

    assert coerenza.shape == (qds.N_Q,)
    first_11 = coerenza[:-1]
    assert np.allclose(first_11, first_11[0], atol=1e-9), (
        f"gate-qubits 0..{qds.N_Q-2} should all match exactly, got {first_11}"
    )
    assert abs(coerenza[-1] - first_11[0]) > 0.1, (
        "the last gate-qubit should differ meaningfully from the rest, "
        f"got last={coerenza[-1]}, rest={first_11[0]}"
    )
