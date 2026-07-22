"""
vqe_silicon_molecular_optimized_per_bond.py -- per-bond exact-gradient tests
-------------------------------------------------------------------------------
2 tests, target < 20s total.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pb = _import_script("vqe_silicon_molecular_optimized_per_bond")
de = pb.de


def _independent_single_circuit_kinetic(theta_vec: np.ndarray) -> float:
    """Independent reference: single-circuit run_circuit_jit_beast_mode with
    each bond's angle set literally, not the batch/vmap code path."""
    ops = [['x', 0]]
    for q in range(pb.N_BONDS):
        ops += [['cx', q + 1, q], ['ry', q + 1, float(theta_vec[q])], ['cx', q, q + 1],
                ['ry', q + 1, float(-theta_vec[q])], ['cx', q + 1, q]]
    s = de.DenseSVSimulator(n_qubits=pb.N_Q, use_gpu=False, use_float32=False)
    s.set_initial_state()
    s.run_circuit_jit_beast_mode(ops)
    return pb._kinetic_from_sv(np.asarray(s.get_statevector()))


def test_per_bond_gradient_matches_finite_difference_and_independent_reference():
    """The batched per-bond kinetic/gradient function must agree with (a) an
    independent single-circuit reference and (b) a per-bond finite-difference
    gradient -- each bond's gradient must only depend on shifting that bond's
    own gates, not the others'."""
    theta_vec = np.array([0.2, 0.9, 0.38, 1.4, 0.05])
    kinetic, grad = pb.batched_kinetic_and_exact_gradient(theta_vec[None, :])

    kinetic_ref = _independent_single_circuit_kinetic(theta_vec)
    assert kinetic[0] == pytest.approx(kinetic_ref, abs=1e-9)

    h = 1e-6
    for q in range(pb.N_BONDS):
        tp = theta_vec.copy(); tp[q] += h
        tm = theta_vec.copy(); tm[q] -= h
        fd_grad = (_independent_single_circuit_kinetic(tp) - _independent_single_circuit_kinetic(tm)) / (2 * h)
        assert grad[0, q] == pytest.approx(fd_grad, abs=1e-5), (
            f"bond {q}: exact={grad[0, q]}, finite-diff={fd_grad}"
        )


def test_per_bond_optimization_differentiates_bonds_and_beats_shared_theta():
    """A small-scale optimization run (5 R points, 30 epochs) must: (1) let
    bonds settle at genuinely different angles from each other (not collapse
    to a single shared value, which would mean the extra freedom bought
    nothing), and (2) reach an energy at or below the shared-theta optimum
    at every tested R (more free parameters can only help, not hurt)."""
    R_space = np.linspace(1.4, 4.0, 5)

    theta_star, E_star, grad_final = pb.optimize_pec_per_bond(R_space, n_epochs=30, verbose=False)

    spread = theta_star.max(axis=1) - theta_star.min(axis=1)
    assert np.all(spread > 0.05), (
        f"bonds should differentiate (spread > 0.05 rad), got {spread}"
    )

    import vqe_silicon_molecular_optimized as shared
    theta_shared, E_shared, _ = shared.optimize_pec(R_space, n_epochs=30, verbose=False)

    assert np.all(E_star <= E_shared + 1e-3), (
        f"per-bond optimum should be at or below the shared-theta optimum: "
        f"E_star={E_star}, E_shared={E_shared}"
    )
