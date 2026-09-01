"""
Tests for scripts/vqe_pauli_sum_zne_autodiff.py -- the "killer example"
combining PauliSumOperator (matrix-free Pauli-sum Hamiltonian), autodiff
VQE, and ZNE in one differentiable pipeline. The full script (real H10
active-space Hartree-Fock + 80-step VQE + 40-shot-per-scale ZNE) takes
several minutes, so these tests exercise the same real functions at a
scale cheap enough for a normal test run instead of re-running the whole
experiment.
"""
import sys
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import vqe_pauli_sum_zne_autodiff as vqe_script

import dense_evolution as de
from dense_evolution.physics.observables import pauli_hamiltonian_to_matrix


def setup_module(module):
    de.set_precision(True)


def test_hardware_efficient_ansatz_qasm_has_expected_gate_counts():
    qasm = vqe_script.hardware_efficient_ansatz_qasm(n_qubits=4, n_layers=2)
    assert qasm.count("ry(") == 4 * 2
    assert qasm.count("cx ") == 3 * 2
    assert "qreg q[4];" in qasm


def test_hardware_efficient_ansatz_qasm_parses_and_has_expected_param_count():
    qasm = vqe_script.hardware_efficient_ansatz_qasm(n_qubits=4, n_layers=2)
    circuit = de.QASMParser().parse(qasm)
    energy_fn, n_params = de.circuit_to_energy_fn(circuit, n_qubits=4)
    assert n_params == 4 * 2


def test_selftest_pauli_sum_operator_matches_dense_passes():
    # Exercises the exact function main() calls first -- if this ever
    # regresses (e.g. a future change reintroduces the float32-by-default
    # gotcha documented on pauli_sum_matvec_jax), this test catches it
    # without needing the full H10 pipeline.
    vqe_script.selftest_pauli_sum_operator_matches_dense()


def test_exact_ground_state_energy_matches_manual_diagonalization():
    rng = np.random.default_rng(5)
    n_qubits = 3
    letters = ["I", "X", "Y", "Z"]
    terms = [
        (float(rng.normal()), "".join(rng.choice(letters) for _ in range(n_qubits)))
        for _ in range(8)
    ]
    expected = float(np.min(np.linalg.eigvalsh(pauli_hamiltonian_to_matrix(terms, n_qubits))))
    actual = vqe_script.exact_ground_state_energy(terms, n_qubits)
    assert actual == pytest.approx(expected, abs=1e-9)


@pytest.mark.slow
def test_build_h10_active_space_terms_returns_documented_qubit_count():
    # Real PennyLane Hartree-Fock, same call main() makes -- slow-ish
    # (~10-20s) but a genuine integration check that active_orbitals=6
    # really does give 12 qubits (not just asserted in the docstring).
    terms, n_qubits = vqe_script.build_h10_active_space_terms(active_orbitals=6)
    assert n_qubits == 12
    assert len(terms) > 0


@pytest.mark.slow
def test_end_to_end_pipeline_runs_at_tiny_scale():
    # Full ansatz -> circuit_to_energy_fn -> PauliSumOperator -> jax.grad
    # -> ZNE pipeline, same code path as main(), but at a size (4 qubits,
    # a handful of Adam steps, few noise shots) cheap enough to run in a
    # normal test suite -- catches wiring bugs (e.g. the single-shot-vs-
    # averaged-trajectory ZNE bug found building this script, where 3
    # noise scales returned bit-identical energies) without waiting
    # minutes for the real H10 system.
    n_qubits = 4
    rng = np.random.default_rng(1)
    letters = ["I", "X", "Y", "Z"]
    terms = [
        (float(rng.normal()), "".join(rng.choice(letters) for _ in range(n_qubits)))
        for _ in range(6)
    ]

    qasm = vqe_script.hardware_efficient_ansatz_qasm(n_qubits, n_layers=1)
    circuit = de.QASMParser().parse(qasm)
    energy_fn, n_params = de.circuit_to_energy_fn(circuit, n_qubits)
    h_op = de.PauliSumOperator(terms, n_qubits)

    @jax.jit
    def loss(theta):
        e, _ = energy_fn(theta, h_op)
        return e

    theta = jnp.zeros(n_params)
    val0, grad0 = jax.value_and_grad(loss)(theta)
    assert jnp.isfinite(val0)
    assert jnp.all(jnp.isfinite(grad0))

    # A few gradient-descent steps should not increase the energy.
    theta1 = theta - 0.2 * grad0
    val1, _ = jax.value_and_grad(loss)(theta1)
    assert float(val1) <= float(val0) + 1e-6

    # ZNE wiring: different noise scales on a NON-trivial (post-descent)
    # circuit must give genuinely different single-shot energies at least
    # some of the time -- catches the "identical across scales" bug this
    # script's own development hit (fixed by averaging many shots per
    # scale; here we just check a handful of raw single-shot draws are
    # not all bit-identical, which is what the original bug looked like).
    energies = []
    for scale, seed in [(1.0, 1), (2.0, 2), (3.0, 3)]:
        noise = de.NoiseSpec(model="depolarizing", p=0.3 * scale, jax_key=jax.random.PRNGKey(seed))
        e_noisy, _ = energy_fn(theta1, h_op, noise=noise)
        energies.append(float(e_noisy))
    assert len(set(energies)) > 1, f"noise had no effect across scales: {energies}"
