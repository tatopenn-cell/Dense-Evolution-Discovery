"""
Every circuit function of the Dashboard on every preset (ideal, dense) and
on every noise model and backend for the Bell and GHZ presets: circuit drawing, histogram, Q-sphere, Bloch spheres, step-by-step
state differences, pairwise half-matrix, per-qubit summary and ket notation.
The last step of the step-by-step view must equal the engine's statevector.
"""
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("streamlit")
dc = pytest.importorskip("dashboard_core")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dashboard"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sota_views as V

NOISES = ["ideal", "depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"]


CASES = [(name, "ideal", "dense") for name in dc.QASM_LIBRARY] + [
    (name, noise, backend)
    for name in ("Bell state (2 qubit)", "GHZ state (3 qubit)")
    for noise in NOISES
    for backend in ("dense", "mps")
    if (noise, backend) != ("ideal", "dense")
]


@pytest.mark.parametrize("name,noise,backend", CASES)
def test_every_circuit_function_runs(name, noise, backend):
    r = dc.run_circuit_from_qasm(dc.QASM_LIBRARY[name], n_shots=50, seed=1, noise_model=noise,
                                 noise_p=0.0 if noise == "ideal" else 0.05, backend=backend)
    n, s = r.n_qubits, r.statevector
    for fig in (dc.draw_circuit_figure(r.ops, n), dc.histogram_figure(r.counts, statevector=s),
                dc.qsphere_figure(s), dc.bloch_multivector_figure(s)):
        plt.close(fig)
    if n <= 10 and r.ops:
        states = V.step_states(r.ops, n)
        for a, b in zip(states[:-1], states[1:]):
            V.difference_rows(a, b, n)
        if noise == "ideal" and backend == "dense":
            assert abs(np.vdot(states[-1], s)) == pytest.approx(1, abs=1e-6)
    if 2 <= n <= 10:
        plt.close(V.half_matrix_figure(s, n))
    assert len(V.qubit_table(s, n)) == n
    assert "|" in V.ket_string(s, n)


def test_bell_pair_views_match_known_values():
    r = dc.run_circuit_from_qasm(dc.QASM_LIBRARY["Bell state (2 qubit)"], n_shots=10, seed=1)
    assert V.concurrence(V._rho(r.statevector, 2, [0, 1])) == pytest.approx(1, abs=1e-9)
    assert V.zz_correlation(r.statevector, 2, 0, 1) == pytest.approx(1, abs=1e-9)
    prod = np.kron([1, 0], [2 ** -0.5, 2 ** -0.5]).astype(complex)
    assert V.concurrence(V._rho(prod, 2, [0, 1])) == pytest.approx(0, abs=1e-9)
