"""
Integration smoke tests -- import and execute the REAL production scripts
----------------------------------------------------------------------------
Every other test file in this repo re-derives the circuit/observable math
by hand instead of calling the actual script functions. That validates the
physics but not the scripts: both real bugs found during the dense-evolution
8.1.21 audit (quantum_defect_scanner.py's wrong batch grid, vqe_jax_grad.py's
wrong batch grid) lived in scripts that had zero test coverage of their own
-- nothing that actually executed them could have caught either one.

This file imports vqe_gradient.py, zne_mitigation.py, scan_ising.py and
next_gen_silicon.py as modules (all four now guard their expensive
sweep/CSV/plot pipeline behind `if __name__ == "__main__":` specifically
so they CAN be imported without side effects) and calls their real
functions directly, cross-checked against an independent reference
(PennyLane, or a closed-form value) -- not a re-implementation of the
same formula.

22 tests, target < 30s total.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import pennylane as qml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    """Import a top-level repo script as a module without running it as
    __main__ -- their sweep/CSV/plot pipelines are guarded behind
    `if __name__ == "__main__":` precisely so this is side-effect-free."""
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vqe_gradient = _import_script("vqe_gradient")
zne_mitigation = _import_script("zne_mitigation")
scan_ising = _import_script("scan_ising")
next_gen_silicon = _import_script("next_gen_silicon")


# ── vqe_gradient.py: calcola_energia_vqe cross-validated against PennyLane ──

_N_Q = vqe_gradient.N_Q
_T_HOP = vqe_gradient.t_hopping
_dev = qml.device("default.qubit", wires=_N_Q)

_OBS_PERIODIC_XY = sum(
    -(_T_HOP / 2.0) * (
        qml.PauliX(q) @ qml.PauliX((q + 1) % _N_Q)
        + qml.PauliY(q) @ qml.PauliY((q + 1) % _N_Q)
    )
    for q in range(_N_Q)
)


def _pl_energy(theta: float) -> float:
    @qml.qnode(_dev)
    def circ():
        qml.PauliX(wires=0)
        for q in range(_N_Q - 1):
            qml.CNOT(wires=[q + 1, q])
            qml.RY(float(theta), wires=q + 1)
            qml.CNOT(wires=[q, q + 1])
            qml.RY(-float(theta), wires=q + 1)
            qml.CNOT(wires=[q + 1, q])
        return qml.expval(_OBS_PERIODIC_XY)
    return float(circ())


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.4, 4.0, 5.7])
def test_vqe_gradient_calcola_energia_vqe_matches_pennylane(theta):
    """Calls the REAL calcola_energia_vqe from vqe_gradient.py (not a
    hand-copied version) and cross-checks it against an independent
    PennyLane circuit of the same ansatz."""
    e_de = vqe_gradient.calcola_energia_vqe(theta)
    e_pl = _pl_energy(theta)
    assert abs(e_de - e_pl) < 1e-9, (
        f"theta={theta}: dense-evolution={e_de:.10f}, pennylane={e_pl:.10f}, "
        f"diff={abs(e_de - e_pl):.2e}"
    )


@pytest.mark.parametrize("theta", [0.0, 0.4471315065697962, 1.3416407865, 2.0,
                                    3.14069479915893, 4.7, 2 * np.pi - 0.01])
def test_vqe_gradient_closed_form_matches_real_circuit_exactly(theta):
    """energia_forma_chiusa(theta) -- no quantum circuit simulation, O(N_Q)
    to evaluate -- must reproduce the REAL calcola_energia_vqe(theta) to
    machine precision. This isn't PennyLane cross-validation (a different
    library computing the same thing); it's a genuine analytic identity:
    the periodic kinetic sum, expanded via the exact amplitude cascade of
    the shared-theta Givens ansatz, has a closed form."""
    e_circuit = vqe_gradient.calcola_energia_vqe(theta)
    e_closed_form = vqe_gradient.energia_forma_chiusa(theta)
    assert e_circuit == pytest.approx(e_closed_form, abs=1e-9), (
        f"theta={theta}: circuit={e_circuit:.12f}, closed_form={e_closed_form:.12f}, "
        f"diff={abs(e_circuit - e_closed_form):.2e}"
    )


# ── zne_mitigation.py: real functions, closed-form + internal-consistency ──

def test_zne_mitigation_real_functions_are_internally_consistent():
    """Calls the REAL generate_bloch_state / calcola_aspettazione_hamiltoniana /
    measure_energy_with_shots / apply_stochastic_dephasing from
    zne_mitigation.py (not re-implementations) and checks:
    1. At noise_scale=0.0, measure_energy_with_shots short-circuits to the
       exact ideal energy -- must equal calling calcola_aspettazione_hamiltoniana
       directly on the same Bloch state (same function, called two ways).
    2. At k=0.0, the ideal energy must hit the exact analytic ground-state
       bound E = -2*t_hopping (closed form, cited in the README).
    3. apply_stochastic_dephasing with p_error=1.0 is fully deterministic
       (every qubit dephases on every call, no randomness left) -- verified
       against a hand-computed Z-parity phase pattern, an independent
       reference not reusing the function's own bit-masking code.
    """
    k0 = 0.0
    sv = zne_mitigation.generate_bloch_state(k0)

    e_direct = zne_mitigation.calcola_aspettazione_hamiltoniana(sv)
    e_via_shots = zne_mitigation.measure_energy_with_shots(k0, noise_scale=0.0, base_seed=0)
    assert e_direct == e_via_shots, (
        f"noise_scale=0.0 should short-circuit to the exact ideal energy: "
        f"{e_direct} != {e_via_shots}"
    )

    assert e_direct == pytest.approx(-2.0 * zne_mitigation.t_hopping, abs=1e-9), (
        f"E(k=0) should hit the exact ground-state bound -2*t_hopping "
        f"= {-2.0 * zne_mitigation.t_hopping}, got {e_direct}"
    )

    sv_dephased = zne_mitigation.apply_stochastic_dephasing(sv, p_error=1.0, seed=123)
    n = zne_mitigation.N_Q
    dim = 1 << n
    idx = np.arange(dim)
    expected_phase = np.ones(dim, dtype=np.float64)
    for q in range(n):
        expected_phase *= np.where((idx & (1 << q)) >> q == 1, -1.0, 1.0)
    expected = np.asarray(sv) * expected_phase
    assert np.allclose(sv_dephased, expected, atol=1e-12), (
        "apply_stochastic_dephasing(p_error=1.0) must deterministically "
        "apply a Z-dephasing phase on every qubit"
    )


# ── scan_ising.py: real functions cross-validated against PennyLane ─────────

_N_Q_ISING = scan_ising.N_Q
_dev_ising = qml.device("default.qubit", wires=_N_Q_ISING)
_OBS_ZZ_ISING = sum(
    qml.PauliZ(q) @ qml.PauliZ(q + 1) for q in range(_N_Q_ISING - 1)
) / (_N_Q_ISING - 1)


def _pl_ising_zz(g: float) -> float:
    @qml.qnode(_dev_ising)
    def circ():
        for q in range(_N_Q_ISING - 1):
            qml.CNOT(wires=[q, q + 1])
            qml.RZ(1.2, wires=q + 1)
            qml.CNOT(wires=[q, q + 1])
        for q in range(_N_Q_ISING):
            qml.RX(float(g * 0.6), wires=q)
        return qml.expval(_OBS_ZZ_ISING)
    return float(circ())


@pytest.mark.parametrize("g", [0.0, 0.8, 1.6, 2.5])
def test_scan_ising_correlazione_zz_matches_pennylane(g):
    """Calls the REAL esegui_circuito_ising_reale + calcola_vera_correlazione_zz
    from scan_ising.py (not hand-copied versions) and cross-checks against
    an independent PennyLane circuit of the same ansatz, at the script's
    actual N_Q=12 (not a smaller stand-in scale)."""
    prob = scan_ising.esegui_circuito_ising_reale(g)
    e_de = scan_ising.calcola_vera_correlazione_zz(prob)
    e_pl = _pl_ising_zz(g)
    assert abs(e_de - e_pl) < 1e-9, (
        f"g={g}: dense-evolution={e_de:.10f}, pennylane={e_pl:.10f}, "
        f"diff={abs(e_de - e_pl):.2e}"
    )


# ── next_gen_silicon.py: real functions cross-validated against PennyLane ──

_N_Q_SI = next_gen_silicon.N_Q
_dev_si = qml.device("default.qubit", wires=_N_Q_SI)
_OBS_PERIODIC_XY_SI = sum(
    qml.PauliX(q) @ qml.PauliX((q + 1) % _N_Q_SI) + qml.PauliY(q) @ qml.PauliY((q + 1) % _N_Q_SI)
    for q in range(_N_Q_SI)
)


def _pl_total_kinetic(k: float) -> float:
    sv = next_gen_silicon.genera_stato_bloch_puro(k, _N_Q_SI)

    @qml.qnode(_dev_si)
    def circ():
        qml.StatePrep(sv, wires=range(_N_Q_SI))
        return qml.expval(_OBS_PERIODIC_XY_SI)
    return float(circ())


@pytest.mark.parametrize("k", [-2.5, -1.0, 0.0, 1.3, 2.8])
def test_next_gen_silicon_hopping_expectation_matches_pennylane(k):
    """Calls the REAL genera_stato_bloch_puro + compute_jordan_wigner_hopping_expectation
    from next_gen_silicon.py (not hand-copied versions), summed over the same
    periodic N_Q bonds the script itself sums over, cross-checked against
    an independent PennyLane observable on the same Bloch state."""
    sv = next_gen_silicon.genera_stato_bloch_puro(k, _N_Q_SI)
    total_de = sum(
        next_gen_silicon.compute_jordan_wigner_hopping_expectation(sv, q, _N_Q_SI)
        for q in range(_N_Q_SI)
    )
    total_pl = _pl_total_kinetic(k)
    assert abs(total_de - total_pl) < 1e-9, (
        f"k={k}: dense-evolution={total_de:.10f}, pennylane={total_pl:.10f}, "
        f"diff={abs(total_de - total_pl):.2e}"
    )
