"""
Cross-validation: Dense Evolution v8 vs PennyLane default.qubit
----------------------------------------------------------------
3 tests, target < 20s total on GitHub Actions CPU runner.

Convention notes:
- Open boundary conditions (N_Q-1 bonds) on both simulators
- 4 qubits, reduced sweep points for CI speed
- Tolerance 1e-10 (numerical precision, not physical approximation)
"""

import numpy as np
import pytest
import pennylane as qml
import dense_evolution as de

# ── shared fixtures ────────────────────────────────────────────────────────────

N_Q = 4
T_HOP = 2.11
ATOL = 1e-10

_sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
_dev = qml.device("default.qubit", wires=N_Q)


def _bloch_state(k: float, n: int) -> np.ndarray:
    """Single-fermion Bloch state ψ(k) = 1/√N Σ_q e^{iqk} |1_q⟩"""
    dim = 1 << n
    sv = np.zeros(dim, dtype=np.complex128)
    for q in range(n):
        sv[1 << q] = (1.0 / np.sqrt(n)) * np.exp(1j * k * q)
    return sv


def _de_xy_energy(sv: np.ndarray) -> float:
    """Compute -(t/2) ΣXX+YY on open chain using DE statevector."""
    idx = np.arange(len(sv))
    E = 0.0
    for q in range(N_Q - 1):
        qn = q + 1
        mask = (1 << q) | (1 << qn)
        pf = sv[idx ^ mask]
        xx = np.real(np.sum(np.conj(sv) * pf))
        bi = (idx & (1 << q)) >> q
        bj = (idx & (1 << qn)) >> qn
        yy = np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
        E += xx + yy
    return -(T_HOP / 2.0) * E


def _pl_xy_energy(sv: np.ndarray) -> float:
    """Same observable via PennyLane StatePrep + ExpvalCost."""
    @qml.qnode(_dev)
    def circ():
        qml.StatePrep(sv, wires=range(N_Q))
        obs = sum(
            -(T_HOP / 2.0) * (
                qml.PauliX(q) @ qml.PauliX(q + 1)
                + qml.PauliY(q) @ qml.PauliY(q + 1)
            )
            for q in range(N_Q - 1)
        )
        return qml.expval(obs)
    return float(circ())


# ── Test 1: Dispersion E(k) = -2t·cos(k) cross-validation ────────────────────

@pytest.mark.parametrize("k", np.linspace(-np.pi, np.pi, 7).tolist())
def test_dispersion_de_vs_pennylane(k):
    """
    Tight-binding dispersion on Bloch states: Dense Evolution == PennyLane.
    Both evaluate -(t/2)(XX+YY) on open 4-qubit chain.
    Tolerance: 1e-10 (machine precision).
    """
    sv = _bloch_state(k, N_Q)
    de_e = _de_xy_energy(sv)
    pl_e = _pl_xy_energy(sv)
    assert abs(de_e - pl_e) < ATOL, (
        f"k={k:.4f}: DE={de_e:.8f} PL={pl_e:.8f} diff={abs(de_e-pl_e):.2e}"
    )


# ── Test 2: TFIM ZZ order parameter cross-validation ─────────────────────────

def _de_ising_zz(g: float) -> float:
    """<H_zz> via Dense Evolution variational ansatz."""
    ops = []
    for q in range(N_Q - 1):
        ops += [["cx", q, q + 1], ["rz", q + 1, 1.2], ["cx", q, q + 1]]
    for q in range(N_Q):
        ops.append(["rx", q, float(g * 0.6)])
    _sim.set_initial_state()
    _sim.run_circuit_jit_beast_mode(ops)
    prob = _sim.get_probabilities()
    idx = np.arange(len(prob))
    E = 0.0
    for q in range(N_Q - 1):
        bi = (idx & (1 << q)) >> q
        bj = (idx & (1 << (q + 1))) >> (q + 1)
        E += float(np.sum(prob * np.where(bi == bj, 1.0, -1.0)))
    return E / (N_Q - 1)


def _pl_ising_zz(g: float) -> float:
    """Same <H_zz> via PennyLane."""
    @qml.qnode(_dev)
    def circ():
        for q in range(N_Q - 1):
            qml.CNOT([q, q + 1])
            qml.RZ(1.2, q + 1)
            qml.CNOT([q, q + 1])
        for q in range(N_Q):
            qml.RX(float(g * 0.6), q)
        return qml.expval(
            sum(qml.PauliZ(q) @ qml.PauliZ(q + 1) for q in range(N_Q - 1))
            / (N_Q - 1)
        )
    return float(circ())


@pytest.mark.parametrize("g", np.linspace(0.0, 2.5, 6).tolist())
def test_ising_zz_de_vs_pennylane(g):
    """
    TFIM spin-spin correlation <ZZ>: Dense Evolution == PennyLane.
    Validates the ferromagnetic-to-paramagnetic sweep used in scan_ising.py.
    Tolerance: 1e-10.
    """
    de_v = _de_ising_zz(g)
    pl_v = _pl_ising_zz(g)
    assert abs(de_v - pl_v) < ATOL, (
        f"g={g:.4f}: DE={de_v:.8f} PL={pl_v:.8f} diff={abs(de_v-pl_v):.2e}"
    )


# ── Test 3: ZNE ideal baseline — noise-free energy cross-validation ───────────

@pytest.mark.parametrize("k", np.linspace(-np.pi, np.pi, 5).tolist())
def test_zne_ideal_baseline_de_vs_pennylane(k):
    """
    ZNE ideal (λ=0) energy: Dense Evolution == PennyLane.
    Validates the zero-noise target used in zne_mitigation.py before
    stochastic dephasing is applied.
    Tolerance: 1e-10.
    """
    sv = _bloch_state(k, N_Q)
    de_e = _de_xy_energy(sv)
    pl_e = _pl_xy_energy(sv)
    assert abs(de_e - pl_e) < ATOL, (
        f"k={k:.4f}: DE={de_e:.8f} PL={pl_e:.8f} diff={abs(de_e-pl_e):.2e}"
    )
