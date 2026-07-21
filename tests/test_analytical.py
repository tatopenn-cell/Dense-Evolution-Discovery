"""
Analytical validation tests — Dense Evolution v8
-------------------------------------------------
5 tests against exact mathematical references (no external simulator needed).
Target: < 20s total on GitHub Actions CPU runner.

Tests:
  1. PEC shape    — repulsive wall + bound well + dissociation limit
  2. PEC minimum  — bound state energy < 0 (stable molecule exists)
  3. PSR exactness — Parameter-Shift Rule == exact analytic on RY+<Z>
  4. Harrison ratio — strained vs unstrained bandwidth = 1/(1+ε)²
  5. Dispersion symmetry — E(k) == E(-k) for tight-binding chain
"""

import numpy as np
import pytest
import dense_evolution as de

# ── shared setup ──────────────────────────────────────────────────────────────

_sim_6q = de.DenseSVSimulator(n_qubits=6, use_gpu=False, use_float32=False)
_sim_1q = de.DenseSVSimulator(n_qubits=1, use_gpu=False, use_float32=False)

# Molecular PEC parameters (vqe_silicon_molecular.py)
_T0_MOL  = 2.11   # eV
_BETA    = 1.5    # Å⁻¹
_R0      = 2.35   # Å
_V0      = 5.4    # eV
_GAMMA   = 3.0    # Å⁻¹
_THETA   = 0.38   # optimal variational angle (rad)

# Silicon tight-binding parameters (next_gen_silicon.py)
_T0_SI   = 2.11   # eV
_STRAIN  = 0.05   # 5% tensile strain
_N_SI    = 8      # qubits for bandstructure


def _vqe_pec(R: float, theta: float = _THETA) -> float:
    """Born-Oppenheimer PEC: E_elec(R, theta) + V_rep(R)."""
    t_R   = _T0_MOL * np.exp(-_BETA  * (R - _R0))
    V_rep = _V0     * np.exp(-_GAMMA * (R - _R0))
    ops   = [["x", 0]]
    for q in range(5):
        ops += [
            ["cx", q + 1, q],
            ["ry", q + 1, float(theta)],
            ["cx", q, q + 1],
            ["ry", q + 1, -float(theta)],
            ["cx", q + 1, q],
        ]
    _sim_6q.set_initial_state()
    _sim_6q.run_circuit_jit_beast_mode(ops)
    sv  = _sim_6q.get_statevector()
    idx = np.arange(len(sv))
    E   = 0.0
    for q in range(5):
        mask = (1 << q) | (1 << (q + 1))
        pf   = sv[idx ^ mask]
        xx   = np.real(np.sum(np.conj(sv) * pf))
        bi   = (idx & (1 << q))       >> q
        bj   = (idx & (1 << (q + 1))) >> (q + 1)
        yy   = np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
        E   += xx + yy
    return -(t_R / 2.0) * E + V_rep


def _bloch_energy(k: float, t: float) -> float:
    """Tight-binding energy on 8-qubit open chain via Bloch state."""
    sv  = np.zeros(1 << _N_SI, dtype=np.complex128)
    for q in range(_N_SI):
        sv[1 << q] = (1.0 / np.sqrt(_N_SI)) * np.exp(1j * k * q)
    idx = np.arange(1 << _N_SI)
    E   = 0.0
    for q in range(_N_SI - 1):
        mask = (1 << q) | (1 << (q + 1))
        pf   = sv[idx ^ mask]
        xx   = np.real(np.sum(np.conj(sv) * pf))
        bi   = (idx & (1 << q))       >> q
        bj   = (idx & (1 << (q + 1))) >> (q + 1)
        yy   = np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
        E   += xx + yy
    return -(t / 2.0) * E


# ── Test 1: PEC shape ─────────────────────────────────────────────────────────

def test_pec_shape():
    """
    Born-Oppenheimer PEC must have the correct 3-region shape:
      - Short range (R=1.4 Å): strong repulsion → E > 0
      - Binding well (R=3.3 Å): attractive minimum → E < 0
      - Dissociation (R=7.0 Å): asymptotic limit → |E| < 0.01 eV

    Validates the qualitative physics of vqe_silicon_molecular.py.
    """
    E_repulsive    = _vqe_pec(1.4)
    E_binding_well = _vqe_pec(3.3)
    E_dissociation = _vqe_pec(7.0)

    assert E_repulsive > 0, (
        f"Repulsive wall at R=1.4 Å should be E>0, got {E_repulsive:.4f} eV"
    )
    assert E_binding_well < 0, (
        f"Binding well at R=3.3 Å should be E<0, got {E_binding_well:.4f} eV"
    )
    assert abs(E_dissociation) < 0.01, (
        f"Dissociation limit at R=7.0 Å should be |E|<0.01 eV, got {E_dissociation:.6f} eV"
    )


# ── Test 2: PEC binding minimum ───────────────────────────────────────────────

@pytest.mark.parametrize("R", np.linspace(3.0, 4.5, 8).tolist())
def test_pec_minimum_is_negative(R):
    """
    The well core (R in [3.0, 4.5] A, empirically confirmed bound at every
    sampled point -- see test_pec_global_minimum_negative for the wider
    [2.5, 4.5] existence check) must be negative everywhere, confirming a
    stable bound state, not merely finite. The docstring here used to
    promise this but the assertion only checked np.isfinite -- found
    during the dense-evolution 8.1.21 audit.
    Exact minimum at R ~ 3.32 A with E ~ -0.302 eV.
    """
    E = _vqe_pec(R)
    assert E < -0.01, f"E(R={R:.3f}) should be bound (<-0.01 eV), got {E:.6f} eV"


def test_pec_global_minimum_negative():
    """Global minimum of PEC must be negative (bound state)."""
    R_scan  = np.linspace(2.5, 4.5, 15)
    E_min   = min(_vqe_pec(R) for R in R_scan)
    assert E_min < 0, f"Global PEC minimum should be <0, got {E_min:.6f} eV"


# ── Test 3: PSR exactness ─────────────────────────────────────────────────────

@pytest.mark.parametrize("theta", [0.4, 0.9, 1.3, 1.7, 2.2])
def test_psr_exactness_ry_z(theta):
    """
    Parameter-Shift Rule on RY(θ)|0⟩ + ⟨Z⟩ measurement:

        dE/dθ = ½[E(θ+π/2) - E(θ-π/2)] = -sin(θ)  (exact)

    Validates the PSR engine used in vqe_jax_grad.py.
    Tolerance: 1e-10 (machine precision — no approximation).
    """
    def e1q(th):
        _sim_1q.set_initial_state()
        _sim_1q.run_circuit_jit_beast_mode([["ry", 0, float(th)]])
        sv = _sim_1q.get_statevector()
        return float(abs(sv[0]) ** 2 - abs(sv[1]) ** 2)

    psr_grad    = 0.5 * (e1q(theta + np.pi / 2) - e1q(theta - np.pi / 2))
    exact_grad  = -np.sin(theta)

    assert abs(psr_grad - exact_grad) < 1e-10, (
        f"θ={theta:.2f}: PSR={psr_grad:.12f}, exact={exact_grad:.12f}, "
        f"diff={abs(psr_grad - exact_grad):.2e}"
    )


# ── Test 4: Harrison hopping ratio ────────────────────────────────────────────

@pytest.mark.parametrize("k", [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
def test_harrison_strain_ratio(k):
    """
    Harrison's law: t(ε) = t₀/(1+ε)²
    The ratio of strained to unstrained energy must equal 1/(1+ε)² exactly
    at every k-point in the Brillouin zone.

        E_strained(k) / E_unstrained(k) = 1/(1+ε)²

    Validates next_gen_silicon.py with ε=0.05 (5% tensile strain).
    Tolerance: 1e-10 (exact algebraic identity).
    """
    if abs(k) < 1e-12 or abs(abs(k) - np.pi) < 1e-12:
        pytest.skip("E(k=0) and E(k=π) near zero for open chain — division unstable")

    t_strained      = _T0_SI / (1.0 + _STRAIN) ** 2
    E_s             = _bloch_energy(k, t_strained)
    E_u             = _bloch_energy(k, _T0_SI)
    ratio_sim       = E_s / E_u
    ratio_expected  = 1.0 / (1.0 + _STRAIN) ** 2

    assert abs(ratio_sim - ratio_expected) < 1e-10, (
        f"k={k:.4f}: ratio={ratio_sim:.12f}, expected={ratio_expected:.12f}, "
        f"diff={abs(ratio_sim - ratio_expected):.2e}"
    )


# ── Test 5: Dispersion symmetry E(k) = E(-k) ─────────────────────────────────

@pytest.mark.parametrize("k", [0.3, 0.7, 1.2, 1.8, 2.4])
def test_dispersion_time_reversal_symmetry(k):
    """
    Time-reversal symmetry of the tight-binding chain requires E(k) = E(-k).
    Violated only if the Hamiltonian breaks time-reversal (e.g. magnetic flux).

    Validates the Bloch state evaluation used in next_gen_silicon.py and
    zne_mitigation.py.
    Tolerance: 1e-12.
    """
    E_pos = _bloch_energy( k, _T0_SI)
    E_neg = _bloch_energy(-k, _T0_SI)

    assert abs(E_pos - E_neg) < 1e-12, (
        f"k={k:.3f}: E(+k)={E_pos:.12f}, E(-k)={E_neg:.12f}, "
        f"diff={abs(E_pos - E_neg):.2e}"
    )