"""
Manufacturing thermodynamics validation — Debye phonon model
---------------------------------------------------------------
4 tests against exact/closed-form references (no external simulator needed,
no CSV/plot I/O). Mirrors the physics of manufacturing_thermodynamics.py.
Target: < 5s total on GitHub Actions CPU runner.

Tests:
  1. Kinetic expectation closed form — <XX+YY> on a Bloch state == (N/2)*cos(k)
  2. Bose-Einstein occupancy monotonic in T, bounded in [0, 1]
  3. Effective hopping monotonically decreasing in T, stays positive
  4. Low-temperature freeze-out — n_bose -> 0, t_effettivo -> bare hopping
"""

import numpy as np
import pytest

# Thermodynamics parameters (manufacturing_thermodynamics.py)
N_Q = 8
HBAR_OMEGA = 0.032
KB = 8.617333e-5
T0 = 2.11        # bare hopping (eV), matches _T0_SI/_T0_MOL convention
ALPHA_SCATTER = 0.15
T_SWEEP = np.linspace(10, 400, 3500)


def _generate_bloch_state(k_val: float) -> np.ndarray:
    dim = 1 << N_Q
    state = np.zeros(dim, dtype=np.complex128)
    for q in range(N_Q):
        state[1 << q] = (1.0 / np.sqrt(N_Q)) * np.exp(1j * k_val * q)
    return state


def _calcola_aspettazione_hamiltoniana(statevector: np.ndarray) -> float:
    dim = len(statevector)
    indices = np.arange(dim)
    total_kinetic = 0.0
    for q in range(N_Q):
        q_next = (q + 1) % N_Q
        mask = (1 << q) | (1 << q_next)
        psi_flipped = statevector[indices ^ mask]
        xx_exp = np.real(np.sum(np.conj(statevector) * psi_flipped))
        bit_i = (indices & (1 << q)) >> q
        bit_j = (indices & (1 << q_next)) >> q_next
        phase = np.where(bit_i == bit_j, -1.0, 1.0)
        yy_exp = np.real(np.sum(np.conj(statevector) * psi_flipped * phase))
        total_kinetic += float(xx_exp + yy_exp)
    return total_kinetic


def _n_bose(temp_k):
    return 1.0 / (np.exp(HBAR_OMEGA / (KB * temp_k)) - 1.0)


def _t_effettivo(n_bose):
    return T0 * (1.0 - ALPHA_SCATTER * n_bose)


@pytest.mark.parametrize("k", [0.0, np.pi / 4, np.pi / 2, np.pi])
def test_kinetic_expectation_closed_form(k):
    sv = _generate_bloch_state(k)
    total_kinetic = _calcola_aspettazione_hamiltoniana(sv)
    assert total_kinetic == pytest.approx((N_Q / 2.0) * np.cos(k), abs=1e-10)


def test_bose_einstein_occupancy_monotonic():
    n_bose = _n_bose(T_SWEEP)
    assert np.all(np.diff(n_bose) >= 0)
    assert n_bose.min() >= 0.0
    assert n_bose.max() < 1.0


def test_effective_hopping_stays_positive_and_decreasing():
    n_bose = _n_bose(T_SWEEP)
    t_eff = _t_effettivo(n_bose)
    assert np.all(np.diff(t_eff) <= 0)
    assert t_eff.min() > 0.0
    assert t_eff.max() <= T0


def test_low_temperature_freeze_out():
    n_bose_low = _n_bose(T_SWEEP[0])
    t_eff_low = _t_effettivo(n_bose_low)
    assert n_bose_low == pytest.approx(0.0, abs=1e-6)
    assert t_eff_low == pytest.approx(T0, abs=1e-6)
