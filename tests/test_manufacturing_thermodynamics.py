"""
Real-implementation tests for scripts/manufacturing_thermodynamics.py --
same import pattern as test_sophia_reflection.py and
test_photonic_predictive_zne.py.

Rewritten 2026-08-10 alongside the script itself: the previous version
of this test file reimplemented the OLD formulas locally and never
imported the actual script at all, so it validated a hand-copied
snapshot of the physics rather than the real module -- and the module
it snapshotted never actually used its own quantum simulator or any
real decoherence process (see the script's own docstring). These tests
exercise the real, current implementation directly.
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


mt = _import_script("manufacturing_thermodynamics")


@pytest.mark.parametrize("k", [0.0, np.pi / 4, np.pi / 2, np.pi])
def test_hamiltonian_matrix_matches_closed_form_at_zero_dephasing(k):
    # Closed-form reference: <psi(k)|XX+YY|psi(k)> = (N/2)*cos(k) for the
    # ideal (undecohered) Bloch state -- the same check the old test
    # file had, generalized here to the new matrix-based Tr(rho @ H)
    # formula, at p=0 (should reduce exactly to the pure-state formula).
    import jax.numpy as jnp
    sv = mt.generate_bloch_state(k)
    rho = jnp.asarray(np.outer(sv, sv.conj()), dtype=jnp.complex128)
    H = mt.xy_hamiltonian_matrix()
    E = float(jnp.real(jnp.trace(rho @ H)))
    assert E == pytest.approx((mt.N_QUBITS / 2.0) * np.cos(k), abs=1e-10)


def test_bose_einstein_occupancy_monotonic_and_bounded():
    n_bar = mt.debye_bose_einstein_occupation(mt.T_SWEEP)
    assert np.all(np.diff(n_bar) >= 0)
    assert n_bar.min() >= 0.0


def test_dephasing_probability_bounded_and_monotonic():
    n_bar = mt.debye_bose_einstein_occupation(mt.T_SWEEP)
    p = mt.dephasing_probability(n_bar)
    assert np.all(np.diff(p) >= -1e-12)
    assert p.min() >= 0.0
    assert p.max() < 0.5  # a valid single-channel dephasing probability


def test_low_temperature_freeze_out():
    n_bar_low = mt.debye_bose_einstein_occupation(mt.T_SWEEP[0])
    assert n_bar_low == pytest.approx(0.0, abs=1e-6)


def test_apply_local_dephasing_exact_is_identity_at_zero_probability():
    import jax.numpy as jnp
    sv = mt.generate_bloch_state(np.pi / 4)
    rho = jnp.asarray(np.outer(sv, sv.conj()), dtype=jnp.complex128)
    rho_out = mt.apply_local_dephasing_exact(rho, 0.0)
    np.testing.assert_allclose(np.asarray(rho_out), np.asarray(rho), atol=1e-12)


def test_apply_local_dephasing_exact_output_is_a_valid_density_matrix():
    import jax.numpy as jnp
    sv = mt.generate_bloch_state(np.pi / 4)
    rho = jnp.asarray(np.outer(sv, sv.conj()), dtype=jnp.complex128)
    rho_out = np.asarray(mt.apply_local_dephasing_exact(rho, 0.3))
    np.testing.assert_allclose(rho_out, rho_out.conj().T, atol=1e-9)
    assert np.trace(rho_out).real == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.eigvalsh(rho_out).min() >= -1e-9


def test_thermal_sweep_energy_and_fidelity_decrease_with_temperature():
    # Real regression guard for the headline physics claim: with a
    # genuine dephasing channel (not a classical scalar formula),
    # coherent energy and fidelity with the ideal state must both
    # degrade monotonically as temperature (and thus phonon population)
    # rises. Reduced sweep for CI speed.
    df = mt.run_thermal_decoherence_sweep(T_sweep=np.linspace(10, 400, 12))
    assert np.all(np.diff(df["Fidelity"].values) < 1e-9)
    assert np.all(np.diff(df["Energia_eV"].values) < 1e-9)
    assert df["Fidelity"].iloc[0] > df["Fidelity"].iloc[-1]


def test_thermal_sweep_fidelities_are_valid_probabilities():
    df = mt.run_thermal_decoherence_sweep(T_sweep=np.linspace(10, 400, 6))
    assert (df["Fidelity"] >= 0.0).all()
    assert (df["Fidelity"] <= 1.0 + 1e-9).all()
