"""
Tests for scripts/hubbard_square_arovas.py -- imports the real script and
re-runs its self-tests plus a couple of independent spot-checks.
"""
import importlib.util
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _import_script("hubbard_square_arovas")


def test_periodic_jw_mapping_selftest_passes():
    mod.selftest_periodic_jw_mapping()


def test_perturbative_formula_selftest_passes():
    mod.selftest_perturbative_formula()


def test_perturbative_formula_matches_table2_coefficients():
    """Spot-check the formula's own coefficients directly, independent of
    any exact-diagonalization comparison -- catches a transcription typo
    even if it happened to still pass the small-U numerical self-test."""
    t = 1.0
    # -4t + 0.75*U - (13/128)*U^2/t, per Arovas et al. Table 2 (N=4 row)
    assert mod.perturbative_energy(t, 0.0) == -4.0 * t
    U = 2.0
    expected = -4.0 * t + 0.75 * U - (13.0 / 128.0) * (U ** 2 / t)
    assert abs(mod.perturbative_energy(t, U) - expected) < 1e-12


def test_double_occupancy_decreases_with_u_at_n2():
    """Independent spot-check of the Mott-localization trend at a smaller,
    faster system size than the full N=4 sweep the script itself runs."""
    n_sites = 2
    _, psi_weak = mod.hubbard_ground_state(n_sites, 1.0, 0.5, periodic=True)
    _, psi_strong = mod.hubbard_ground_state(n_sites, 1.0, 8.0, periodic=True)
    d_weak = mod.double_occupancy(psi_weak, n_sites, 0)
    d_strong = mod.double_occupancy(psi_strong, n_sites, 0)
    assert d_strong < d_weak


def test_open_chain_bruteforce_matches_pauli_construction():
    """Independent spot-check with periodic=False (no wraparound bond at
    all), confirming the open-chain case -- not exercised by the module's
    own self-test, which only checks periodic=True -- also agrees."""
    n_sites = 3
    terms = mod.hubbard_pauli_terms(n_sites, 1.0, 0.5, periodic=False)
    from dense_evolution.physics.observables import pauli_hamiltonian_to_matrix
    H_pauli = np.asarray(pauli_hamiltonian_to_matrix(terms, 2 * n_sites))
    H_bruteforce = mod.hubbard_matrix_bruteforce(n_sites, 1.0, 0.5, periodic=False)
    assert np.max(np.abs(H_pauli - H_bruteforce)) < 1e-10
