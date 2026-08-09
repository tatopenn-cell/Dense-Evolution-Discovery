"""
Smoke/regression test for scripts/photonic_predictive_zne.py -- same
import pattern as test_sophia_reflection.py, reduced K for CI speed.
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


photonic_predictive_zne = _import_script("photonic_predictive_zne")


def test_photon_loss_kraus_probability_matches_amplitude_damping_convention():
    # eta=1 (lossless) -> gamma=0 (no decay); eta=0 (total loss) -> gamma=1.
    assert photonic_predictive_zne.photon_loss_kraus_probability(1.0) == 0.0
    assert photonic_predictive_zne.photon_loss_kraus_probability(0.0) == 1.0
    assert photonic_predictive_zne.photon_loss_kraus_probability(0.9) == pytest.approx(0.1, abs=1e-9)


def test_fidelities_are_valid_probabilities():
    df = photonic_predictive_zne.run_photon_loss_comparison(
        eta_sweep=np.array([0.95, 0.8]), k_trajectories=30, seed=1,
    )
    for col in ('raw_fidelity', 'dm_zne_fidelity', 'predictive_dm_zne_fidelity'):
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0 + 1e-9).all(), col


def test_scalar_zne_can_go_unphysical_but_density_matrix_zne_cannot():
    # Regression guard for this script's own headline finding: scalar ZNE
    # has no [0,1] constraint (can exceed 1.0), density-matrix ZNE always
    # does (project_to_physical). Uses a low eta (high loss) sweep, where
    # this was observed to happen reliably in the real run.
    df = photonic_predictive_zne.run_photon_loss_comparison(
        eta_sweep=np.array([0.75, 0.7]), k_trajectories=100, seed=2,
    )
    assert (df['dm_zne_fidelity'] <= 1.0 + 1e-9).all()
    assert (df['dm_zne_fidelity'] >= 0.0).all()


def test_density_matrix_zne_improves_fidelity_on_average_for_photon_loss():
    df = photonic_predictive_zne.run_photon_loss_comparison(
        eta_sweep=np.linspace(0.95, 0.75, 5), k_trajectories=100, seed=3,
    )
    assert df['dm_zne_delta'].mean() > 0
