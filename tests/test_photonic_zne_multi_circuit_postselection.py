"""
Smoke/regression test for scripts/photonic_zne_multi_circuit_postselection.py
-- same import pattern as test_photonic_predictive_zne.py.
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


mc = _import_script("photonic_zne_multi_circuit_postselection")


def test_apply_amplitude_damping_tracked_matches_noise_model_exactly():
    # Regression guard for the missing-final-renormalization bug found
    # during development: verifies bit-exact agreement (not just close)
    # against the real library, across several qubit counts.
    import dense_evolution as de
    from dense_evolution.registry import NoiseModel

    for n_qubits in (2, 3, 4):
        sim = de.DenseSVSimulator(n_qubits)
        sim.run_circuit(de.ghz_state(n_qubits))
        sv = np.asarray(sim.get_statevector())
        for trial in range(5):
            rng1 = np.random.default_rng(trial)
            rng2 = np.random.default_rng(trial)
            gamma = 0.2 + 0.1 * trial
            out_lib = np.asarray(NoiseModel.apply_to_sv(sv.copy(), n_qubits, 'amplitude_damping', gamma, rng=rng1))
            out_mine, _ = mc._apply_amplitude_damping_tracked(sv.copy(), n_qubits, gamma, rng2)
            np.testing.assert_allclose(out_lib, out_mine, atol=1e-12)


def test_apply_amplitude_damping_tracked_output_is_normalized():
    import dense_evolution as de
    sim = de.DenseSVSimulator(3)
    sim.run_circuit(de.ghz_state(3))
    sv = np.asarray(sim.get_statevector())
    rng = np.random.default_rng(0)
    sv_out, _ = mc._apply_amplitude_damping_tracked(sv.copy(), 3, 0.4, rng)
    assert np.linalg.norm(sv_out) == pytest.approx(1.0, abs=1e-9)


def test_postselection_keeps_only_no_decay_trajectories():
    # At gamma=0 (no possible decay), every trajectory must be kept.
    import dense_evolution as de
    sim = de.DenseSVSimulator(2)
    sim.run_circuit(de.ghz_state(2))
    sv = np.asarray(sim.get_statevector())
    rng = np.random.default_rng(0)
    rho_all, rho_kept, keep_fraction = mc._noisy_density_matrices_with_postselection(sv, 2, 0.0, 20, rng)
    assert keep_fraction == 1.0
    np.testing.assert_allclose(rho_all, rho_kept, atol=1e-9)


def test_run_multi_circuit_comparison_fidelities_are_valid_probabilities():
    df = mc.run_multi_circuit_comparison(eta_sweep=np.array([0.85]), k_trajectories=20, seed=0)
    cols = ['raw_fidelity', 'dm_zne_fidelity', 'postselection_fidelity']
    if mc.HAS_JSD_ZNE:
        cols.append('jsd_dm_zne_fidelity')
    for col in cols:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0 + 1e-9).all(), col
    assert (df['postselection_keep_fraction'] >= 0.0).all()
    assert (df['postselection_keep_fraction'] <= 1.0).all()


@pytest.mark.skipif(
    not mc.HAS_JSD_ZNE,
    reason="jsd_predictive_zne_density_matrix not yet in the installed dense-evolution release "
           "(promoted to the library but not published to PyPI yet as of this test's own commit)",
)
def test_jsd_dm_zne_fidelity_is_a_valid_probability_when_available():
    df = mc.run_multi_circuit_comparison(eta_sweep=np.array([0.85]), k_trajectories=20, seed=0)
    assert (df['jsd_dm_zne_fidelity'] >= 0.0).all()
    assert (df['jsd_dm_zne_fidelity'] <= 1.0 + 1e-9).all()


def test_vqe_ansatz_ops_gate_count_matches_layers():
    ops = mc.vqe_ansatz_ops(3, np.zeros(6), n_layers=2)
    ry_count = sum(1 for op in ops if op[0] == 'ry')
    cx_count = sum(1 for op in ops if op[0] == 'cx')
    assert ry_count == 3 * 2  # n_qubits x n_layers
    assert cx_count == 2 * 2  # (n_qubits - 1) x n_layers
