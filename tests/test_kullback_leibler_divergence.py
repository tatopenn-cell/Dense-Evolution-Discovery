"""
Smoke test for scripts/kullback_leibler_divergence.py -- imports the real
script and calls its real kl_divergence function directly.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.stats import entropy as scipy_entropy

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kld = _import_script("kullback_leibler_divergence")


def test_matches_scipy_entropy_reference():
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(2, 8)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        mine = kld.kl_divergence(p, q)
        ref_bits = scipy_entropy(p, q) / np.log(2.0)
        assert abs(mine - ref_bits) < 1e-9


def test_gibbs_inequality_never_negative():
    rng = np.random.default_rng(43)
    for _ in range(200):
        n = rng.integers(2, 10)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        assert kld.kl_divergence(p, q) >= -1e-9


def test_zero_at_equality():
    p = np.array([0.5, 0.3, 0.2])
    assert kld.kl_divergence(p, p) < 1e-12


def test_asymmetric_in_general():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    d_pq = kld.kl_divergence(p, q)
    d_qp = kld.kl_divergence(q, p)
    assert abs(d_pq - d_qp) > 0.1


def test_support_violation_gives_infinity():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.5, 0.0, 0.5])
    assert np.isinf(kld.kl_divergence(p, q))


def test_zero_mass_in_p_does_not_force_infinity():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.3, 0.3, 0.4])
    d = kld.kl_divergence(p, q)
    assert np.isfinite(d) and d > 0


def test_full_experiment_script_runs_end_to_end():
    p_ideal, divergences, noise_levels = kld.part4_real_use_measurement_distributions()
    assert divergences == sorted(divergences)
    kl_values, healing_values = kld.part5_compare_against_healing_scalar_on_the_same_states()
    assert len(kl_values) == len(healing_values) == 5
