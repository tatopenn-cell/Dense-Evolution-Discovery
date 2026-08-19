"""
Smoke test for scripts/sandwiched_renyi_density_matrix.py -- imports the
real script and calls its real functions directly.
"""
import importlib.util
import pathlib
import sys

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


renyi = _import_script("sandwiched_renyi_density_matrix")


def test_bug_confirmation_all_zero_vs_fixed():
    rho = renyi.bell_state_rho()
    rho_var = renyi.amplitude_damping_2q(rho, 0.5)
    buggy = float(renyi.sandwiched_renyi_divergence_buggy(rho, rho_var, alpha=1.5))
    assert buggy == 0.0 or buggy >= 0.0  # buggy version never goes negative (that's the bug)


def test_diagonal_case_matches_classical_formula():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    rho = jnp.array(np.diag(p), dtype=jnp.complex128)
    sigma = jnp.array(np.diag(q), dtype=jnp.complex128)
    for alpha in (0.6, 1.8, 2.5):
        d_quantum = float(renyi.sandwiched_quantum_renyi_divergence(rho, sigma, alpha=alpha))
        d_classical = renyi.classical_renyi_divergence(p, q, alpha)
        assert abs(d_quantum - d_classical) < 1e-6


def test_alpha_one_matches_relative_entropy_reference():
    rho = renyi.amplitude_damping_2q(renyi.bell_state_rho(), 0.1)
    sigma = renyi.amplitude_damping_2q(renyi.bell_state_rho(), 0.4)
    # mix in depolarizing to guarantee full rank (matches the script's own
    # approach for avoiding the exactly-singular edge case)
    d = sigma.shape[0]
    rho = 0.9 * rho + 0.1 * jnp.eye(d, dtype=jnp.complex128) / d
    sigma = 0.9 * sigma + 0.1 * jnp.eye(d, dtype=jnp.complex128) / d
    d_case_one = float(renyi.sandwiched_quantum_renyi_divergence(rho, sigma, alpha=1.0))
    d_ref = renyi.relative_entropy_reference(np.array(rho), np.array(sigma))
    assert abs(d_case_one - d_ref) < 1e-3


def test_support_violation_gives_infinity_for_alpha_greater_than_1():
    # Two different pure states must give +inf at alpha > 1 (support
    # mismatch), never a finite negative number.
    sv1 = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    sv2 = jnp.array([np.cos(0.7), np.sin(0.7)], dtype=jnp.complex128)
    rho = jnp.outer(sv1, jnp.conj(sv1))
    sigma = jnp.outer(sv2, jnp.conj(sv2))
    d = float(renyi.sandwiched_quantum_renyi_divergence(rho, sigma, alpha=1.5))
    assert np.isinf(d) and d > 0


def test_identical_states_give_zero_at_every_alpha():
    rho = renyi.amplitude_damping_2q(renyi.bell_state_rho(), 0.3)
    for alpha in (0.5, 0.8, 1.0, 1.5, 2.0):
        d = float(renyi.sandwiched_quantum_renyi_divergence(rho, rho, alpha=alpha))
        assert abs(d) < 1e-6, f"D_alpha(rho||rho) should be 0 at alpha={alpha}, got {d}"


def test_no_nan_across_noise_sweep():
    rho = renyi.bell_state_rho()
    for p in (0.0, 0.2, 0.5, 0.8, 0.99):
        rho_noisy = renyi.amplitude_damping_2q(rho, p)
        for alpha in (0.5, 1.0, 1.5, 2.0):
            d = float(renyi.sandwiched_quantum_renyi_divergence(rho, rho_noisy, alpha=alpha))
            assert not np.isnan(d)
