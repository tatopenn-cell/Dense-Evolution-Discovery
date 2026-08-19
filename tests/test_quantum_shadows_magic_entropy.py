"""
Smoke test for scripts/quantum_shadows_magic_entropy.py -- imports the
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


shadows = _import_script("quantum_shadows_magic_entropy")


def test_shadow_snapshots_are_unbiased_on_average():
    psi = shadows.t_state()
    rho = jnp.outer(psi, jnp.conj(psi))
    snaps = shadows.sample_shadow_snapshots(psi, 100_000, seed=42)
    empirical_mean = jnp.mean(snaps, axis=0)
    assert float(jnp.max(jnp.abs(empirical_mean - rho))) < 0.03


def test_einsum_bug_is_confirmed_and_fix_matches_true_trace():
    buggy, fixed, true_val = shadows.purity_bug_verification()
    assert abs(fixed - true_val) < 1e-9
    assert abs(buggy - true_val) > 1.0


def test_fixed_purity_estimator_converges_for_pure_state():
    psi = shadows.t_state()
    snaps = shadows.sample_shadow_snapshots(psi, 60_000, seed=7)
    p = shadows.estimate_purity_fixed(snaps)
    assert abs(p - 1.0) < 0.1


def test_o_operators_reproduce_exact_reduced_matrix():
    # With EXACT (noiseless) copies of rho fed through the same linear
    # construction used by the shadow estimator, R built from the O_ab
    # operators must exactly match exact_self_convolve_3's reduced matrix.
    psi = shadows.t_state()
    rho = jnp.outer(psi, jnp.conj(psi))
    rho3 = jnp.kron(jnp.kron(rho, rho), rho)
    r_exact = shadows.exact_self_convolve_3(rho)
    for (a, b), o_ab in shadows._O_AB.items():
        val = complex(jnp.trace(o_ab @ rho3))
        assert abs(val - complex(r_exact[a, b])) < 1e-9


def test_shadow_magic_entropy_converges_for_t_state():
    psi = shadows.t_state()
    rho = jnp.outer(psi, jnp.conj(psi))
    m_exact = shadows.exact_magic_entropy(rho)
    snaps = shadows.sample_shadow_snapshots(psi, 150_000, seed=3)
    m_hat = shadows.estimate_magic_entropy_from_shadows(snaps)
    assert abs(m_hat - m_exact) < 0.15


def test_shadow_magic_entropy_near_zero_for_stabilizer_state():
    psi = shadows.plus_state()
    snaps = shadows.sample_shadow_snapshots(psi, 150_000, seed=4)
    m_hat = shadows.estimate_magic_entropy_from_shadows(snaps)
    assert m_hat < 0.15
