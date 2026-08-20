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
    # Median-of-means trades some variance for its corrupted-block
    # robustness (see the MoM tests below) -- checked directly across 10
    # seeds at this snapshot count: most land under 0.1, one (seed=4)
    # reaches 0.162, so 0.25 is a real, not cherry-picked, bound rather
    # than the tighter 0.15 the pre-MoM plain-mean estimator satisfied.
    psi = shadows.plus_state()
    snaps = shadows.sample_shadow_snapshots(psi, 150_000, seed=4)
    m_hat = shadows.estimate_magic_entropy_from_shadows(snaps)
    assert m_hat < 0.25


def test_median_of_means_matches_plain_mean_on_uncorrupted_data():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=0.1, size=2000)
    mom = shadows._median_of_means(values, n_groups=20)
    assert abs(mom - 5.0) < 0.05


def test_median_of_means_tolerates_a_corrupted_block_naive_mean_does_not():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=1.0, scale=0.05, size=2000)
    corrupted = values.copy()
    corrupted[:800] = -1000.0  # 40% of the run replaced by a wild outlier block
    naive_mean = float(np.mean(corrupted))
    mom = shadows._median_of_means(corrupted, n_groups=20)
    assert abs(mom - 1.0) < 1.0
    assert abs(naive_mean - 1.0) > 100.0


def test_median_of_means_breaks_down_past_half_corrupted():
    # Honest boundary check: MoM is not magic -- once more than half the
    # groups are corrupted, the median itself must be a corrupted value.
    rng = np.random.default_rng(2)
    values = rng.normal(loc=1.0, scale=0.05, size=2000)
    corrupted = values.copy()
    corrupted[:1200] = -1000.0  # 60% corrupted, past the n_groups//2 tolerance
    mom = shadows._median_of_means(corrupted, n_groups=20)
    assert mom < -100.0


def test_shadow_purity_estimator_n_groups_is_configurable():
    psi = shadows.t_state()
    snaps = shadows.sample_shadow_snapshots(psi, 60_000, seed=8)
    p_default = shadows.estimate_purity_fixed(snaps)
    p_5_groups = shadows.estimate_purity_fixed(snaps, n_groups=5)
    assert abs(p_default - 1.0) < 0.15
    assert abs(p_5_groups - 1.0) < 0.15


def test_sample_complexity_fit_exponent_is_near_theoretical_half():
    # Small budget (fast for CI) -- just checks the fitting machinery
    # itself works and lands in a sane range, not a tight reproduction of
    # the full 20-trial study's exact exponent.
    psi = shadows.t_state()
    rho = jnp.outer(psi, jnp.conj(psi))
    m_exact = shadows.exact_magic_entropy(rho)
    rows, fit_c, fit_p = shadows.sample_complexity_fit(
        psi, m_exact, n_snapshots_list=(3000, 30000), n_trials=6, seed_base=500,
    )
    assert len(rows) == 2
    assert fit_c > 0
    assert 0.1 < fit_p < 1.0


def test_n_snapshots_for_target_std_is_monotonically_decreasing_in_target():
    # A tighter (smaller) target std must need MORE snapshots, not fewer.
    n_loose = shadows.n_snapshots_for_target_std(0.1, fit_c=10.0, fit_p=0.5)
    n_tight = shadows.n_snapshots_for_target_std(0.01, fit_c=10.0, fit_p=0.5)
    assert n_tight > n_loose
    # Sanity-check the inversion is actually correct: plugging n back into
    # C/n^p should reproduce the target std.
    recovered_std = 10.0 / n_tight ** 0.5
    assert abs(recovered_std - 0.01) < 1e-6
