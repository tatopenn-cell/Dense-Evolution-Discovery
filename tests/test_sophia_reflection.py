"""
Smoke test for scripts/sophia_reflection.py -- imports the real script
(guarded behind `if __name__ == "__main__":`, same pattern as
test_integration_smoke.py) and calls its real `run_trajectory` function
directly, at reduced size for CI speed (K=20 instead of 200, 4 points
instead of 16 -- the production run in the script itself uses the full
size; this only checks the pipeline is wired correctly and the physics
invariants hold, not the exact production numbers).
"""
import importlib.util
import pathlib
import sys

import jax
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


sophia_reflection = _import_script("sophia_reflection")


def test_trajectory_fidelities_are_valid_probabilities():
    rows = sophia_reflection.run_trajectory(
        base_p_sweep=np.linspace(0.05, 0.4, 4), k_trajectories=20, seed=1,
    )
    for row in rows:
        assert 0.0 <= row['raw_fidelity'] <= 1.0
        assert 0.0 <= row['corrected_fidelity'] <= 1.0


def test_higher_noise_gives_lower_raw_fidelity():
    # sanity: raw fidelity (no correction) must degrade monotonically-ish
    # with more depolarizing noise -- this is the physical invariant the
    # whole experiment depends on, not something to assume unchecked.
    rows = sophia_reflection.run_trajectory(
        base_p_sweep=np.array([0.05, 0.4]), k_trajectories=50, seed=2,
    )
    assert rows[0]['raw_fidelity'] > rows[1]['raw_fidelity']


def test_correction_improves_fidelity_on_average():
    rows = sophia_reflection.run_trajectory(
        base_p_sweep=np.linspace(0.05, 0.4, 5), k_trajectories=30, seed=3,
    )
    deltas = np.array([row['delta'] for row in rows])
    assert deltas.mean() > 0


def test_uhlmann_fidelity_core_gradient_is_finite_at_degenerate_eigenvalues():
    # Regression test for the dense-evolution 8.1.55 upgrade: this
    # script's own local eigh_degenerate_safe/uhlmann_fidelity_
    # degenerate_safe duplicate was removed once the same fix (JAX
    # issues #2311/#8732; Kasim, arXiv:2011.04366) shipped upstream in
    # dense_evolution.mitigation._uhlmann_fidelity_core. Confirms the
    # installed package actually has the fix (not silently still on an
    # older, NaN-at-degeneracy version) using this script's own import
    # path, sharpest case: the fully mixed state, all eigenvalues tied.
    from dense_evolution.mitigation.zne import _uhlmann_fidelity_core

    d = 4
    rho_mixed = jnp.eye(d, dtype=jnp.complex128) / d
    rng = np.random.default_rng(42)
    m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho_other = m @ m.conj().T
    rho_other = jnp.asarray(rho_other / np.trace(rho_other), dtype=jnp.complex128)

    grad = jax.grad(lambda a: jnp.real(_uhlmann_fidelity_core(a, rho_other)))(rho_mixed)
    assert not bool(jnp.any(jnp.isnan(grad)))
