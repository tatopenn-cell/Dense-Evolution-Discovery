"""
Smoke test for scripts/channel_order_noncommutativity.py -- imports the
real script (guarded behind `if __name__ == "__main__":`, same pattern as
test_integration_smoke.py) and calls its real `run_trial` function
directly, at reduced size for CI speed.
"""
import importlib.util
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


channel_order = _import_script("channel_order_noncommutativity")


def test_distributions_are_valid_probabilities():
    dist_ab, dist_ba, observed_js, null_js, p_value = channel_order.run_trial(
        k_trajectories=500, n_permutations=30, seed=1,
    )
    assert abs(dist_ab.sum() - 1.0) < 1e-9
    assert abs(dist_ba.sum() - 1.0) < 1e-9
    assert (dist_ab >= 0).all() and (dist_ba >= 0).all()


def test_js_divergence_is_nonnegative():
    dist_ab, dist_ba, observed_js, null_js, p_value = channel_order.run_trial(
        k_trajectories=500, n_permutations=30, seed=2,
    )
    assert observed_js >= 0
    assert (null_js >= 0).all()
    assert 0.0 <= p_value <= 1.0


def test_same_order_gives_zero_divergence():
    # sanity: comparing "AB" against itself (same random draws) must give
    # exactly zero JS divergence -- confirms the metric and sampling
    # pipeline aren't silently injecting a difference from nothing.
    import numpy as np
    sv0 = channel_order.ideal_sv()
    rng = np.random.default_rng(3)
    outcomes = [o for o in (channel_order.sample_outcome(sv0, 'AB', rng) for _ in range(500)) if o is not None]
    dist = channel_order.empirical_dist(np.array(outcomes))
    assert channel_order.js_divergence(dist, dist) < 1e-12
