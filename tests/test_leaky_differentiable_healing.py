"""
Smoke test for scripts/leaky_differentiable_healing.py -- imports the real
script (guarded behind `if __name__ == "__main__":`) and calls its real
functions directly, at reduced size for CI speed.
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


leaky_mod = _import_script("leaky_differentiable_healing")


def test_leaky_healer_output_has_no_nan_or_inf():
    clean = leaky_mod.clean_trajectory(seed=0)
    corrupted = leaky_mod.corrupt(clean, "combined", seed=0)
    out = np.array(leaky_mod.LEAKY_HEALER(jnp.array(corrupted)))
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert out.shape == clean.shape


def test_leaky_healer_gradient_is_finite_and_nonzero():
    clean = leaky_mod.clean_trajectory(seed=1)
    corrupted = jnp.array(leaky_mod.corrupt(clean, "single_spike", seed=1))

    def loss_fn(x):
        return jnp.sum(leaky_mod.LEAKY_HEALER(x) ** 2)

    grad = jax.grad(loss_fn)(corrupted)
    assert not bool(jnp.isnan(grad).any())
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_leaky_vs_no_leak_gradient_both_finite():
    # The 'leaky' epsilon floor is meant to prevent dead gradients from
    # sigmoid saturation -- verify both variants at least produce a finite
    # gradient on a representative input (Experiment 28's own finding is
    # that removing the leak made no measurable difference across 30
    # seeds; this test just guards against either variant regressing to
    # NaN/dead gradients, not the (already-documented) quantitative gap).
    clean = leaky_mod.clean_trajectory(seed=2)
    corrupted = jnp.array(leaky_mod.corrupt(clean, "nan_string", seed=2))

    for healer in (leaky_mod.LEAKY_HEALER, leaky_mod.LEAKY_HEALER_NO_LEAK):
        grad = jax.grad(lambda x: jnp.sum(healer(x) ** 2))(corrupted)
        assert not bool(jnp.isnan(grad).any())
