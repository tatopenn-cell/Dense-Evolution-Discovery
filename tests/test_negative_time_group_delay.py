"""
Tests for scripts/negative_time_group_delay.py -- imports the real script
and re-runs its self-tests plus a couple of independent spot-checks not
covered by the module's own main() run.
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


mod = _import_script("negative_time_group_delay")


def test_narrowband_selftest_passes():
    mod.selftest_narrowband_limit()


def test_external_published_ratio_selftest_passes():
    mod.selftest_external_published_ratio()


def test_zero_optical_depth_gives_zero_excitation_time():
    """tau0=0 means no atoms at all -- tau_T and tau_bar_0 must both be exactly 0."""
    tau_t, tau_bar_0, p_t = mod.excitation_times(0.0, 1.0)
    assert abs(float(tau_t)) < 1e-10
    assert abs(float(tau_bar_0)) < 1e-10
    assert abs(float(p_t) - 1.0) < 1e-10


def test_on_resonance_group_delay_matches_exact_formula():
    """Eq. (34) at delta=0 reduces to exactly -tau0 (Gamma=1 units) --
    checked directly on the group_delay() function, independent of the
    spectral-integration machinery. Tolerance set by JAX's default
    float32 precision, not a physics approximation."""
    import jax.numpy as jnp
    for tau0 in (0.5, 2.0, 7.3):
        assert abs(float(mod.group_delay(jnp.array(0.0), tau0)) - (-tau0)) < 1e-6


def test_fig2_broad_pulse_crosses_from_negative_to_positive():
    """Independent spot-check of the qualitative Fig. 2 shape at a single
    sigma not exercised by the module's own crossing assertions: a
    moderately broad pulse (sigma=1.0) must start near zero, dip negative,
    then cross to positive well within tau0<=9."""
    tau0_low, _, _ = mod.excitation_times(0.5, 1.0)
    tau0_high, _, _ = mod.excitation_times(9.0, 1.0)
    assert float(tau0_low) < 0.0
    assert float(tau0_high) > 0.0
