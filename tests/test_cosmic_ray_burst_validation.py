"""
Tests for scripts/cosmic_ray_burst_validation.py -- imports the real script
(runs the full experiment on import, same convention as
test_germanium_iswap_validation.py) and checks its real module-level
results.
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


try:
    cr = _import_script("cosmic_ray_burst_validation")
except ImportError as exc:
    # continuous_dissipative_evolve is still on Dense-Evolution PR #122,
    # not yet merged/released -- CI installs dense_evolution from PyPI
    # (requirements-ci.txt), so it won't have the function until that PR
    # ships. Skip cleanly here instead of failing the whole suite; this
    # starts running for real the moment a release picks it up.
    pytest.skip(f"continuous_dissipative_evolve not available yet (pending "
                f"Dense-Evolution PR #121/#122): {exc}", allow_module_level=True)


def test_scaling_factor_is_one_at_baseline():
    assert abs(float(cr.scaling_factor(0.0)) - 1.0) < 1e-9


def test_scaling_factor_reaches_paper_peak_ratio_before_decaying_away():
    # At t=1ms the two rise stages are essentially saturated (tau1=3us,
    # tau2=300us, both << 1ms) while the 25ms decay has barely acted yet,
    # so scaling_factor(1ms) should sit close to the paper's real peak
    # ratio (15/4 = 3.75x).
    value_at_1ms = float(cr.scaling_factor(1000.0))
    assert abs(value_at_1ms - cr.RATIO_PEAK) < 0.2


def test_scaling_factor_decays_with_the_papers_fitted_time_constant():
    # The excess above baseline (scaling_factor - 1) must decay as
    # exp(-t/TAU_DECAY_MS), the paper's own directly fitted quantity --
    # checked here between two late times where both rise stages are long
    # since saturated, so only the decay term is changing.
    t1_ms, t2_ms = 10.0, 35.0   # 25ms apart == exactly one tau_decay
    excess1 = float(cr.scaling_factor(t1_ms * 1000.0)) - 1.0
    excess2 = float(cr.scaling_factor(t2_ms * 1000.0)) - 1.0
    ratio = excess1 / excess2
    assert abs(ratio - np.e) / np.e < 0.02


def test_effective_t1_drops_and_recovers():
    t1_baseline = float(cr.t1_baseline_us)
    t1_peak = float(cr.effective_t1_us(cr.P_PEAK_1US))
    assert t1_peak < t1_baseline
    assert cr.T1_DROP_FACTOR > 1.0
    # T1_eff at the end of the 150ms window must have recovered close to
    # baseline (6 decay time constants have passed).
    assert abs(float(cr.t1_event_us[-1]) - t1_baseline) / t1_baseline < 0.01


def test_event_survival_worse_than_baseline_at_checkpoint():
    assert cr.SURVIVAL_AT_CHECKPOINT_EVENT < cr.SURVIVAL_AT_CHECKPOINT_BASELINE


def test_final_density_matrices_are_trace_preserving():
    assert abs(float(np.real(np.trace(np.array(cr.final_rho_event)))) - 1.0) < 1e-9
    assert abs(float(np.real(np.trace(np.array(cr.final_rho_baseline)))) - 1.0) < 1e-9


def test_amplitude_damping_channel_only_moves_population_one_way():
    # Starting from |0><0| (ground state), the channel must leave it
    # completely unchanged -- the real asymmetry the paper reports (decay
    # errors only, no excess excitation errors).
    rho0 = np.zeros((2, 2), dtype=complex)
    rho0[0, 0] = 1.0
    out = np.array(cr.amplitude_damping_channel(rho0, 0.5))
    assert np.allclose(out, rho0)
