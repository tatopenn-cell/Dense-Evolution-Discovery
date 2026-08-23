"""
Tests for scripts/cosmic_ray_erasure_decoding.py -- imports the real script
(runs the full Monte Carlo sweep on import, same convention as this repo's
other tests) and checks its real module-level results.
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


cr = _import_script("cosmic_ray_erasure_decoding")


def test_hot_spot_error_rate_exceeds_baseline():
    assert cr.P_HOTSPOT_PEAK > cr.BASELINE_P


def test_some_shots_actually_get_a_real_herald():
    assert cr.RESULTS["n_heralded_shots"] > 0


def test_erasure_aware_strategy_never_worse_than_blind_overall():
    assert cr.RESULTS["rate_erasure"] <= cr.RESULTS["rate_blind"]


def test_erasure_aware_strategy_beats_blind_on_heralded_shots():
    # This is where the effect should actually live -- on the subset of
    # shots where the burst really did hit the hot spot.
    assert cr.RESULTS["heralded_rate_erasure"] < cr.RESULTS["heralded_rate_blind"]
