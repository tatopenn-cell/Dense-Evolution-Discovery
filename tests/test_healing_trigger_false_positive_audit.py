"""
Smoke test for scripts/healing_trigger_false_positive_audit.py -- imports
the real script and calls its real functions directly.
"""
import importlib.util
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _import_script("healing_trigger_false_positive_audit")


def test_phi_trigger_replaces_returns_expected_length():
    clean = audit.clean_trajectory(seed=0)
    replaces = audit.phi_trigger_replaces(clean)
    assert replaces.shape == (audit.N_STEPS - 2,)
    assert replaces.dtype == bool


def test_adaptive_trigger_replaces_returns_expected_length():
    clean = audit.clean_trajectory(seed=0)
    replaces = audit.adaptive_trigger_replaces(clean)
    assert replaces.shape == (audit.N_STEPS - 2,)
    assert replaces.dtype == bool


def test_adaptive_trigger_has_far_fewer_false_positives_than_phi():
    clean = audit.clean_trajectory(seed=1)
    phi_rate = audit.phi_trigger_replaces(clean).mean()
    adaptive_rate = audit.adaptive_trigger_replaces(clean).mean()
    assert adaptive_rate < phi_rate
    assert phi_rate > 0.5
    assert adaptive_rate < 0.35


def test_adaptive_trigger_always_flags_nan_rows():
    clean = audit.clean_trajectory(seed=2)
    corrupted, indices = audit.corrupt(clean, "nan_string", seed=2)
    replaces = audit.adaptive_trigger_replaces(corrupted)
    idx_arr = np.array(indices) - 2
    assert replaces[idx_arr].all()


def test_corrupt_indices_are_actually_altered():
    clean = audit.clean_trajectory(seed=3)
    for scenario in audit.SCENARIOS:
        corrupted, indices = audit.corrupt(clean, scenario, seed=3)
        for i in indices:
            assert not np.allclose(corrupted[i], clean[i]) or np.isnan(corrupted[i]).any()
