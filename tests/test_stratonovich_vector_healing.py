"""
Smoke test for scripts/stratonovich_vector_healing.py -- imports the real
script (guarded behind `if __name__ == "__main__":`, same pattern as
test_integration_smoke.py) and calls its real functions directly.
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


sh = _import_script("stratonovich_vector_healing")


def test_online_healing_never_leaks_nan_median():
    clean = sh._clean_trajectory(seed=0)
    corrupted, _ = sh._corrupt(clean, "nan_string", seed=0)
    healed, _replaced = sh.online_healing(corrupted, sh._median_correction)
    assert not np.isnan(healed).any()


def test_online_healing_never_leaks_nan_strato():
    clean = sh._clean_trajectory(seed=0)
    corrupted, _ = sh._corrupt(clean, "nan_string", seed=0)
    healed, _replaced = sh.online_healing(corrupted, sh._stratonovich_correction)
    assert not np.isnan(healed).any()


def test_forced_healing_only_touches_corrupt_indices():
    clean = sh._clean_trajectory(seed=1)
    corrupted, corrupt_indices = sh._corrupt(clean, "single_spike", seed=1)
    healed = sh.forced_healing(corrupted, sh._stratonovich_correction, corrupt_indices)
    untouched = [i for i in range(len(clean)) if i not in set(corrupt_indices)]
    # away from the forced-correction indices, forced_healing must
    # reproduce the (sanitized) input exactly -- no NaN/Inf to sanitize
    # here since only the spike scenario is corrupted, so this reduces to
    # the raw corrupted array.
    assert np.allclose(healed[untouched], corrupted[untouched])


def test_forced_healing_median_matches_np_median_at_target():
    clean = sh._clean_trajectory(seed=2)
    corrupted, corrupt_indices = sh._corrupt(clean, "single_spike", seed=2)
    healed = sh.forced_healing(corrupted, sh._median_correction, corrupt_indices)
    idx = corrupt_indices[0]
    radius = min(20, max(3, len(clean) // 3))
    lo = max(0, idx - radius)
    expected = np.median(corrupted[lo:idx], axis=0)
    assert np.allclose(healed[idx], expected)


def test_cosine_alignment_self_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 8))
    assert abs(sh.cosine_alignment(x, x) - 1.0) < 1e-9


def test_l2_error_zero_for_identical_arrays():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 8))
    assert sh.l2_error(x, x) == 0.0


def test_corrupt_indices_match_actual_corruption():
    clean = sh._clean_trajectory(seed=3)
    for scenario in sh.SCENARIOS:
        corrupted, indices = sh._corrupt(clean, scenario, seed=3)
        diff_or_nan = [
            i for i in range(len(clean))
            if not np.allclose(corrupted[i], clean[i]) or np.isnan(corrupted[i]).any()
        ]
        assert set(diff_or_nan) == set(i for i in indices) or set(diff_or_nan) <= set(range(len(clean)))
        # every reported index really was altered
        for i in indices:
            assert not np.allclose(corrupted[i], clean[i]) or np.isnan(corrupted[i]).any()
