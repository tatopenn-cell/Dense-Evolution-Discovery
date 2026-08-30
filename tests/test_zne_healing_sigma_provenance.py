"""
Tests for scripts/zne_healing_sigma_provenance.py -- imports the real
script and calls its real functions directly, at reduced scope for CI
speed (2 configs x 3 seeds instead of the full 45-config sweep).
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


mod = _import_script("zne_healing_sigma_provenance")


def test_ideal_bell_state_zz_is_one():
    sv_ideal = mod.build_bell_ideal()
    import dense_evolution as de
    ideal = float(np.real(de.pauli_expectation(sv_ideal, mod.PAULI_STRING)))
    assert abs(ideal - 1.0) < 1e-9


def test_run_one_returns_real_base_std():
    sv_ideal = mod.build_bell_ideal()
    import dense_evolution as de
    ideal = float(np.real(de.pauli_expectation(sv_ideal, mod.PAULI_STRING)))
    sigma_ref = mod.measure_sigma_ideal_ref(sv_ideal, "depolarizing", seed=0)
    means, base_std, err_plain, err_healing = mod.run_one(
        sv_ideal, ideal, "depolarizing", 0.05, seed=0, sigma_ideal_ref=sigma_ref)
    assert len(means) == 3
    assert base_std > 0.0
    assert err_plain >= 0.0 and err_healing >= 0.0


def test_negative_control_permutation_matches_real_pairing():
    """Regression test for the confound finding itself: on a small,
    reduced-scope sweep, the shuffled-sigma control's win rate must stay
    close to the real pairing's win rate (within 25 percentage points) --
    if a future change to zero_noise_extrapolation's healing branch makes
    it start discriminating real signal from noise, this test should be
    revisited (a big gap here would mean the confound no longer holds)."""
    sv_ideal = mod.build_bell_ideal()
    import dense_evolution as de
    ideal = float(np.real(de.pauli_expectation(sv_ideal, mod.PAULI_STRING)))

    rows = []
    for noise_model in ["depolarizing", "bitflip"]:
        for seed in range(3):
            sigma_ref = mod.measure_sigma_ideal_ref(sv_ideal, noise_model, seed)
            means, base_std, err_plain, err_healing = mod.run_one(
                sv_ideal, ideal, noise_model, 0.05, seed, sigma_ref)
            rows.append(dict(
                base_std=base_std, means=means, err_plain=err_plain,
                err_healing=err_healing, sigma_ideal_ref=sigma_ref,
            ))

    real_win = float(np.mean([r["err_plain"] > r["err_healing"] for r in rows]))
    ctrl_mean, ctrl_win = mod.negative_control_permutation(rows, ideal)
    assert abs(real_win - ctrl_win) < 0.25, (
        f"real win_rate={real_win:.2f} vs control win_rate={ctrl_win:.2f} -- "
        "diverged more than expected, confound finding may no longer hold"
    )