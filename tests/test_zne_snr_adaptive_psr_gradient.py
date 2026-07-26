"""
zne_snr_adaptive_psr_gradient.py -- SNR-based adaptive ZNE-before-PSR tests
-------------------------------------------------------------------------------
4 tests. This is a NEGATIVE-RESULT study (see the top-of-file note in
zne_snr_adaptive_psr_gradient.py): attenuating the static Richardson
correction from zne_stabilized_psr_gradient.py based on the correction
term's own signal-to-noise ratio does not deliver a clean win, and does not
fix the theta=0.62 zero-crossing regression that motivated it (it makes
that case worse, not better at the full study budget). These tests lock in
the honest, verified findings, not the originally-hoped-for outcome.

Note: an attempt to also lock in "raising SNR_TARGET makes theta=0.62
worse" as a strict CI regression test was dropped -- verified directly,
that comparison is too fragile at CI-sized budgets to reproduce reliably
(same fragility already documented for theta=0.62 comparisons in
tests/test_zne_adaptive_psr_gradient.py). The finding itself is real and
reported via the one-time, full-budget (40 trials/200 shots) measurement
in the top-of-file note, not asserted here.
"""

import importlib.util
import pathlib
import sys
import zlib

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


def _stable_seed(*parts) -> int:
    """Deterministic, cross-process-stable seed. Python's built-in hash()
    is NOT stable across process invocations for tuples containing strings
    (hash randomization, PYTHONHASHSEED defaults to random) -- verified
    directly during development: the same hash((tag, theta, trial)) % 2**32
    expression gave a different value on every fresh Python process, so
    trial data (and any RMSE comparison built on it) silently differed
    between runs even with "the same" seed expression, causing a test to
    pass once and fail on an immediate re-run. zlib.crc32 has no such
    randomization and is stable across processes/platforms."""
    s = "_".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 32)


m = _import_script("zne_snr_adaptive_psr_gradient")

_TEST_THETAS = [0.38]
_N_TRIALS = 15
_N_SHOTS = 60


def _collect_trials(theta):
    exact = m.exact_kinetic_gradient(theta)
    static_vals = np.empty(_N_TRIALS)
    snr_vals = np.empty(_N_TRIALS)
    for trial in range(_N_TRIALS):
        r_static = np.random.default_rng(_stable_seed("static", theta, trial))
        static_vals[trial] = m.psr_gradient_zne_static(theta, m.BASE_P, _N_SHOTS, r_static)
        r_snr = np.random.default_rng(_stable_seed("snr", theta, trial))
        snr_vals[trial] = m.psr_gradient_snr_adaptive_zne(
            theta, m.BASE_P, _N_SHOTS, r_snr, m.SNR_TARGET, m.K_SENSITIVITY)
    return exact, static_vals, snr_vals


_RESULTS = {theta: _collect_trials(theta) for theta in _TEST_THETAS}


def test_exact_kinetic_gradient_matches_finite_difference():
    """Same regression guard as the other ZNE-PSR scripts -- this script
    replicates the ansatz locally, so it needs its own check."""
    h = 1e-6
    for theta in [0.2, 0.38, 0.62, 1.0]:
        sv_plus = m._run_circuit_direct(m._base_row(theta + h))
        sv_minus = m._run_circuit_direct(m._base_row(theta - h))
        fd_grad = (m._kinetic_from_sv(sv_plus) - m._kinetic_from_sv(sv_minus)) / (2 * h)
        exact = m.exact_kinetic_gradient(theta)
        assert exact == pytest.approx(fd_grad, abs=1e-4), f"theta={theta}: PSR {exact} != finite-diff {fd_grad}"


def test_zero_noise_reduces_naive_static_and_snr_adaptive_to_exact_gradient():
    """At p_error=0, naive, static, and SNR-adaptive must all reduce
    exactly to the noise-free PSR gradient."""
    theta = 0.5
    exact = m.exact_kinetic_gradient(theta)
    naive = m.psr_gradient_naive_noisy(theta, 0.0, 5, np.random.default_rng(0))
    static = m.psr_gradient_zne_static(theta, 0.0, 5, np.random.default_rng(0))
    snr_adaptive = m.psr_gradient_snr_adaptive_zne(theta, 0.0, 5, np.random.default_rng(0), 3.0, 1.0)
    assert naive == pytest.approx(exact, abs=1e-9)
    assert static == pytest.approx(exact, abs=1e-9)
    assert snr_adaptive == pytest.approx(exact, abs=1e-9)


def test_snr_adaptive_reduces_exactly_to_static_at_high_snr():
    """When snr_target is set to the ACTUAL measured SNR for this exact
    call (same seed => same noise draws => same SNR), delta_p=0 and the
    attenuation vanishes -- the adaptive formula must then coincide EXACTLY
    with the static one. (Setting snr_target far below the measured SNR
    would also zero the attenuation once clamped, but using the exact
    measured value is the precise, unambiguous boundary case.)"""
    theta = 0.5
    base = m._base_row(theta)
    clean_sv = m._run_circuit_direct(base)
    seed = 123

    snr, _, _ = m._correction_snr(clean_sv, m.BASE_P, 100, np.random.default_rng(seed))

    static = m.zne_corrected_kinetic(clean_sv, m.BASE_P, 100, np.random.default_rng(seed))
    snr_adaptive = m.snr_adaptive_zne_corrected_kinetic(clean_sv, m.BASE_P, 100, np.random.default_rng(seed),
                                                         snr_target=snr, k_sensitivity=1.0)
    assert snr_adaptive == pytest.approx(static, abs=1e-9), (
        f"at snr==snr_target the adaptive formula must match the static one exactly: "
        f"static={static}, snr_adaptive={snr_adaptive}"
    )


def test_snr_adaptive_beats_static_at_a_large_gradient_theta():
    """The one robust, reproducible win found in this study: at theta=0.38
    (large exact-gradient magnitude), the SNR-adaptive correction reliably
    beats the static one -- confirmed at both the full study budget (40
    trials/200 shots: 0.862 vs 0.920) and this test's smaller CI budget."""
    theta = 0.38
    exact, static_vals, snr_vals = _RESULTS[theta]
    static_rmse = m._rmse(static_vals, exact)
    snr_rmse = m._rmse(snr_vals, exact)
    assert snr_rmse < static_rmse, (
        f"theta={theta}: SNR-adaptive RMSE {snr_rmse:.4f} should beat static RMSE {static_rmse:.4f}"
    )
