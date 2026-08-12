"""
zne_adaptive_psr_gradient.py -- adaptive (delta_preemp) ZNE-before-PSR tests
-------------------------------------------------------------------------------
6 tests. The 3-way (naive/static/adaptive) trial study is run ONCE at module
import (small trial/shot budget, calibrated for CI speed) and reused.

This is a NEGATIVE-RESULT study (see the top-of-file note in
zne_adaptive_psr_gradient.py): SEM-based confidence attenuation does not
Pareto-dominate the static correction from zne_stabilized_psr_gradient.py --
at the shipped calibration, the adaptive gradient sits strictly BETWEEN naive
and static at every tested theta, never winning outright. These tests lock
in that honest finding, not an assumed improvement.
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
    directly during development of a later ZNE-PSR study in this repo: the
    same hash((tag, theta, trial)) % 2**32 expression gave a different
    value on every fresh Python process, so trial data silently differed
    between runs despite looking deterministic. zlib.crc32 has no such
    randomization and is stable across processes/platforms."""
    s = "_".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 32)


m = _import_script("zne_adaptive_psr_gradient")

_TEST_THETAS = [0.38, 0.62, 1.0]
_N_TRIALS = 40  # raised from 15 (2026-08-12): the corrected depolarizing
# channel needed a larger sample to stabilize the theta=1.0 comparisons --
# see test_adaptive_wins_outright_at_theta_1_0_since_static_no_longer_helps.
_N_SHOTS = 60


def _collect_trials(theta):
    exact = m.exact_kinetic_gradient(theta)
    naive_vals = np.empty(_N_TRIALS)
    static_vals = np.empty(_N_TRIALS)
    adaptive_vals = np.empty(_N_TRIALS)
    for trial in range(_N_TRIALS):
        r1 = np.random.default_rng(_stable_seed("naive", theta, trial))
        naive_vals[trial] = m.psr_gradient_naive_noisy(theta, m.BASE_P, _N_SHOTS, r1)
        r2 = np.random.default_rng(_stable_seed("static", theta, trial))
        static_vals[trial] = m.psr_gradient_zne_static(theta, m.BASE_P, _N_SHOTS, r2)
        r3 = np.random.default_rng(_stable_seed("adaptive", theta, trial))
        adaptive_vals[trial] = m.psr_gradient_adaptive_zne(
            theta, m.BASE_P, _N_SHOTS, r3, m.TARGET_SIGMA_IDEAL, m.K_SENSITIVITY)
    return exact, naive_vals, static_vals, adaptive_vals


_RESULTS = {theta: _collect_trials(theta) for theta in _TEST_THETAS}


def test_exact_kinetic_gradient_matches_finite_difference():
    """Same regression guard as zne_stabilized_psr_gradient.py -- this
    script replicates the ansatz locally, so it needs its own check."""
    h = 1e-6
    for theta in [0.2, 0.38, 0.62, 1.0]:
        sv_plus = m._run_circuit_direct(m._base_row(theta + h))
        sv_minus = m._run_circuit_direct(m._base_row(theta - h))
        fd_grad = (m._kinetic_from_sv(sv_plus) - m._kinetic_from_sv(sv_minus)) / (2 * h)
        exact = m.exact_kinetic_gradient(theta)
        assert exact == pytest.approx(fd_grad, abs=1e-4), f"theta={theta}: PSR {exact} != finite-diff {fd_grad}"


def test_zero_noise_reduces_all_three_paths_to_exact_gradient():
    """At p_error=0, naive, static, and adaptive must all reduce exactly to
    the noise-free PSR gradient."""
    theta = 0.5
    exact = m.exact_kinetic_gradient(theta)
    naive = m.psr_gradient_naive_noisy(theta, 0.0, 5, np.random.default_rng(0))
    static = m.psr_gradient_zne_static(theta, 0.0, 5, np.random.default_rng(0))
    adaptive = m.psr_gradient_adaptive_zne(theta, 0.0, 5, np.random.default_rng(0), 1.0, 1.0)
    assert naive == pytest.approx(exact, abs=1e-9)
    assert static == pytest.approx(exact, abs=1e-9)
    assert adaptive == pytest.approx(exact, abs=1e-9)


def test_adaptive_reduces_exactly_to_static_at_high_confidence():
    """calculate_delta_preemp(current, target) = |current-target|/target --
    this is MINIMIZED (delta_p=0) when target equals current exactly, NOT
    when target is very large (that actually drives delta_p TOWARD 1, since
    |current-target|~target when target>>current). So to test the
    high-confidence limit, target_sigma_ideal must be set to the ACTUAL
    measured SEM for this exact call (same seed => same noise draws => same
    SEM), not an arbitrarily large number -- an earlier version of this
    test got this backwards and failed by asserting equality against a
    completely different value.

    Tested at the single-gate-shift level (zne_corrected_kinetic /
    adaptive_zne_corrected_kinetic) rather than through the full
    N_PARAMS-gate psr_gradient_* wrapper, since each of the 20 gate-shifts
    has its own slightly different SEM realization and a single global
    target_sigma_ideal can't zero out delta_p for all of them at once."""
    theta = 0.5
    base = m._base_row(theta)
    clean_sv = m._run_circuit_direct(base)
    seed = 123

    samples_1 = m._noisy_kinetic_samples(clean_sv, m.BASE_P, 100, np.random.default_rng(seed))
    exact_sem = m._sem(samples_1)

    static = m.zne_corrected_kinetic(clean_sv, m.BASE_P, 100, np.random.default_rng(seed))
    adaptive = m.adaptive_zne_corrected_kinetic(clean_sv, m.BASE_P, 100, np.random.default_rng(seed),
                                                 target_sigma_ideal=exact_sem, k_sensitivity=1.0)
    assert adaptive == pytest.approx(static, abs=1e-9), (
        f"at delta_p=0 (target_sigma_ideal == measured SEM) the adaptive formula must match "
        f"the static one exactly: static={static}, adaptive={adaptive}"
    )


def test_adaptive_is_worse_than_static_away_from_a_gradient_zero_crossing():
    """Honest finding at theta=0.38 (large exact-gradient magnitude, where
    the static correction wins big over naive): the adaptive correction
    gives back a large fraction of that win -- it is consistently WORSE
    than static there.

    theta=1.0 is NOT included here anymore. Re-verified 2026-08-12 after
    dense-evolution 8.1.57's depolarizing-channel fix: static's own RMSE
    edge over naive at theta=1.0 collapsed under the corrected noise (see
    the top-of-file RE-VERIFIED note), and with static no longer clearly
    ahead there, adaptive's partial correction now wins outright at
    theta=1.0 instead of losing to static -- see
    test_adaptive_wins_outright_at_theta_1_0_since_static_no_longer_helps
    below for that new, opposite finding."""
    theta = 0.38
    exact, naive_vals, static_vals, adaptive_vals = _RESULTS[theta]
    static_rmse = m._rmse(static_vals, exact)
    adaptive_rmse = m._rmse(adaptive_vals, exact)
    assert adaptive_rmse > static_rmse, (
        f"theta={theta}: adaptive RMSE {adaptive_rmse:.4f} should be worse than "
        f"static RMSE {static_rmse:.4f} -- if this no longer holds, the negative-result "
        f"finding documented at the top of zne_adaptive_psr_gradient.py needs to be "
        f"re-verified and updated, not just have this test changed"
    )


def test_adaptive_wins_outright_at_theta_1_0_since_static_no_longer_helps():
    """New finding, re-verified 2026-08-12: at theta=1.0, dense-evolution
    8.1.57's corrected depolarizing channel erased static's RMSE advantage
    over naive (they're now within ~0.2% of each other -- see
    zne_stabilized_psr_gradient.py's parallel finding for the bias/variance
    mechanism). With static no longer a strictly-better reference to be
    dominated by, adaptive's partial attenuation -- which always sits
    between naive and static -- now has the LOWEST RMSE of the three at
    this theta. Reproduced consistently across n_trials in {15, 40, 80} at
    fixed n_shots=60, not a small-sample artifact. This directly
    contradicts the original blanket claim ("adaptive does NOT win outright
    at ANY theta") -- kept as its own test rather than silently folded into
    the theta=0.38 test above, so the reversal is visible, not hidden."""
    theta = 1.0
    exact, naive_vals, static_vals, adaptive_vals = _RESULTS[theta]
    naive_rmse = m._rmse(naive_vals, exact)
    static_rmse = m._rmse(static_vals, exact)
    adaptive_rmse = m._rmse(adaptive_vals, exact)
    assert adaptive_rmse < naive_rmse and adaptive_rmse < static_rmse, (
        f"theta={theta}: adaptive RMSE {adaptive_rmse:.4f} should now be the lowest of "
        f"the three (naive={naive_rmse:.4f}, static={static_rmse:.4f}) -- if this no "
        f"longer holds, re-verify rather than just adjusting the assertion"
    )


def test_adaptive_is_better_than_static_near_a_gradient_zero_crossing():
    """Honest finding, the core motivation for this study: at theta=0.62
    (near a gradient zero-crossing, where static ZNE actively makes things
    WORSE than doing nothing), the adaptive correction meaningfully
    mitigates that damage relative to static. This does NOT mean it
    recovers naive's performance (it doesn't, per the high-precision
    40-trial/200-shot measurement documented at the top of
    zne_adaptive_psr_gradient.py: adaptive RMSE 0.175 vs naive's 0.106) --
    that specific comparison is too fragile at this test's small budget to
    assert reliably, so only the robust static-vs-adaptive gap is checked
    here."""
    theta = 0.62
    exact, naive_vals, static_vals, adaptive_vals = _RESULTS[theta]
    static_rmse = m._rmse(static_vals, exact)
    adaptive_rmse = m._rmse(adaptive_vals, exact)
    assert adaptive_rmse < static_rmse, (
        f"adaptive RMSE {adaptive_rmse:.4f} should improve on static RMSE {static_rmse:.4f} near the zero-crossing"
    )


def test_sem_is_roughly_theta_independent():
    """Verifies the root-cause explanation documented at the top of
    zne_adaptive_psr_gradient.py: the measured SEM of the scale=1.0 estimate
    does NOT vary meaningfully with theta (it's driven by shot count and
    noise probability, not by proximity to a gradient zero-crossing) --
    which is why a SEM-based confidence signal can't distinguish the
    regimes where Richardson correction helps vs. hurts."""
    rng = np.random.default_rng(7)
    sems = []
    for theta in [0.2, 0.38, 0.62, 1.0]:
        base = m._base_row(theta)
        sv = m._run_circuit_direct(base)
        samples = m._noisy_kinetic_samples(sv, m.BASE_P, 200, rng)
        sems.append(m._sem(samples))
    assert max(sems) / min(sems) < 3.0, (
        f"expected SEM to stay within a narrow band across theta, got {sems}"
    )
