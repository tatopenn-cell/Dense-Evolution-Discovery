"""
zne_stabilized_psr_gradient.py -- ZNE-before-PSR gradient stabilization tests
-------------------------------------------------------------------------------
6 tests. The noisy-trial study (naive vs. ZNE-pre-PSR) is run ONCE at module
import (small trial/shot budget, calibrated for CI speed) for a handful of
theta values and reused by the tests that need it.
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
    between runs even with "the same" seed expression. zlib.crc32 has no
    such randomization and is stable across processes/platforms."""
    s = "_".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 32)


m = _import_script("zne_stabilized_psr_gradient")

# theta=0.38 and theta=1.0: exact gradient has large magnitude -- this is
# where ZNE-pre-PSR is expected to win on bias and RMSE (see the top-of-file
# note in zne_stabilized_psr_gradient.py). theta=0.62: exact gradient is
# near zero (a gradient zero-crossing) -- this is the documented EXCEPTION
# where ZNE-pre-PSR loses on every axis (bias, std, RMSE), because there's
# little systematic bias left to correct and Richardson's variance
# amplification has nothing to buy against.
_TEST_THETAS = [0.38, 1.0]
_ZERO_CROSSING_THETA = 0.62
_N_TRIALS = 40  # raised from 15 (2026-08-12): the corrected depolarizing
# channel's changed variance made 15 trials too unstable at theta=1.0 --
# see test_zne_pre_psr_rmse_is_roughly_tied_at_theta_1_0.
_N_SHOTS = 60


def _collect_trials(theta):
    exact = m.exact_kinetic_gradient(theta)
    naive_vals = np.empty(_N_TRIALS)
    zne_vals = np.empty(_N_TRIALS)
    for trial in range(_N_TRIALS):
        rng_naive = np.random.default_rng(_stable_seed("naive", theta, trial))
        naive_vals[trial] = m.psr_gradient_naive_noisy(theta, m.BASE_P, _N_SHOTS, rng_naive)
        rng_zne = np.random.default_rng(_stable_seed("zne", theta, trial))
        zne_vals[trial] = m.psr_gradient_zne_stabilized(theta, m.BASE_P, _N_SHOTS, rng_zne)
    return exact, naive_vals, zne_vals


_RESULTS = {theta: _collect_trials(theta) for theta in _TEST_THETAS}
_ZERO_CROSSING_RESULT = _collect_trials(_ZERO_CROSSING_THETA)


def _rmse(vals, exact):
    bias = vals.mean() - exact
    return np.sqrt(bias ** 2 + vals.std() ** 2)


def test_exact_kinetic_gradient_matches_finite_difference():
    """The noise-free chain-rule PSR kinetic gradient (w.r.t. the shared
    scalar theta) must match an independent finite-difference reference
    that shifts the scalar theta directly (valid for a classical derivative
    regardless of how many internal gates depend on it) -- same ground-truth
    math already verified elsewhere in this repo (e.g.
    vqe_silicon_molecular_optimized.py). This is also a regression guard
    against the scalar-shift PSR bug found during development (shifting the
    shared scalar directly by +-pi/2 inside the circuit is NOT valid PSR
    when two gates both depend on it -- see the top-of-file caution note)."""
    h = 1e-6
    for theta in [0.2, 0.38, 0.62, 1.0]:
        sv_plus = m._run_circuit_direct(m._base_row(theta + h))
        sv_minus = m._run_circuit_direct(m._base_row(theta - h))
        fd_grad = (m._kinetic_from_sv(sv_plus) - m._kinetic_from_sv(sv_minus)) / (2 * h)
        exact = m.exact_kinetic_gradient(theta)
        assert exact == pytest.approx(fd_grad, abs=1e-4), f"theta={theta}: PSR {exact} != finite-diff {fd_grad}"


def test_zero_noise_reduces_naive_and_zne_to_exact_gradient():
    """At p_error=0, both the naive and ZNE-corrected paths must reduce
    exactly to the noise-free PSR gradient -- trivial consistency check
    (ZNE of two identical noise-free values collapses to that same value,
    and averaging noise-free 'shots' is a no-op)."""
    theta = 0.5
    exact = m.exact_kinetic_gradient(theta)
    rng = np.random.default_rng(0)
    naive = m.psr_gradient_naive_noisy(theta, 0.0, 5, rng)
    zne = m.psr_gradient_zne_stabilized(theta, 0.0, 5, rng)
    assert naive == pytest.approx(exact, abs=1e-9)
    assert zne == pytest.approx(exact, abs=1e-9)


def test_zne_pre_psr_reduces_bias_away_from_a_gradient_zero_crossing():
    """Where the exact gradient has a large magnitude (theta=0.38, 1.0),
    ZNE-correcting each single-gate PSR term before combining them via the
    chain rule must reduce the systematic bias vs. the naive noisy PSR
    gradient. This does NOT hold everywhere -- see the zero-crossing test
    below for the documented exception."""
    for theta, (exact, naive_vals, zne_vals) in _RESULTS.items():
        naive_bias = abs(naive_vals.mean() - exact)
        zne_bias = abs(zne_vals.mean() - exact)
        assert zne_bias < naive_bias, (
            f"theta={theta}: ZNE bias {zne_bias:.4f} should be smaller than naive bias {naive_bias:.4f}"
        )


def test_zne_pre_psr_has_higher_trial_to_trial_variance_than_naive():
    """Honest, counter-intuitive finding: ZNE-pre-PSR does NOT reduce the
    raw std across repeated noisy trials -- it INCREASES it, because the
    2-point Richardson formula (2*E1 - E2) amplifies statistical noise
    (the textbook Richardson bias/variance tradeoff). This is locked in as
    a regression test so it isn't silently 'fixed' into a false claim of
    reduced variance later -- the real benefit (where there is one) is in
    RMSE, not in raw fluctuation."""
    for theta, (exact, naive_vals, zne_vals) in _RESULTS.items():
        assert zne_vals.std() > naive_vals.std(), (
            f"theta={theta}: expected ZNE std ({zne_vals.std():.4f}) > naive std "
            f"({naive_vals.std():.4f}) -- if this no longer holds, the bias/variance "
            f"tradeoff finding documented at the top of zne_stabilized_psr_gradient.py "
            f"needs to be re-verified and updated, not just have this test changed"
        )


def test_zne_pre_psr_has_lower_rmse_away_from_a_gradient_zero_crossing():
    """The metric that matters for 'stabilized' where the exact gradient
    magnitude is large: RMSE = sqrt(bias^2 + std^2) against the exact
    gradient. At theta=0.38, ZNE-pre-PSR's much larger bias reduction nets
    a clearly lower RMSE than the naive noisy PSR gradient.

    theta=1.0 is NOT asserted here as a reliable win anymore. Re-verified
    2026-08-12 after dense-evolution 8.1.57's depolarizing-channel fix (see
    the top-of-file note in zne_stabilized_psr_gradient.py): at theta=1.0
    the RMSE gap has collapsed to ~0.4% (well inside run-to-run noise --
    bias improved a lot, but variance amplification grew enough to erase
    almost the entire win), unlike theta=0.38's robust ~1.6x margin. See
    test_zne_pre_psr_rmse_is_roughly_tied_at_theta_1_0 below for that
    honest, weaker claim."""
    exact, naive_vals, zne_vals = _RESULTS[0.38]
    naive_rmse = _rmse(naive_vals, exact)
    zne_rmse = _rmse(zne_vals, exact)
    assert zne_rmse < naive_rmse, (
        f"theta=0.38: ZNE RMSE {zne_rmse:.4f} should be lower than naive RMSE {naive_rmse:.4f}"
    )


def test_zne_pre_psr_rmse_is_roughly_tied_at_theta_1_0():
    """Honest, re-verified 2026-08-12 finding: under dense-evolution
    8.1.57's corrected depolarizing channel, ZNE-pre-PSR's RMSE advantage
    at theta=1.0 has essentially disappeared -- it is no longer a reliable
    win (nor a reliable loss), just within noise of naive. This replaces
    the old claim (a clear ~2x RMSE win at theta=1.0, see the historical
    table in zne_stabilized_psr_gradient.py) rather than silently deleting
    it. Asserted as a loose bound, not a directional winner, since the sign
    of the (tiny) gap is not reproducible run to run at this budget."""
    exact, naive_vals, zne_vals = _RESULTS[1.0]
    naive_rmse = _rmse(naive_vals, exact)
    zne_rmse = _rmse(zne_vals, exact)
    assert zne_rmse < naive_rmse * 1.2, (
        f"theta=1.0: ZNE RMSE {zne_rmse:.4f} should be within ~20% of naive RMSE "
        f"{naive_rmse:.4f} (roughly tied, not a clear loss) -- if ZNE is now clearly "
        f"WORSE than this, that is a bigger change than the 2026-08-12 re-verification "
        f"found and needs its own investigation, not just a threshold bump"
    )


def test_zne_pre_psr_variance_penalty_persists_near_a_gradient_zero_crossing():
    """Documented exception, not an oversight: at theta=0.62 the exact
    gradient is itself near zero. A high-precision one-time measurement (40
    trials, 200 shots -- see the top-of-file note in
    zne_stabilized_psr_gradient.py) found ZNE-pre-PSR WORSE than naive on
    every axis there (bias, std, RMSE): near a zero-crossing there is very
    little systematic error left for Richardson extrapolation to correct,
    so its variance amplification has nothing to buy against and just adds
    noise. 'ZNE stabilizes the gradient' is therefore a regime-dependent
    claim, not a universal one.

    At the small trial/shot budget this test suite can afford, the BIAS
    comparison near this zero-crossing is too noisy to reproduce reliably
    (verified: it flips direction across different shot counts in this
    low-signal regime) -- asserting a specific bias ordering here would be
    a flaky test, not a real regression guard. What IS robust even at a
    small budget is the variance-amplification mechanism itself (structural
    to the 2*E1-E2 Richardson formula, not dependent on the bias magnitude):
    ZNE-pre-PSR's std must still exceed naive's std here too."""
    exact, naive_vals, zne_vals = _ZERO_CROSSING_RESULT
    assert zne_vals.std() > naive_vals.std(), (
        f"expected ZNE std ({zne_vals.std():.4f}) > naive std ({naive_vals.std():.4f}) "
        f"near the theta=0.62 gradient zero-crossing too"
    )
