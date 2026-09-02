"""
scripts/cusum_detectability_theory/arl_theory.py
====================================================
Closed-form Average Run Length (ARL) approximation for a two-sided
CUSUM change-point detector, grounding a real, testable answer to the
question Experiment 42's lidar work raised informally ("1.29 sigma of
local noise, below the n_sigmas=3.0 threshold, so detection was poor")
-- turning that one-off, after-the-fact ratio into a general, PRE-
REGISTERED prediction: given a detector's real local noise level and
parameters, how long would it take, on average, to detect a shift of a
given size, or to false-alarm with none present?

THEORY: Page, E.S. (1954), "Continuous Inspection Schemes", Biometrika
41(1-2), 100-115 (dense-armor's own cusum.py citation) introduced
CUSUM. The closed-form Brownian-motion/Wald-type ARL approximation used
here follows the derivation in Reynolds, M.R. Jr. (1975),
"Approximations to the Average Run Length in Cumulative Sum Control
Charts", Technometrics 17(1), 65-71 (verified real: fetched and read
directly, indexed in quantumrag's statistica_controllo_processo
collection) -- the boundary/continuity correction constant (Reynolds'
own empirical value ~1.2, refined by Siegmund (1985), Sequential
Analysis: Tests and Confidence Intervals, Theorem 10.16, to ~1.166) is
used here.

Two more recent (2022-2026) CUSUM papers were checked and explicitly
NOT used: Wei & Xie, "Online Kernel CUSUM for Change-Point Detection"
(arXiv:2211.15070, J. Royal Statist. Soc. B 2026) and an adaptive-
control-limit CUSUM variant (arXiv:2303.04628) both derive ARL/EDD
formulas for algorithmically DIFFERENT CUSUM variants (kernel/MMD-
based, adaptive-limit) than dense-armor's simple linear CUSUM --
citing their formulas here would be a real citation mismatch, not a
more "modern" version of the same theory.

MONTE CARLO VALIDATION (see validate_arl_theory.py for the frozen,
reproducible numbers), two real, honest findings:
  1. Against a DIRECT simulation of the idealized model (a pure random
     walk with the exact assumed mean/variance) -- the Siegmund-
     corrected one-sided formula matches to within ~1% (e.g. delta=0:
     theory 38.02 vs simulated 38.23). The UNCORRECTED Wald formula
     underestimates badly (25.00 vs 38.23), confirming Reynolds' own
     1975 finding that the correction is not optional for small h.
  2. Against dense-armor's REAL cusum_detector (reference="fixed" --
     the only mode this classical theory describes; the default
     "adaptive" mode continuously re-estimates its own reference and
     is NOT the process this theory models): the out-of-control
     (post-shift) ARL is a reasonably good practical estimate (4-17%
     relative error across mu=0.5/1.0/1.5). The in-control (false-
     alarm) ARL is UNDERESTIMATED by the theory -- the real detector
     has real false alarms sooner than the idealized theory predicts,
     because the fixed reference's median/MAD is estimated from a
     FINITE window, not known exactly. This gap shrinks monotonically
     with window size (54% relative error at span=40, down to 11% at
     span=1000) -- confirmed directly, not assumed, consistent with
     finite-sample scale-estimation noise as the real cause.
"""
import numpy as np

BOUNDARY_CORRECTION = 1.166  # Siegmund (1985) refinement of Reynolds' (1975) empirical ~1.2


def one_sided_arl(delta: float, h: float, corrected: bool = True) -> float:
    """Wald/Brownian-motion approximation to the ARL of a one-sided
    CUSUM with drift `delta` (true standardized mean minus reference
    value k) and decision boundary `h`.

    Parameters
    ----------
    delta : float
        Standardized drift, (true mean - k). delta=0 is the in-control
        (null) case for that branch.
    h : float
        CUSUM decision boundary, in the same standardized (robust-
        sigma) units as delta.
    corrected : bool, default True
        Apply Siegmund's boundary/continuity correction (h -> h +
        1.166) -- verified via direct Monte Carlo to matter a lot for
        small h (see module docstring); leave True unless deliberately
        reproducing the plain, uncorrected textbook Wald formula.

    Returns
    -------
    float
        Approximate average run length (in samples).
    """
    hh = h + BOUNDARY_CORRECTION if corrected else h
    if abs(delta) < 1e-12:
        return hh ** 2
    return (np.exp(-2 * delta * hh) - 1 + 2 * delta * hh) / (2 * delta ** 2)


def two_sided_arl(mu: float, k: float, h: float, corrected: bool = True) -> float:
    """ARL of dense-armor's symmetric two-sided CUSUM (same k, h on
    both the S+ and S- accumulators, matching utility.cusum.
    cusum_detector's own convention) under a true standardized mean
    shift `mu`. Combines the two one-sided branches via 1/ARL =
    1/ARL+ + 1/ARL- (the standard two-sided-from-one-sided combination
    rule, e.g. Reynolds 1975 eq. 13)."""
    arl_pos = one_sided_arl(mu - k, h, corrected=corrected)
    arl_neg = one_sided_arl(-mu - k, h, corrected=corrected)
    return 1.0 / (1.0 / arl_pos + 1.0 / arl_neg)


def detectability_report(local_noise_scale: float, k: float, h: float, candidate_shift: float) -> dict:
    """Convert the standardized-units theory above into REAL units for
    a specific detector deployment -- the practical entry point.

    Parameters
    ----------
    local_noise_scale : float
        The detector's own real local noise scale (e.g. a causal
        window's median/MAD*1.4826, the same quantity cusum_detector
        computes internally) -- real units (e.g. meters, g, seconds).
    k, h : float
        The CUSUM's own k, h parameters, in STANDARDIZED (robust-
        sigma) units, matching cusum_detector's own convention.
    candidate_shift : float
        A real-unit shift magnitude to evaluate detectability for
        (e.g. "a +10m persistent offset").

    Returns
    -------
    dict with:
        false_alarm_arl : expected samples between false alarms (mu=0).
        detection_arl    : expected samples to detect `candidate_shift`.
        shift_in_sigma   : candidate_shift / local_noise_scale, the
                            same "offset/MAD" ratio Experiment 42
                            computed ad hoc -- now derivable directly.
    """
    mu_std = candidate_shift / local_noise_scale
    return dict(
        false_alarm_arl=two_sided_arl(0.0, k, h),
        detection_arl=two_sided_arl(mu_std, k, h),
        shift_in_sigma=mu_std,
    )
