"""
Unit tests for scripts/cusum_detectability_theory/arl_theory.py and the
frozen Monte Carlo validation in validate_arl_theory.py. See
docs/cusum_detectability_theory.md for the full write-up.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts" / "cusum_detectability_theory"))

from arl_theory import one_sided_arl, two_sided_arl, detectability_report  # noqa: E402

_FROZEN_PATH = _ROOT / "scripts" / "cusum_detectability_theory" / "arl_theory_validation_frozen.json"


@pytest.fixture(scope="module")
def frozen():
    with open(_FROZEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_null_arl_grows_with_h():
    assert one_sided_arl(0.0, 3.0) < one_sided_arl(0.0, 8.0)


def test_corrected_arl_exceeds_uncorrected():
    """Siegmund's boundary correction always makes the ARL prediction
    larger (the plain Wald approximation systematically underestimates,
    verified directly via Monte Carlo, see the frozen results)."""
    for delta in (0.1, 0.5, 1.0):
        assert one_sided_arl(delta, 5.0, corrected=True) > one_sided_arl(delta, 5.0, corrected=False)


def test_larger_shift_gives_shorter_detection_arl():
    k, h = 0.5, 5.0
    arls = [two_sided_arl(mu, k, h) for mu in (0.0, 0.5, 1.0, 1.5)]
    assert all(arls[i] > arls[i + 1] for i in range(len(arls) - 1))


def test_detectability_report_shift_in_sigma():
    r = detectability_report(local_noise_scale=7.72, k=0.5, h=5.0, candidate_shift=10.0)
    assert r["shift_in_sigma"] == pytest.approx(10.0 / 7.72, rel=1e-6)
    # This reproduces Experiment 42's own ad hoc ratio (offset/MAD = 1.29),
    # now derivable from a general, reusable function.
    assert r["shift_in_sigma"] == pytest.approx(1.30, abs=0.02)


def test_frozen_ideal_simulation_matches_corrected_theory_closely(frozen):
    """The formula itself, checked against a direct simulation of the
    exact idealized model it describes -- should agree closely."""
    for row in frozen["ideal_vs_theory"]:
        assert row["rel_err_pct"] < 5.0, f"mu={row['mu']}: theory/simulation mismatch too large"


def test_frozen_ideal_simulation_shows_uncorrected_formula_is_worse(frozen):
    for row in frozen["ideal_vs_theory"]:
        corr_err = abs(row["empirical"] - row["theory_corrected"])
        uncorr_err = abs(row["empirical"] - row["theory_uncorrected"])
        assert corr_err < uncorr_err, f"mu={row['mu']}: corrected formula should fit better"


def test_frozen_real_cusum_detector_null_arl_is_lower_than_theory(frozen):
    """Real, honest finding: the real detector's false-alarm ARL is
    LOWER than the idealized theory predicts (more false alarms sooner
    than theory suggests) -- finite-sample MAD estimation noise, not a
    bug. Re-verified here against the frozen numbers, not just asserted
    in prose."""
    null_row = next(r for r in frozen["real_cusum_vs_theory"] if r["mu"] == 0.0)
    assert null_row["empirical"] < null_row["theory"]
    assert null_row["rel_err_pct"] > 20.0, "expected a real, substantial gap at span=40"


def test_frozen_span_sensitivity_shows_convergence_toward_theory(frozen):
    """The false-alarm-ARL gap shrinks monotonically as the reference
    window grows -- confirms finite-sample noise as the real cause,
    not a structural flaw in the theory or the detector."""
    rows = sorted(frozen["span_sensitivity"], key=lambda r: r["span"])
    errs = [r["rel_err_pct"] for r in rows]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), \
        "expected relative error to shrink monotonically with larger span"


def test_frozen_out_of_control_arl_is_reasonably_predictive(frozen):
    """Post-shift (out-of-control) detection delay is a reasonably
    useful practical estimate, unlike the in-control/false-alarm case."""
    shifted_rows = [r for r in frozen["real_cusum_vs_theory"] if r["mu"] > 0]
    assert all(r["rel_err_pct"] < 30.0 for r in shifted_rows)
