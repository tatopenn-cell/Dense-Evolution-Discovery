"""
Tests for scripts/syndrome_rate_estimator.py: per-qubit error probabilities
estimated from the firing frequency of a code's checks.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from syndrome_rate_estimator import estimate_error_rates_from_syndromes as est


def rep_code(n):
    return np.eye(n - 1, n, dtype=int) + np.eye(n - 1, n, 1, dtype=int)


def sample(h, p, cycles, seed):
    e = (np.random.default_rng(seed).random((cycles, h.shape[1])) < p).astype(int)
    return (e @ h.T) % 2


def test_single_check_single_qubit_recovers_the_firing_frequency_exactly():
    hist = np.array([[1]] * 20 + [[0]] * 80)
    assert est(np.array([[1]]), hist, lam=0.0)[0] == pytest.approx(0.2, abs=1e-9)


def test_single_check_with_pull_matches_the_closed_form():
    hist = np.array([[1]] * 20 + [[0]] * 80)
    a = (-np.log(0.6) + 0.1 * -np.log(0.98)) / 1.1
    assert est(np.array([[1]]), hist, p0=0.01, lam=0.1)[0] == pytest.approx((1 - np.exp(-a)) / 2, abs=1e-9)


def test_output_has_one_probability_per_qubit_and_stays_in_range():
    h = rep_code(9)
    r = est(h, sample(h, np.full(9, 0.05), 500, 1))
    assert r.shape == (9,)
    assert (r >= 0).all() and (r < 0.5).all()


def test_uniform_noise_is_recovered_on_every_qubit():
    h = rep_code(9)
    r = est(h, sample(h, np.full(9, 0.05), 20000, 0))
    assert (r > 0.04).all() and (r < 0.06).all()


def test_a_noisy_zone_is_found_and_the_quiet_qubits_stay_low():
    h = rep_code(9)
    p = np.full(9, 0.01)
    p[2:7] = 0.25
    r = est(h, sample(h, p, 2000, 3))
    assert (r[2:7] > 0.2).all() and (r[2:7] < 0.3).all()
    assert (np.delete(r, np.arange(2, 7)) < 0.05).all()


def test_qubit_shared_by_both_checks_gets_the_highest_probability():
    h = np.array([[1, 1, 0], [0, 1, 1]])
    hist = np.array([[1, 1]] * 25 + [[1, 0]] * 5 + [[0, 1]] * 4 + [[0, 0]] * 66)
    r = est(h, hist)
    assert r[1] > r[0] and r[1] > r[2]


def test_empty_alarm_history_gives_probabilities_below_the_baseline():
    h = rep_code(9)
    r = est(h, np.zeros((50, 8), dtype=int))
    assert (r >= 0).all() and (r < 0.002).all()


def test_history_where_every_check_always_rings_stays_finite_and_below_one_half():
    h = rep_code(9)
    r = est(h, np.ones((50, 8), dtype=int))
    assert np.isfinite(r).all() and (r > 0.35).all() and (r < 0.5).all()


def test_without_the_pull_the_result_is_still_non_negative():
    h = rep_code(9)
    assert (est(h, sample(h, np.full(9, 0.05), 2000, 4), lam=0.0) >= 0).all()


@pytest.mark.parametrize("bad", [
    dict(h=np.ones(3), hist=np.zeros((5, 3))),
    dict(h=np.ones((2, 3)), hist=np.zeros((5, 3))),
    dict(h=np.ones((2, 3)), hist=np.zeros(5)),
    dict(h=np.ones((2, 3)), hist=np.zeros((0, 2))),
    dict(h=np.ones((2, 3)), hist=np.full((5, 2), 2)),
    dict(h=np.ones((2, 3)), hist=np.zeros((5, 2)), p0=0.6),
    dict(h=np.ones((2, 3)), hist=np.zeros((5, 2)), p0=0.0),
    dict(h=np.ones((2, 3)), hist=np.zeros((5, 2)), lam=-1.0),
])
def test_wrong_inputs_raise_value_error(bad):
    with pytest.raises(ValueError):
        est(**bad)


def test_estimated_weights_make_the_decoder_correct_the_noisy_zone():
    pm = pytest.importorskip("pymatching")
    h = rep_code(9)
    p = np.full(9, 0.01)
    p[2:7] = 0.25
    w = -np.log(est(h, sample(h, p, 2000, 3)) + 1e-3)
    ee = (np.random.default_rng(11).random((20000, 9)) < p).astype(int)
    ss = ((ee @ h.T) % 2).astype(np.uint8)
    rates = []
    for wt in (None, w):
        m = pm.Matching.from_check_matrix(h, weights=wt)
        rates.append(float(((ee ^ m.decode_batch(ss)).sum(1) == 9).mean()))
    assert rates[1] < rates[0]
