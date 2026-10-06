"""
Tests for scripts/noise_mitigation_validation/cosmic_ray_zone_weighted_decoding.py and
scripts/noise_mitigation_validation/cosmic_ray_spreading_zone_decoding.py -- exact coset-level maximum-
likelihood decoders, so the checks below are structural guarantees plus the
qualitative findings of the two experiments.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "noise_mitigation_validation"))

import cosmic_ray_zone_weighted_decoding as zw
import cosmic_ray_spreading_zone_decoding as sp


def test_coset_probabilities_sum_to_one():
    assert abs(zw.coset_probs(zw.gammas((0, 1), 0.2)).sum() - 1.0) < 1e-12


def test_x_only_model_matches_the_erasure_script_blind_rate():
    assert abs(zw.compare(2, 1000.0, 3.75, twirl=False)["blind"] - 0.0055) < 1e-4


def test_zone_prior_never_worse_than_blind():
    for k in (1, 2, 3, 4):
        for ratio in (3.75, 30.0, 100.0):
            c = zw.compare(k, 1000.0, ratio)
            assert c["zone"] <= c["blind"] + 1e-12


def test_weak_burst_gives_no_zone_gain():
    c = zw.compare(2, 1000.0, 3.75)
    assert 1 - c["zone"] / c["blind"] < 1e-3


def test_strong_small_zone_halves_failures():
    c = zw.compare(2, 1000.0, 30.0)
    assert c["zone"] < 0.6 * c["blind"]


def test_wrong_zone_is_worse_than_blind_for_strong_burst():
    c = zw.compare(2, 1000.0, 30.0)
    assert c["mislocated"] > c["blind"]


def test_front_starts_at_initial_patch_and_saturates():
    assert sp.front(0.0) == sp.K0
    assert abs(sp.front(1e6) - zw.N) < 1e-6


def test_tracking_prior_never_worse_than_blind_while_spreading():
    order = np.arange(zw.N)
    wrong = order[::-1]
    q_blind = zw.coset_probs(zw.gammas((), 0))
    for t in (10.0, 100.0, 300.0):
        r = sp.failures_at(t, order, wrong, 100.0, q_blind)
        assert r[2] <= r[0] + 1e-12
