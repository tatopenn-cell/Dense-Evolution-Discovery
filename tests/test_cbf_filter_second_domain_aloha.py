"""
Loads the frozen results from scripts/robot_sensor_validation/
cbf_filter_second_domain_aloha.py -- a second, independent real physical
domain (ALOHA, bimanual 14-DOF, real 50Hz) for geometric_cbf_filter,
after the SO-101 evaluation. No re-run needed. See
docs/geometric_cbf_filter_real_joint_commands.md's second-domain section.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "robot_sensor_validation" / "cbf_filter_second_domain_aloha_frozen.json"
)


@pytest.fixture(scope="module")
def result():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_real_aloha_trials_checked(result):
    assert result["domain"] == "aloha_static_coffee"
    assert result["n_invariance_trials"] > 0
    assert result["n_invasiveness_checks"] > 0


def test_invariance_holds_on_every_real_aloha_trial_too(result):
    assert result["n_invariance_ok"] == result["n_invariance_trials"]


def test_minimal_invasiveness_is_exact_on_this_second_domain(result):
    """Even cleaner than SO-101 here: 0 nonzero deviations across all
    real per-step checks, not just >99.9%."""
    assert result["n_invasiveness_nonzero"] == 0
    assert result["max_invasiveness_deviation"] == 0.0
