"""
Loads the frozen results from scripts/trajectory_planning/
kinematic_controller_validation_so101_aloha.py (kinematic_tracking_
controller chained with quintic_trajectory, checked against real
SO-101 and real ALOHA joint excursions with a real disclosed initial
tracking-error perturbation) and checks the real findings -- no
re-download/re-run of the dataset needed here, this only reads the
already-committed frozen JSON. See docs/kinematic_tracking_
controller.md for the full write-up.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "trajectory_planning" / "kinematic_controller_validation_frozen.json"
)


@pytest.fixture(scope="module")
def results():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_both_domains_present(results):
    assert len(results["so101"]) == 6
    assert len(results["aloha"]) == 14


def test_every_real_joint_excursion_converges(results):
    for r in results["so101"] + results["aloha"]:
        assert abs(r["initial_error"]) > 0.01, "perturbation must be real and nonzero for this to test anything"
        assert abs(r["final_error"]) < abs(r["initial_error"]) * 0.05
