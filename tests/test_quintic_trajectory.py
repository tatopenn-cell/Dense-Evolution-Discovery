"""
Loads the frozen results from scripts/trajectory_planning/
quintic_validation_so101_aloha.py (quintic_trajectory checked against
real SO-101 and real ALOHA joint excursions) and checks the real
findings -- no re-download/re-run of the dataset needed here, this
only reads the already-committed frozen JSON. See
docs/quintic_trajectory_planner.md for the full write-up.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "trajectory_planning" / "quintic_validation_frozen.json"
)


@pytest.fixture(scope="module")
def results():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_both_domains_present(results):
    assert len(results["so101"]) == 6
    assert len(results["aloha"]) == 14


def test_boundary_conditions_exact_in_every_real_case(results):
    for r in results["so101"] + results["aloha"]:
        assert abs(r["boundary_q_start"] - r["q0"]) < 1e-6
        assert abs(r["boundary_q_end"] - r["qf"]) < 1e-6


def test_quintic_never_needs_more_peak_speed_than_the_real_trajectory(results):
    ratios = [r["ratio"] for r in results["so101"] + results["aloha"] if r["ratio"] is not None]
    assert len(ratios) == 20
    assert all(r < 1.0 for r in ratios)
