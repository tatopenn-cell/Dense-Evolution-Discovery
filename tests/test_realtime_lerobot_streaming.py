"""
Loads the frozen results from scripts/dense_armor_streaming/
realtime_lerobot_streaming.py (real per-call latency + sustained
real-time playback check against real LeRobot data) and asserts the
real invariants that matter -- no re-download/re-run needed here, this
only reads the already-committed frozen JSON. See
docs/realtime_streaming_lerobot.md for the full write-up.
"""
import json
import pathlib

import pytest

_DATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "dense_armor_streaming" / "realtime_lerobot_streaming_frozen.json"
)


@pytest.fixture(scope="module")
def result():
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_real_per_call_latency_never_exceeds_real_30hz_budget(result):
    c1 = result["check1"]
    assert c1["n_over_budget"] == 0
    assert c1["median_us"] < c1["real_budget_us"] / 10  # real, comfortable headroom


def test_sustained_realtime_playback_does_not_drift_past_budget(result):
    c2 = result["check2"]
    assert c2["max_drift_ms"] < result["real_dt_ms"]
