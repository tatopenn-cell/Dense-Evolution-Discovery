"""
Correctness tests for scripts/crypto/di_qkd_ghz.py -- Protocol 1 of the
crypto-q RFC (DI-QKD via GHZ(3)). Uses a smaller N than the module's own
validation run (N=3000) since these tests must run fast in CI; the
statistical bands below are correspondingly wider, not tighter, so a
passing test here doesn't replace the fuller validation already documented
in docs/crypto/di_qkd_ghz.md, just checks the same scenarios haven't
regressed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "crypto"))

from di_qkd_ghz import parity_chsh_win_rate, expected_win_rate, key_qber


def test_ideal_channel_beats_classical_bound():
    p_win = parity_chsh_win_rate(500, p_dep=0.0, seed=0)
    assert p_win > 0.75


def test_ideal_channel_matches_quantum_maximum():
    p_win = parity_chsh_win_rate(500, p_dep=0.0, seed=0)
    assert abs(p_win - expected_win_rate(0.0)) < 0.05


def test_depolarizing_sweep_matches_closed_form():
    for p in [0.05, 0.10]:
        p_win = parity_chsh_win_rate(500, p_dep=p, seed=1)
        assert abs(p_win - expected_win_rate(p)) < 0.05


def test_key_generation_zero_qber_on_ideal_channel():
    qber_b1, qber_b2 = key_qber(500, p_dep=0.0, seed=2)
    assert qber_b1 == 0.0
    assert qber_b2 == 0.0


def test_key_generation_qber_rises_with_noise():
    _, qber_ideal = key_qber(500, p_dep=0.0, seed=3)
    _, qber_noisy = key_qber(500, p_dep=0.10, seed=3)
    assert qber_noisy > qber_ideal
