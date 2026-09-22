"""
Correctness tests for scripts/crypto/dicka_protocol2.py -- Protocol 2 of the
crypto-q RFC (the multi-round DICKA structure around Protocol 1's Parity-
CHSH game). Uses a smaller N than the module's own validation run (N=2000)
since these tests must run fast in CI; the statistical bands below are
correspondingly wider, not tighter, so a passing test here doesn't replace
the fuller validation already documented in docs/crypto/dicka_protocol2.md,
just checks the same scenarios haven't regressed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "crypto"))

from dicka_protocol2 import run_protocol, CLASSICAL_BOUND, QUANTUM_MAX

BETA = (CLASSICAL_BOUND + QUANTUM_MAX) / 2


def test_ideal_channel_proceeds():
    result = run_protocol(400, gamma=0.7, beta=BETA, p_dep=0.0, seed=0)
    assert not result["aborted"]
    assert result["qber_b1"] == 0.0
    assert result["qber_b2"] == 0.0


def test_noisy_channel_aborts():
    result = run_protocol(400, gamma=0.7, beta=BETA, p_dep=0.15, seed=1)
    assert result["aborted"]


def test_qber_rises_with_noise():
    low = run_protocol(400, gamma=0.7, beta=BETA, p_dep=0.02, seed=2)
    high = run_protocol(400, gamma=0.7, beta=BETA, p_dep=0.15, seed=2)
    assert high["qber_b1"] > low["qber_b1"]


def test_round_counts_match_gamma_roughly():
    result = run_protocol(1000, gamma=0.7, beta=BETA, p_dep=0.0, seed=3)
    assert result["n_test"] + result["n_key"] == 1000
    assert 600 < result["n_test"] < 800
