"""
Protocol 2 of the crypto-q RFC (issue #189): the full multi-round DICKA
structure from Ribeiro, Murta & Wehner 2018 (arXiv:1708.00798), Appendix
Protocol 2 -- "a more detailed version of Protocol 1". Builds directly on
`di_qkd_ghz.py`'s already-validated single-round primitives (GHZ(3)
preparation, measurements, depolarizing noise, the Parity-CHSH game, the
all-Z key round); this module adds the round-selection and parameter-
estimation layer around them.

Per-round structure (paper's own text, step 1):
    (a) prepare the shared GHZ(3) state.
    (b) Alice picks T_i ~ Bernoulli(gamma) and announces it.
    (c) T_i=0 -> a key round: everyone measures Z (Xi, Y = 0, 2, 0).
        T_i=1 -> a test round: Alice/Bob1 get random questions, the other
        Bob gets question=1 (Definition 8) -- this is the Parity-CHSH game.
    Repeated for n rounds; the protocol aborts if the observed winning
    frequency on test rounds falls below a threshold beta in
    ]3/4, 1/2+1/(2 sqrt2)[.

What this does NOT reproduce, and why: Theorem 4's exact secure key length
`l` is a function of f-tilde(beta), a bound on the single-round von Neumann
entropy defined in the paper's own Lemma 3 as the unique tangent to a convex
function at an optimized point `popt` -- not a closed-form expression, but
the output of a separate numerical (SDP-style) optimization the paper
carries out on its own. Inventing a number for it here would mean reporting
something the paper never actually gives in closed form, which this
project's own rule is not to do (see the module docstring convention
established in di_qkd_ghz.py and the crypto-q RFC). What IS reproduced,
against the real simulator, is everything Protocol 2 actually specifies as
a physical procedure: the round selection, the honest-implementation
winning frequency (parameter estimation), the abort decision, and the raw
key's QBER before error correction -- i.e. everything up to, but not
including, the privacy-amplification key-rate number.
"""

import numpy as np

from di_qkd_ghz import (
    prepare_ghz3,
    measure_alice,
    measure_bob1,
    measure_other_bob,
    apply_depolarizing,
    expected_win_rate,
)

CLASSICAL_BOUND = 0.75
QUANTUM_MAX = 0.5 + 1 / (2 * np.sqrt(2))


def run_round(rng, gamma, p_dep=0.0):
    """One round of Protocol 2, step 1(a)-(c). Returns either
    ('key', a, b1, b2) or ('test', win)."""
    sim = prepare_ghz3()
    apply_depolarizing(sim, p_dep, rng)
    if rng.random() < gamma:
        x = int(rng.integers(0, 2))
        y = int(rng.integers(0, 2))
        a = measure_alice(sim, x, rng)
        b1 = measure_bob1(sim, y, rng)
        b2 = measure_other_bob(sim, 1, rng)
        win = int((a + b1) % 2 == (x * (y ^ b2)) % 2)
        return ("test", win)
    a = measure_alice(sim, 0, rng)
    b1 = measure_bob1(sim, 2, rng)
    b2 = measure_other_bob(sim, 0, rng)
    return ("key", a, b1, b2)


def run_protocol(n_rounds, gamma, beta, p_dep=0.0, seed=None):
    """Runs Protocol 2 for n_rounds, then applies the abort rule from step
    1: abort if the observed test-round win frequency is below `beta`.
    Returns a dict of everything the protocol itself produces (not the
    privacy-amplified final key length -- see module docstring)."""
    rng = np.random.default_rng(seed)
    test_wins = test_total = 0
    key_a, key_b1, key_b2 = [], [], []
    for _ in range(n_rounds):
        outcome = run_round(rng, gamma, p_dep)
        if outcome[0] == "test":
            test_total += 1
            test_wins += outcome[1]
        else:
            _, a, b1, b2 = outcome
            key_a.append(a)
            key_b1.append(b1)
            key_b2.append(b2)

    p_hat = test_wins / test_total if test_total else 0.0
    aborted = p_hat < beta
    key_a, key_b1, key_b2 = map(np.array, (key_a, key_b1, key_b2))
    qber_b1 = float(np.mean(key_a != key_b1)) if len(key_a) else 0.0
    qber_b2 = float(np.mean(key_a != key_b2)) if len(key_a) else 0.0

    return {
        "n_rounds": n_rounds,
        "n_test": test_total,
        "n_key": len(key_a),
        "p_hat": p_hat,
        "beta": beta,
        "aborted": aborted,
        "qber_b1": qber_b1,
        "qber_b2": qber_b2,
    }


def summarize(result, p_dep):
    status = "ABORT" if result["aborted"] else "proceed"
    print(
        f"p_dep={p_dep:.2f}  n_test={result['n_test']:4d}  n_key={result['n_key']:4d}  "
        f"p_hat={result['p_hat']:.4f}  beta={result['beta']:.4f}  [{status}]  "
        f"QBER(A,B1)={result['qber_b1']:.4f}  QBER(A,B2)={result['qber_b2']:.4f}"
    )


if __name__ == "__main__":
    N = 2000
    GAMMA = 0.7
    print("=" * 88)
    print(f"DICKA Protocol 2 -- N={N} rounds, gamma={GAMMA} (fraction of key rounds)")
    print("=" * 88)

    print("\n[TEST 1] Threshold set halfway between classical bound and quantum max:")
    beta = (CLASSICAL_BOUND + QUANTUM_MAX) / 2
    print(f"  beta = {beta:.4f}  (classical={CLASSICAL_BOUND:.4f}, quantum max={QUANTUM_MAX:.4f})")
    for p in [0.0, 0.05, 0.10, 0.15]:
        result = run_protocol(N, GAMMA, beta, p_dep=p, seed=0)
        summarize(result, p)

    print("\n[TEST 2] Same sweep, but with beta close to the quantum max (harder to pass):")
    beta_strict = QUANTUM_MAX - 0.01
    print(f"  beta = {beta_strict:.4f}")
    for p in [0.0, 0.05, 0.10, 0.15]:
        result = run_protocol(N, GAMMA, beta_strict, p_dep=p, seed=1)
        summarize(result, p)

    print(
        "\nNote: this reports the protocol's own observables (parameter estimation,\n"
        "abort decision, raw-key QBER) -- not a secure key length. See the module\n"
        "docstring for why Theorem 4's exact `l` is not reproduced here."
    )
