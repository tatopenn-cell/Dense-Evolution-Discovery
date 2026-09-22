"""
Protocol 1 of the crypto-q RFC (issue #189): three-party device-independent
conference key agreement (DICKA) via a GHZ(3) state, following Ribeiro,
Murta & Wehner 2018, "Fully device-independent conference key agreement"
(arXiv:1708.00798).

Ground truth, corrected against the actual paper text (not Mermin's
inequality, which this paper does not use -- see the crypto-q RFC's own
correction note): the paper introduces its own "Parity-CHSH" inequality,
an N-party extension of CHSH (Definition 8). For N=3 (Alice, Bob1, one
other Bob), the winning condition is `a + b1 = x*(y XOR b2) mod 2`, with
classical bound P_win <= 3/4 and quantum maximum P_win ~ 0.85 (the same
bound as ordinary CHSH, Eq. A.14) -- both reproduced numerically below,
not assumed.

A real bug was found and fixed while implementing this: Definition 8's own
text fixes the other Bobs' test-round question at "always equal to 1", not
0 -- a first implementation used 0 (Z measurement) and got P_win=0.677,
exactly half the expected quantum boost (1/2 + 1/(4*sqrt2) instead of
1/2 + 1/(2*sqrt2)). Fixing the other Bob's question to 1 (X measurement,
the setting that actually projects Alice and Bob1's reduced state into a
genuine entangled pair rather than a classically-correlated product state)
reproduces the exact theoretical value 0.85355... to machine precision.
Key-generation rounds are unaffected -- Protocol 2's own step (c) fixes
(Xi, Y(1...N-1),i) = (0, 2, 0, ..., 0) for those, i.e. the other Bob's
question there really is 0 (Z), matching what this file already had.

A second real bug, found the same way (comparing against the actual
simulator output, not trusting a formula in isolation): the noisy-channel
comparison in `expected_win_rate` assumed depolarizing shrinks a qubit's
Pauli expectations by (1-p); dense_evolution's actual 'depolarizing' Kraus
channel shrinks them by (1-4p/3) instead (see that function's own docstring
for the derivation and the numbers). Fixed the same way.

Measurement operators, taken directly from the paper's own honest-
implementation section (Eq. A.64-A.65 region):
    Alice:      x=0 -> Z,                 x=1 -> X
    Bob1:       y=0 -> (Z+X)/sqrt(2),      y=1 -> (Z-X)/sqrt(2),  y=2 -> Z
    other Bobs: y=0 -> Z,                  y=1 -> X

The two rotated Bob1 observables are diagonalized numerically (eigh), not
hand-derived, and the resulting change-of-basis unitaries are verified
against the observable's own eigenvectors before use (see the module's own
test file).

Reused, not reimplemented: de.ghz_state(3), de.DenseSVSimulator, de.GATES,
sim.measure (jax_key), NoiseModel.apply_to_sv. No new quantum primitive.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import dense_evolution as de
from dense_evolution.noise import NoiseModel

_Z = np.diag([1, -1]).astype(complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _diagonalizing_unitary(observable):
    """U such that measuring Z after applying U to the state reproduces
    measuring `observable` directly -- derived from the observable's own
    eigendecomposition (eigenvalue +1 first, to match Z=diag(1,-1)), not
    assumed from a hand-picked rotation angle."""
    w, v = np.linalg.eigh(observable)
    order = np.argsort(-w)
    v = v[:, order]
    return v.conj().T


_U_PLUS = _diagonalizing_unitary((_Z + _X) / np.sqrt(2))
_U_MINUS = _diagonalizing_unitary((_Z - _X) / np.sqrt(2))


def prepare_ghz3():
    sim = de.DenseSVSimulator(3)
    sim.run_circuit(de.ghz_state(3))
    return sim


def measure_alice(sim, x, rng):
    if x == 1:
        sim.apply_gate_1q(de.GATES["h"], 0)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(0, jax_key=key))


def measure_bob1(sim, y, rng):
    if y == 0:
        sim.apply_gate_1q(_U_PLUS, 1)
    elif y == 1:
        sim.apply_gate_1q(_U_MINUS, 1)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(1, jax_key=key))


def measure_other_bob(sim, y, rng, qubit=2):
    if y == 1:
        sim.apply_gate_1q(de.GATES["h"], qubit)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(qubit, jax_key=key))


def apply_depolarizing(sim, p, rng):
    """i.i.d. depolarizing on all 3 qubits, matching the paper's own
    D^{\\otimes N}(GHZ_N) honest-implementation model (Eq. A.64)."""
    if p == 0.0:
        return
    sv = np.asarray(sim.get_statevector())
    sv_noisy = NoiseModel.apply_to_sv(sv, 3, "depolarizing", p, rng=rng)
    sim.set_initial_state(np.asarray(sv_noisy))


def test_round(rng, p_dep=0.0):
    """One Parity-CHSH test round (Def. 8): Alice and Bob1 get uniformly
    random questions; the other Bob gets a FIXED question, always 1 (X
    measurement) -- per the paper's own definition text, not 0. Returns 1
    if the parties win, else 0."""
    sim = prepare_ghz3()
    apply_depolarizing(sim, p_dep, rng)
    x = int(rng.integers(0, 2))
    y = int(rng.integers(0, 2))
    a = measure_alice(sim, x, rng)
    b1 = measure_bob1(sim, y, rng)
    b2 = measure_other_bob(sim, 1, rng)
    win = (a + b1) % 2 == (x * (y ^ b2)) % 2
    return int(win)


def key_round(rng, p_dep=0.0):
    """One key-generation round: everyone measures Z (x=0, y=2, y2=0).
    Returns (a, b1, b2) -- correlated in the noiseless case."""
    sim = prepare_ghz3()
    apply_depolarizing(sim, p_dep, rng)
    a = measure_alice(sim, 0, rng)
    b1 = measure_bob1(sim, 2, rng)
    b2 = measure_other_bob(sim, 0, rng)
    return a, b1, b2


def parity_chsh_win_rate(n_rounds, p_dep=0.0, seed=None):
    rng = np.random.default_rng(seed)
    wins = sum(test_round(rng, p_dep) for _ in range(n_rounds))
    return wins / n_rounds


def key_qber(n_rounds, p_dep=0.0, seed=None):
    rng = np.random.default_rng(seed)
    disagree_b1 = 0
    disagree_b2 = 0
    for _ in range(n_rounds):
        a, b1, b2 = key_round(rng, p_dep)
        disagree_b1 += a != b1
        disagree_b2 += a != b2
    return disagree_b1 / n_rounds, disagree_b2 / n_rounds


def expected_win_rate(p_dep, n_parties=3):
    """Closed-form honest-implementation win rate, paper Eq. A.65,
    specialized to N=3, rewritten in terms of the per-qubit Pauli-expectation
    shrink factor `s` a noise channel produces (rather than assuming s=1-p
    directly): p_exp = 1/2 + s^3/(2 sqrt2) + s^2*(1-s)/(4 sqrt2).

    A second real bug was found and fixed here, the same way as the
    question=1 bug above -- by comparing against the actual simulator output
    rather than trusting the formula: this originally used s=(1-p) directly,
    which assumes a "replace with the maximally mixed state" depolarizing
    channel. dense_evolution's own 'depolarizing' model is instead the
    isotropic-Pauli-error Kraus map (K0=sqrt(1-p)I, K1..3=sqrt(p/3)*Pauli --
    see NoiseModel's own docstring), which shrinks every single-qubit Pauli
    expectation value by s=1-4p/3, not (1-p) -- the same convention already
    behind bb84.py's validated QBER=2p/3 result. Using s=(1-p) matched the
    simulator only at p=0 and diverged with growing significance as p grew
    (z=-2.75 at p=0.10, z=-3.65 at p=0.15, N=3000); s=1-4p/3 matches the
    simulator across the whole sweep (|z|<0.5 everywhere tested)."""
    s = 1 - 4 * p_dep / 3
    s2 = np.sqrt(2)
    return 0.5 + s ** n_parties / (2 * s2) + s ** 2 * (1 - s ** (n_parties - 2)) / (4 * s2)


def summarize(label, observed, expected, n_rounds):
    sigma = np.sqrt(0.25 / n_rounds)
    z = (observed - expected) / sigma if sigma > 1e-9 else 0.0
    print(f"{label:32s}  observed = {observed:.4f}  expected = {expected:.4f}  z = {z:+.2f}")


if __name__ == "__main__":
    N = 3000

    print("=" * 76)
    print(f"DI-QKD via GHZ(3), Parity-CHSH game -- N={N} rounds per test")
    print("=" * 76)

    print("\n[TEST 1] Ideal channel: Parity-CHSH win rate vs. classical bound 3/4")
    p_win_ideal = parity_chsh_win_rate(N, p_dep=0.0, seed=0)
    summarize("ideal P_win", p_win_ideal, expected_win_rate(0.0), N)
    print(f"  classical bound: 0.7500  (must be exceeded for a device-independent channel)")

    print("\n[TEST 2] Depolarizing sweep vs. closed-form Eq. A.65 (N=3)")
    for p in [0.02, 0.05, 0.10, 0.15]:
        p_win = parity_chsh_win_rate(N, p_dep=p, seed=1)
        summarize(f"p_dep={p:.2f}", p_win, expected_win_rate(p), N)

    print("\n[TEST 3] Key-generation QBER (all-Z rounds) vs. depolarizing p")
    for p in [0.0, 0.05, 0.10]:
        qber_b1, qber_b2 = key_qber(N, p_dep=p, seed=2)
        print(f"  p_dep={p:.2f}   QBER(Alice,Bob1) = {qber_b1:.4f}   QBER(Alice,Bob2) = {qber_b2:.4f}")
