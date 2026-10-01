"""
Cosmic-ray burst with zone-weighted decoding on a Steane [[7,1,3]] block.

Follow-up to cosmic_ray_erasure_decoding.py, changed to match what
arXiv:2104.05219 (McEwen et al.) measures:
  - errors are decay errors (|1> -> |0>), no excitation errors (Fig. S1). Each
    hit qubit gets amplitude damping with decay probability gamma, Pauli-twirled
    to X and Y with gamma/4 each and Z with ((1-sqrt(1-gamma))/2)**2. The twirl
    is an approximation of the non-Pauli channel.
  - the paper gives a hot ZONE with elevated error probability (time-averaged
    qubit heatmaps, Fig. 3c), not the list of qubits that erred in a given
    run. The decoder here therefore receives the zone as a prior (per-qubit
    error probabilities), not as erasures.

Decoders, all maximum-likelihood at the level of stabilizer cosets (exact,
no sampling): they differ only in the prior they assume.
  blind:      every qubit at the baseline rate.
  zone:       knows the hot-spot qubits and the burst strength at time t.
  mislocated: same strength, but the zone is placed on the wrong qubits.

The failure probability is computed by summing over all 4**7 Pauli errors under
the true prior. Burst strength follows cosmic_ray_burst_profile; its peak ratio
R multiplies the baseline decay probability. R=3.75 is the chip-wide count ratio
(15/4) used in this repo's Experiment 34; the paper's hot-spot qubits reach
higher rates, so R=10 is also shown.
"""
import itertools

import numpy as np
import jax.numpy as jnp

from dense_evolution import cosmic_ray_burst_profile

N = 7
STABILIZERS = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']
BASELINE_P = 0.01
TIMES_US = (0.0, 10.0, 100.0, 1000.0, 1500.0, 10000.0, 25000.0, 100000.0)
SIZES = tuple(range(1, N + 1))

_XZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
_IDX = np.array(list(itertools.product(range(4), repeat=N)))
_X, _Z = _XZ[_IDX][..., 0], _XZ[_IDX][..., 1]


def _sym(s):
    return np.array([[c in 'XY', c in 'ZY'] for c in s], dtype=int)


_SYN = np.zeros(len(_IDX), dtype=int)
for st in STABILIZERS:
    sx, sz = _sym(st).T
    _SYN = _SYN * 2 + ((_X @ sz + _Z @ sx) % 2)
_CLS = _SYN * 4 + (_X.sum(1) % 2) * 2 + (_Z.sum(1) % 2)


def pauli_probs(g, twirl=True):
    g = np.asarray(g, dtype=float)
    if not twirl:
        return np.stack([1 - g, g, 0 * g, 0 * g], axis=-1)
    s = np.sqrt(1 - g)
    return np.stack([((1 + s) / 2) ** 2, g / 4, g / 4, ((1 - s) / 2) ** 2], axis=-1)


def coset_probs(gammas, twirl=True):
    p = pauli_probs(gammas, twirl)
    pe = np.prod(p[np.arange(N), _IDX], axis=1)
    return np.bincount(_CLS, weights=pe, minlength=256).reshape(64, 4)


def failure_probability(q_true, q_dec):
    best = q_dec.argmax(axis=1)
    return float((q_true.sum(1) - q_true[np.arange(64), best]).sum())


def gammas(hot, g_hot):
    g = np.full(N, BASELINE_P)
    g[list(hot)] = g_hot
    return g


def hot_gamma(t_us, ratio):
    return float(cosmic_ray_burst_profile(
        jnp.array([t_us]), baseline_gamma=BASELINE_P,
        ratio_intermediate=ratio * 2.5 / 3.75, ratio_peak=ratio)[0])


def compare(k, t_us, ratio, twirl=True):
    g_hot = hot_gamma(t_us, ratio)
    hot = tuple(range(k))
    wrong = tuple((q + k) % N for q in range(k))
    q_true = coset_probs(gammas(hot, g_hot), twirl)
    return dict(
        blind=failure_probability(q_true, coset_probs(gammas((), 0), twirl)),
        zone=failure_probability(q_true, q_true),
        mislocated=failure_probability(q_true, coset_probs(gammas(wrong, g_hot), twirl)),
    )


if __name__ == "__main__":
    print("Check against the X-only model of cosmic_ray_erasure_decoding.py (k=2, t=1ms, R=3.75):")
    c = compare(2, 1000.0, 3.75, twirl=False)
    print(f"  blind {c['blind']:.4f}  zone {c['zone']:.4f}  mislocated {c['mislocated']:.4f}")

    for ratio in (3.75, 10.0):
        print(f"\nDecay errors (Pauli-twirled), peak ratio R={ratio}, t=1ms: failure probability by zone size k")
        print("  k   blind    zone     mislocated   zone gain")
        for k in SIZES:
            c = compare(k, 1000.0, ratio)
            gain = 1 - c['zone'] / c['blind']
            print(f"  {k}   {c['blind']:.4f}   {c['zone']:.4f}   {c['mislocated']:.4f}       {gain:6.1%}")
        print(f"  k=3, by time t (us): blind / zone / mislocated")
        for t in TIMES_US:
            c = compare(3, t, ratio)
            print(f"  t={t:>8.0f}   {c['blind']:.4f} / {c['zone']:.4f} / {c['mislocated']:.4f}")
