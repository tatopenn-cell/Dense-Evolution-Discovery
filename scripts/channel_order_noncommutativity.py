"""
Channel-order non-commutativity: does applying one noise channel then
another give a different result than the reverse order?

Origin: Regole 100-109 in the "matrix" archive (an August-September 2025
personal research project, the direct predecessor of Dense-Evolution --
see SOPHIA_REFLECTION.md/scripts/sophia_reflection.py for that lineage)
claimed noise-channel ORDER *always* leaves a measurable fingerprint on
the final state -- the "Attrattori Crono-Topologici" ("chrono-topological
attractors") idea. Unlike most other claims from that archive (tested and
found to be no different from generic noise physics -- see
tests/test_channel_order_noncommutativity.py's sibling investigation),
this one is partly real -- but not for the reason, or with the generality,
the archive claimed.

Testing two channel pairs on the same Regola 16 circuit settles it:

- dephasing (phaseflip) -> amplitude damping vs the reverse: REAL signal,
  p=0.0033 (Jensen-Shannon divergence far outside the permutation null).
- dephasing (phaseflip) -> bit-flip vs the reverse: NO signal, p~0.17.

The dividing line is not "any two channels", it's Pauli vs non-Pauli.
Phaseflip and bitflip are both Pauli channels, and Pauli channels commute
with each other as superoperators (they're simultaneously diagonal in the
Pauli-transfer-matrix basis) -- so of course their order doesn't matter.
Amplitude damping is non-unital (it has a preferred fixed point, |0>) and
is *not* a Pauli channel, which is exactly why composing it with a Pauli
channel is order-dependent. So Regole 100-109's blanket "order always
matters" is an overgeneralization from one true instance -- the real,
precise rule is: order matters iff at least one of the two channels is
non-Pauli.

Method: Monte Carlo unraveling. For each of K trajectories, apply
channel 1 then channel 2 (order AB) vs channel 2 then channel 1 (order BA)
to the SAME ideal statevector, each with its own fresh random draw, then
sample a measurement outcome. Averaging many single-trajectory outer
products over a channel's random Kraus realization exactly reproduces
that channel's true density-matrix action, so the empirical outcome
distribution over K trajectories is a genuine Monte Carlo estimate of
each order's true output distribution.

Compares the two empirical distributions with Jensen-Shannon divergence,
then a permutation test (reshuffle which trajectory belongs to order AB
vs BA many times) for an honest p-value against the null "order doesn't
matter, sampling noise alone explains it."

Produces `data/channel_order_noncommutativity.csv` and
`images/channel_order_noncommutativity.png`.

    python scripts/channel_order_noncommutativity.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de
from dense_evolution.registry import NoiseModel

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 3
CHANNEL_1 = 'phaseflip'          # "Dephasing" -- default pair, kept for
CHANNEL_2 = 'amplitude_damping'  # "AD"        -- backward-compatible calls
P1 = 0.3
P2 = 0.3

CHANNEL_PAIRS = {
    'dephasing_vs_AD': ('phaseflip', 'amplitude_damping'),   # non-Pauli involved
    'dephasing_vs_bitflip': ('phaseflip', 'bitflip'),        # both Pauli
}


def ideal_sv():
    """Regola 16 circuit: GHZ(3q) -> X(Q0) -> Z(Q1) -> X(Q2) -> CNOT(Q0,Q2)."""
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([('h', 0), ('cx', 0, 1), ('cx', 1, 2), ('x', 0), ('z', 1), ('x', 2), ('cx', 0, 2)])
    return np.asarray(sim.get_statevector())


def sample_outcome(sv0, order, rng, p1=P1, p2=P2, channel_1=CHANNEL_1, channel_2=CHANNEL_2):
    """order='AB': channel_1 then channel_2. order='BA': channel_2 then channel_1."""
    sv = sv0.copy()
    if order == 'AB':
        sv = NoiseModel.apply_to_sv(sv, N_QUBITS, channel_1, p1, rng=rng)
        sv = NoiseModel.apply_to_sv(sv, N_QUBITS, channel_2, p2, rng=rng)
    else:
        sv = NoiseModel.apply_to_sv(sv, N_QUBITS, channel_2, p2, rng=rng)
        sv = NoiseModel.apply_to_sv(sv, N_QUBITS, channel_1, p1, rng=rng)
    probs = np.abs(sv) ** 2
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    probs = probs / total
    return rng.choice(len(probs), p=probs)


def empirical_dist(outcomes, dim=8):
    counts = np.bincount(outcomes, minlength=dim).astype(float)
    return counts / counts.sum()


def js_divergence(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_trial(k_trajectories, n_permutations, seed, channel_1=CHANNEL_1, channel_2=CHANNEL_2):
    """Returns (dist_ab, dist_ba, observed_js, null_js_array, p_value)."""
    sv0 = ideal_sv()
    rng = np.random.default_rng(seed)

    outcomes_ab = [o for o in (sample_outcome(sv0, 'AB', rng, channel_1=channel_1, channel_2=channel_2) for _ in range(k_trajectories)) if o is not None]
    outcomes_ba = [o for o in (sample_outcome(sv0, 'BA', rng, channel_1=channel_1, channel_2=channel_2) for _ in range(k_trajectories)) if o is not None]

    dist_ab = empirical_dist(np.array(outcomes_ab))
    dist_ba = empirical_dist(np.array(outcomes_ba))
    observed_js = js_divergence(dist_ab, dist_ba)

    pooled = np.array(outcomes_ab + outcomes_ba)
    n_ab = len(outcomes_ab)
    null_js = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(pooled)
        null_js[i] = js_divergence(empirical_dist(pooled[:n_ab]), empirical_dist(pooled[n_ab:]))

    p_value = (np.sum(null_js >= observed_js) + 1) / (n_permutations + 1)
    return dist_ab, dist_ba, observed_js, null_js, p_value


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    results = {}
    for pair_name, (c1, c2) in CHANNEL_PAIRS.items():
        results[pair_name] = run_trial(
            k_trajectories=8192, n_permutations=300, seed=99, channel_1=c1, channel_2=c2
        )

    rows = []
    for pair_name, (c1, c2) in CHANNEL_PAIRS.items():
        dist_ab, dist_ba, observed_js, null_js, p_value = results[pair_name]
        for i in range(8):
            rows.append({
                'pair': pair_name, 'channel_1': c1, 'channel_2': c2,
                'state': format(i, '03b'),
                f'P_{c1}_then_{c2}': dist_ab[i],
                f'P_{c2}_then_{c1}': dist_ba[i],
            })
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "channel_order_noncommutativity.csv", index=False)

    for pair_name, (c1, c2) in CHANNEL_PAIRS.items():
        dist_ab, dist_ba, observed_js, null_js, p_value = results[pair_name]
        print(f"\n=== {pair_name}: {c1} <-> {c2} ===")
        for i in range(8):
            print(f"{format(i,'03b')}  P({c1}->{c2})={dist_ab[i]:.4f}  P({c2}->{c1})={dist_ba[i]:.4f}")
        print(f"Observed Jensen-Shannon divergence: {observed_js:.6f}")
        print(f"Null JS (order doesn't matter): mean={null_js.mean():.6f} +/- {null_js.std():.6f}, "
              f"max={null_js.max():.6f}")
        print(f"p-value (observed JS is just sampling noise): {p_value:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = np.arange(8)
    width = 0.35
    for row, (pair_name, (c1, c2)) in enumerate(CHANNEL_PAIRS.items()):
        dist_ab, dist_ba, observed_js, null_js, p_value = results[pair_name]
        ax1, ax2 = axes[row]

        ax1.bar(x - width / 2, dist_ab, width, label=f'{c1} -> {c2}', color='#2980b9')
        ax1.bar(x + width / 2, dist_ba, width, label=f'{c2} -> {c1}', color='#c0392b')
        ax1.set_xticks(x)
        ax1.set_xticklabels([format(i, '03b') for i in range(8)], rotation=45)
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Regola 16, {pair_name}')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')

        ax2.hist(null_js, bins=30, color='#95a5a6', alpha=0.8, label='Null (order shuffled)')
        ax2.axvline(observed_js, color='#c0392b', linewidth=2, label=f'Observed JS={observed_js:.5f}')
        ax2.set_xlabel('Jensen-Shannon divergence')
        ax2.set_ylabel('Count (permutations)')
        ax2.set_title(f'{pair_name}: p={p_value:.4f}')
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "channel_order_noncommutativity.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'channel_order_noncommutativity.png'}")

    print("\n=== Conclusion ===")
    print("Order matters iff at least one channel is non-Pauli (e.g. amplitude")
    print("damping). Two Pauli channels (dephasing, bit-flip) commute as")
    print("superoperators -- Regole 100-109's blanket 'order always matters'")
    print("does not hold in general.")
