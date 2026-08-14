"""Test: 'Topologie Operative Resilienti' -- a claim from the pre-Dense-
Evolution "matrix" archive (Desktop/matrix/catterizzazione delle
'Topologie Ope.txt, TUREQ/TREC lineage, same origin as
channel_order_noncommutativity.py's Regole 100-109).

The archive's claim: for a fixed 3-operation gate set (one CX + two
single-qubit gates) on a 3-qubit circuit, SOME pairs of gate-order
permutations show "low maximum Jensen-Shannon distance across the whole
noise range" -- i.e. under noise, those two orderings' output
distributions stay close together (a "resilient operational topology"),
while other permutation pairs presumably don't. Tested across three
connectivity labels: Linear_3Q, Ring_3Q, Complete_3Q.

Important caveat, found and worth stating plainly: the archive file
itself contains ONLY qualitative discussion (the same boilerplate
paragraph repeated for every listed pair) -- no raw JS numbers, no
noise model/intensity, no script. There is nothing to re-run; this is a
fresh implementation of the claim as stated, not a reproduction of the
original computation. Two things follow from that:

1. The archive only lists SOME pairs as "resilient" (6 of the 15
   possible unordered pairs among 6 permutations, per topology), with
   no stated selection rule visible in the text. Testing only those
   6 would silently inherit whatever selection bias produced that list.
   This script tests all 15 pairs per topology instead, so "resilient"
   here is discovered here, not copied from the archive.
2. Dense-Evolution's DenseSVSimulator does not model hardware
   connectivity constraints -- any 2-qubit gate can act on any qubit
   pair regardless of a topology's edge list. So "Linear_3Q" and
   "Ring_3Q" (same ops, same CX target qubits, different *labeled*
   connectivity) simulate to IDENTICAL circuits here -- the
   connectivity label is conceptual bookkeeping in the archive, not a
   constraint enforced by this simulator. Included anyway, to test the
   claim exactly as partitioned, and to make this equivalence visible
   rather than silently skip one.

Method: for a fixed op set, gate-by-gate noisy simulation -- depolarizing
noise (NoiseModel, now the v8.1.57-fixed version) applied to the gate's
target qubit(s) immediately after each gate, single-shot Monte Carlo
unraveling (same style as channel_order_noncommutativity.py). For each
of the 15 permutation pairs, sweep several noise levels, compute the
Jensen-Shannon divergence between the two orderings' empirical output
distributions at each level (permutation test for significance), and
report the MAXIMUM JS distance across the swept range as the
"resilience" metric -- low max JS = order barely matters here ("resilient"),
high max JS = order matters somewhere in the range.

    python scripts/resilient_operational_topologies.py
"""
import itertools
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import dense_evolution as de
from dense_evolution.registry import NoiseModel

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 3
NOISE_LEVELS = (0.1, 0.2, 0.3, 0.4)
K_TRAJECTORIES = 800
N_PERMUTATIONS = 100

# (topology label, connectivity edges [cosmetic only, see module docstring],
#  op set: one CX + two single-qubit gates)
TOPOLOGIES = {
    "Linear_3Q": ([(0, 1), (1, 2)], [('x', [0]), ('z', [1]), ('cx', [0, 1])]),
    "Ring_3Q": ([(0, 1), (1, 2), (2, 0)], [('x', [0]), ('z', [1]), ('cx', [0, 1])]),
    "Complete_3Q": ([(0, 1), (0, 2), (1, 2)], [('x', [0]), ('z', [1]), ('cx', [0, 2])]),
}


def _target_qubits(op):
    name, qubits = op
    return qubits


def sample_outcome_gate_order(ops_ordered, noise_p, rng):
    """One Monte Carlo trajectory: apply ops_ordered in sequence, a fresh
    depolarizing draw on each gate's own target qubit(s) immediately
    after that gate, then sample a computational-basis outcome."""
    sim = de.DenseSVSimulator(N_QUBITS)
    for name, qubits in ops_ordered:
        sim.run_circuit([(name, *qubits)])
        sv = np.asarray(sim.get_statevector())
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, N_QUBITS, 'depolarizing', noise_p, rng=rng, qubits=qubits)
        sim.set_state(sv)
    sv = np.asarray(sim.get_statevector())
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


def js_at_noise_level(ops_a, ops_b, noise_p, seed):
    rng = np.random.default_rng(seed)
    outcomes_a = [o for o in (sample_outcome_gate_order(ops_a, noise_p, rng) for _ in range(K_TRAJECTORIES)) if o is not None]
    outcomes_b = [o for o in (sample_outcome_gate_order(ops_b, noise_p, rng) for _ in range(K_TRAJECTORIES)) if o is not None]
    dist_a = empirical_dist(np.array(outcomes_a))
    dist_b = empirical_dist(np.array(outcomes_b))
    observed_js = js_divergence(dist_a, dist_b)

    pooled = np.array(outcomes_a + outcomes_b)
    n_a = len(outcomes_a)
    null_js = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        null_js[i] = js_divergence(empirical_dist(pooled[:n_a]), empirical_dist(pooled[n_a:]))
    p_value = (np.sum(null_js >= observed_js) + 1) / (N_PERMUTATIONS + 1)
    return observed_js, p_value


def run_topology(topo_name, edges, op_set, seed_base):
    perms = list(itertools.permutations(op_set))  # Permutation 1..6, matching the archive's numbering
    pair_indices = list(itertools.combinations(range(6), 2))  # all 15 pairs, not just the archive's 6

    rows = []
    for pair_i, (i, j) in enumerate(pair_indices):
        ops_a, ops_b = list(perms[i]), list(perms[j])
        js_values, p_values = [], []
        for level_i, noise_p in enumerate(NOISE_LEVELS):
            seed = seed_base + 1000 * pair_i + level_i
            observed_js, p_value = js_at_noise_level(ops_a, ops_b, noise_p, seed)
            js_values.append(observed_js)
            p_values.append(p_value)
        max_js = max(js_values)
        min_p = min(p_values)
        rows.append({
            "topology": topo_name,
            "perm_i": i + 1, "perm_j": j + 1,
            "order_i": ",".join(n for n, _ in ops_a), "order_j": ",".join(n for n, _ in ops_b),
            "max_js": max_js, "min_p_value": min_p,
            "js_at_each_level": ";".join(f"{v:.5f}" for v in js_values),
        })
        print(f"  {topo_name} perm{i+1} vs perm{j+1}: max_js={max_js:.5f}  min_p={min_p:.4f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    all_dfs = []
    for topo_i, (topo_name, (edges, op_set)) in enumerate(TOPOLOGIES.items()):
        print(f"\n=== {topo_name} (connectivity {edges}, ops {op_set}) ===")
        df_topo = run_topology(topo_name, edges, op_set, seed_base=42 + topo_i * 100000)
        all_dfs.append(df_topo)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(_DATA_DIR / "resilient_operational_topologies.csv", index=False)

    median_max_js = df["max_js"].median()
    df["resilient"] = df["max_js"] < median_max_js

    print("\n=== Summary ===")
    for topo_name in TOPOLOGIES:
        sub = df[df["topology"] == topo_name]
        print(f"{topo_name}: {sub['resilient'].sum()}/{len(sub)} pairs below median max_js "
              f"(median={median_max_js:.5f}), range [{sub['max_js'].min():.5f}, {sub['max_js'].max():.5f}]")

    lin = df[df["topology"] == "Linear_3Q"]["max_js"].values
    ring = df[df["topology"] == "Ring_3Q"]["max_js"].values
    identical = np.allclose(lin, ring, atol=1e-9)
    print(f"\nLinear_3Q vs Ring_3Q max_js identical (expected -- simulator doesn't model connectivity): {identical}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, topo_name in zip(axes, TOPOLOGIES):
        sub = df[df["topology"] == topo_name].sort_values("max_js")
        labels = [f"{int(r.perm_i)} vs {int(r.perm_j)}" for r in sub.itertuples()]
        colors = ['#2980b9' if r < median_max_js else '#c0392b' for r in sub["max_js"]]
        ax.barh(range(len(sub)), sub["max_js"], color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(median_max_js, color='#888888', linestyle='--', linewidth=1, label=f'median={median_max_js:.4f}')
        ax.set_xlabel("max JS distance across noise range")
        ax.set_title(topo_name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='x')
    fig.suptitle("Resilient Operational Topologies: max JS distance per permutation pair, all 15 pairs/topology", fontweight='bold')
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "resilient_operational_topologies.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'resilient_operational_topologies.png'}")
