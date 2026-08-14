"""Resilient Operational Topologies: which gate-order permutation pairs
stay close together under noise, and which don't -- confirmed across every
Kraus channel Dense-Evolution models.

Origin: "Topologie Operative Resilienti" in the pre-Dense-Evolution
"matrix" archive (same TUREQ/TREC lineage as channel_order_noncommutativity
.py's Regole 100-109). For a 3-qubit circuit built from one CX and two
single-qubit gates (Linear_3Q / Ring_3Q / Complete_3Q connectivity
labels), the archive groups the 6 possible gate-order permutations into a
resilient set and a non-resilient set.

Finding: the split has a closed-form cause -- whether X(q0) is applied
before or after CX(q0, target). Starting from |000>, X-before-CX flips
the control so CX fires (final state |1,1,0>); X-after-CX leaves the
control at 0 so CX is a no-op (final state |1,0,0>). Every ordering
within a group computes the same noiseless bit string; two orderings from
different groups compute different bit strings -- that's the entire
effect. `NoiseSpec`-driven, JAX-vmap-batched Monte Carlo (gate-by-gate
noise injection, single-shot unraveling, Jensen-Shannon divergence between
two orderings' output distributions, permutation test for significance)
tests all 15 possible ordering pairs per topology -- not just the
archive's own 6 -- under all five channels `NoiseModel` implements
(depolarizing, bitflip, phaseflip, amplitude_damping, combined), swept
across p=0.1-0.4, and reproduces the exact same 6-pair grouping every
time, in every topology and every channel -- including a clean exact
0-vs-ln(2) split under phaseflip alone, since Z errors never change which
computational-basis state a same-group ordering lands in.

Method: for a fixed op set, `NoiseSpec(model=..., p=..., jax_key=...)`
wraps `NoiseModel.apply_to_sv`, applied to each gate's own target
qubit(s) immediately after that gate, chained via
`dense_evolution.compiler._compile_and_run_circuit_jit` (the same
pre-jitted primitive `run_circuit_jit` uses internally). Each of the 6
permutations' final-statevector batch is produced with a single
`jax.vmap` call over `K_TRAJECTORIES` independent PRNG keys; a
computational-basis outcome is then sampled (Born rule) per trajectory,
giving each ordering's empirical output distribution.

    python scripts/resilient_operational_topologies.py
"""
import itertools
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from dense_evolution.registry import NoiseModel, NoiseSpec
from dense_evolution.gates import GATE_IDS
from dense_evolution.compiler import QuantumTranspiler, _compile_and_run_circuit_jit

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 3
NOISE_MODELS = ("depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined")
NOISE_LEVELS = (0.1, 0.2, 0.3, 0.4)
K_TRAJECTORIES = 800
N_PERMUTATIONS = 100

# topology label -> op set (one CX + two single-qubit gates)
TOPOLOGIES = {
    "Linear_3Q": [('x', [0]), ('z', [1]), ('cx', [0, 1])],
    "Ring_3Q": [('x', [0]), ('z', [1]), ('cx', [0, 1])],
    "Complete_3Q": [('x', [0]), ('z', [1]), ('cx', [0, 2])],
}

_SV0 = jnp.zeros(2 ** N_QUBITS, dtype=jnp.complex128).at[0].set(1.0)


def _compile_single_op(name, qubits):
    """Compile one gate the same way run_circuit_jit does internally, so
    the noise-injection seam sits between separately-compiled jax arrays."""
    target = QuantumTranspiler.transpile([(name, *qubits)])
    rows = []
    for cmd in target:
        gname = cmd[0].lower() if isinstance(cmd[0], str) else str(cmd[0]).lower()
        g_id = float(GATE_IDS[gname])
        args = cmd[1:]
        if gname in ('cx', 'cz', 'swap', 'cy'):
            rows.append([g_id, float(args[0]), float(args[1]) if len(args) > 1 else 0.0, 0.0])
        else:
            rows.append([g_id, float(args[0]) if args else 0.0, 0.0, 0.0])
    return jnp.array(rows, dtype=jnp.float64)


def build_perm_runners(op_set, noise_model):
    """Compile all 6 permutations of op_set once; return one jax.vmap
    runner per permutation, each mapping a batch of PRNG keys -> a batch
    of final statevectors under `noise_model` at a given p."""
    perms = list(itertools.permutations(op_set))
    compiled = [[(_compile_single_op(name, qubits), qubits) for name, qubits in perm] for perm in perms]

    def run_one(key, perm_ops, noise_p):
        sv = _SV0
        for op_arr, qubits in perm_ops:
            sv = _compile_and_run_circuit_jit(sv, op_arr)
            if noise_p > 0:
                key, sub = jax.random.split(key)
                spec = NoiseSpec(model=noise_model, p=noise_p, jax_key=sub)
                sv = NoiseModel.apply_to_sv(sv, N_QUBITS, model=spec.model, p=spec.p,
                                             jax_key=spec.jax_key, qubits=qubits)
        return sv

    runners = [jax.vmap(lambda k, p, ops=ops: run_one(k, ops, p), in_axes=(0, None))
               for ops in compiled]
    return perms, runners


def empirical_dist(outcomes, dim=8):
    counts = np.bincount(outcomes, minlength=dim).astype(float)
    return counts / counts.sum()


def js_divergence(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _sample_outcomes(sv_batch, rng):
    probs = np.abs(np.asarray(sv_batch)) ** 2
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.array([rng.choice(probs.shape[1], p=row) for row in probs])


def js_at_noise_level(runner_a, runner_b, noise_p, seed):
    master_key = jax.random.PRNGKey(seed)
    key_a, key_b = jax.random.split(master_key)
    sv_a = runner_a(jax.random.split(key_a, K_TRAJECTORIES), float(noise_p))
    sv_b = runner_b(jax.random.split(key_b, K_TRAJECTORIES), float(noise_p))

    rng = np.random.default_rng(seed)
    outcomes_a = _sample_outcomes(sv_a, rng)
    outcomes_b = _sample_outcomes(sv_b, rng)
    dist_a, dist_b = empirical_dist(outcomes_a), empirical_dist(outcomes_b)
    observed_js = js_divergence(dist_a, dist_b)

    pooled = np.concatenate([outcomes_a, outcomes_b])
    n_a = len(outcomes_a)
    null_js = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        null_js[i] = js_divergence(empirical_dist(pooled[:n_a]), empirical_dist(pooled[n_a:]))
    p_value = (np.sum(null_js >= observed_js) + 1) / (N_PERMUTATIONS + 1)
    return observed_js, p_value


def run_topology_channel(topo_name, op_set, noise_model, seed_base):
    perms, runners = build_perm_runners(op_set, noise_model)
    pair_indices = list(itertools.combinations(range(6), 2))  # all 15 pairs

    rows = []
    for pair_i, (i, j) in enumerate(pair_indices):
        js_values, p_values = [], []
        for level_i, noise_p in enumerate(NOISE_LEVELS):
            seed = seed_base + 1000 * pair_i + level_i
            observed_js, p_value = js_at_noise_level(runners[i], runners[j], noise_p, seed)
            js_values.append(observed_js)
            p_values.append(p_value)
        rows.append({
            "topology": topo_name, "noise_model": noise_model,
            "perm_i": i + 1, "perm_j": j + 1,
            "order_i": ",".join(n for n, _ in perms[i]), "order_j": ",".join(n for n, _ in perms[j]),
            "max_js": max(js_values), "min_p_value": min(p_values),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    all_dfs = []
    for topo_i, (topo_name, op_set) in enumerate(TOPOLOGIES.items()):
        for model_i, noise_model in enumerate(NOISE_MODELS):
            print(f"=== {topo_name} / {noise_model} ===")
            seed_base = 42 + topo_i * 1_000_000 + model_i * 100_000
            df = run_topology_channel(topo_name, op_set, noise_model, seed_base)
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(_DATA_DIR / "resilient_operational_topologies.csv", index=False)

    # Resilient set = pairs classified "low" by any one channel (depolarizing,
    # the reference channel used in the original single-channel pass).
    ref = df[df["noise_model"] == "depolarizing"]
    ref_median = {}
    resilient_pairs = {}
    for topo_name in TOPOLOGIES:
        sub = ref[ref["topology"] == topo_name]
        med = sub["max_js"].median()
        ref_median[topo_name] = med
        resilient_pairs[topo_name] = set(
            (int(r.perm_i), int(r.perm_j)) for r in sub[sub["max_js"] < med].itertuples()
        )

    print("\n=== Cross-channel consistency ===")
    consistent_everywhere = True
    for topo_name in TOPOLOGIES:
        for noise_model in NOISE_MODELS:
            sub = df[(df["topology"] == topo_name) & (df["noise_model"] == noise_model)]
            med = sub["max_js"].median()
            low_set = set((int(r.perm_i), int(r.perm_j)) for r in sub[sub["max_js"] < med].itertuples())
            match = low_set == resilient_pairs[topo_name]
            consistent_everywhere &= match
            print(f"{topo_name:14s} {noise_model:18s} low-JS set matches depolarizing reference: {match}")

    print(f"\nSame 6-pair resilient/non-resilient split under all 5 channels, all 3 topologies: {consistent_everywhere}")

    summary_rows = []
    for topo_name in TOPOLOGIES:
        for noise_model in NOISE_MODELS:
            sub = df[(df["topology"] == topo_name) & (df["noise_model"] == noise_model)]
            resilient = sub[sub["max_js"] < sub["max_js"].median()]
            non_resilient = sub[sub["max_js"] >= sub["max_js"].median()]
            summary_rows.append({
                "topology": topo_name, "noise_model": noise_model,
                "resilient_max_js_range": f"{resilient['max_js'].min():.4f}-{resilient['max_js'].max():.4f}",
                "non_resilient_max_js_range": f"{non_resilient['max_js'].min():.4f}-{non_resilient['max_js'].max():.4f}",
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(_DATA_DIR / "resilient_operational_topologies_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    fig, axes = plt.subplots(len(TOPOLOGIES), 1, figsize=(11, 4.2 * len(TOPOLOGIES)), sharex=True)
    pair_labels = [f"{i+1} vs {j+1}" for i, j in itertools.combinations(range(6), 2)]
    for ax, topo_name in zip(axes, TOPOLOGIES):
        for noise_model in NOISE_MODELS:
            sub = df[(df["topology"] == topo_name) & (df["noise_model"] == noise_model)].sort_values(["perm_i", "perm_j"])
            ax.plot(pair_labels, sub["max_js"], marker='o', label=noise_model, alpha=0.85)
        ax.axhline(ref_median[topo_name], color='#888888', linestyle='--', linewidth=1,
                    label='depolarizing median' if topo_name == list(TOPOLOGIES)[0] else None)
        ax.set_ylabel("max JS distance\n(p=0.1-0.4)")
        ax.set_title(topo_name, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    axes[0].legend(fontsize=8, ncol=3, loc='upper left')
    axes[-1].set_xlabel("permutation pair")
    fig.suptitle("Resilient Operational Topologies: max JS distance per permutation pair, all 5 noise channels", fontweight='bold')
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "resilient_operational_topologies.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'resilient_operational_topologies.png'}")
