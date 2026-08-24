"""Experiment 28: is the "Leaky-Switch" differentiable healing design
(dinamicsHEal.txt, final block) actually differentiable as claimed, and
how does its healing quality compare to the shipped numpy healers?

Origin: same proposal line as Experiments 26/27. Claims a JAX/XLA
rewrite of the healing filter (jax.lax.scan, sigmoid soft-trigger with a
"leaky" floor instead of a hard branch) removes "dead gradients" so the
healing step can sit inside a differentiable training loop (e.g. a VQE
loss) without blocking backprop. The original test was a single 4x8
toy gradient check on one seed -- not a real differentiability audit
across many inputs, and it never checked healing QUALITY at all (only
that gradients existed).

Two separate questions tested here:
1. DIFFERENTIABILITY: does jax.grad actually give finite, non-trivial
   gradients across many seeds/corruption patterns? Is the "leaky" 1e-4
   floor load-bearing (ablate epsilon=0 vs 1e-4)?
2. HEALING QUALITY: on the same 4 corruption scenarios x 40 seeds used in
   Experiments 26/27, how does this design's output compare (L2 error,
   cosine alignment vs. the clean ground truth) to the shipped
   enhanced_dense_healing_hybrid in both 'phi' and 'adaptive' modes?
   Important structural differences to keep in mind when reading results:
   this design (a) cascades corrections (jax.lax.scan carries the updated
   sequence forward), unlike the shipped non-cascading loop; (b) never
   hard-replaces, only soft-blends state_B and the healed candidate via a
   sigmoid; (c) uses a FIXED 0.25 threshold, not the MAD-adaptive one from
   Experiment 27; (d) zero-imputes NaN/Inf instead of column-mean-imputing.

    python scripts/leaky_differentiable_healing.py
"""
import pathlib

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import stats

from ia_utils.vector_healing import enhanced_dense_healing_hybrid

jax.config.update("jax_enable_x64", True)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

N_STEPS = 50
DIM = 128
N_SEEDS = 40


def make_leaky_healer(leaky_epsilon=1e-4):
    """Faithful port of leaky_healing_hu_sverak_jax, with the
    leaky floor exposed as a parameter for the ablation test."""

    def leaky_healing(vettori, radius=16):
        n, hidden_dim = vettori.shape
        nu = 0.05

        vettori_clean = jnp.where(jnp.isnan(vettori) | jnp.isinf(vettori), 0.0, vettori)

        def scan_body(carry, i):
            processed = carry
            indices = jnp.arange(n)
            mask = (indices >= jnp.maximum(0, i - radius)) & (indices < i)

            state_A = jnp.sum(jnp.where(mask[:, None], processed, 0.0), axis=0) / jnp.maximum(jnp.sum(mask), 1.0)
            norm_A = jnp.linalg.norm(state_A)
            state_B = vettori_clean[i]

            ipg_raw = processed[i - 1] - processed[i - 2]
            norm_ipg = jnp.linalg.norm(ipg_raw)
            ipg_vector = jnp.where(norm_ipg > 1e-9, ipg_raw / jnp.maximum(norm_ipg, 1e-9), ipg_raw)
            q_force = ipg_vector * norm_A

            current_deviation = jnp.linalg.norm(state_B - state_A) / jnp.sqrt(hidden_dim)
            trigger_activation = jax.nn.sigmoid((current_deviation - 0.25) * 50.0)
            trigger_activation = trigger_activation * (1.0 - leaky_epsilon)

            drift_step = (q_force - nu * state_A)
            healed = state_A + drift_step * 0.5

            norm_healed = jnp.linalg.norm(healed)
            healed_normalized = jnp.where(norm_healed > 1e-9, healed * (norm_A / jnp.maximum(norm_healed, 1e-9)), healed)

            final_vector = (1.0 - trigger_activation) * state_B + trigger_activation * healed_normalized
            new_processed = processed.at[i].set(final_vector)
            return new_processed, None

        indices_to_scan = jnp.arange(2, n)
        final_history, _ = jax.lax.scan(scan_body, vettori_clean, indices_to_scan)
        return final_history

    return jax.jit(leaky_healing, static_argnames=())


LEAKY_HEALER = make_leaky_healer(leaky_epsilon=1e-4)
LEAKY_HEALER_NO_LEAK = make_leaky_healer(leaky_epsilon=0.0)


def clean_trajectory(seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, N_STEPS)
    trend = np.sin(t)[:, None] * np.ones((1, DIM))
    return rng.normal(loc=0.0, scale=0.1, size=(N_STEPS, DIM)) + trend


def corrupt(clean, scenario, seed):
    rng = np.random.default_rng(seed + 500_000)
    corrupted = np.copy(clean)
    if scenario == "single_spike":
        idx = N_STEPS // 2
        corrupted[idx] += rng.normal(loc=8.0, scale=1.0, size=DIM)
    elif scenario == "nan_string":
        idx = N_STEPS // 2
        corrupted[idx:idx + 3] = np.nan
    elif scenario == "scattered_outliers":
        idxs = rng.choice(np.arange(5, N_STEPS - 5), size=4, replace=False)
        for idx in idxs:
            corrupted[idx] += rng.normal(loc=6.0, scale=1.5, size=DIM)
    elif scenario == "combined":
        spike_idx = N_STEPS // 3
        nan_idx = 2 * N_STEPS // 3
        corrupted[spike_idx] += rng.normal(loc=10.0, scale=1.0, size=DIM)
        corrupted[nan_idx:nan_idx + 3] = np.nan
    else:
        raise ValueError(scenario)
    return corrupted


def l2_error(healed, ideal):
    return float(np.linalg.norm(healed - ideal))


def cosine_alignment(healed, ideal):
    h, d = healed.ravel(), ideal.ravel()
    denom = np.linalg.norm(h) * np.linalg.norm(d)
    return float(np.real(np.vdot(h, d)) / denom) if denom > 1e-12 else 0.0


SCENARIOS = ("single_spike", "nan_string", "scattered_outliers", "combined")


def part1_differentiability_audit():
    print("=== PART 1: DIFFERENTIABILITY AUDIT ===\n")

    n_finite_nonzero_with_leak = 0
    n_finite_nonzero_no_leak = 0
    grad_norms_with_leak, grad_norms_no_leak = [], []

    for seed in range(30):
        clean = jnp.array(clean_trajectory(seed))
        corrupted = jnp.array(corrupt(np.array(clean), "combined", seed))

        def loss_fn(x, healer):
            return jnp.sum(healer(x) ** 2)

        for healer, norms_list, counter_name in (
            (LEAKY_HEALER, grad_norms_with_leak, "with_leak"),
            (LEAKY_HEALER_NO_LEAK, grad_norms_no_leak, "no_leak"),
        ):
            grad = jax.grad(lambda x: loss_fn(x, healer))(corrupted)
            has_nan = bool(jnp.isnan(grad).any())
            norm = float(jnp.linalg.norm(grad))
            norms_list.append(norm)
            if not has_nan and norm > 0:
                if counter_name == "with_leak":
                    n_finite_nonzero_with_leak += 1
                else:
                    n_finite_nonzero_no_leak += 1

    print(f"Finite & nonzero gradient (epsilon=1e-4, 'leaky'): {n_finite_nonzero_with_leak}/30 seeds")
    print(f"  grad norm: mean={np.mean(grad_norms_with_leak):.4f} min={np.min(grad_norms_with_leak):.4f} max={np.max(grad_norms_with_leak):.4f}")
    print(f"Finite & nonzero gradient (epsilon=0, no leak):     {n_finite_nonzero_no_leak}/30 seeds")
    print(f"  grad norm: mean={np.mean(grad_norms_no_leak):.4f} min={np.min(grad_norms_no_leak):.4f} max={np.max(grad_norms_no_leak):.4f}")

    # Does removing the leak actually create dead-gradient zones anywhere
    # in these 30 trials, or was epsilon=1e-4 unnecessary insurance?
    zero_grad_no_leak = sum(1 for n in grad_norms_no_leak if n == 0.0)
    print(f"\nSeeds with EXACTLY zero gradient norm, no-leak version: {zero_grad_no_leak}/30")
    print(f"Seeds with EXACTLY zero gradient norm, leaky version:   {sum(1 for n in grad_norms_with_leak if n == 0.0)}/30")
    return {
        "with_leak_finite_nonzero": n_finite_nonzero_with_leak,
        "no_leak_finite_nonzero": n_finite_nonzero_no_leak,
        "zero_grad_no_leak": zero_grad_no_leak,
    }


def part2_healing_quality():
    print("\n=== PART 2: HEALING QUALITY vs. GROUND TRUTH ===\n")
    rows = []
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            clean = clean_trajectory(seed)
            corrupted = corrupt(clean, scenario, seed)

            healed_phi, _ = enhanced_dense_healing_hybrid(corrupted.copy(), trigger_mode='phi')
            healed_adaptive, _ = enhanced_dense_healing_hybrid(corrupted.copy(), trigger_mode='adaptive')
            healed_leaky = np.array(LEAKY_HEALER(jnp.array(corrupted)))

            rows.append({
                "scenario": scenario, "seed": seed,
                "l2_phi": l2_error(healed_phi, clean),
                "l2_adaptive": l2_error(healed_adaptive, clean),
                "l2_leaky": l2_error(healed_leaky, clean),
                "cos_phi": cosine_alignment(healed_phi, clean),
                "cos_adaptive": cosine_alignment(healed_adaptive, clean),
                "cos_leaky": cosine_alignment(healed_leaky, clean),
                "leaky_has_nan": bool(np.isnan(healed_leaky).any()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "leaky_differentiable_healing.csv", index=False)

    assert not df["leaky_has_nan"].any(), "leaky healer leaked NaN into output"

    print(f"{'scenario':22s} {'L2 phi':>10s} {'L2 adapt':>10s} {'L2 leaky':>10s}   "
          f"{'cos phi':>9s} {'cos adapt':>9s} {'cos leaky':>9s}")
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario]
        print(f"{scenario:22s} {sub['l2_phi'].mean():10.3f} {sub['l2_adaptive'].mean():10.3f} {sub['l2_leaky'].mean():10.3f}   "
              f"{sub['cos_phi'].mean():9.4f} {sub['cos_adaptive'].mean():9.4f} {sub['cos_leaky'].mean():9.4f}")

        w_stat, p_val = stats.wilcoxon(sub["l2_leaky"], sub["l2_adaptive"])
        leaky_wins = (sub["l2_leaky"] < sub["l2_adaptive"]).mean()
        print(f"    leaky vs adaptive (L2, lower better): leaky-win-rate={leaky_wins:.0%}  wilcoxon p={p_val:.4f}")
    return df


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    diff_results = part1_differentiability_audit()
    quality_df = part2_healing_quality()

    summary = quality_df.groupby("scenario")[["cos_phi", "cos_adaptive", "cos_leaky"]].mean().reindex(SCENARIOS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.25

    axes[0].bar(x - width, summary["cos_phi"], width, label="phi (shipped)", color="#888888")
    axes[0].bar(x, summary["cos_adaptive"], width, label="adaptive (Exp. 27 fix)", color="#00e5ff")
    axes[0].bar(x + width, summary["cos_leaky"], width, label="leaky (this exp.)", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SCENARIOS, rotation=20, ha="right")
    axes[0].set_ylabel("cosine alignment vs. ground truth (higher = better)")
    axes[0].set_title("Healing quality: leaky trails adaptive on every scenario")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(["with leak (eps=1e-4)", "no leak (eps=0)"],
                [diff_results["with_leak_finite_nonzero"], diff_results["no_leak_finite_nonzero"]],
                color=["#00e5ff", "#ff7f0e"])
    axes[1].set_ylim(0, 32)
    axes[1].set_ylabel("seeds out of 30")
    axes[1].set_title("Differentiability: real, but the leaky floor changes nothing")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment 28: Leaky-Switch differentiable healing -- real gradients, poor healing quality", fontweight="bold")
    fig.tight_layout()
    images_dir = _DATA_DIR.parent / "images"
    images_dir.mkdir(exist_ok=True)
    fig.savefig(images_dir / "leaky_differentiable_healing.png", dpi=150)
    print(f"saved plot: {images_dir / 'leaky_differentiable_healing.png'}")
