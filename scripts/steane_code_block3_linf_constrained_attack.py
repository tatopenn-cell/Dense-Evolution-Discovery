"""
Steane code [[7,1,3]] -- block 3: L2-cap + L-infinity-cap constrained PGD
attack (fix for block 2's degenerate solution).

Motivation
-----------
Block 2 (scripts/steane_code_block2_adversarial_noise.py) found that
unconstrained (L2-ball-only) PGD on syndrome_leakage degenerately dumps the
entire epsilon budget onto the single qubit shared by all 3 X-stabilizer
generators. A coherent rz error concentrated on ONE qubit always collapses
to weight-0 or weight-1 syndrome, which the single-error decoder corrects
exactly every time -- so that "adversarial" direction had decoder failure
rate 0.0 at every epsilon tested, while random multi-qubit-spread directions
of the same L2 norm failed readily (mean failure 12% by eps=0.35, >=94% by
eps=0.75). The raw-sum leakage proxy's unconstrained optimum is a safe
direction, not a dangerous one.

Fix tested here: project each PGD step into the INTERSECTION of the L2
epsilon-ball (same total budget as block 2) AND an L-infinity ball of radius
linf_cap (a per-qubit cap |delta_q| <= linf_cap) -- forbidding the
one-qubit-takes-everything solution and forcing the search to distribute
budget across multiple qubits, the regime block 2 already showed causes real
decoder failures.

Exact L2-cap ∩ L-infinity-cap projection
-------------------------------------------
Projecting onto B2(eps) ∩ B_inf(c) is NOT "clip to [-c, c] then rescale to
norm eps" in general -- rescaling after clipping can push coordinates back
outside the box. The correct approach (standard KKT result for minimizing
||z-y||^2 s.t. z in box [-c,c]^n and ||z||_2 <= eps):
  1. First just box-clip y: if its L2 norm is already <= eps, that IS the
     exact projection onto the intersection (the closest point in the box
     happens to already satisfy the ball constraint, so it's feasible and a
     lower bound over the box implies optimality over the smaller
     intersection too).
  2. Otherwise the ball constraint is active. For a Lagrange multiplier
     lambda >= 0 on ||z||_2^2 <= eps^2, the coordinate-wise stationarity
     condition (ignoring the box) is z_i = y_i / (1 + lambda); re-imposing
     the box gives z_i(lambda) = clip(y_i / (1+lambda), -c, c). ||z(lambda)||_2
     is monotonically non-increasing in lambda, so bisect on lambda >= 0
     until ||z(lambda)||_2 == eps.
This is an EXACT projection (up to bisection tolerance), not an
approximation -- implemented as project_l2_linf() below and checked by an
explicit assertion after every PGD step.

Reused, not hand-copied, from earlier blocks
------------------------------------------------
- block1: encode_logical_zero, apply_logical_x, build_syndrome_table,
  decode_and_correct_stochastic (the real, discrete, projective decoder).
- block2: syndrome_leakage / _leakage_grad (the differentiable proxy
  objective, evaluated against |+>_L -- block2 proved |0>_L is a
  mathematical blind spot for Z-type errors, not repeated here),
  apply_rz_all, decoder_failure_rate, and its own real CSV output
  (data/steane_adversarial_vs_random_coherent_noise.csv) as the random-noise
  baseline -- NOT recomputed, reused verbatim since it was already a real,
  300/1800-trial run at the exact same epsilons used below.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

sys.path.insert(0, r'C:\Users\Admin\Desktop\Dense-Evolution-main\Dense-Evolution-main')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import steane_code_block1 as block1
import steane_code_block2_adversarial_noise as block2

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_IMAGES_DIR = _REPO_ROOT / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

N_QUBITS = block2.N_QUBITS


# ─────────────────────────────────────────────────────────────────────────
# Exact projection onto B2(epsilon) ∩ B_inf(linf_cap)
# ─────────────────────────────────────────────────────────────────────────

def project_l2_linf(y: np.ndarray, epsilon: float, linf_cap: float, n_bisect: int = 60) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    box = np.clip(y, -linf_cap, linf_cap)
    if np.linalg.norm(box) <= epsilon + 1e-12:
        return box
    lo, hi = 0.0, 1e8
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        z = np.clip(y / (1.0 + mid), -linf_cap, linf_cap)
        if np.linalg.norm(z) > epsilon:
            lo = mid
        else:
            hi = mid
    return np.clip(y / (1.0 + hi), -linf_cap, linf_cap)


# ─────────────────────────────────────────────────────────────────────────
# PGD with the L2 ∩ L-infinity projection (mirrors block2.craft_adversarial_delta,
# only the projection step differs)
# ─────────────────────────────────────────────────────────────────────────

def craft_adversarial_delta_constrained(sv0: jnp.ndarray, epsilon: float, linf_cap: float,
                                         n_steps: int = 150, step_size: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    init_dir = rng.normal(size=N_QUBITS)
    init_dir /= np.linalg.norm(init_dir)
    init_norm = min(epsilon, 1e-2)
    delta_np = project_l2_linf(init_dir * init_norm, epsilon, linf_cap)
    delta = jnp.array(delta_np)

    best_delta = delta
    best_leakage = float(block2.syndrome_leakage(delta, sv0))
    history = [best_leakage]

    for _ in range(n_steps):
        grad = block2._leakage_grad(delta, sv0)
        grad_norm = jnp.linalg.norm(grad)
        step = jnp.where(grad_norm > 1e-12, grad / grad_norm, jnp.zeros_like(grad))
        delta_raw = np.asarray(delta) + step_size * np.asarray(step)
        delta_np = project_l2_linf(delta_raw, epsilon, linf_cap)
        assert np.max(np.abs(delta_np)) <= linf_cap + 1e-8, "L-infinity cap violated by projection"
        assert np.linalg.norm(delta_np) <= epsilon + 1e-6, "L2 cap violated by projection"
        delta = jnp.array(delta_np)

        current = float(block2.syndrome_leakage(delta, sv0))
        history.append(current)
        if current > best_leakage:
            best_leakage = current
            best_delta = delta

    return np.asarray(best_delta), best_leakage, history


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SETUP: reuse block 1's encoding + syndrome table, test vs |+>_L (block 2's finding)")
    print("=" * 70)
    sv0_np = block1.encode_logical_zero()
    sv1_np = block1.apply_logical_x(sv0_np)
    sv_plus_np = (sv0_np + sv1_np) / np.linalg.norm(sv0_np + sv1_np)
    qubit_to_syndrome = block1.build_syndrome_table(sv0_np)
    syndrome_to_qubit = {s: q for q, s in qubit_to_syndrome.items()}
    assert len(syndrome_to_qubit) == N_QUBITS
    sv_plus_jax = jnp.array(sv_plus_np)

    print("\nsanity check -- exact projection: clip-then-renormalize would violate the L-infinity "
          "cap, project_l2_linf must not:")
    probe_y = np.array([5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    naive_clip_then_renorm = np.clip(probe_y, -0.2, 0.2)
    naive_clip_then_renorm = naive_clip_then_renorm / np.linalg.norm(naive_clip_then_renorm) * 1.0
    exact_proj = project_l2_linf(probe_y, epsilon=1.0, linf_cap=0.2)
    print(f"   naive clip-then-rescale-to-eps: max|.|={np.max(np.abs(naive_clip_then_renorm)):.4f} "
          f"(cap=0.2, VIOLATED: {np.max(np.abs(naive_clip_then_renorm)) > 0.2 + 1e-9})")
    print(f"   exact project_l2_linf:          max|.|={np.max(np.abs(exact_proj)):.4f}, "
          f"||.||_2={np.linalg.norm(exact_proj):.4f} (cap=0.2, eps=1.0, both respected)")

    baseline_csv = _DATA_DIR / "steane_adversarial_vs_random_coherent_noise.csv"
    baseline_df = pd.read_csv(baseline_csv).set_index("epsilon")
    print(f"\nReusing block 2's real random-baseline CSV (not recomputed): {baseline_csv}")

    EPSILONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    CAPS = {"tight (linf_cap=0.15)": 0.15, "loose (linf_cap=2.0, sanity check)": 2.0}
    N_TRIALS_ADV = 300

    print("\n" + "=" * 70)
    print(f"SWEEP: {len(CAPS)} linf_cap values x {len(EPSILONS)} epsilon points")
    print("=" * 70)

    rows = []
    rng_eval = np.random.default_rng(42)
    for cap_label, linf_cap in CAPS.items():
        for eps in EPSILONS:
            adv_delta, adv_leakage, _ = craft_adversarial_delta_constrained(sv_plus_jax, eps, linf_cap, seed=0)
            adv_delta_norm = float(np.linalg.norm(adv_delta))
            adv_delta_linf = float(np.max(np.abs(adv_delta)))
            n_spread = int(np.sum(np.abs(adv_delta) > 1e-3))
            adv_fail_rate = block2.decoder_failure_rate(adv_delta, sv_plus_np, syndrome_to_qubit, N_TRIALS_ADV, rng_eval)

            rand_mean = float(baseline_df.loc[eps, "random_mean_failure_rate"])
            rand_max = float(baseline_df.loc[eps, "random_max_failure_rate"])

            rows.append(dict(cap_label=cap_label, linf_cap=linf_cap, epsilon=eps,
                              adv_delta_norm=adv_delta_norm, adv_delta_linf=adv_delta_linf,
                              n_qubits_spread=n_spread, adv_leakage=adv_leakage,
                              adv_failure_rate=adv_fail_rate,
                              random_mean_failure_rate=rand_mean, random_max_failure_rate=rand_max))
            print(f"   [{cap_label}] eps={eps:.3f}: ||delta||_2={adv_delta_norm:.4f} "
                  f"||delta||_inf={adv_delta_linf:.4f} spread_over={n_spread} qubits  "
                  f"adv_fail={adv_fail_rate:.4f}  |  random mean={rand_mean:.4f} max={rand_max:.4f}")

    df = pd.DataFrame(rows)
    csv_path = _DATA_DIR / "steane_linf_constrained_attack.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    loose_label = "loose (linf_cap=2.0, sanity check)"
    tight_label = "tight (linf_cap=0.15)"
    loose_df = df[df.cap_label == loose_label]
    tight_df = df[df.cap_label == tight_label]

    loose_matches_block2 = np.allclose(loose_df["adv_failure_rate"].to_numpy(), 0.0)
    print(f"Sanity check (loose cap, never binds since cap > every epsilon tested): "
          f"adv_failure_rate at every epsilon = {loose_df['adv_failure_rate'].tolist()} "
          f"-- {'REPRODUCES' if loose_matches_block2 else 'DOES NOT reproduce'} block 2's degenerate "
          f"0.0-failure-rate finding.")

    tight_df = tight_df.assign(
        beats_random_max=tight_df["adv_failure_rate"] > tight_df["random_max_failure_rate"] + 1e-9,
        beats_random_mean=tight_df["adv_failure_rate"] > tight_df["random_mean_failure_rate"] + 1e-9,
    )
    n_beats_max = int(tight_df["beats_random_max"].sum())
    n_beats_mean = int(tight_df["beats_random_mean"].sum())
    bound_mask = tight_df["epsilon"] > tight_df["linf_cap"]
    max_spread_bound = int(tight_df.loc[bound_mask, "n_qubits_spread"].max()) if bound_mask.any() else 1
    print(f"\nTight cap (linf_cap=0.15): below the cap (eps<=0.15) a single qubit already satisfies both "
          f"constraints so nothing is forced; once eps exceeds the cap, spread grows to as many as "
          f"{max_spread_bound} qubits (spread counts by epsilon {tight_df['epsilon'].tolist()}: "
          f"{tight_df['n_qubits_spread'].tolist()}).")
    print(f"adv_failure_rate by epsilon: {tight_df['adv_failure_rate'].tolist()}")
    print(f"random_max_failure_rate by epsilon (block 2's real baseline): {tight_df['random_max_failure_rate'].tolist()}")

    if n_beats_max > 0:
        conclusion = (f"GENUINE BLIND SPOT FOUND: with linf_cap=0.15 forcing spread over up to {max_spread_bound} "
                       f"qubits, PGD's adversarial failure rate exceeds the random baseline's MAX (of 30 "
                       f"directions) at {n_beats_max}/{len(EPSILONS)} epsilon points, and exceeds the "
                       f"random MEAN at {n_beats_mean}/{len(EPSILONS)}.")
    elif n_beats_mean > 0:
        conclusion = (f"PARTIAL: with linf_cap=0.15, adversarial failure rate exceeds the random MEAN at "
                       f"{n_beats_mean}/{len(EPSILONS)} epsilon points but never exceeds the random MAX "
                       f"(best of 30 random directions) -- the L-infinity constraint forces real, "
                       f"nonzero decoder failures (unlike block 2's 0.0 everywhere), but the code remains "
                       f"at least as robust to this constrained worst-case search as to random multi-qubit "
                       f"noise of the same L2 budget.")
    else:
        conclusion = (f"NO BLIND SPOT: with linf_cap=0.15 forcing spread over up to {max_spread_bound} qubits, the "
                       f"adversarial failure rate never exceeds the random baseline's mean OR max at any "
                       f"tested epsilon. Forcing the L2-ball-only degenerate single-qubit solution out of "
                       f"reach makes the leakage-proxy-driven search perform no better than (and often "
                       f"worse than) chance -- the code appears genuinely robust under this constrained "
                       f"worst-case coherent-error search, not merely because the search found a spurious "
                       f"safe optimum.")
    print(f"\n{conclusion}")

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.errorbar(EPSILONS, [baseline_df.loc[e, "random_mean_failure_rate"] for e in EPSILONS],
                fmt='o-', color='#888888', linewidth=1.6, markersize=4,
                label='Random coherent noise (mean of 30 directions, block 2)')
    ax.plot(EPSILONS, [baseline_df.loc[e, "random_max_failure_rate"] for e in EPSILONS], 's--',
            color='#FFD700', linewidth=1.4, markersize=4,
            label='Random coherent noise (max of 30 directions, block 2)')
    ax.plot(loose_df["epsilon"], loose_df["adv_failure_rate"], 'D:', color='#00BFFF', linewidth=1.4,
            markersize=5, label='PGD, loose L-inf cap=2.0 (sanity check, reproduces block 2)')
    ax.plot(tight_df["epsilon"], tight_df["adv_failure_rate"], 'o-', color='#FF007F', linewidth=1.8,
            markersize=5, label='PGD, tight L-inf cap=0.15 (forces multi-qubit spread)')
    ax.set_xlabel('L2 epsilon budget on per-qubit rz coherent-error angles', color='#888888')
    ax.set_ylabel('Empirical decoder logical-failure rate', color='#888888')
    ax.set_title('Steane [[7,1,3]]: L2 ∩ L-infinity constrained adversarial search\nvs. block 2\'s random baseline (real projective decoder)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc='upper left', fontsize=8.5)
    plt.tight_layout()
    png_path = _IMAGES_DIR / "steane_linf_constrained_attack.png"
    plt.savefig(png_path, dpi=300)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(conclusion)
    print(f"CSV: {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
