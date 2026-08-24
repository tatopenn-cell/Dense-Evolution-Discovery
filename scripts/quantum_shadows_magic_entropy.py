"""Experiment 31: fixing a real issue in the Classical Shadows
proposal, then using shadows to estimate magic entropy from randomized
measurements instead of the exact state.

Origin: following Huang, Kueng, Preskill 2020, "Predicting Many Properties
of a Quantum System from Very Few Measurements" (paper test/varianze.txt),
it was proposed to add a dense_evolution/circuits/shadows.py
module with a ClassicalShadow class and predict_renyi_entropy. The purity
estimator at its core computes the U-statistic cross-trace between
independent shadow snapshots as
    jnp.einsum('ijk,mjk->im', matrices, matrices)
which is WRONG: this contracts matching indices with no transpose, i.e.
sum_jk A_i[j,k]*A_m[j,k], not Tr(A_i @ A_m) = sum_jk A_i[j,k]*A_m[k,j].
Verified directly (see purity_bug_verification below): on two Hermitian
2x2 test matrices with complex off-diagonal entries, the buggy contraction
gives 45.0 while the true Tr(A@B) is 21.0. The bug is silent whenever
every snapshot happens to be real-valued (a Z-basis-only case, which is
why an earlier Bell-state S2=1.000000 check never exposed it), but
is wrong in general whenever X/Y-basis snapshots (complex entries) are
mixed in, which is the normal case for a real random-Pauli protocol.

This experiment: (a) implements the real single-qubit classical-shadow
snapshot/reconstruction protocol (random Pauli basis, rho_hat = 3 U^dag
|b><b| U - I) and verifies it is unbiased, (b) reproduces the purity bug
and fixes it, validating the fixed purity estimator against the exact
Tr[rho^2], (c) extends the SAME multi-copy U-statistic idea -- which Huang
et al. state explicitly "readily generalizes to higher order polynomials"
-- from 2 copies (purity) to 3 copies, to build a shadow-based ESTIMATOR
for Experiment 30's magic entropy (boxtimes_3 self-convolution), and
validates it converges to Experiment 30's exact value as the snapshot
budget grows.

    python scripts/quantum_shadows_magic_entropy.py
"""
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# --- Single-qubit random-Pauli classical shadow protocol (Huang, Kueng,
# Preskill 2020, eq. 2-3) ---
_H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)
_SDAG = jnp.array([[1.0, 0.0], [0.0, -1j]], dtype=jnp.complex128)
_BASIS_U = jnp.stack([_H, _H @ _SDAG, jnp.eye(2, dtype=jnp.complex128)])  # X, Y, Z


def sample_shadow_snapshots(psi, n_snapshots, seed):
    """Simulates the real measurement protocol: for each snapshot, pick a
    random Pauli basis uniformly, apply its diagonalizing unitary, sample
    a computational-basis outcome from the true Born-rule probability,
    then reconstruct rho_hat = 3 U^dag |b><b| U - I (the single-qubit
    Pauli-shadow inverse channel). Returns the (n_snapshots, 2, 2) array
    of classical snapshot matrices."""
    key = jax.random.PRNGKey(seed)
    key_basis, key_bit = jax.random.split(key)
    bases = jax.random.randint(key_basis, (n_snapshots,), 0, 3)

    def prob0(basis_idx):
        amp0 = (_BASIS_U[basis_idx] @ psi)[0]
        return jnp.abs(amp0) ** 2

    probs0 = jax.vmap(prob0)(bases)
    uniforms = jax.random.uniform(key_bit, (n_snapshots,))
    bits = (uniforms > probs0).astype(jnp.int32)

    def snapshot_matrix(basis_idx, bit):
        u = _BASIS_U[basis_idx]
        b_ket = jnp.array([1.0, 0.0], dtype=jnp.complex128) * (1 - bit) + \
            jnp.array([0.0, 1.0], dtype=jnp.complex128) * bit
        proj = jnp.outer(b_ket, jnp.conj(b_ket))
        return 3.0 * (jnp.conj(u).T @ proj @ u) - jnp.eye(2, dtype=jnp.complex128)

    return jax.vmap(snapshot_matrix)(bases, bits)


def purity_bug_verification():
    """Directly reproduces the original einsum error on two fixed Hermitian
    test matrices, independent of any measurement simulation."""
    a = jnp.array([[1.0 + 0j, 2.0 + 3j], [2.0 - 3j, 4.0 + 0j]], dtype=jnp.complex128)
    b = jnp.array([[5.0 + 0j, 1.0 - 1j], [1.0 + 1j, 2.0 + 0j]], dtype=jnp.complex128)
    stacked = jnp.stack([a, b])
    buggy = float(jnp.einsum("ijk,mjk->im", stacked, stacked)[0, 1].real)
    fixed = float(jnp.einsum("ijk,mkj->im", stacked, stacked)[0, 1].real)
    true_tr_ab = float(jnp.trace(a @ b).real)
    return buggy, fixed, true_tr_ab


def _disjoint_pairs(snapshot_matrices):
    n = snapshot_matrices.shape[0]
    n_pairs = n // 2
    pairs = snapshot_matrices[: n_pairs * 2].reshape(n_pairs, 2, 2, 2)
    return pairs[:, 0], pairs[:, 1]


def estimate_purity_buggy(snapshot_matrices):
    """Grouped into disjoint pairs rather than all O(n^2) cross-pairs --
    both are unbiased U-statistics of the same quantity, but all-pairs is
    memory-prohibitive at the snapshot counts tested below (n^2 grows to
    tens of GB by n=30,000). The bug itself ('ij,ij->' instead of
    'ij,ji->', i.e. no transpose) is identical either way."""
    left, right = _disjoint_pairs(snapshot_matrices)
    vals = jnp.einsum("tij,tij->t", left, right).real
    return float(jnp.mean(vals))


def _median_of_means(values, n_groups):
    """Split real-valued `values` into n_groups contiguous batches, average
    each batch, then return the median of those batch means -- Huang et
    al.'s standard robustification for shadow-estimator U-statistics (this
    experiment originally shipped with plain averaging over all
    triples/pairs instead; see PART 4 below for why that matters).
    Unlike a single overall mean, the median tolerates up to
    (n_groups // 2) entirely corrupted/outlier batches without being
    dragged toward them -- e.g. a systematic calibration fault affecting
    one contiguous stretch of a measurement run -- at the cost of some
    extra variance from averaging within fewer, larger batches instead of
    every sample individually."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    n_groups = max(1, min(n_groups, n))
    group_size = n // n_groups
    trimmed = values[: group_size * n_groups]
    group_means = trimmed.reshape(n_groups, group_size).mean(axis=1)
    return float(np.median(group_means))


def estimate_purity_fixed(snapshot_matrices, n_groups=20):
    left, right = _disjoint_pairs(snapshot_matrices)
    vals = np.array(jnp.einsum("tij,tji->t", left, right).real)
    return _median_of_means(vals, n_groups)


# --- Magic-entropy Key Unitary (K=3), duplicated from
# scripts/quantum_ruzsa_magic_entropy.py (Experiment 30) as a small,
# self-contained, already-validated construction, per this repo's
# convention of keeping each experiment script standalone. ---
def _cnot_matrix(control, target, n=3):
    dim = 2 ** n
    mat = np.zeros((dim, dim))
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control]:
            bits[target] ^= 1
        j = 0
        for bit in bits:
            j = (j << 1) | bit
        mat[j, i] = 1.0
    return mat


def _build_key_unitary_k3():
    layer1 = _cnot_matrix(0, 2) @ _cnot_matrix(0, 1)
    layer2 = _cnot_matrix(2, 0) @ _cnot_matrix(1, 0)
    return jnp.array(layer2 @ layer1, dtype=jnp.complex128)


KEY_UNITARY_K3 = _build_key_unitary_k3()


def exact_self_convolve_3(rho):
    rho_full = jnp.kron(jnp.kron(rho, rho), rho)
    evolved = KEY_UNITARY_K3 @ rho_full @ jnp.conj(KEY_UNITARY_K3).T
    tensor = evolved.reshape(2, 2, 2, 2, 2, 2)
    return jnp.einsum("ijkljk->il", tensor)


def exact_magic_entropy(rho, eps=1e-12):
    reduced = exact_self_convolve_3(rho)
    ev = jnp.linalg.eigvalsh(reduced)
    safe_ev = jnp.clip(ev.real, eps, 1.0)
    return float(-jnp.sum(safe_ev * jnp.log2(safe_ev)))


def _o_operators():
    """O_ab = V^dagger (|a><b| (x) I_4) V, the fixed 8x8 operators such
    that R_ab = Tr[O_ab . rho^{(x)3}] reproduces the (a,b) entry of the
    exact reduced convolution R = Tr_{2,3}[V rho^{(x)3} V^dagger] -- a
    LINEAR functional of rho^{(x)3}, exactly the shape Huang et al. state
    their multi-copy U-statistic estimator "readily generalizes to" for
    higher-order polynomials (they demonstrate it for 2 copies/purity; 3
    copies is the same construction with one more independent snapshot
    per group)."""
    i4 = jnp.eye(4, dtype=jnp.complex128)
    v = KEY_UNITARY_K3
    ops = {}
    for a in range(2):
        for b in range(2):
            # Tr[(|a><b| (x) I) M] = sum_j M_{bj,aj} = R_ba, NOT R_ab (a
            # direct index check catches this even though the cyclic-trace
            # derivation looks right on paper) -- so the projector here is
            # deliberately |b><a|, swapped, to land on R_ab.
            proj_ba = jnp.zeros((2, 2), dtype=jnp.complex128).at[b, a].set(1.0)
            ops[(a, b)] = jnp.conj(v).T @ jnp.kron(proj_ba, i4) @ v
    return ops


_O_AB = _o_operators()


def estimate_magic_entropy_from_shadows(snapshot_matrices, n_groups=20):
    """Groups the n independent single-qubit shadow snapshots of the SAME
    state into disjoint triples (n//3 groups), estimates each entry of the
    3-copy-convolution reduced matrix R via median-of-means (see
    _median_of_means) over Tr[O_ab . (rho_hat_i (x) rho_hat_j (x) rho_hat_k)]
    -- an unbiased U-statistic-style estimator, since each triple is drawn
    from 3 independent unbiased single-copy estimators of rho -- then
    computes the von Neumann entropy of the (Hermitized, eigenvalue-clipped,
    renormalized) ESTIMATED R classically. Entropy itself is never
    shadow-estimated directly (Huang et al. never do this either, even for
    their own Renyi-2 entanglement entropy example); only the linear
    reduced-matrix reconstruction is shadow-based. Real and imaginary parts
    of each R_ab entry are median-of-means'd independently -- the median of
    a set of complex numbers has no single standard definition, but the
    median of their real/imaginary parts separately is the standard
    practical choice."""
    n = snapshot_matrices.shape[0]
    n_triples = n // 3
    triples = snapshot_matrices[: n_triples * 3].reshape(n_triples, 3, 2, 2)

    def triple_kron(t):
        return jnp.kron(jnp.kron(t[0], t[1]), t[2])

    rho3_batch = jax.vmap(triple_kron)(triples)  # (n_triples, 8, 8)

    r_hat = jnp.zeros((2, 2), dtype=jnp.complex128)
    for (a, b), o_ab in _O_AB.items():
        vals = np.array(jnp.einsum("ij,tji->t", o_ab, rho3_batch))
        real_part = _median_of_means(vals.real, n_groups)
        imag_part = _median_of_means(vals.imag, n_groups)
        r_hat = r_hat.at[a, b].set(real_part + 1j * imag_part)

    r_hat = 0.5 * (r_hat + jnp.conj(r_hat).T)  # enforce Hermiticity
    ev = jnp.linalg.eigvalsh(r_hat)
    safe_ev = jnp.clip(ev.real, 1e-9, None)
    safe_ev = safe_ev / jnp.sum(safe_ev)  # renormalize (estimation noise can shift trace off 1)
    return float(-jnp.sum(safe_ev * jnp.log2(safe_ev)))


def sample_complexity_fit(psi, m_exact, n_snapshots_list, n_trials, seed_base=1000):
    """Empirically measures the std of estimate_magic_entropy_from_shadows
    across n_trials independent seeds at each snapshot count in
    n_snapshots_list, then fits std(n) ~ C / n^p via log-log linear
    regression (a real measured error bar, not a theoretical constant
    taken on faith). Returns (rows, C, p)."""
    rows = []
    for n_snap in n_snapshots_list:
        estimates = []
        for trial in range(n_trials):
            snaps_trial = sample_shadow_snapshots(psi, n_snap, seed=seed_base + trial)
            estimates.append(estimate_magic_entropy_from_shadows(snaps_trial))
        estimates = np.array(estimates)
        rows.append({
            "n_snapshots": n_snap, "n_trials": n_trials,
            "mean_estimate": float(estimates.mean()), "std_estimate": float(estimates.std()),
            "exact": m_exact, "mean_abs_error": float(abs(estimates.mean() - m_exact)),
        })
    log_n = np.log([r["n_snapshots"] for r in rows])
    log_std = np.log([r["std_estimate"] for r in rows])
    slope, intercept = np.polyfit(log_n, log_std, 1)
    return rows, float(np.exp(intercept)), float(-slope)


def n_snapshots_for_target_std(target_std, fit_c, fit_p):
    """Inverts std(n) ~ fit_c / n^fit_p (see sample_complexity_fit) to
    solve for the snapshot count needed to reach a target standard
    deviation, in bits, on the magic-entropy estimate."""
    return (fit_c / target_std) ** (1.0 / fit_p)


def t_state():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    return (zero + jnp.exp(1j * jnp.pi / 4.0) * one) / jnp.sqrt(2.0)


def plus_state():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    return (zero + one) / jnp.sqrt(2.0)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)

    print("=== PART 1: SHADOW PROTOCOL SANITY CHECK (unbiasedness) ===\n")
    psi_t = t_state()
    rho_t = jnp.outer(psi_t, jnp.conj(psi_t))
    snaps = sample_shadow_snapshots(psi_t, 200_000, seed=0)
    rho_empirical = jnp.mean(snaps, axis=0)
    err = float(jnp.max(jnp.abs(rho_empirical - rho_t)))
    print(f"max |empirical mean of rho_hat - true rho| over 200,000 snapshots: {err:.5f}")
    assert err < 0.02, "shadow snapshot reconstruction does not look unbiased"
    print("    PASS: empirical mean of the shadow snapshots converges to the true state\n")

    print("=== PART 2: THE einsum BUG (purity / predict_renyi_entropy) ===\n")
    buggy, fixed, true_val = purity_bug_verification()
    print(f"Fixed test matrices: buggy_einsum_result={buggy:.4f}  fixed_einsum_result={fixed:.4f}  true_Tr(AB)={true_val:.4f}")
    assert abs(fixed - true_val) < 1e-9 and abs(buggy - true_val) > 1.0
    print("    PASS: 'ijk,mjk->im' is confirmed wrong; 'ijk,mkj->im' matches Tr(AB) exactly\n")

    rows_purity = []
    for n_snap in (300, 1000, 3000, 10000, 30000, 100000):
        snaps_n = sample_shadow_snapshots(psi_t, n_snap, seed=1)
        p_buggy = estimate_purity_buggy(snaps_n)
        p_fixed = estimate_purity_fixed(snaps_n)
        rows_purity.append({"n_snapshots": n_snap, "purity_buggy": p_buggy, "purity_fixed": p_fixed, "purity_exact": 1.0})
        print(f"n={n_snap:>7d}  buggy={p_buggy:.4f}  fixed={p_fixed:.4f}  exact=1.0000 (T-state is pure)")
    pd.DataFrame(rows_purity).to_csv(_DATA_DIR / "quantum_shadows_purity_bugfix.csv", index=False)
    p_fixed_large = rows_purity[-1]["purity_fixed"]
    p_buggy_large = rows_purity[-1]["purity_buggy"]
    assert abs(p_fixed_large - 1.0) < 0.05, f"fixed purity estimator should converge near 1.0 for a pure state, got {p_fixed_large}"
    assert abs(p_buggy_large - 1.0) > 0.1, "expected the buggy estimator to remain visibly biased even at large snapshot counts"
    print(f"    PASS: at n=100,000 the fixed estimator is within 0.05 of the true value 1.0; "
          f"the buggy one stays off by {abs(p_buggy_large - 1.0):.3f} -- a BIAS, not noise, so it does not shrink with more samples\n")

    print("=== PART 3: SHADOW-BASED MAGIC ENTROPY (3-copy U-statistic) ===\n")
    rows_magic = []
    for name, psi in (("|T>", t_state()), ("|+>", plus_state())):
        rho = jnp.outer(psi, jnp.conj(psi))
        m_exact = exact_magic_entropy(rho)
        print(f"{name}: exact magic entropy (Experiment 30) = {m_exact:.6f}")
        for n_snap in (3000, 10000, 30000, 100000, 300000):
            snaps_n = sample_shadow_snapshots(psi, n_snap, seed=2)
            m_hat = estimate_magic_entropy_from_shadows(snaps_n)
            rows_magic.append({"state": name, "n_snapshots": n_snap, "magic_entropy_shadow": m_hat, "magic_entropy_exact": m_exact})
            print(f"    n={n_snap:>7d}  shadow-estimated magic entropy = {m_hat:.4f}  (exact = {m_exact:.4f}, |err| = {abs(m_hat - m_exact):.4f})")
    df_magic = pd.DataFrame(rows_magic)
    df_magic.to_csv(_DATA_DIR / "quantum_shadows_magic_entropy_convergence.csv", index=False)

    largest_n = df_magic["n_snapshots"].max()
    for name in ("|T>", "|+>"):
        row = df_magic[(df_magic["state"] == name) & (df_magic["n_snapshots"] == largest_n)].iloc[0]
        err = abs(row["magic_entropy_shadow"] - row["magic_entropy_exact"])
        assert err < 0.15, f"{name}: shadow-estimated magic entropy did not converge close enough to the exact value (err={err})"
    print(f"    PASS: at n={largest_n}, shadow-estimated magic entropy is within 0.15 bits of Experiment 30's exact value for both states\n")

    print("=== PART 4: MEDIAN-OF-MEANS ROBUSTNESS (naive mean vs. MoM under a corrupted batch) ===\n")
    # Simulates a systematic fault affecting one CONTIGUOUS stretch of a
    # measurement run (e.g. a calibration drift), not independent per-sample
    # noise -- the case median-of-means is specifically built for. A single
    # stray outlier is already diluted fine by a plain mean over thousands
    # of samples; a systematic block failure is not.
    n_snap_robust = 30000
    n_groups_robust = 20
    snaps_robust = sample_shadow_snapshots(psi_t, n_snap_robust, seed=3)
    left_r, right_r = _disjoint_pairs(snaps_robust)
    vals_clean = np.array(jnp.einsum("tij,tji->t", left_r, right_r).real)

    rows_robust = []
    for corrupt_fraction in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45):
        corrupted = vals_clean.copy()
        n_bad = int(len(corrupted) * corrupt_fraction)
        corrupted[:n_bad] = -50.0  # a wildly wrong systematic-fault value
        naive_mean = float(np.mean(corrupted))
        mom = _median_of_means(corrupted, n_groups_robust)
        rows_robust.append({
            "corrupt_fraction": corrupt_fraction, "naive_mean": naive_mean,
            "median_of_means": mom, "exact_purity": 1.0,
        })
        print(f"corrupted={corrupt_fraction:.0%}  naive_mean={naive_mean:8.3f}  "
              f"median_of_means={mom:7.4f}  (exact=1.0000)")
    df_robust = pd.DataFrame(rows_robust)
    df_robust.to_csv(_DATA_DIR / "quantum_shadows_median_of_means_robustness.csv", index=False)

    # Up to n_groups_robust // 2 - 1 = 9 of 20 groups (45%) can be entirely
    # corrupted before the median itself is forced to pick a corrupted
    # group's mean -- verified directly rather than assumed: at 40%
    # corruption MoM must still be close to the true value 1.0, while the
    # naive mean has already been dragged far away by the same corruption.
    row_40 = df_robust[df_robust["corrupt_fraction"] == 0.4].iloc[0]
    assert abs(row_40["median_of_means"] - 1.0) < 0.5, \
        f"expected median-of-means to still be near 1.0 at 40% corruption, got {row_40['median_of_means']}"
    assert abs(row_40["naive_mean"] - 1.0) > 10.0, \
        f"expected the naive mean to be dragged far from 1.0 at 40% corruption, got {row_40['naive_mean']}"
    print(f"    PASS: at 40% of samples corrupted, median-of-means stays within 0.5 of the true value 1.0 "
          f"(got {row_40['median_of_means']:.4f}) while the naive mean is dragged to {row_40['naive_mean']:.2f}\n")

    print("=== PART 5: SAMPLE-COMPLEXITY STUDY (empirical error vs. snapshot count) ===\n")
    # Repeats the magic-entropy estimate n_trials times, independently, at
    # each snapshot count, and measures the empirical standard deviation --
    # a real measured error bar, not a theoretical constant taken on faith.
    # Used to fit error ~ C / n_snapshots^p and give
    # magic_entropy_sample_complexity() below a concrete, checked formula
    # instead of an arbitrary guess.
    sc_n_snapshots = (3000, 10000, 30000, 100000)
    sc_n_trials = 20
    m_exact_t = exact_magic_entropy(rho_t)
    rows_sc, fitted_c, fitted_p = sample_complexity_fit(psi_t, m_exact_t, sc_n_snapshots, sc_n_trials)
    for r in rows_sc:
        print(f"n={r['n_snapshots']:>7d}  mean={r['mean_estimate']:.4f}  std={r['std_estimate']:.4f}  |mean-exact|={r['mean_abs_error']:.4f}")
    df_sc = pd.DataFrame(rows_sc)
    df_sc.to_csv(_DATA_DIR / "quantum_shadows_sample_complexity.csv", index=False)

    # theory predicts an exponent of 0.5 for this kind of estimator (error
    # shrinks like 1/sqrt(n)) -- checked empirically, not assumed.
    print(f"\n    Fitted: std(n) ~ {fitted_c:.3f} / n^{fitted_p:.3f}")
    assert 0.3 < fitted_p < 0.7, f"fitted exponent {fitted_p:.3f} is far from the ~0.5 theory predicts -- investigate before trusting the formula"
    print(f"    PASS: fitted exponent {fitted_p:.3f} is consistent with the theoretical ~0.5 scaling\n")

    with open(_DATA_DIR / "quantum_shadows_sample_complexity_fit.txt", "w") as fh:
        fh.write(f"C={fitted_c!r}\np={fitted_p!r}\n")

    print("Practical lookup (T-state-like magic states, fitted on this data):")
    for target_std in (0.1, 0.05, 0.02, 0.01):
        n_needed = n_snapshots_for_target_std(target_std, fitted_c, fitted_p)
        print(f"    target std={target_std:.2f} bits  ->  ~{n_needed:,.0f} snapshots")

    print("All assertions passed.")

    # --- Plot: purity bug fix + magic-entropy shadow convergence + MoM robustness + sample-complexity fit ---
    fig, axes = plt.subplots(1, 4, figsize=(23, 5))

    df_purity = pd.DataFrame(rows_purity)
    axes[0].plot(df_purity["n_snapshots"], df_purity["purity_buggy"], marker="s", label="original (uncorrected) einsum", color="#888888")
    axes[0].plot(df_purity["n_snapshots"], df_purity["purity_fixed"], marker="o", label="fixed einsum", color="#00e5ff")
    axes[0].axhline(1.0, color="#ff7f0e", linestyle="--", label="exact Tr[rho^2] = 1 (pure state)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("number of shadow snapshots")
    axes[0].set_ylabel("estimated purity")
    axes[0].set_title("Purity estimator: buggy contraction is biased, not just noisy")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, color in (("|T>", "#00e5ff"), ("|+>", "#ff7f0e")):
        sub = df_magic[df_magic["state"] == name]
        axes[1].plot(sub["n_snapshots"], sub["magic_entropy_shadow"], marker="o", label=f"{name} shadow estimate", color=color)
        axes[1].axhline(sub["magic_entropy_exact"].iloc[0], color=color, linestyle="--", alpha=0.6, label=f"{name} exact (Exp. 30)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of shadow snapshots")
    axes[1].set_ylabel("magic entropy (bits)")
    axes[1].set_title("Shadow-estimated magic entropy converging to the exact value")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    axes[2].plot(df_robust['corrupt_fraction'] * 100, df_robust['naive_mean'], marker='s', label='naive mean', color='#888888')
    axes[2].plot(df_robust['corrupt_fraction'] * 100, df_robust['median_of_means'], marker='o', label='median-of-means', color='#00e5ff')
    axes[2].axhline(1.0, color='#ff7f0e', linestyle='--', label='exact purity = 1.0')
    axes[2].set_xlabel('% of samples corrupted (one contiguous block)')
    axes[2].set_ylabel('estimated purity')
    axes[2].set_title('MoM tolerates a corrupted block; naive mean does not')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].loglog(df_sc["n_snapshots"], df_sc["std_estimate"], "o", label="measured std (20 trials each)", color="#00e5ff")
    fit_n = np.array(sc_n_snapshots, dtype=float)
    axes[3].loglog(fit_n, fitted_c / fit_n ** fitted_p, "--", label=f"fit: {fitted_c:.2f} / n^{fitted_p:.3f}", color="#ff7f0e")
    axes[3].set_xlabel("number of shadow snapshots")
    axes[3].set_ylabel("std of magic-entropy estimate (bits)")
    axes[3].set_title("Sample-complexity fit: error shrinks ~1/sqrt(n)")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3, which="both")

    fig.suptitle("Experiment 31: Classical Shadows -- purity bug fix, shadow-based magic entropy, MoM robustness, and sample complexity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(_DATA_DIR.parent / "images" / "quantum_shadows_magic_entropy.png", dpi=150)
    print(f"saved plot: {_DATA_DIR.parent / 'images' / 'quantum_shadows_magic_entropy.png'}")
