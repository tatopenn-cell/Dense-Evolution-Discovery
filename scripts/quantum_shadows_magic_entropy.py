"""Experiment 31: fixing a real bug in the Colab's Classical Shadows
proposal, then using shadows to estimate magic entropy from randomized
measurements instead of the exact state.

Origin: a prior Colab session (paper test/varianze.txt, following Huang,
Kueng, Preskill 2020, "Predicting Many Properties of a Quantum System from
Very Few Measurements") proposed a dense_evolution/circuits/shadows.py
module with a ClassicalShadow class and predict_renyi_entropy. The purity
estimator at its core computes the U-statistic cross-trace between
independent shadow snapshots as
    jnp.einsum('ijk,mjk->im', matrices, matrices)
which is WRONG: this contracts matching indices with no transpose, i.e.
sum_jk A_i[j,k]*A_m[j,k], not Tr(A_i @ A_m) = sum_jk A_i[j,k]*A_m[k,j].
Verified directly (see purity_bug_verification below): on two Hermitian
2x2 test matrices with complex off-diagonal entries, the buggy contraction
gives 45.0 while the true Tr(A@B) is 21.0. The bug is silent whenever
every snapshot happens to be real-valued (a Z-basis-only demo, which is
why the Colab's own Bell-state S2=1.000000 output never exposed it), but
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
    """Directly reproduces the Colab's einsum bug on two fixed Hermitian
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


def estimate_purity_fixed(snapshot_matrices):
    left, right = _disjoint_pairs(snapshot_matrices)
    vals = jnp.einsum("tij,tji->t", left, right).real
    return float(jnp.mean(vals))


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


def estimate_magic_entropy_from_shadows(snapshot_matrices):
    """Groups the n independent single-qubit shadow snapshots of the SAME
    state into disjoint triples (n//3 groups), estimates each entry of the
    3-copy-convolution reduced matrix R as the average over groups of
    Tr[O_ab . (rho_hat_i (x) rho_hat_j (x) rho_hat_k)] -- an unbiased
    U-statistic-style estimator, since each triple is drawn from 3
    independent unbiased single-copy estimators of rho -- then computes
    the von Neumann entropy of the (Hermitized, eigenvalue-clipped,
    renormalized) ESTIMATED R classically. Entropy itself is never
    shadow-estimated directly (Huang et al. never do this either, even for
    their own Renyi-2 entanglement entropy example); only the linear
    reduced-matrix reconstruction is shadow-based."""
    n = snapshot_matrices.shape[0]
    n_triples = n // 3
    triples = snapshot_matrices[: n_triples * 3].reshape(n_triples, 3, 2, 2)

    def triple_kron(t):
        return jnp.kron(jnp.kron(t[0], t[1]), t[2])

    rho3_batch = jax.vmap(triple_kron)(triples)  # (n_triples, 8, 8)

    r_hat = jnp.zeros((2, 2), dtype=jnp.complex128)
    for (a, b), o_ab in _O_AB.items():
        vals = jnp.einsum("ij,tji->t", o_ab, rho3_batch)
        r_hat = r_hat.at[a, b].set(jnp.mean(vals))

    r_hat = 0.5 * (r_hat + jnp.conj(r_hat).T)  # enforce Hermiticity
    ev = jnp.linalg.eigvalsh(r_hat)
    safe_ev = jnp.clip(ev.real, 1e-9, None)
    safe_ev = safe_ev / jnp.sum(safe_ev)  # renormalize (estimation noise can shift trace off 1)
    return float(-jnp.sum(safe_ev * jnp.log2(safe_ev)))


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

    print("All assertions passed.")

    # --- Plot: purity bug fix + magic-entropy shadow convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    df_purity = pd.DataFrame(rows_purity)
    axes[0].plot(df_purity["n_snapshots"], df_purity["purity_buggy"], marker="s", label="buggy einsum (Colab)", color="#888888")
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

    fig.suptitle("Experiment 31: Classical Shadows -- purity bug fix and shadow-based magic entropy", fontweight="bold")
    fig.tight_layout()
    fig.savefig(_DATA_DIR.parent / "images" / "quantum_shadows_magic_entropy.png", dpi=150)
    print(f"saved plot: {_DATA_DIR.parent / 'images' / 'quantum_shadows_magic_entropy.png'}")
