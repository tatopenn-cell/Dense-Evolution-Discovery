"""Experiment 30: the real Quantum Ruzsa / magic-entropy construction
(Key Unitary, Bu-Gu-Jaffe arXiv:2306.09292 Definitions 7-8) as a noise
diagnostic.

Origin: Experiment 29 fixed the Sandwiched Quantum Renyi Divergence, but a
separate proposal -- a "Quantum Ruzsa Divergence" built from a
pairwise convolution rho boxtimes sigma parametrized by s,t with
s^2+t^2=1 mod d -- turned out to have no valid solution for qubits (d=2).
Confirmed by reading both the original Ruzsa paper (arXiv:2401.14385) and
its companion (arXiv:2306.09292, "Stabilizer testing and magic entropy"):
the companion paper does NOT extend the pairwise formula to qubits. It
defines a structurally different, minimum-3-input "Key Unitary"
convolution instead (Definition 7/8: K quantum registers, K must be ODD,
K>=3 -- there is no K=2 case; two layers of CNOTs: first fan register 1's
value into every other register, then XOR every register back into
register 1). For qubits the smallest valid object is therefore the 3-fold
SELF-convolution of one state with itself, boxtimes_3(psi,psi,psi), and
the entropy of its reduced output register is what the paper calls "magic
entropy": zero for stabilizer states, positive for non-stabilizer ("magic")
states (Examples 32/33, p.16).

This experiment: (a) builds the real K=3, n=1-qubit Key Unitary circuit
exactly as specified in Definition 7, (b) verifies it against the paper's
own combinatorial identity (Lemma 9, eq. 15: V|x1 x2 x3> = |x1+x2+x3> (x)
|x2+x1> (x) |x3+x1>, all mod 2), (c) validates magic_entropy against the
paper's own claim (all six single-qubit stabilizer states give entropy
~0; the T-state and H-state, the two standard single-qubit magic states,
give entropy > 0), then (d) uses it as a NEW noise diagnostic on a T-state
under depolarizing and amplitude-damping noise, compared against the two
diagnostics this repo already has: uhlmann_fidelity (existing) and the
fixed sandwiched Renyi divergence (Experiment 29, duplicated here as a
small self-contained function per this repo's existing per-script
convention).

    python scripts/quantum_ruzsa_magic_entropy.py
"""
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from dense_evolution.mitigation import uhlmann_fidelity

jax.config.update("jax_enable_x64", True)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _cnot_matrix(control, target, n=3):
    """Permutation matrix for a CNOT(control->target) gate on n qubits,
    computational basis ordered as |q0 q1 ... q(n-1)>."""
    dim = 2 ** n
    mat = np.zeros((dim, dim))
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control]:
            bits[target] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        mat[j, i] = 1.0
    return mat


def _build_key_unitary_k3():
    """Definition 7 (Bu, Gu, Jaffe, arXiv:2306.09292, p.6), specialized to
    K=3 registers of n=1 qubit each (the minimum valid case -- K must be
    odd). U = (CNOT_{2->1} CNOT_{3->1}) (CNOT_{1->2} CNOT_{1->3}): layer 1
    fans register 1 into registers 2,3; layer 2 XORs registers 2,3 (their
    post-layer-1 values) back into register 1. The two gates within each
    layer act on disjoint qubit pairs modulo their shared qubit and
    commute, so intra-layer order does not matter."""
    layer1 = _cnot_matrix(0, 2) @ _cnot_matrix(0, 1)
    layer2 = _cnot_matrix(2, 0) @ _cnot_matrix(1, 0)
    v = layer2 @ layer1
    return jnp.array(v, dtype=jnp.complex128)


KEY_UNITARY_K3 = _build_key_unitary_k3()


@jax.jit
def self_convolve_3(rho):
    """boxtimes_3(rho,rho,rho) = Tr_{2,3}[V (rho (x) rho (x) rho) V^dagger]
    (Definition 8). Works for pure or mixed single-qubit rho."""
    rho_full = jnp.kron(jnp.kron(rho, rho), rho)
    evolved = KEY_UNITARY_K3 @ rho_full @ jnp.conj(KEY_UNITARY_K3).T
    tensor = evolved.reshape(2, 2, 2, 2, 2, 2)
    return jnp.einsum("ijkljk->il", tensor)


def magic_entropy(rho, eps=1e-12):
    """Von Neumann entropy (bits) of the reduced output register of the
    3-fold self-convolution. Zero for stabilizer states, positive for
    magic states -- this is the paper's "magic entropy" (p.16)."""
    reduced = self_convolve_3(rho)
    ev = jnp.linalg.eigvalsh(reduced)
    safe_ev = jnp.clip(ev.real, eps, 1.0)
    return float(-jnp.sum(safe_ev * jnp.log2(safe_ev)))


@jax.jit
def sandwiched_quantum_renyi_divergence(rho, sigma, alpha=1.5):
    """Duplicated from scripts/sandwiched_renyi_density_matrix.py
    (Experiment 29) as a small, self-contained, already-fixed-and-
    validated function, per this repo's existing convention of keeping
    each experiment script standalone rather than importing across
    scripts/ (which would execute that script's own __main__ block)."""
    eps = 1e-12

    def case_half(_):
        ev_r, ec_r = jnp.linalg.eigh(rho)
        safe_ev_r = jnp.where(ev_r > eps, ev_r, 0.0)
        sqrt_rho = (ec_r * jnp.sqrt(safe_ev_r)) @ jnp.conj(ec_r).T
        uhlmann_mat = sqrt_rho @ sigma @ sqrt_rho
        ev_u = jnp.linalg.eigvalsh(uhlmann_mat)
        safe_ev_u = jnp.where(ev_u > eps, ev_u, 0.0)
        fidelity = jnp.clip(jnp.sum(jnp.sqrt(safe_ev_u)), 0.0, 1.0)
        return -2.0 * jnp.log2(jnp.maximum(fidelity, eps))

    def case_one(_):
        ev_r, ec_r = jnp.linalg.eigh(rho)
        safe_ev_r = jnp.where(ev_r > eps, ev_r, 1.0)
        log_rho = (ec_r * jnp.where(ev_r > eps, jnp.log2(safe_ev_r), 0.0)) @ jnp.conj(ec_r).T
        ev_s, ec_s = jnp.linalg.eigh(sigma)
        safe_ev_s = jnp.where(ev_s > eps, ev_s, 1.0)
        log_sigma = (ec_s * jnp.where(ev_s > eps, jnp.log2(safe_ev_s), 0.0)) @ jnp.conj(ec_s).T
        return jnp.trace(rho @ (log_rho - log_sigma)).real

    def case_general(_):
        exponent = (1.0 - alpha) / (2.0 * alpha)
        ev_s, ec_s = jnp.linalg.eigh(sigma)
        mask_s = ev_s > eps
        safe_ev_s = jnp.where(mask_s, ev_s, 1.0)
        pow_ev_s = jnp.where(mask_s, safe_ev_s ** exponent, 0.0)
        sigma_pow = (ec_s * pow_ev_s) @ jnp.conj(ec_s).T
        int_m = sigma_pow @ rho @ sigma_pow
        v_int = jnp.linalg.eigvalsh(int_m)
        mask_int = v_int > eps
        safe_v_int = jnp.where(mask_int, v_int, 1.0)
        pow_v_int = jnp.where(mask_int, safe_v_int ** alpha, 0.0)
        tr_inner = jnp.sum(pow_v_int)
        is_support_violation = (alpha > 1.0) & (tr_inner < 1.0 - 1e-9)
        tr_inner_safe = jnp.maximum(tr_inner, eps)
        finite_result = ((1.0 / (alpha - 1.0)) * jnp.log2(tr_inner_safe)).real
        return jnp.where(is_support_violation, jnp.inf, finite_result)

    is_half = jnp.isclose(alpha, 0.5)
    is_one = jnp.isclose(alpha, 1.0)
    return jax.lax.cond(
        is_half, case_half,
        lambda _: jax.lax.cond(is_one, case_one, case_general, operand=None),
        operand=None,
    )


def _stabilizer_states():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    plus = (zero + one) / jnp.sqrt(2.0)
    minus = (zero - one) / jnp.sqrt(2.0)
    plus_i = (zero + 1j * one) / jnp.sqrt(2.0)
    minus_i = (zero - 1j * one) / jnp.sqrt(2.0)
    kets = {"|0>": zero, "|1>": one, "|+>": plus, "|->": minus, "|+i>": plus_i, "|-i>": minus_i}
    return {name: jnp.outer(sv, jnp.conj(sv)) for name, sv in kets.items()}


def t_state_rho():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    sv = (zero + jnp.exp(1j * jnp.pi / 4.0) * one) / jnp.sqrt(2.0)
    return jnp.outer(sv, jnp.conj(sv))


def h_state_rho():
    zero = jnp.array([1.0, 0.0], dtype=jnp.complex128)
    one = jnp.array([0.0, 1.0], dtype=jnp.complex128)
    beta = jnp.pi / 8.0
    sv = jnp.cos(beta) * zero + jnp.sin(beta) * one
    return jnp.outer(sv, jnp.conj(sv))


def depolarize_1q(rho, p):
    return (1.0 - p) * rho + p * jnp.eye(2, dtype=jnp.complex128) / 2.0


def amplitude_damping_1q(rho, p):
    k0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - p)]], dtype=jnp.complex128)
    k1 = jnp.array([[0.0, jnp.sqrt(p)], [0.0, 0.0]], dtype=jnp.complex128)
    return (k0 @ rho @ jnp.conj(k0).T) + (k1 @ rho @ jnp.conj(k1).T)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)

    print("=== PART 1: KEY UNITARY VERIFICATION (Lemma 9 combinatorial identity) ===\n")
    # V|x1 x2 x3> = |x1+x2+x3> (x) |x2+x1> (x) |x3+x1>  (mod 2, all bits)
    all_ok = True
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                idx_in = (x1 << 2) | (x2 << 1) | x3
                sv_in = jnp.zeros(8, dtype=jnp.complex128).at[idx_in].set(1.0)
                sv_out = KEY_UNITARY_K3 @ sv_in
                y1, y2, y3 = x1 ^ x2 ^ x3, x2 ^ x1, x3 ^ x1
                idx_expected = (y1 << 2) | (y2 << 1) | y3
                out_idx = int(jnp.argmax(jnp.abs(sv_out)))
                ok = out_idx == idx_expected and abs(complex(sv_out[out_idx])) > 0.999
                all_ok &= ok
                print(f"|{x1}{x2}{x3}> -> |{y1}{y2}{y3}>  (predicted idx={idx_expected}, got={out_idx})  ok={ok}")
    assert all_ok, "Key Unitary circuit does not match Lemma 9's stated combinatorial identity"
    print("    PASS: circuit matches the paper's own Lemma 9 exactly\n")

    print("=== PART 2: MAGIC ENTROPY -- STABILIZER STATES VS. MAGIC STATES ===\n")
    rows_states = []
    stab = _stabilizer_states()
    for name, rho in stab.items():
        m = magic_entropy(rho)
        rows_states.append({"state": name, "magic_entropy": m, "is_stabilizer": True})
        print(f"{name:6s}  magic_entropy = {m:.10f}   (stabilizer)")
    for name, rho in (("|T>", t_state_rho()), ("|H>", h_state_rho())):
        m = magic_entropy(rho)
        rows_states.append({"state": name, "magic_entropy": m, "is_stabilizer": False})
        print(f"{name:6s}  magic_entropy = {m:.10f}   (magic state)")
    pd.DataFrame(rows_states).to_csv(_DATA_DIR / "quantum_ruzsa_magic_entropy_states.csv", index=False)

    max_stab = max(r["magic_entropy"] for r in rows_states if r["is_stabilizer"])
    min_magic = min(r["magic_entropy"] for r in rows_states if not r["is_stabilizer"])
    assert max_stab < 1e-8, f"expected all stabilizer states to give ~0 magic entropy, got max={max_stab}"
    assert min_magic > 1e-4, f"expected T/H magic states to give nonzero magic entropy, got min={min_magic}"
    print(f"    PASS: stabilizer states all ~0 (max={max_stab:.2e}); T/H states clearly nonzero (min={min_magic:.6f})\n")

    print("=== PART 3: NOISE DIAGNOSTIC -- MAGIC ENTROPY VS. EXISTING METRICS ===\n")
    t_ideal = t_state_rho()
    rows_noise = []
    for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        rho_depol = depolarize_1q(t_ideal, p)
        rho_ad = amplitude_damping_1q(t_ideal, p)
        m_depol = magic_entropy(rho_depol)
        m_ad = magic_entropy(rho_ad)
        fid_depol = float(uhlmann_fidelity(t_ideal, rho_depol))
        fid_ad = float(uhlmann_fidelity(t_ideal, rho_ad))
        renyi_depol = float(sandwiched_quantum_renyi_divergence(t_ideal, rho_depol, alpha=1.5))
        renyi_ad = float(sandwiched_quantum_renyi_divergence(t_ideal, rho_ad, alpha=1.5))
        rows_noise.append({
            "p": p,
            "magic_entropy_depolarizing": m_depol, "magic_entropy_amp_damping": m_ad,
            "uhlmann_fidelity_depolarizing": fid_depol, "uhlmann_fidelity_amp_damping": fid_ad,
            "renyi_1.5_depolarizing": renyi_depol, "renyi_1.5_amp_damping": renyi_ad,
        })
        print(f"p={p:.1f}  magic(depol)={m_depol:.4f}  magic(amp.damp)={m_ad:.4f}  "
              f"fid(depol)={fid_depol:.4f}  D_1.5(depol)={renyi_depol:.4f}")
    df_noise = pd.DataFrame(rows_noise)
    df_noise.to_csv(_DATA_DIR / "quantum_ruzsa_magic_entropy_noise_sweep.csv", index=False)

    # A first hypothesis here was wrong and worth keeping visible: at p=1,
    # depolarizing noise drives rho to I/2, which is NOT a pure stabilizer
    # state -- it is maximally mixed, so magic_entropy(I/2) = 1 (its own
    # intrinsic entropy), not 0. Confirmed by direct computation below.
    # Amplitude damping is different: at p=1 it drives ANY state to the
    # pure state |0>, which IS a stabilizer state, so magic entropy there
    # genuinely returns to ~0. So magic entropy's endpoint behavior depends
    # on whether the noise channel's fixed point is a pure stabilizer state
    # or a generic mixed state -- it is not simply "distance from magic"
    # for mixed inputs, it conflates that with the state's own mixedness.
    # This is a real, qualitatively distinct signature from fidelity/Renyi
    # divergence (which saturate monotonically under both channels): under
    # amplitude damping, magic entropy rises then falls back to exactly 0;
    # under depolarizing, it rises monotonically to 1 and stays there.
    m_at_1_depol = df_noise.loc[df_noise["p"] == 1.0, "magic_entropy_depolarizing"].iloc[0]
    m_at_1_ad = df_noise.loc[df_noise["p"] == 1.0, "magic_entropy_amp_damping"].iloc[0]
    assert abs(m_at_1_depol - 1.0) < 1e-6, f"expected magic entropy -> 1 (I/2's own entropy) under full depolarizing, got {m_at_1_depol}"
    assert m_at_1_ad < 1e-6, f"expected magic entropy -> 0 under full amplitude damping (fixed point |0> is a stabilizer state), got {m_at_1_ad}"
    m_ad_series = df_noise["magic_entropy_amp_damping"].to_numpy()
    peak_idx = int(np.argmax(m_ad_series))
    assert 0 < peak_idx < len(m_ad_series) - 1, "expected amplitude-damping magic entropy to rise then fall, not be monotonic"
    print(f"    PASS: depolarizing -> 1 (I/2's own entropy, not zero -- corrects an initial wrong assumption); "
          f"amplitude damping -> 0 (its p=1 fixed point |0> is a stabilizer state), after rising to a peak at p={df_noise['p'].iloc[peak_idx]:.1f}. "
          f"Neither fidelity nor Renyi divergence show this non-monotonic, channel-dependent endpoint behavior.\n")

    print("All assertions passed.")

    # --- Plot: state validation + noise-diagnostic comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    df_states = pd.DataFrame(rows_states)
    colors = ["#00e5ff" if s else "#ff7f0e" for s in df_states["is_stabilizer"]]
    axes[0].bar(df_states["state"], df_states["magic_entropy"], color=colors)
    axes[0].set_ylabel("magic entropy (bits)")
    axes[0].set_title("Stabilizer states (cyan, ~0) vs. T/H magic states (orange, >0)")
    axes[0].grid(alpha=0.3)

    # Amplitude damping shown here (not depolarizing) because it is the
    # qualitatively richer, more honest comparison: magic entropy rises
    # then returns cleanly to exactly 0 (its p=1 fixed point |0> is a
    # stabilizer state), fidelity rises back up as the state re-purifies,
    # and the Renyi divergence instead DIVERGES to +inf at p=1 (a genuine
    # support-mismatch case from Experiment 29's fix, capped here for
    # display) -- three diagnostics, three different endpoint behaviors on
    # the same noise sweep.
    renyi_ad_capped = df_noise["renyi_1.5_amp_damping"].replace([np.inf], 3.0)
    axes[1].plot(df_noise["p"], df_noise["magic_entropy_amp_damping"], marker="o", label="magic entropy (new)", color="#00e5ff")
    axes[1].plot(df_noise["p"], df_noise["uhlmann_fidelity_amp_damping"], marker="s", label="uhlmann_fidelity (existing)", color="#888888")
    axes[1].plot(df_noise["p"], renyi_ad_capped, marker="^", label="Renyi D_1.5 (Exp. 29, inf capped at 3.0)", color="#ff7f0e")
    axes[1].set_xlabel("amplitude-damping noise probability p")
    axes[1].set_ylabel("metric value")
    axes[1].set_title("Magic entropy as a noise diagnostic vs. existing metrics")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Experiment 30: Quantum Ruzsa Key Unitary -> magic entropy as a noise diagnostic", fontweight="bold")
    fig.tight_layout()
    fig.savefig(_DATA_DIR.parent / "images" / "quantum_ruzsa_magic_entropy.png", dpi=150)
    print(f"saved plot: {_DATA_DIR.parent / 'images' / 'quantum_ruzsa_magic_entropy.png'}")
