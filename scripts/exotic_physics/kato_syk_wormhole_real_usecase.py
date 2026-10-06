"""Does the Kato degeneracy fix (dense_evolution.physics.spectral) matter
for the real wormhole/SYK teleportation signal used elsewhere in this
repo? Answers the gap explicitly left open in
docs/spectral_evolve_kato_degeneracy.md: "not yet wired into a real
Discovery use case with genuine exact degeneracy."

Scans the SYK+coupling wormhole Hamiltonian's teleportation signal
(N_MAJ=8, dim=256, one seed) across g -> 0, comparing gradients from
`dense_evolution.physics.spectral.spectral_evolve` (Kato) against plain
`jnp.linalg.eigh` (std) and central finite differences.

HONEST RESULT: std eigh does NOT diverge from finite differences at any
g tested, all the way down to g=1e-6, even though H_total's min_gap dips
below the 1e-8 degeneracy tolerance in that range (min_gap ~2.8e-14 at
g=1e-6). max|Kato-fd| and max|std-fd| are equal to 4 significant figures
(~1.67e-05) -- Kato provides no measurable benefit here.

Why this differs from the synthetic case that motivated Kato
(docs/spectral_evolve_kato_degeneracy.md's H with four EXACT
doubly-degenerate eigenvalue pairs, std error 0.98): that H was built as
U diag([1,1,2,2,...]) U^dagger, a mathematically exact tie. Real physical
Hamiltonians built from random disorder (SYK's random J tensor) generate
near-degeneracies that are numerically small but never mathematically
exact -- eigh's division by a merely-small (not exactly zero) gap doesn't
catastrophically fail the way division by an exactly-zero gap does. The
degeneracy tolerance (1e-8) used to decide "is this degenerate" is a
useful diagnostic threshold, not the precision floor where eigh's
gradient actually breaks.

Conclusion: `spectral_evolve` remains a validated, correct utility for
the case it targets (genuine exact ties), but this real Discovery
Hamiltonian isn't that case -- plain `jnp.linalg.eigh` stays correct and
faster for the wormhole teleportation protocol as actually used here.

A real usage gotcha found writing this script: `spectral_evolve` calls
`ensure_x64()` internally, but that can't retroactively upcast arrays
the CALLER already built in float32/complex64 before ever calling it.
Omitting this file's own `jax.config.update("jax_enable_x64", True)`
line and relying on `spectral_evolve`'s internal call alone produced a
spurious NaN at g=1e-6 on the first (JIT-tracing) call only -- callers
building their own Hamiltonians/state vectors before calling
`spectral_evolve` still need to enable x64 themselves, up front.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dense_evolution.physics.spectral import spectral_evolve

DEGENERACY_TOL = 1e-8


def spectral_evolve_std(H, t):
    w, v = jnp.linalg.eigh(H)
    return v @ jnp.diag(jnp.exp(-1j * w * t)) @ v.conj().T


def _majorana_matrices(n_qubits):
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def kron_list(mats):
        out = np.array([[1.0]], dtype=np.complex128)
        for m in mats:
            out = np.kron(out, m)
        return out

    chis = []
    for i in range(2 * n_qubits):
        j = i // 2
        letter = X if (i % 2 == 0) else Y
        mats = [Z] * j + [letter] + [I2] * (n_qubits - j - 1)
        chis.append(kron_list(mats))
    return chis


def build_syk_hamiltonian(n_majorana, seed, J_scale=1.0):
    n_qubits = n_majorana // 2
    chis = _majorana_matrices(n_qubits)
    key = jax.random.PRNGKey(seed)
    indices = [(i, j, k, l)
               for i in range(n_majorana)
               for j in range(i + 1, n_majorana)
               for k in range(j + 1, n_majorana)
               for l in range(k + 1, n_majorana)]
    J = jax.random.normal(key, (len(indices),)) * J_scale / np.sqrt(len(indices))
    H = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=np.complex128)
    for coeff, (i, j, k, l) in zip(np.array(J), indices):
        H = H + float(coeff) * (chis[i] @ chis[j] @ chis[k] @ chis[l])
    return jnp.asarray(H)


def build_coupling(chis_L, chis_R, g, coeffs, skip=2):
    dim_side = chis_L[0].shape[0]
    H_c = jnp.zeros((dim_side * dim_side, dim_side * dim_side), dtype=jnp.complex128)
    for idx, (cL, cR, a) in enumerate(zip(chis_L, chis_R, coeffs)):
        if idx < skip:
            continue
        H_c = H_c + g * a * jnp.kron(cL, cR)
    return H_c


def make_tfd_true(H_side, beta):
    E, V = jnp.linalg.eigh(H_side)
    weights = jnp.exp(-beta * E / 2.0)
    Z = jnp.sum(weights ** 2)
    weights = weights / jnp.sqrt(Z)
    dim_side = H_side.shape[0]
    psi_tfd = jnp.zeros(dim_side * dim_side, dtype=jnp.complex128)
    for n in range(dim_side):
        psi_tfd = psi_tfd + weights[n] * jnp.kron(V[:, n], V[:, n])
    return psi_tfd / jnp.linalg.norm(psi_tfd)


if __name__ == "__main__":
    N_MAJ = 8
    DIM_SIDE = 2 ** (N_MAJ // 2)

    H_side = build_syk_hamiltonian(N_MAJ, seed=42)
    chis = [jnp.asarray(c) for c in _majorana_matrices(N_MAJ // 2)]
    COEFFS = jnp.array([1.0, 0.5, 0.25, 0.125, 0.7, 0.3, 0.9, 0.15])

    X_side = jnp.asarray(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    I2 = jnp.eye(2, dtype=np.complex128)

    def kron_n(ops):
        out = jnp.array([[1.0 + 0j]], dtype=jnp.complex128)
        for o in ops:
            out = jnp.kron(out, o)
        return out

    X_L_full = jnp.kron(kron_n([X_side, I2, I2, I2]), kron_n([I2, I2, I2, I2]))
    X_R_full = jnp.kron(kron_n([I2, I2, I2, I2]), kron_n([X_side, I2, I2, I2]))

    psi_tfd = make_tfd_true(H_side, beta=2.0)
    psi_msg = X_R_full @ psi_tfd
    psi_msg = psi_msg / jnp.linalg.norm(psi_msg)
    ref = X_L_full @ psi_tfd
    ref = ref / jnp.linalg.norm(ref)

    H_L_full = jnp.kron(H_side, jnp.eye(DIM_SIDE))
    H_R_full = jnp.kron(jnp.eye(DIM_SIDE), H_side)

    def signal_kato(g):
        H_total = H_L_full + H_R_full + build_coupling(chis, chis, g, COEFFS, skip=2)
        U = spectral_evolve(H_total, 1.0)
        return jnp.abs(jnp.vdot(ref, U @ psi_msg)) ** 2

    def signal_std(g):
        H_total = H_L_full + H_R_full + build_coupling(chis, chis, g, COEFFS, skip=2)
        U = spectral_evolve_std(H_total, 1.0)
        return jnp.abs(jnp.vdot(ref, U @ psi_msg)) ** 2

    grad_kato = jax.jit(jax.grad(signal_kato))
    grad_std = jax.jit(jax.grad(signal_std))

    g_scan = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5]
    print(f"{'g':<10}{'min_gap':<12}{'Kato':<15}{'std':<15}{'fd':<15}{'diff_kato':<12}{'diff_std':<12}")
    max_dk, max_ds = 0.0, 0.0
    for g_val in g_scan:
        H_total = H_L_full + H_R_full + build_coupling(chis, chis, g_val, COEFFS, skip=2)
        w_np = np.asarray(jnp.linalg.eigvalsh(H_total))
        gaps = np.diff(np.sort(w_np))
        pos_gaps = gaps[np.abs(gaps) > 1e-14]
        min_gap = float(np.abs(pos_gaps).min()) if len(pos_gaps) else 0.0

        gk = float(grad_kato(g_val))
        gs = float(grad_std(g_val))
        h_g = max(g_val * 0.05, 1e-7)
        fd = (float(signal_kato(g_val + h_g)) - float(signal_kato(g_val - h_g))) / (2 * h_g)
        dk, ds = abs(gk - fd), abs(gs - fd)
        max_dk, max_ds = max(max_dk, dk), max(max_ds, ds)
        print(f"{g_val:<10.1e}{min_gap:<12.2e}{gk:<+15.6e}{gs:<+15.6e}{fd:<+15.6e}{dk:<12.2e}{ds:<12.2e}")

    print(f"\nmax |Kato - fd|: {max_dk:.4e}")
    print(f"max |std  - fd|: {max_ds:.4e}")
    print("Kato and std comparable -- no measurable benefit for this real Hamiltonian."
          if max_ds < max_dk * 2 else "Kato measurably better here.")
