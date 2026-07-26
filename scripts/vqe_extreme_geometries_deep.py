import pathlib
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════════
# Deeper ansatz (12 parameters, N_Q=7 / 6 bonds) version of
# vqe_extreme_geometries.py (10 parameters, N_Q=6 / 5 bonds). The model and
# every function here are IDENTICAL in structure to that script -- N_Q is
# the only thing that changes, since everything downstream is already
# parametric in N_BONDS = N_Q - 1. This is not a different ansatz, just one
# more bond in the same open tight-binding chain.
#
# HONEST FINDING: the qualitative pattern at 12 parameters is NOT identical
# to the one at 10. Measured deficit_fraction (see deficit_fraction() below,
# same scale-normalized metric as the 10-parameter script):
#
#   uniform (equilibrium/compressed/dissociated): 0.163 (all three
#     identical -- scale invariance holds at this depth too)
#   mutazione_localizzata:       0.188 (ABOVE the uniform baseline)
#   distorsione_alternata:       0.264 (ABOVE the uniform baseline)
#   mutazioni_congiunte_estremi: 0.464 (~2.8x baseline, still the worst)
#
# At 10 parameters, mutazione_localizzata (0.103) and distorsione_alternata
# (0.027) sat BELOW the uniform baseline (0.169) -- at 12 parameters both
# sit ABOVE it instead. Only mutazioni_congiunte_estremi is a robust
# standout across both depths. Whether a given irregular-geometry pattern
# helps or hurts the rigid shared-angle approximation is not a fixed
# property of the pattern alone -- it depends on the chain depth too. This
# is reported as measured, not adjusted to match the 10-parameter script.
#
# CONFORMATIONAL SEARCH -- a second, distinct feature this script adds
# beyond the 10-parameter one: instead of only comparing rigid vs. adaptive
# theta at hand-picked geometries, optimize_geometry_and_theta_jointly()
# also optimizes the bond distances R_q themselves (classical analytic
# gradient, no PSR needed -- R only enters through t_q(R_q) and the
# repulsion term), searching for a genuine minimum-energy conformation.
#
# This REQUIRES a per-bond repulsion term (_local_repulsion_per_bond),
# NOT the mean-based _global_repulsion used by the fixed-geometry benchmark
# above. Under free R optimization, mean-based repulsion dilutes the
# repulsive cost by 1/N_BONDS, so a single bond can compress almost without
# limit -- verified empirically: with mean-based repulsion, every tested
# starting point drove one bond straight to an artificial clip boundary
# with a suspiciously large negative energy, not a genuine interior
# minimum. Per-bond repulsion gives every bond its own local repulsive
# wall, independent of the others, producing a genuine stationary point.
# ═══════════════════════════════════════════════════════════════════════════

N_Q = 7
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

T0_MOL, BETA, R0_MOL, V0_MOL, GAMMA = 2.11, 1.5, 2.35, 5.4, 3.0

N_BONDS = N_Q - 1                  # 6
N_PARAMS = 2 * N_BONDS              # 12
ROWS_PER_POINT = 1 + 4 * N_BONDS    # 25: 1 base + 2 shifts x 2 gates/bond

RIGID_THETA = 0.38  # the fixed shared angle from vqe_silicon_molecular.py --
                     # the "rigid commercial-style" approximation under test


def _build_ops() -> list:
    ops = [['x', 0]]
    for q in range(N_BONDS):
        ops += [['cx', q + 1, q], ['ry', q + 1, 'A'], ['cx', q, q + 1], ['ry', q + 1, 'B'], ['cx', q + 1, q]]
    return ops


def _base_row(theta_per_bond: np.ndarray) -> np.ndarray:
    row = np.empty(N_PARAMS, dtype=np.float64)
    row[0::2] = theta_per_bond
    row[1::2] = -theta_per_bond
    return row


def _kinetic_per_bond_from_sv(sv: np.ndarray) -> np.ndarray:
    """Per-bond <XX+YY> contribution, kept as an array (not summed) --
    needed because an irregular geometry weights each bond's kinetic term
    by its OWN local hopping strength t_q(R_q), not a single shared t(R)."""
    dim = len(sv)
    idx = np.arange(dim)
    kin = np.empty(N_BONDS)
    for q in range(N_BONDS):
        mask = (1 << q) | (1 << (q + 1))
        pf = sv[idx ^ mask]
        xx = np.real(np.sum(np.conj(sv) * pf))
        bi = (idx & (1 << q)) >> q
        bj = (idx & (1 << (q + 1))) >> (q + 1)
        yy = np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
        kin[q] = xx + yy
    return kin


def batched_per_bond_kinetic_and_jacobian(theta_matrix: np.ndarray):
    """theta_matrix: shape (n_geo, N_BONDS). Returns
    (kinetic[n_geo, N_BONDS], jacobian[n_geo, N_BONDS, N_BONDS]) where
    jacobian[g, q, r] = d(kinetic_q)/d(theta_r) at geometry g.

    Exact per-gate Parameter-Shift Rule: shifting only bond r's own two
    gates by +/- pi/2 gives the exact derivative of ANY observable
    (including every per-bond kinetic_q, not just their sum) with respect
    to theta_r, regardless of which qubits that observable acts on."""
    n_g = theta_matrix.shape[0]
    grid = np.zeros((n_g * ROWS_PER_POINT, N_PARAMS), dtype=np.float64)
    for i in range(n_g):
        base = _base_row(theta_matrix[i])
        off = i * ROWS_PER_POINT
        grid[off] = base
        r = off + 1
        for k in range(N_PARAMS):
            plus = base.copy();  plus[k]  += np.pi / 2
            minus = base.copy(); minus[k] -= np.pi / 2
            grid[r] = plus
            grid[r + 1] = minus
            r += 2

    batch = np.asarray(sim.run_parametric_batch_jit(_build_ops(), jnp.array(grid, dtype=jnp.float64)))

    kinetic = np.empty((n_g, N_BONDS))
    jacobian = np.zeros((n_g, N_BONDS, N_BONDS))
    for i in range(n_g):
        off = i * ROWS_PER_POINT
        kinetic[i] = _kinetic_per_bond_from_sv(batch[off])
        r = off + 1
        for k in range(N_PARAMS):
            k_plus = _kinetic_per_bond_from_sv(batch[r])
            k_minus = _kinetic_per_bond_from_sv(batch[r + 1])
            partial = 0.5 * (k_plus - k_minus)   # shape (N_BONDS,): d(k_q)/d(gate_k)
            bond_idx = k // 2
            sign = 1.0 if (k % 2 == 0) else -1.0
            jacobian[i, :, bond_idx] += partial * sign
            r += 2

    return kinetic, jacobian


def _local_hopping(R_matrix: np.ndarray) -> np.ndarray:
    """Per-bond hopping strength t_q(R_q) -- local, bond-by-bond."""
    return T0_MOL * np.exp(-BETA * (R_matrix - R0_MOL))


def _global_repulsion(R_matrix: np.ndarray) -> np.ndarray:
    """Global steric/electrostatic repulsion evaluated at the geometry's
    MEAN bond length -- used ONLY by the fixed-geometry benchmark below.
    Same single-term SHAPE as vqe_silicon_molecular's V_rep(R) (this is
    what makes the 10-parameter script, N_Q=6/5 bonds, collapse exactly
    onto that reference formula in the uniform-R limit -- because there
    N_BONDS matches the reference's hardcoded 5 bonds). This 12-parameter
    script has N_BONDS=6, a DIFFERENT chain length / different physical
    system, so it does NOT reduce to that same reference formula even at
    uniform R -- there is no single-formula reference for a 6-bond chain
    in this repo to check against; correctness here is instead verified
    against an independent fresh-circuit calculation in the test suite.
    NOT suitable for free geometry optimization -- see
    optimize_geometry_and_theta_jointly / the per-bond repulsion note at
    the top of this file."""
    return V0_MOL * np.exp(-GAMMA * (R_matrix.mean(axis=1) - R0_MOL))


def _local_repulsion_per_bond(R_matrix: np.ndarray) -> np.ndarray:
    """Per-bond repulsion V0*exp(-GAMMA*(R_q-R0)), NOT summed here (caller
    sums) -- used ONLY by the conformational search. Gives every bond its
    own repulsive wall, independent of the others, which the mean-based
    _global_repulsion does not provide under free R optimization (see the
    top-of-file note)."""
    return V0_MOL * np.exp(-GAMMA * (R_matrix - R0_MOL))


def energy_from_theta(R_matrix: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
    """R_matrix, theta_matrix: shape (n_geo, N_BONDS). Returns total energy
    per geometry: E = -sum_q (t_q(R_q)/2) * k_q(theta) + V_rep(mean(R))."""
    kinetic, _ = batched_per_bond_kinetic_and_jacobian(theta_matrix)
    t_local = _local_hopping(R_matrix)
    return -np.sum((t_local / 2.0) * kinetic, axis=1) + _global_repulsion(R_matrix)


def optimize_theta_per_geometry(R_matrix: np.ndarray, n_epochs: int = 120, lr: float = 0.15,
                                 theta_init: float = RIGID_THETA, verbose: bool = True):
    """Adam-optimizes an independent per-bond theta vector at EVERY geometry
    (row of R_matrix) in parallel, minimizing E(R_row, theta_row) with the
    exact per-bond Jacobian, chain-ruled through each bond's own local
    weight t_q(R_q)."""
    n_geo = R_matrix.shape[0]
    t_local = _local_hopping(R_matrix)
    v_global = _global_repulsion(R_matrix)
    theta = np.full((n_geo, N_BONDS), theta_init, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        kinetic, jac = batched_per_bond_kinetic_and_jacobian(theta)
        grad = -np.einsum('gq,gqr->gr', t_local / 2.0, jac)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        theta -= (lr / (np.sqrt(v_hat) + eps)) * m_hat

        if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == n_epochs):
            e_now = -np.sum((t_local / 2.0) * kinetic, axis=1) + v_global
            print(f"Epoch {epoch:03d}/{n_epochs} | mean|grad|: {np.mean(np.abs(grad)):.6f} "
                  f"| mean E: {e_now.mean():+.6f} eV | elapsed: {time.perf_counter()-t0:.2f}s")

    kinetic_f, _ = batched_per_bond_kinetic_and_jacobian(theta)
    E_final = -np.sum((t_local / 2.0) * kinetic_f, axis=1) + v_global
    return theta, E_final


def optimize_shared_theta_per_geometry(R_matrix: np.ndarray, n_epochs: int = 120, lr: float = 0.15,
                                        theta_init: float = RIGID_THETA):
    """Adam-optimizes ONE shared scalar theta per geometry -- the best a
    rigid, single-parameter solver could EVER do for that geometry, as
    opposed to the arbitrary hardcoded RIGID_THETA."""
    n_geo = R_matrix.shape[0]
    t_local = _local_hopping(R_matrix)
    theta_scalar = np.full(n_geo, theta_init, dtype=np.float64)
    m = np.zeros(n_geo)
    v = np.zeros(n_geo)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(1, n_epochs + 1):
        theta_row = np.repeat(theta_scalar[:, None], N_BONDS, axis=1)
        kinetic, jac = batched_per_bond_kinetic_and_jacobian(theta_row)
        grad_perbond = -np.einsum('gq,gqr->gr', t_local / 2.0, jac)
        grad_scalar = grad_perbond.sum(axis=1)

        m = beta1 * m + (1 - beta1) * grad_scalar
        v = beta2 * v + (1 - beta2) * (grad_scalar ** 2)
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        theta_scalar -= (lr / (np.sqrt(v_hat) + eps)) * m_hat

    theta_row = np.repeat(theta_scalar[:, None], N_BONDS, axis=1)
    kinetic_f, _ = batched_per_bond_kinetic_and_jacobian(theta_row)
    return theta_scalar, kinetic_f


def deficit_fraction(R_matrix: np.ndarray, kinetic_perbond: np.ndarray, kinetic_shared: np.ndarray) -> np.ndarray:
    """Scale-normalized measure of how much weighted kinetic energy the best
    possible SHARED angle leaves on the table relative to per-bond adaptive
    optimization: (K_perbond - K_shared) / K_perbond."""
    t_local = _local_hopping(R_matrix)
    k_pb = np.sum(t_local * kinetic_perbond, axis=1)
    k_sh = np.sum(t_local * kinetic_shared, axis=1)
    return (k_pb - k_sh) / k_pb


def build_extreme_geometries() -> dict:
    """6 hand-picked bond-length patterns (one distance R_q per bond, in
    Angstrom, N_BONDS=6) -- same 6 scenario types as the 10-parameter
    script, generalized to a 6-bond chain."""
    eq, compressa, dissociata = R0_MOL, 1.30, 4.30
    return {
        "uniforme_equilibrio":          np.full(N_BONDS, eq),
        "uniforme_compressa":           np.full(N_BONDS, compressa),
        "uniforme_dissociata":          np.full(N_BONDS, dissociata),
        "mutazione_localizzata":        np.array([eq, eq, eq, dissociata, eq, eq]),
        "distorsione_alternata":        np.array([compressa, dissociata, compressa, dissociata, compressa, dissociata]),
        "mutazioni_congiunte_estremi":  np.array([dissociata, eq, eq, eq, eq, dissociata]),
    }


def optimize_geometry_and_theta_jointly(R_init: np.ndarray, n_epochs: int = 400,
                                         lr_theta: float = 0.15, lr_R: float = 0.04,
                                         R_bounds=(0.6, 8.0), verbose: bool = False):
    """Jointly optimizes BOTH the per-bond theta vector AND the bond
    distances R_q -- not hand-picked geometries, a genuine search for the
    minimum-energy conformation. R's gradient is analytic and classical (no
    PSR needed, R only enters through t_q(R_q) and the per-bond repulsion):

        E = -sum_q (t_q(R_q)/2) * k_q(theta) + sum_q V0*exp(-GAMMA*(R_q-R0))
        dE/dR_q = (BETA/2) * t_q(R_q) * k_q(theta) - GAMMA * V_rep_q(R_q)

    Uses _local_repulsion_per_bond, NOT _global_repulsion -- see the
    top-of-file note on why the mean-based term is unsuitable here.
    R_bounds are a numerical safety guardrail, not a constraint the minimum
    is expected to reach -- callers should verify R* doesn't sit at either
    bound (see tests/test_vqe_extreme_geometries_deep.py).

    R_init: shape (n_start, N_BONDS) -- different rows = different starting
    points, to check whether they converge to the same conformation or to
    distinct ones."""
    n_start = R_init.shape[0]
    R = R_init.copy().astype(np.float64)
    theta = np.full((n_start, N_BONDS), RIGID_THETA, dtype=np.float64)
    m_t, v_t = np.zeros_like(theta), np.zeros_like(theta)
    m_R, v_R = np.zeros_like(R), np.zeros_like(R)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        kinetic, jac = batched_per_bond_kinetic_and_jacobian(theta)
        t_local = _local_hopping(R)
        v_perbond = _local_repulsion_per_bond(R)

        grad_theta = -np.einsum('gq,gqr->gr', t_local / 2.0, jac)
        grad_R = (BETA / 2.0) * t_local * kinetic - GAMMA * v_perbond

        m_t = beta1 * m_t + (1 - beta1) * grad_theta
        v_t = beta2 * v_t + (1 - beta2) * (grad_theta ** 2)
        m_t_hat = m_t / (1 - beta1 ** epoch); v_t_hat = v_t / (1 - beta2 ** epoch)
        theta -= (lr_theta / (np.sqrt(v_t_hat) + eps)) * m_t_hat

        m_R = beta1 * m_R + (1 - beta1) * grad_R
        v_R = beta2 * v_R + (1 - beta2) * (grad_R ** 2)
        m_R_hat = m_R / (1 - beta1 ** epoch); v_R_hat = v_R / (1 - beta2 ** epoch)
        R -= (lr_R / (np.sqrt(v_R_hat) + eps)) * m_R_hat
        R = np.clip(R, R_bounds[0], R_bounds[1])

        if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == n_epochs):
            kinetic_now, _ = batched_per_bond_kinetic_and_jacobian(theta)
            e_now = -np.sum((_local_hopping(R) / 2.0) * kinetic_now, axis=1) + np.sum(_local_repulsion_per_bond(R), axis=1)
            print(f"Epoch {epoch:04d}/{n_epochs} | mean E: {e_now.mean():+.6f} eV "
                  f"| elapsed: {time.perf_counter()-t0:.2f}s")

    kinetic_f, jac_f = batched_per_bond_kinetic_and_jacobian(theta)
    t_local_f = _local_hopping(R)
    E_final = -np.sum((t_local_f / 2.0) * kinetic_f, axis=1) + np.sum(_local_repulsion_per_bond(R), axis=1)
    grad_theta_f = -np.einsum('gq,gqr->gr', t_local_f / 2.0, jac_f)
    grad_R_f = (BETA / 2.0) * t_local_f * kinetic_f - GAMMA * _local_repulsion_per_bond(R)
    return R, theta, E_final, grad_R_f, grad_theta_f


def build_conformational_starting_points() -> dict:
    """Distinct starting geometries for optimize_geometry_and_theta_jointly,
    to check whether the joint search converges to the same conformation
    regardless of starting point, or to distinct stable ones."""
    rng = np.random.default_rng(42)
    return {
        "uniforme_R0":       np.full(N_BONDS, R0_MOL),
        "perturbata_random": R0_MOL + rng.normal(0, 0.4, N_BONDS),
        "alternata":         np.array([1.6, 3.2, 1.6, 3.2, 1.6, 3.2]),
    }


def _run_full_sweep():
    scenari = build_extreme_geometries()
    nomi = list(scenari.keys())
    R_matrix = np.stack([scenari[n] for n in nomi])

    print("============================================================")
    print(f"MOLECULAR VQE: EXTREME/IRREGULAR GEOMETRY BENCHMARK ({N_PARAMS} PARAMETERS)")
    print("   Rigid shared theta=0.38 vs. per-bond adaptive theta*(R_q)")
    print("============================================================")

    theta_rigid = np.full((len(nomi), N_BONDS), RIGID_THETA)
    E_rigid = energy_from_theta(R_matrix, theta_rigid)

    t_global_start = time.perf_counter()
    theta_star, E_star = optimize_theta_per_geometry(R_matrix, n_epochs=120, verbose=False)
    theta_shared_star, kinetic_shared = optimize_shared_theta_per_geometry(R_matrix, n_epochs=120)
    tempo_totale = time.perf_counter() - t_global_start

    kinetic_perbond, _ = batched_per_bond_kinetic_and_jacobian(theta_star)
    deficit = deficit_fraction(R_matrix, kinetic_perbond, kinetic_shared)

    miglioramento = E_rigid - E_star
    spread = theta_star.max(axis=1) - theta_star.min(axis=1)

    df = pd.DataFrame({
        "Scenario": nomi,
        **{f"R_Legame_{q}": R_matrix[:, q] for q in range(N_BONDS)},
        "Energia_Rigida_eV": E_rigid,
        "Energia_Adattiva_eV": E_star,
        "Miglioramento_eV": miglioramento,
        "Theta_Shared_Ottimale": theta_shared_star,
        "Deficit_Frazionario_vs_Shared_Ottimale": deficit,
        "Spread_Theta_Adattivo": spread,
        **{f"Theta_Adattivo_Legame_{q}": theta_star[:, q] for q in range(N_BONDS)},
    })
    df.to_csv(_DATA_DIR / "vqe_extreme_geometries_deep.csv", index=False)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(nomi))
    ax1.bar(x - 0.2, E_rigid, width=0.4, color='#888888', label='Angolo rigido (theta=0.38)')
    ax1.bar(x + 0.2, E_star, width=0.4, color='#FF007F', label=f'Per-legame adattivo ({N_PARAMS} parametri)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomi, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel("Energia Totale (eV)", color='#888888')
    ax1.set_title(f"Rigido vs. Adattivo ({N_PARAMS} parametri) per Scenario", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax1.legend(loc="best", fontsize=8)

    colors = ['#FF007F', '#FFFF00', '#00FFFF', '#00FF00', '#FF8800', '#AA88FF']
    for i, nome in enumerate(nomi):
        ax2.plot(range(N_BONDS), theta_star[i], marker='o', color=colors[i % len(colors)], label=nome)
    ax2.axhline(RIGID_THETA, color='#888888', linestyle=':', label='Theta rigido (0.38)')
    ax2.set_xlabel("Indice Legame")
    ax2.set_ylabel("Theta Adattivo (rad)", color='#888888')
    ax2.set_title(f"Profilo Angolare Adattivo ({N_BONDS} legami)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "vqe_extreme_geometries_deep.png", dpi=300)

    print("============================================================")
    print(f"BENCHMARK COMPLETATO IN {tempo_totale:.2f} s")
    for i, nome in enumerate(nomi):
        print(f"   {nome:28s} | E_rigida={E_rigid[i]:+.4f} eV | E_adattiva={E_star[i]:+.4f} eV "
              f"| deficit_frazionario={deficit[i]:.4f} | spread_theta={spread[i]:.4f} rad")
    idx_worst = int(np.argmax(deficit))
    print(f"   -> scenario con maggior deficit frazionario: {nomi[idx_worst]} ({deficit[idx_worst]:.4f})")
    print("============================================================")


def _run_conformational_search():
    partenze = build_conformational_starting_points()
    nomi = list(partenze.keys())
    R_init = np.stack([partenze[n] for n in nomi])

    print("============================================================")
    print(f"RICERCA CONFORMAZIONALE: minimo congiunto (R_q*, theta_q*), {N_BONDS} legami")
    print("============================================================")

    t0 = time.perf_counter()
    R_star, theta_star, E_star, grad_R, grad_theta = optimize_geometry_and_theta_jointly(
        R_init, n_epochs=600, verbose=True)
    print(f"\nCompletato in {time.perf_counter()-t0:.2f}s\n")

    for i, nome in enumerate(nomi):
        print(f"Partenza '{nome}':")
        print(f"   E_min = {E_star[i]:+.6f} eV")
        print(f"   R*    = {np.round(R_star[i], 4)}")
        print(f"   theta*= {np.round(theta_star[i], 4)}")
        print(f"   max|grad_R|={np.max(np.abs(grad_R[i])):.2e}, max|grad_theta|={np.max(np.abs(grad_theta[i])):.2e}")

    df = pd.DataFrame({
        "Partenza": nomi,
        "E_min_eV": E_star,
        **{f"R_star_Legame_{q}": R_star[:, q] for q in range(N_BONDS)},
        **{f"Theta_star_Legame_{q}": theta_star[:, q] for q in range(N_BONDS)},
    })
    df.to_csv(_DATA_DIR / "vqe_extreme_geometries_deep_conformazioni.csv", index=False)
    print("\nCSV salvato: data/vqe_extreme_geometries_deep_conformazioni.csv")
    print("============================================================")


if __name__ == "__main__":
    _run_full_sweep()
    print()
    _run_conformational_search()
