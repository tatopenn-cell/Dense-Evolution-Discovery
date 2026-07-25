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
# Extreme / irregular molecular-geometry benchmark.
#
# Every other script in this repo sweeps a single, UNIFORM interatomic
# distance R across the whole open chain (vqe_silicon_molecular.py,
# ..._optimized.py, ..._optimized_per_bond.py). That's the easy regime: a
# rigid, shared variational angle theta=0.38 (vqe_silicon_molecular.py's
# fixed choice -- the kind of rigid approximation a commercial solver bakes
# in) only ever has to fit ONE hopping strength t(R).
#
# Here each of the N_BONDS=5 bonds gets its OWN interatomic distance R_q --
# modeling an irregular / distorted chain (extreme compression, near-
# dissociation stretch, a single localized "mutated" bond, or several at
# once). This is a strictly local generalization: hopping t_q(R_q) is
# per-bond (nearest-neighbor overlap only depends on that bond's own
# distance), while the steric/electrostatic repulsion term keeps the exact
# SAME single-formula shape as the uniform scripts, evaluated at the
# geometry's mean bond length -- so this benchmark's energy_from_theta
# collapses EXACTLY onto vqe_silicon_molecular.calcola_energia_molecolare
# in the uniform-R, uniform-theta limit (checked in the test suite).
#
# NOTE on scope: this is a tight-binding / single-excitation hopping toy
# model, not ab-initio electronic-structure. "Rigid angle fails" here means
# concretely: a single shared theta cannot simultaneously satisfy N_BONDS
# different per-bond stationarity conditions once the R_q differ, so it
# leaves variational energy on the table that a per-bond-optimized theta
# vector recovers. That is the property under test -- nothing here claims
# to model real electron correlation or nuclear QM repulsion.
#
# CAUTION -- naive comparison against the fixed theta=0.38 baseline is
# scale-confounded: any uniform-R geometry (equilibrium, compressed,
# dissociated) already shows a LARGE energy "improvement" from per-bond
# optimization, purely because the per-bond ansatz can reach the true
# sine-mode kinetic maximum (see vqe_silicon_molecular_optimized_per_bond.py)
# while a single shared angle structurally cannot -- regardless of geometry
# irregularity. That gap is constant in RELATIVE terms across all uniform-R
# scenarios (verified: identical deficit_fraction, see below) and must not
# be attributed to "extreme conditions."
#
# The metric that actually isolates the effect of geometry IRREGULARITY is
# deficit_fraction = (K_perbond - K_shared_optimal) / K_perbond, comparing
# per-bond adaptive kinetic against the BEST achievable SINGLE shared theta
# (re-optimized per geometry, not the hardcoded 0.38) -- this cancels the
# overall t(R) energy scale. Measured on build_extreme_geometries():
# uniform scenarios all give the SAME deficit_fraction ~0.169 (pure
# ansatz-expressivity cost, independent of R). Irregular-weight scenarios
# do NOT uniformly exceed that baseline: mutazione_localizzata (~0.10) and
# distorsione_alternata (~0.03) actually sit BELOW it, while
# mutazioni_congiunte_estremi (~0.50) sits far ABOVE it. The rigid single
# angle is not simply "worse under any distortion" -- it specifically
# struggles when the geometry forces it to reconcile two strongly-weighted
# but topologically distant bonds (opposite ends of the chain) at once,
# something a single scalar parameter cannot do but per-bond adaptation can.
# ═══════════════════════════════════════════════════════════════════════════

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

T0_MOL, BETA, R0_MOL, V0_MOL, GAMMA = 2.11, 1.5, 2.35, 5.4, 3.0

N_BONDS = N_Q - 1                  # 5
N_PARAMS = 2 * N_BONDS              # 10
ROWS_PER_POINT = 1 + 4 * N_BONDS    # 21: 1 base + 2 shifts x 2 gates/bond

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
    to theta_r, regardless of which qubits that observable acts on -- same
    principle used for the scalar gradient in
    vqe_silicon_molecular_optimized_per_bond.py, just keeping the
    per-bond breakdown instead of summing over q immediately."""
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
    """Per-bond hopping strength t_q(R_q) -- local, bond-by-bond (nearest-
    neighbor overlap depends only on that bond's own distance)."""
    return T0_MOL * np.exp(-BETA * (R_matrix - R0_MOL))


def _global_repulsion(R_matrix: np.ndarray) -> np.ndarray:
    """Global steric/electrostatic repulsion -- the SAME single-term
    formula as vqe_silicon_molecular.py's V_rep(R), evaluated at the
    geometry's mean bond length, so it reduces EXACTLY to the original
    scalar model when every bond shares one R."""
    return V0_MOL * np.exp(-GAMMA * (R_matrix.mean(axis=1) - R0_MOL))


def energy_from_theta(R_matrix: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
    """R_matrix, theta_matrix: shape (n_geo, N_BONDS). Returns total energy
    per geometry: E = -sum_q (t_q(R_q)/2) * k_q(theta) + V_rep(mean(R))."""
    kinetic, _ = batched_per_bond_kinetic_and_jacobian(theta_matrix)
    t_local = _local_hopping(R_matrix)
    return -np.sum((t_local / 2.0) * kinetic, axis=1) + _global_repulsion(R_matrix)


def optimize_theta_per_geometry(R_matrix: np.ndarray, n_epochs: int = 80, lr: float = 0.15,
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
        # dE/dtheta_r = -sum_q (t_local_q/2) * d(k_q)/dtheta_r
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
    """Adam-optimizes ONE shared scalar theta per geometry (same value used
    at every bond, like vqe_silicon_molecular_optimized.py's optimize_pec,
    generalized to per-bond-weighted energy) -- the best a rigid,
    single-parameter solver could EVER do for that geometry, as opposed to
    the arbitrary hardcoded RIGID_THETA. Returns (theta_scalar[n_geo],
    kinetic[n_geo, N_BONDS]) so callers can build whatever weighted energy
    or kinetic comparison they need.

    Chain rule: since every bond shares the same scalar, d(E)/d(theta_shared)
    = sum_r d(E)/d(theta_r), i.e. just the row-sum of the per-bond gradient
    already computed by batched_per_bond_kinetic_and_jacobian."""
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
    optimization: (K_perbond - K_shared) / K_perbond, using each geometry's
    own local weights t_q(R_q) -- cancels the overall t(R) energy scale so
    scenarios with very different R magnitudes are directly comparable."""
    t_local = _local_hopping(R_matrix)
    k_pb = np.sum(t_local * kinetic_perbond, axis=1)
    k_sh = np.sum(t_local * kinetic_shared, axis=1)
    return (k_pb - k_sh) / k_pb


def build_extreme_geometries() -> dict:
    """Hand-picked bond-length patterns (one distance R_q per bond, in
    Angstrom) that stress-test the rigid single-angle approximation the
    way a real chain would under extreme conditions or a localized
    structural mutation -- not the smooth, uniform-R sweep every other
    script in this repo uses."""
    eq, compressa, dissociata = R0_MOL, 1.30, 4.30
    return {
        "uniforme_equilibrio":          np.full(N_BONDS, eq),
        "uniforme_compressa":           np.full(N_BONDS, compressa),
        "uniforme_dissociata":          np.full(N_BONDS, dissociata),
        "mutazione_localizzata":        np.array([eq, eq, dissociata, eq, eq]),
        "distorsione_alternata":        np.array([compressa, dissociata, compressa, dissociata, compressa]),
        "mutazioni_congiunte_estremi":  np.array([dissociata, eq, eq, eq, dissociata]),
    }


def _run_full_sweep():
    scenari = build_extreme_geometries()
    nomi = list(scenari.keys())
    R_matrix = np.stack([scenari[n] for n in nomi])

    print("============================================================")
    print("🔬 MOLECULAR VQE: EXTREME/IRREGULAR GEOMETRY BENCHMARK")
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
    df.to_csv(_DATA_DIR / "vqe_extreme_geometries.csv", index=False)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(nomi))
    ax1.bar(x - 0.2, E_rigid, width=0.4, color='#888888', label='Angolo rigido (theta=0.38)')
    ax1.bar(x + 0.2, E_star, width=0.4, color='#FF007F', label='Per-bond adattivo theta*(R_q)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomi, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel("Energia Totale (eV)", color='#888888')
    ax1.set_title("Rigido vs. Adattivo per Scenario Geometrico", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax1.legend(loc="best", fontsize=8)

    colors = ['#FF007F', '#FFFF00', '#00FFFF', '#00FF00', '#FF8800', '#AA88FF']
    for i, nome in enumerate(nomi):
        ax2.plot(range(N_BONDS), theta_star[i], marker='o', color=colors[i % len(colors)], label=nome)
    ax2.axhline(RIGID_THETA, color='#888888', linestyle=':', label='Theta rigido (0.38)')
    ax2.set_xlabel("Indice Legame")
    ax2.set_ylabel("Theta Adattivo (rad)", color='#888888')
    ax2.set_title("Profilo Angolare Adattivo per Legame", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "vqe_extreme_geometries.png", dpi=300)

    print("============================================================")
    print(f"✅ BENCHMARK COMPLETATO IN {tempo_totale:.2f} s")
    for i, nome in enumerate(nomi):
        print(f"   {nome:28s} | E_rigida={E_rigid[i]:+.4f} eV | E_adattiva={E_star[i]:+.4f} eV "
              f"| miglioramento={miglioramento[i]:+.4f} eV | deficit_frazionario={deficit[i]:.4f} "
              f"| spread_theta={spread[i]:.4f} rad")
    idx_worst = int(np.argmax(deficit))
    print(f"   -> scenario con maggior deficit frazionario (angolo condiviso piu' penalizzato): "
          f"{nomi[idx_worst]} ({deficit[idx_worst]:.4f})")
    print("============================================================")


if __name__ == "__main__":
    _run_full_sweep()
