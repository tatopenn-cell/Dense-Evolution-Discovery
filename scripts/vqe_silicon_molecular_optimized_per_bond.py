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
# Molecular VQE with an INDEPENDENT Givens angle per bond (5 free parameters
# per R), instead of one angle shared by all 5 bonds
# (vqe_silicon_molecular_optimized.py).
#
# That earlier script found theta*(R) = argmax_theta K(theta) is R-independent
# BY CONSTRUCTION for a shared theta: R only enters through the scalar
# prefactor t(R), which doesn't move the location of the kinetic term's
# extremum. A genuinely R-dependent optimum needs the CIRCUIT itself to
# have R-independent structure but bond-differentiated freedom -- this is
# exactly what real hardware-efficient VQE ansatze do (each excitation
# amplitude optimized independently), so this script asks the more
# realistic and more honest question: does the true multi-parameter
# optimum stay uniform across bonds, or does structure emerge?
#
# Per-bond Givens rotation kept intact (gate_B_q = -gate_A_q, so particle-
# number conservation on the single-excitation subspace still holds) --
# only the SHARING across bonds is removed. Each bond's own angle gets an
# exact single-gate-pair Parameter-Shift Rule gradient (same grid/shift
# machinery as vqe_silicon_molecular_optimized.py, just no chain-rule sum
# across bonds -- each bond's gradient is independent of the others').
# ═══════════════════════════════════════════════════════════════════════════

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

T0_MOL, BETA, R0_MOL, V0_MOL, GAMMA = 2.11, 1.5, 2.35, 5.4, 3.0

N_BONDS = N_Q - 1                  # 5
N_PARAMS = 2 * N_BONDS              # 10
ROWS_PER_POINT = 1 + 4 * N_BONDS    # 21: 1 base + 2 shifts x 2 gates/bond


def _build_ops() -> list:
    ops = [['x', 0]]
    for q in range(N_BONDS):
        ops += [['cx', q + 1, q], ['ry', q + 1, 'A'], ['cx', q, q + 1], ['ry', q + 1, 'B'], ['cx', q + 1, q]]
    return ops


def _base_row(theta_per_bond: np.ndarray) -> np.ndarray:
    """theta_per_bond: shape (N_BONDS,) -- one independent Givens angle per bond."""
    row = np.empty(N_PARAMS, dtype=np.float64)
    row[0::2] = theta_per_bond
    row[1::2] = -theta_per_bond
    return row


def _kinetic_from_sv(sv: np.ndarray) -> float:
    dim = len(sv)
    idx = np.arange(dim)
    kinetic = 0.0
    for q in range(N_BONDS):
        mask = (1 << q) | (1 << (q + 1))
        pf = sv[idx ^ mask]
        xx = np.real(np.sum(np.conj(sv) * pf))
        bi = (idx & (1 << q)) >> q
        bj = (idx & (1 << (q + 1))) >> (q + 1)
        yy = np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
        kinetic += xx + yy
    return float(kinetic)


def batched_kinetic_and_exact_gradient(theta_matrix: np.ndarray):
    """theta_matrix: shape (n_R, N_BONDS). Returns (kinetic[n_R], gradient[n_R, N_BONDS])
    -- the exact PSR gradient of the kinetic term w.r.t. EACH bond's own angle,
    computed for every R in a single run_parametric_batch_jit call. Unlike the
    shared-theta version, no chain-rule summation across bonds: each bond's
    gradient depends only on shifting that bond's own 2 gates."""
    n_r = theta_matrix.shape[0]
    grid = np.zeros((n_r * ROWS_PER_POINT, N_PARAMS), dtype=np.float64)
    for i in range(n_r):
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

    kinetic = np.empty(n_r)
    gradient = np.zeros((n_r, N_BONDS))
    for i in range(n_r):
        off = i * ROWS_PER_POINT
        kinetic[i] = _kinetic_from_sv(batch[off])
        r = off + 1
        for k in range(N_PARAMS):
            e_plus = _kinetic_from_sv(batch[r])
            e_minus = _kinetic_from_sv(batch[r + 1])
            partial = 0.5 * (e_plus - e_minus)
            bond_idx = k // 2
            sign = 1.0 if (k % 2 == 0) else -1.0   # dtheta_A/dtheta_q=1, dtheta_B/dtheta_q=-1
            gradient[i, bond_idx] += partial * sign
            r += 2

    return kinetic, gradient


def optimize_pec_per_bond(R_space: np.ndarray, n_epochs: int = 60, lr: float = 0.15,
                           theta_init: float = 0.38, verbose: bool = True):
    """Adam-optimizes N_BONDS independent angles at every R, minimizing
    E(R, theta_vec) = -(t(R)/2) * kinetic(theta_vec) + V_rep(R)."""
    t_R = T0_MOL * np.exp(-BETA * (R_space - R0_MOL))
    V_rep = V0_MOL * np.exp(-GAMMA * (R_space - R0_MOL))

    theta = np.full((len(R_space), N_BONDS), theta_init, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        kinetic, d_kinetic_dtheta = batched_kinetic_and_exact_gradient(theta)
        grad = -(t_R[:, None] / 2.0) * d_kinetic_dtheta

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        theta -= (lr / (np.sqrt(v_hat) + eps)) * m_hat

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == n_epochs):
            e_now = -(t_R / 2.0) * kinetic + V_rep
            spread = theta.max(axis=1) - theta.min(axis=1)
            print(f"Epoch {epoch:03d}/{n_epochs} | mean|grad|: {np.mean(np.abs(grad)):.6f} "
                  f"| mean E: {e_now.mean():+.6f} eV | mean bond-spread: {spread.mean():.6f} rad "
                  f"| elapsed: {time.perf_counter()-t0:.2f}s")

    kinetic_final, grad_final = batched_kinetic_and_exact_gradient(theta)
    E_final = -(t_R / 2.0) * kinetic_final + V_rep
    return theta, E_final, grad_final


def _run_full_sweep():
    R_space = np.linspace(1.2, 4.5, 200)

    print("============================================================")
    print("🔬 MOLECULAR VQE: PER-BOND ADAM-OPTIMIZED PEC (5 INDEPENDENT ANGLES)")
    print("============================================================")

    t_global_start = time.perf_counter()
    theta_star, E_star, grad_final = optimize_pec_per_bond(R_space)
    tempo_totale = time.perf_counter() - t_global_start

    df = pd.DataFrame({
        "Distanza_R": R_space,
        "Energia_Ottimizzata_eV": E_star,
        **{f"Theta_Legame_{q}": theta_star[:, q] for q in range(N_BONDS)},
        **{f"Gradiente_Legame_{q}": grad_final[:, q] for q in range(N_BONDS)},
    })
    df.to_csv(_DATA_DIR / "vqe_molecola_silicio_ottimizzata_per_legame.csv", index=False)

    try:
        df_fixed = pd.read_csv(_DATA_DIR / "vqe_molecola_silicio.csv")
        has_fixed = True
    except FileNotFoundError:
        has_fixed = False
    try:
        df_shared = pd.read_csv(_DATA_DIR / "vqe_molecola_silicio_ottimizzata.csv")
        has_shared = True
    except FileNotFoundError:
        has_shared = False

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    if has_fixed:
        ax1.plot(df_fixed["Distanza_R"], df_fixed["Energia_Totale_eV"], color='#888888',
                  linestyle=':', linewidth=2, label='Fixed theta=0.38')
    if has_shared:
        ax1.plot(df_shared["Distanza_R"], df_shared["Energia_Ottimizzata_eV"], color='#00FFFF',
                  linewidth=1.5, linestyle='--', label='Shared theta*(R)')
    ax1.plot(df["Distanza_R"], df["Energia_Ottimizzata_eV"], color='#FF007F', linewidth=2.5,
              label='Per-bond theta*(R) (5 independent angles)')
    idx_m = df["Energia_Ottimizzata_eV"].idxmin()
    rm, em = df.loc[idx_m, "Distanza_R"], df.loc[idx_m, "Energia_Ottimizzata_eV"]
    ax1.scatter(rm, em, color='#FFFF00', s=50, zorder=5)
    ax1.annotate(f"Min: {em:.3f} eV @ {rm:.2f} Å", xy=(rm, em), xytext=(rm + 0.3, em + 0.5),
                  arrowprops=dict(arrowstyle="->", color='#FFFF00', lw=1), color='#FFFF00', fontsize=9)
    ax1.set_ylabel("Total Molecular Energy (eV)", color='#888888')
    ax1.set_title("Silicon Dimer PEC: Shared vs. Per-Bond Optimized Variational Angles", fontsize=11, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax1.legend(loc="upper right")

    colors = ['#FF007F', '#FFFF00', '#00FFFF', '#00FF00', '#FF8800']
    for q in range(N_BONDS):
        ax2.plot(df["Distanza_R"], df[f"Theta_Legame_{q}"], color=colors[q], linewidth=1.8, label=f'Legame {q}')
    ax2.axhline(0.38, color='#888888', linestyle=':', alpha=0.5, label='Fixed theta=0.38')
    ax2.set_xlabel("Interatomic Distance R (Å)", color='#888888')
    ax2.set_ylabel("Per-Bond Optimal Angle (rad)", color='#888888')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "curva_potenziale_silicio_ottimizzata_per_legame.png", dpi=300)

    theta_spread = theta_star.max(axis=1) - theta_star.min(axis=1)

    print("============================================================")
    print(f"✅ OTTIMIZZAZIONE COMPLETATA IN {tempo_totale:.2f} s")
    print(f"   Minimo: {em:.4f} eV @ R = {rm:.3f} Å")
    print(f"   Spread massimo tra legami: {theta_spread.max():.6f} rad (a R={R_space[np.argmax(theta_spread)]:.3f} Å)")
    print(f"   Spread medio tra legami: {theta_spread.mean():.6f} rad")
    if has_shared:
        improvement = df_shared["Energia_Ottimizzata_eV"].to_numpy() - df["Energia_Ottimizzata_eV"].to_numpy()
        print(f"   Miglioramento medio vs theta condiviso ottimizzato: {improvement.mean():.6f} eV "
              f"(max {improvement.max():.6f} eV)")
    print("============================================================")


if __name__ == "__main__":
    _run_full_sweep()
