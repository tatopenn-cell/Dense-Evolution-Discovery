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
# Molecular VQE with a genuinely per-R optimized variational angle.
#
# vqe_silicon_molecular.py evaluates the PEC at a single hardcoded
# theta = 0.38 rad for every R -- a reasonable approximation, but not the
# true variational minimum, which generally shifts with the bond distance
# R (different R means a different hopping strength t(R), a different
# physical regime). This script instead runs a real Adam optimization at
# EVERY R, using the exact chain-rule Parameter-Shift Rule gradient
# (same machinery verified in vqe_jax_grad.py / tests/test_vqe_jax_gradient.py,
# agreement with finite differences to ~1e-9) to find theta*(R).
#
# All R points are optimized in parallel: each Adam step batches every
# R's current theta into ONE run_parametric_batch_jit call (shape
# (n_R * (1+4*N_BONDS), N_PARAMS)), instead of a nested R-then-epoch loop.
# ═══════════════════════════════════════════════════════════════════════════

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_float32=False)

T0_MOL, BETA, R0_MOL, V0_MOL, GAMMA = 2.11, 1.5, 2.35, 5.4, 3.0

N_BONDS = N_Q - 1                  # 5 -- open chain, matches calcola_energia_molecolare
N_PARAMS = 2 * N_BONDS              # 10
B_COEFF = -1.0                       # theta_B = -theta_A (same ansatz as vqe_gradient.py/vqe_jax_grad.py)
ROWS_PER_POINT = 1 + 4 * N_BONDS    # 21: 1 base + 2 shifts x 2 gates/bond


def _build_ops() -> list:
    ops = [['x', 0]]
    for q in range(N_BONDS):
        ops += [['cx', q + 1, q], ['ry', q + 1, 'A'], ['cx', q, q + 1], ['ry', q + 1, 'B'], ['cx', q + 1, q]]
    return ops


def _base_row(theta: float) -> np.ndarray:
    row = np.empty(N_PARAMS, dtype=np.float64)
    row[0::2] = theta
    row[1::2] = B_COEFF * theta
    return row


def _kinetic_from_sv(sv: np.ndarray) -> float:
    """Open-chain <XX+YY> kinetic term, identical formula to
    vqe_silicon_molecular.calcola_energia_molecolare's inner loop."""
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


def batched_kinetic_and_exact_gradient(thetas: np.ndarray):
    """For a vector of per-R thetas, returns (kinetic[R], d(kinetic)/dtheta[R])
    -- the exact chain-rule Parameter-Shift Rule gradient of the theta-dependent
    kinetic term, computed for every R in a single run_parametric_batch_jit call."""
    n_r = len(thetas)
    grid = np.zeros((n_r * ROWS_PER_POINT, N_PARAMS), dtype=np.float64)
    for i, theta in enumerate(thetas):
        base = _base_row(theta)
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
    gradient = np.empty(n_r)
    for i in range(n_r):
        off = i * ROWS_PER_POINT
        kinetic[i] = _kinetic_from_sv(batch[off])
        grad = 0.0
        r = off + 1
        for k in range(N_PARAMS):
            e_plus = _kinetic_from_sv(batch[r])
            e_minus = _kinetic_from_sv(batch[r + 1])
            partial = 0.5 * (e_plus - e_minus)
            dtheta_dt = 1.0 if (k % 2 == 0) else B_COEFF
            grad += partial * dtheta_dt
            r += 2
        gradient[i] = grad

    return kinetic, gradient


def optimize_pec(R_space: np.ndarray, n_epochs: int = 60, lr: float = 0.15,
                  theta_init: float = 0.38, verbose: bool = True):
    """Adam-optimizes theta independently at every R in R_space, minimizing
    E(R, theta) = -(t(R)/2) * kinetic(theta) + V_rep(R). V_rep doesn't depend
    on theta, so it never enters the gradient -- only shifts the reported
    energy after optimization converges."""
    t_R = T0_MOL * np.exp(-BETA * (R_space - R0_MOL))
    V_rep = V0_MOL * np.exp(-GAMMA * (R_space - R0_MOL))

    theta = np.full(len(R_space), theta_init, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        kinetic, d_kinetic_dtheta = batched_kinetic_and_exact_gradient(theta)
        # dE/dtheta = -(t_R/2) * d(kinetic)/dtheta
        grad = -(t_R / 2.0) * d_kinetic_dtheta

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        theta -= (lr / (np.sqrt(v_hat) + eps)) * m_hat

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == n_epochs):
            e_now = -(t_R / 2.0) * kinetic + V_rep
            print(f"Epoch {epoch:03d}/{n_epochs} | mean|grad|: {np.mean(np.abs(grad)):.6f} "
                  f"| mean E: {e_now.mean():+.6f} eV | elapsed: {time.perf_counter()-t0:.2f}s")

    kinetic_final, grad_final = batched_kinetic_and_exact_gradient(theta)
    E_final = -(t_R / 2.0) * kinetic_final + V_rep
    return theta, E_final, grad_final


def _run_full_sweep():
    R_space = np.linspace(1.2, 4.5, 200)

    print("============================================================")
    print("🔬 MOLECULAR VQE: ADAM-OPTIMIZED PEC (EXACT CHAIN-RULE PSR)")
    print("============================================================")

    t_global_start = time.perf_counter()
    theta_star, E_star, grad_final = optimize_pec(R_space)
    tempo_totale = time.perf_counter() - t_global_start

    df = pd.DataFrame({
        "Distanza_R": R_space,
        "Theta_Ottimale": theta_star,
        "Energia_Ottimizzata_eV": E_star,
        "Gradiente_Finale": grad_final,
    })
    df.to_csv(_DATA_DIR / "vqe_molecola_silicio_ottimizzata.csv", index=False)

    try:
        df_fixed = pd.read_csv(_DATA_DIR / "vqe_molecola_silicio.csv")
        has_fixed = True
    except FileNotFoundError:
        has_fixed = False

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    if has_fixed:
        ax1.plot(df_fixed["Distanza_R"], df_fixed["Energia_Totale_eV"], color='#888888',
                  linestyle=':', linewidth=2, label='Fixed theta=0.38 (vqe_silicon_molecular.py)')
    ax1.plot(df["Distanza_R"], df["Energia_Ottimizzata_eV"], color='#00FFFF', linewidth=2.5,
              label='Adam-optimized theta*(R) (exact PSR gradient)')
    idx_m = df["Energia_Ottimizzata_eV"].idxmin()
    rm, em = df.loc[idx_m, "Distanza_R"], df.loc[idx_m, "Energia_Ottimizzata_eV"]
    ax1.scatter(rm, em, color='#FFFF00', s=50, zorder=5)
    ax1.annotate(f"Min: {em:.3f} eV @ {rm:.2f} Å", xy=(rm, em), xytext=(rm + 0.3, em + 0.5),
                  arrowprops=dict(arrowstyle="->", color='#FFFF00', lw=1), color='#FFFF00', fontsize=9)
    ax1.set_ylabel("Total Molecular Energy (eV)", color='#888888')
    ax1.set_title("Silicon Dimer PEC: Fixed vs. Adam-Optimized Variational Angle", fontsize=11, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax1.legend(loc="upper right")

    ax2.plot(df["Distanza_R"], df["Theta_Ottimale"], color='#FF007F', linewidth=2.5, label='theta*(R)')
    ax2.axhline(0.38, color='#888888', linestyle=':', alpha=0.6, label='Fixed theta=0.38')
    ax2.set_xlabel("Interatomic Distance R (Å)", color='#888888')
    ax2.set_ylabel("Optimal Variational Angle (rad)", color='#888888')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "curva_potenziale_silicio_ottimizzata.png", dpi=300)

    print("============================================================")
    print(f"✅ OTTIMIZZAZIONE COMPLETATA IN {tempo_totale:.2f} s")
    print(f"   Minimo: {em:.4f} eV @ R = {rm:.3f} Å")
    if has_fixed:
        common = np.interp(R_space, df_fixed["Distanza_R"], df_fixed["Energia_Totale_eV"])
        improvement = common - df["Energia_Ottimizzata_eV"].to_numpy()
        print(f"   Miglioramento medio vs theta fisso: {improvement.mean():.6f} eV "
              f"(max {improvement.max():.6f} eV)")
    print("============================================================")


if __name__ == "__main__":
    _run_full_sweep()
