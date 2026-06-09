import time, jax, numpy as np, pandas as pd, matplotlib.pyplot as plt, dense_evolution as de
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def test_vqe_silicon_complete_simulation():
    N_Q, THETA = 6, 0.38
    R_SPACE = np.linspace(1.2, 4.5, 3500)
    sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
    DIM = 1 << N_Q
    IDX = np.arange(DIM)
    MSK = [(1 << q) | (1 << (q + 1)) for q in range(N_Q - 1)]
    PHS = [np.where(((IDX & (1 << q)) >> q) == ((IDX & (1 << (q + 1))) >> (q + 1)), -1.0, 1.0) for q in range(N_Q - 1)]
    
    def _vqe(R):
        tR, vR = 2.11 * np.exp(-1.5 * (R - 2.35)), 5.4 * np.exp(-3.0 * (R - 2.35))
        c = [['x', 0]]
        for q in range(N_Q - 1):
            c.extend([['cx', q + 1, q], ['ry', q + 1, float(THETA)], ['cx', q, q + 1], ['ry', q + 1, -float(THETA)], ['cx', q + 1, q]])
        sim.set_initial_state()
        sim.run_circuit_jit_beast_mode(c)
        sv = sim.get_statevector()
        csv = np.conj(sv)
        tk = sum(float(np.real(np.sum(csv * sv[IDX ^ m])) + np.real(np.sum(csv * sv[IDX ^ m] * p))) for m, p in zip(MSK, PHS))
        return (- (tR / 2.0) * tk) + vR

    t0 = time.perf_counter()
    pec = [{"Distanza_R": r, "Energia_Totale_eV": _vqe(r)} for r in R_SPACE]
    df = pd.DataFrame(pec)
    df.to_csv("vqe_molecola_silicio.csv", index=False)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(df["Distanza_R"], df["Energia_Totale_eV"], color='#FF007F', linewidth=2.5, label='VQE PEC')
    idx_m = df["Energia_Totale_eV"].idxmin()
    rm, em = df.loc[idx_m, "Distanza_R"], df.loc[idx_m, "Energia_Totale_eV"]
    ax.scatter(rm, em, color='#00FFFF', s=50, zorder=5)
    ax.annotate(f"Min: {em:.3f} eV @ {rm:.2f} A", xy=(rm, em), xytext=(rm + 0.3, em + 0.5), arrowprops=dict(arrowstyle="->", color='#00FFFF', lw=1), color='#00FFFF', fontsize=9)
    plt.tight_layout()
    plt.savefig("curva_potenziale_silicio.png", dpi=300)
    plt.close()
    
    assert len(df) == 3500 and not df["Energia_Totale_eV"].isnull().any() and em < 0
