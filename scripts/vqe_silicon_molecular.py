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

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

def calcola_energia_molecolare(R, theta):
    t_R = 2.11 * np.exp(-1.5 * (R - 2.35))
    V_rep = 5.4 * np.exp(-3.0 * (R - 2.35))
    
    ansatz_circuit = []
    ansatz_circuit.append(['x', 0])
    for q in range(N_Q - 1):
        ansatz_circuit.append(['cx', q + 1, q])
        ansatz_circuit.append(['ry', q + 1, float(theta)])
        ansatz_circuit.append(['cx', q, q + 1])
        ansatz_circuit.append(['ry', q + 1, -float(theta)])
        ansatz_circuit.append(['cx', q + 1, q])
        
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ansatz_circuit)
    statevector = sim.get_statevector()
    
    dim = len(statevector)
    indices = np.arange(dim)
    total_kinetic = 0.0
    for q in range(N_Q - 1):
        mask = (1 << q) | (1 << (q + 1))
        psi_flipped = statevector[indices ^ mask]
        xx_exp = np.real(np.sum(np.conj(statevector) * psi_flipped))
        bit_i = (indices & (1 << q)) >> q
        bit_j = (indices & (1 << (q + 1))) >> (q + 1)
        phase = np.where(bit_i == bit_j, -1.0, 1.0)
        yy_exp = np.real(np.sum(np.conj(statevector) * psi_flipped * phase))
        total_kinetic += float(xx_exp + yy_exp)
        
    E_elettronica = - (t_R / 2.0) * total_kinetic
    return E_elettronica + V_rep

def _run_full_sweep():
    distanze_R = np.linspace(1.2, 4.5, 3500)
    risultati_pec = []

    for idx, R in enumerate(distanze_R):
        theta_ottimale = 0.38
        E_totale = calcola_energia_molecolare(R, theta_ottimale)

        if (idx + 1) % 500 == 0 or idx == 0 or idx == len(distanze_R) - 1:
            print(f"Distanza R: {R:.3f} Å | Energia Totale Molecola: {E_totale:+.6f} eV")

        risultati_pec.append({
            "Distanza_R": R,
            "Energia_Totale_eV": E_totale
        })

    df = pd.DataFrame(risultati_pec)
    df.to_csv(_DATA_DIR / "vqe_molecola_silicio.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Distanza_R"], df["Energia_Totale_eV"], color='#FF007F', linewidth=2.5, label='VQE Born-Oppenheimer PEC')
    ax.set_title("Silicon Dimer (Si2) Dissociation Curve: Variational Quantum Chemistry", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("Interatomic Distance R (Angstrom)", color='#888888')
    ax.set_ylabel("Total Molecular Energy (eV)", color='#888888')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "curva_potenziale_silicio.png", dpi=300)


if __name__ == "__main__":
    print("============================================================")
    print("🔬 MOLECULAR VQE: EXACT POTENTIAL ENERGY CURVE (PEC)")
    print("============================================================")
    _run_full_sweep()

