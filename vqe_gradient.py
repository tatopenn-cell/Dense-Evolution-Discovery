import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
t_hopping = 2.11

def calcola_energia_vqe(theta):
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
    
    for q in range(N_Q):
        q_next = (q + 1) % N_Q
        mask = (1 << q) | (1 << q_next)
        psi_flipped = statevector[indices ^ mask]
        
        xx_exp = np.real(np.sum(np.conj(statevector) * psi_flipped))
        bit_i = (indices & (1 << q)) >> q
        bit_j = (indices & (1 << q_next)) >> q_next
        phase = np.where(bit_i == bit_j, -1.0, 1.0)
        yy_exp = np.real(np.sum(np.conj(statevector) * psi_flipped * phase))
        
        total_kinetic += float(xx_exp + yy_exp)
        
    return - (t_hopping / 2.0) * total_kinetic

def _run_full_sweep():
    punti_theta = np.linspace(0.0, 2 * np.pi, 3500)
    dati_gradiente = []
    h = 1e-5

    print("============================================================")
    print("🔬 COMPUTING EXACT ANALYTICAL VQE GRADIENT LANDSCAPE (3500 STEPS)")
    print("============================================================")

    t_global_start = time.perf_counter()

    for idx, theta in enumerate(punti_theta):
        E_plus = calcola_energia_vqe(theta + h)
        E_minus = calcola_energia_vqe(theta - h)
        gradiente_reale = (E_plus - E_minus) / (2 * h)

        E_attuale = calcola_energia_vqe(theta)

        if (idx + 1) % 250 == 0 or idx == 0 or idx == len(punti_theta) - 1:
            print(f"Step {idx+1:04d}/3500 | Theta: {theta:.3f} rad | Energia: {E_attuale:+.4f} eV | Gradiente: {gradiente_reale:+.6f}")

        dati_gradiente.append({
            "Theta": theta,
            "Energia": E_attuale,
            "Gradiente": gradiente_reale
        })

    df = pd.DataFrame(dati_gradiente)
    df.to_csv("vqe_gradient_landscape.csv", index=False)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df["Theta"], df["Energia"], color='#00FFFF', linewidth=2.5, label='VQE Energy Surface E(θ)')
    ax1.set_ylabel("Energy (eV)", color='#888888')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax1.legend(loc="upper right")
    ax1.set_title("VQE Energy Landscape & Exact Numerical Gradients", fontsize=11, fontweight='bold', pad=15)

    ax2.plot(df["Theta"], df["Gradiente"], color='#FFFF00', linewidth=2, label='Exact Gradient (dE/dθ)')
    ax2.axhline(0.0, color='#888888', linestyle=':', alpha=0.5)
    ax2.set_xlabel("Variational Parameter θ (radians)", color='#888888')
    ax2.set_ylabel("Gradient Magnitude", color='#888888')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("vqe_gradient_landscape.png", dpi=300)

    tempo_totale = time.perf_counter() - t_global_start
    print("============================================================")
    print(f"✅ MAPPA DEI GRADIENTI COMPLETATA IN {tempo_totale:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    _run_full_sweep()

