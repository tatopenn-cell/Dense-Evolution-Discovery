import time
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 24
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

# Griglia di angoli theta da 0 a due pi greco
angoli_theta = np.linspace(0.0, 2 * np.pi, 10)
epsilon = 1e-4  # Spostamento infinitesimo per il calcolo del gradiente reale
risultati = []

print("============================================================")
print(f"🔬 VQE GRADIENT TRACKING & BARREN PLATEAU DETECTOR (24 QUBIT)")
print("============================================================")

def calcola_energia(theta_val):
    # Circuito parametrico: Rotazione RX dipendente da theta seguita da CNOT
    rotazioni = [['rx', q, float(theta_val)] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(rotazioni + entanglement)
    prob = sim.get_probabilities()
    return -float(prob[0] + prob[-1])

for idx, theta in enumerate(angoli_theta):
    t_start = time.perf_counter()
    
    # 1. Energia al punto corrente
    E_attuale = calcola_energia(theta)
    
    # 2. Energia al punto spostato (per ricavare il gradiente dE/dtheta)
    E_spostata = calcola_energia(theta + epsilon)
    
    # Calcolo numerico reale del gradiente puro
    gradiente = (E_spostata - E_attuale) / epsilon
    
    latenza = time.perf_counter() - t_start
    print(f"Angolo {idx+1:02d}/10 | Theta: {theta:.2f} | Gradiente: {gradiente:.6f} | Tempo: {latenza:.2f}s")
    risultati.append({"Theta": theta, "Gradiente": gradiente})

# Generazione del grafico del paesaggio del gradiente
df = pd.DataFrame(risultati)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Theta"], df["Gradiente"], marker='s', linestyle='-', color='#00FF00', linewidth=2, label=r'Gradiente reale $\nabla_\theta \langle H_{zz} \rangle$')
ax.axhline(0, color='#FF007F', linestyle='--', alpha=0.5, label='Soglia di Barren Plateau (Gradiente = 0)')

ax.set_title("Analisi di Ottimizzabilità VQE: Paesaggio dei Gradienti (24 Qubit)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Parametro Angolare Variatizionale (Theta)", color='#888888')
ax.set_ylabel("Magnitudo del Gradiente", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("vqe_gradient_landscape.png", dpi=300)
print("\n✅ Analisi del gradiente completata! Mappa salvata in: vqe_gradient_landscape.png")
