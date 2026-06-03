import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 18
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

print("============================================================")
print("🔬 MOLECULAR VQE: QUANTUM PARAMETER-SHIFT OPTIMIZER")
print("============================================================")

distanze_R = np.linspace(1.4, 3.2, 15)
energie_fondamentali = []

def cost_function_vqe(theta_val, distanza_R):
    rotazioni = [['rx', q, float(theta_val)] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(rotazioni + entanglement)
    prob = sim.get_probabilities()
    
    repulsione_nucleare = 12.0 / (distanza_R ** 2)
    # INDEXED TO CAPTURE SCALARS AND PREVENT TYPEERROR
    attrazione_elettronica = - float(prob[0] + prob[-1]) * (7.5 / distanza_R)
    return repulsione_nucleare + attrazione_elettronica

SHIFT = np.pi / 2
learning_rate = 0.05
theta_iniziale = 0.5

for idx, R in enumerate(distanze_R):
    t_start = time.perf_counter()
    
    theta_current = theta_iniziale
    for step in range(10):
        E_plus = cost_function_vqe(theta_current + SHIFT, R)
        E_minus = cost_function_vqe(theta_current - SHIFT, R)
        grad_quantistico = 0.5 * (E_plus - E_minus)
        theta_current -= learning_rate * grad_quantistico
    
    E_minima_ottimizzata = cost_function_vqe(theta_current, R)
    latenza = time.perf_counter() - t_start
    print(f"R: {R:.2f} A | Theta: {theta_current:.4f} | Energy: {E_minima_ottimizzata:.6f} Ha | {latenza:.2f}s")
    
    energie_fondamentali.append({"Distanza_R": R, "Energia_Hartree": E_minima_ottimizzata})

df_chimica = pd.DataFrame(energie_fondamentali)
df_chimica.to_csv("vqe_molecola_silicio.csv", index=False)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_chimica["Distanza_R"], df_chimica["Energia_Hartree"], marker='o', linestyle='-', color='#FF007F', linewidth=2, label='VQE Ground State')

idx_minimo = df_chimica["Energia_Hartree"].idxmin()
R_equilibrio = df_chimica["Distanza_R"].iloc[idx_minimo]
E_legame = df_chimica["Energia_Hartree"].iloc[idx_minimo]
ax.plot(R_equilibrio, E_legame, marker='X', color='#00FFFF', markersize=12, label=f'Equilibrium: {R_equilibrio:.2f} A')

ax.set_title("VQE Molecular Chemistry: Crystalline Silicon Potential Energy Curve", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Interatomic Distance R (Angstrom)", color='#888888')
ax.set_ylabel("Total Energy (Hartree)", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("curva_potenziale_silicio.png", dpi=300)
print("\n✅ VQE Simulation Completed! Plot saved as: curva_potenziale_silicio.png")
