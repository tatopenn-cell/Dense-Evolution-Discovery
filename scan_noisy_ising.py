import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 24
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

punti_g = np.linspace(0.0, 2.5, 10)
risultati = []

print("============================================================")
print(f"🔬 RUNNING DOUBLE SCAN: IDEAL VS NOISY ISING MODEL")
print("============================================================")

for idx, g in enumerate(punti_g):
    rotazioni = [['rx', q, float(0.1 * g)] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    circuito = rotazioni + entanglement
    
    # 1. Canale Ideale
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(circuito)
    p_ideal = sim.get_probabilities()
    E_zz_ideal = -float(p_ideal[0] + p_ideal[-1])
    
    # 2. Canale Noisy (Iniezione Amplitude Damping p=0.04 a fine circuito)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(circuito)
    key = jax.random.PRNGKey(42 + idx)
    sim.sv = de.NoiseModel.apply_to_sv(sv=sim.get_statevector(), n=N_Q, model='amplitude_damping', p=0.04, jax_key=key)
    p_noisy = sim.get_probabilities()
    E_zz_noisy = -float(p_noisy[0] + p_noisy[-1])
    
    print(f"Punto {idx+1:02d}/10 | Campo g: {g:.2f} | Ideale: {E_zz_ideal:.4f} | Noisy: {E_zz_noisy:.4f}")
    risultati.append({"Campo_g": g, "Ideale": E_zz_ideal, "Noisy": E_zz_noisy})

# Generazione del grafico comparativo finale ad alta risoluzione
df = pd.DataFrame(risultati)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Campo_g"], df["Ideale"], marker='o', color='#00FFFF', linewidth=2, label=r'$\langle H_{zz} \rangle$ Ideale (Coerente)')
ax.plot(df["Campo_g"], df["Noisy"], marker='x', linestyle='--', color='#FF007F', linewidth=2, label=r'$\langle H_{zz} \rangle$ Noisy (Thermal Relaxation)')

ax.set_title("Distorsione della Transizione di Fase Quantistica sotto Rumore Termico NISQ (24 Qubit)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Intensità del Campo Trasversale (g)", color='#888888')
ax.set_ylabel("Correlazione Ferromagnetica Z-Z", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("confronto_transizione_noisy.png", dpi=300)
print("\n✅ Doppio screening completato! Grafico salvato in: confronto_transizione_noisy.png")
