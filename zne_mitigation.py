import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 24
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

punti_g = np.linspace(0.0, 2.5, 10)
p_base = 0.04
risultati = []

print("============================================================")
print(f"🔬 RUNNING ERROR MITIGATION: RICHARDSON ZNE SCAN")
print("============================================================")

for idx, g in enumerate(punti_g):
    rotazioni = [['rx', q, float(0.1 * g)] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    circuito = rotazioni + entanglement
    
    # 1. Target Ideale (Nessun rumore aggiunto)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(circuito)
    p_ideal = sim.get_probabilities()
    E_ideal = -float(p_ideal[0] + p_ideal[-1])
    
    # 2. Scala di Rumore Base (lambda = 1.0)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(circuito)
    key1 = jax.random.PRNGKey(42 + idx)
    sim.sv = de.NoiseModel.apply_to_sv(sv=sim.get_statevector(), n=N_Q, model='amplitude_damping', p=p_base, jax_key=key1)
    p_l1 = sim.get_probabilities()
    E_l1 = -float(p_l1[0] + p_l1[-1])
    
    # 3. Scala di Rumore Doppia (lambda = 2.0)
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(circuito)
    key2 = jax.random.PRNGKey(10042 + idx)
    sim.sv = de.NoiseModel.apply_to_sv(sv=sim.get_statevector(), n=N_Q, model='amplitude_damping', p=p_base * 2.0, jax_key=key2)
    p_l2 = sim.get_probabilities()
    E_l2 = -float(p_l2[0] + p_l2[-1])
    
    # --- ESTRAPOLAZIONE LINEARE DI RICHARDSON (ZNE) ---
    # Formula esatta per due punti equidistanti: E(0) = 2*E(1) - E(2)
    E_mitigated = 2.0 * E_l1 - E_l2
    
    print(f"g: {g:.2f} | Ideale: {E_ideal:.4f} | Noisy: {E_l1:.4f} | Mitigato: {E_mitigated:.4f}")
    risultati.append({"Campo_g": g, "Ideale": E_ideal, "Noisy": E_l1, "Mitigato": E_mitigated})

# Generazione del grafico comparativo a tre vie
df = pd.DataFrame(risultati)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Campo_g"], df["Ideale"], marker='o', color='#00FFFF', linewidth=2, label='Ideale (Target)')
ax.plot(df["Campo_g"], df["Noisy"], marker='x', linestyle='--', color='#FF007F', linewidth=1.5, label='Noisy (Grezzo lambda=1.0)')
ax.plot(df["Campo_g"], df["Mitigato"], marker='^', linestyle=':', color='#00FF00', linewidth=2, label='Mitigato (ZNE Richardson)')

ax.set_title("Protocollo di Eradicazione Errore ZNE su Modello di Ising (24 Qubit)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Intensità del Campo Trasversale (g)", color='#888888')
ax.set_ylabel("Correlazione Ferromagnetica Z-Z", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("transizione_ising_mitigata.png", dpi=300)
print("\n✅ Calcolo ZNE completato con successo! Grafico salvato in: transizione_ising_mitigata.png")
