import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

# Precisione doppia float64 nativa
jax.config.update("jax_enable_x64", True)

N_Q = 18
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

print("============================================================")
print(" NEXT-GEN SILICON DESIGNER: HIGH-RESOLUTION SWEEP (100 PTS)")
print("============================================================")
print("ðŸš€ Ingegnerizzazione fine delle bande quantistiche in corso...")

#  100 Point 
punti_k = np.linspace(0.0, 2.0, 100)
risultati_nuovo_silicio = []

def circuito_reticolo_modificato(k_vector, straining_factor):
    angolo_ottimizzato = float(k_vector * np.pi / 4 * (1.0 - straining_factor))
    rotazioni = [['rx', q, angolo_ottimizzato] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(rotazioni + entanglement)
    return sim.get_probabilities()

STRAIN = 0.05

for idx, k in enumerate(punti_k):
    t_start = time.perf_counter()
    
    prob = circuito_reticolo_modificato(k, STRAIN)
    E_nuovo_silicio = -float(prob[0] + prob[-1]) * 1.12 * (1.0 - STRAIN)

    
    latenza = time.perf_counter() - t_start
    # Stampiamo solo un punto ogni 10 per non intasare il terminale
    if idx % 10 == 0 or idx == 99:
        print(f"K-Point {idx+1:03d}/100 | k: {k:.2f} | Nuova Energia: {E_nuovo_silicio:.6f} eV | {latenza:.2f}s")
    
    risultati_nuovo_silicio.append({
        "Wavevector_k": k,
        "Energy_eV_Strained": E_nuovo_silicio
    })

try:
    df_vecchio = pd.read_csv("bande_silicio_ibrido.csv")
    ha_vecchio = True
except:
    ha_vecchio = False

df_nuovo = pd.DataFrame(risultati_nuovo_silicio)
df_nuovo.to_csv("bande_nuovo_silicio.csv", index=False)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

if ha_vecchio:
    # Campioniamo i punti del vecchio file per il confronto grafico
    ax.plot(df_vecchio["Wavevector_k"], df_vecchio["Energy_eV"], linestyle='--', color='#888888', alpha=0.7, label='Silicio Classico Standard (1.12 eV)')

ax.plot(df_nuovo["Wavevector_k"], df_nuovo["Energy_eV_Strained"], color='#FFFF00', linewidth=2.5, label='Nuovo Silicio Ingegnerizzato (Strained - 100 Punti)')
ax.fill_between(df_nuovo["Wavevector_k"], df_nuovo["Energy_eV_Strained"], color='#FFFF00', alpha=0.1)

ax.set_title("Ingegnerizzazione dello Stato Solido ad Alta Risoluzione", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Momento dell'Elettrone / Spazio di Brillouin (k)", color='#888888')
ax.set_ylabel("Energia Modale (eV)", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("confronto_nuovo_silicio.png", dpi=300)
print("============================================================")
print(" SCANSIONE AD ALTA RISOLUZIONE COMPLETATA CON SUCCESSO!")
print(" Grafico continuo salvato in: confronto_nuovo_silicio.png")
print("============================================================")

