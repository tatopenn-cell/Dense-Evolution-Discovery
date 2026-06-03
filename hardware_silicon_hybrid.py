import time
import psutil
import platform
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

# Forza la precisione assoluta a 64 bit
jax.config.update("jax_enable_x64", True)

print("============================================================")
print("ðŸ”¬ HYBRID SCANNER: REAL CPU SILICON TO QUANTUM BANDSTRUCTURE")
print("============================================================")

# 1. ESTRAZIONE DATI IN DIRETTA DALLA TUA CPU FISICA
freq_hardware = psutil.cpu_freq()
cpu_load = psutil.cpu_percent(interval=0.5)

# Ricaviamo la frequenza corrente (es. 2.50 GHz) o usiamo un fallback stabile
current_ghz = freq_hardware.current / 1000.0 if freq_hardware else 2.50
if current_ghz == 0.0: current_ghz = 2.50

print(f"ðŸ“Š Live Hardware Frequency : {current_ghz:.4f} GHz")
print(f"ðŸ”¥ Live CPU Core Load      : {cpu_load}%")

# 2. CALIBRAZIONE DELLA FISICA DEL RETICOLO DEL SILICIO CRISTALLINO VERO
# Il Silicio ha un Bandgap reale di circa 1.12 eV. Usiamo il carico e la frequenza
# della tua CPU come perturbazione cinetica (fluttuazione termica del silicio reale).
N_Q = 18
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

# Mappiamo i punti nello spazio di Brillouin (Vettore d'onda K da 0 a 2)
punti_k = np.linspace(0.0, 2.0, 10)
risultati_ibridi = []

# Formula quantistica per simulare il salto energetico (Tight-Binding perturbato)
def circuito_reticolo_silicio(k_vector, rumore_cpu):
    # L'angolo di rotazione dei qubit simula il vettore d'onda degli elettroni nel reticolo
    angolo_k = float(k_vector * np.pi / 4)
    # Il rumore della CPU agisce come perturbazione di campo esterno g
    perturbazione_termica = float(rumore_cpu * 0.005)
    
    rotazioni = [['rx', q, angolo_k + perturbazione_termica] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(rotazioni + entanglement)
    return sim.get_probabilities()

print("\nâ³ Computazione quantistica delle bande energetiche del Silicio...")
for idx, k in enumerate(punti_k):
    t_start = time.perf_counter()
    
    # Calcolo con iniezione dinamica dei dati hardware reali del tuo PC
    prob = circuito_reticolo_silicio(k, cpu_load)
    
    # Ricaviamo l'energia di banda (Valore di aspettazione proiettato)
    E_banda = -float(prob[0] + prob[-1]) * 1.12  # Scalato sul vero Bandgap del silicio (1.12 eV)
    
    latenza = time.perf_counter() - t_start
    print(f"K-Space Point {idx+1:02d}/10 | Wavevector k: {k:.2f} | Energia Banda: {E_banda:.6f} eV | {latenza:.2f}s")
    
    risultati_ibridi.append({
        "Wavevector_k": k,
        "Energy_eV": E_banda,
        "CPU_Load_Perturbation": cpu_load
    })

# Esportazione del Dataset Ibrido
df = pd.DataFrame(risultati_ibridi)
df.to_csv("bande_silicio_ibrido.csv", index=False)

# Generazione del grafico ad alta risoluzione della struttura a bande
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Wavevector_k"], df["Energy_eV"], marker='o', linestyle='-', color='#00FF00', linewidth=2, label='Banda di Valenza Perturbata (Dati Hardware Reali)')
ax.fill_between(df["Wavevector_k"], df["Energy_eV"], -1.5, color='#00FF00', alpha=0.1)

ax.set_title(f"Silicon Bandstructure on {platform.processor()}\nLive Injection: {current_ghz:.2f}GHz @ {cpu_load}% Load", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Vettore d'onda nello Spazio di Brillouin (k)", color='#888888')
ax.set_ylabel("Energia degli Elettroni (eV)", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("bande_silicio_ibrido.png", dpi=300)
print("\n============================================================")
print("âœ… MAPPATURA IBRIDA EFFETTUATA CON SUCCESSO!")
print("ðŸ“Š Grafico accoppiato salvato in: bande_silicio_ibrido.png")
print("============================================================")
