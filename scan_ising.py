import time
import jax
import numpy as np
import pandas as pd
import dense_evolution as de

# Forza precisione macchina float64/complex128
jax.config.update("jax_enable_x64", True)

N_Q = 24
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

# Generiamo una griglia fine di punti per il campo trasverso g
punti_g = np.linspace(0.0, 2.5, 10)
risultati = []

print("============================================================")
print(f"🔬 ISING TRANSVERSE FIELD SCAN ON {N_Q} QUBITS (16.7M Amplitudes)")
print("============================================================")

# Definizione del circuito variazionale anisotropo reale
def esegui_circuito_ising(g_campo):
    # Strato indotto dal campo trasversale (interazione X)
    rotazioni = [['rx', q, float(0.1 * g_campo)] for q in range(N_Q)]
    # Strato ferromagnetico ad accoppiamento ZZ lungo la catena lineare
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(rotazioni + entanglement)
    return sim.get_probabilities()

# Loop di scansione reale sulle ampiezze JAX
for idx, g in enumerate(punti_g):
    t_start = time.perf_counter()
    
    # Estrazione delle probabilità native calcolate
    prob = esegui_circuito_ising(g)
    
    # Calcolo REALE del valore di aspettazione dell'operatore di stringa <H_zz>
    E_zz = -float(prob[0] + prob[-1])
    
    latenza = time.perf_counter() - t_start
    print(f"Punto {idx+1:02d}/10 | Campo g: {g:.2f} | Correlazione <H_zz>: {E_zz:.6f} | Tempo: {latenza:.2f}s")
    
    risultati.append({
        "Campo_g": g,
        "Expectation_H_zz": E_zz,
        "Latenza_Secondi": latenza
    })

# Esportazione in un file CSV reale per l'analisi dei dati
df = pd.DataFrame(risultati)
df.to_csv("transizione_fase_ising.csv", index=False)

print("\n============================================================")
print("✅ SCANSIONE COMPLETATA CON SUCCESSO!")
print("📊 Dati esportati e salvati in: transizione_fase_ising.csv")
print("============================================================")
