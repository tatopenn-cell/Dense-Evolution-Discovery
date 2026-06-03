import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

# Forza precisione doppia float64 macchina
jax.config.update("jax_enable_x64", True)

# Ridotto a 18 per rientrare perfettamente nei 7.88 GB di RAM fisica ed evitare l'Out of Memory
N_Q = 18
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

print("============================================================")
print(f"ðŸ”¬ BATCH-ENGINE QUANTUM DEFECT SCANNER (18 QUBITS - LIGHTWEIGHT)")
print("============================================================")
print("âš¡ Calcolo parallelo istantaneo ottimizzato per RAM limitata...")

# Definiamo la struttura del circuito parametrico di base
base_ops = [['rx', q, 0.0] for q in range(N_Q)] + [['cx', q, q + 1] for q in range(N_Q - 1)]

t_start = time.perf_counter()

# Generiamo la matrice di parametri (18x18)
griglia_parametri = np.zeros((N_Q, N_Q), dtype=np.float64)
for q in range(N_Q):
    griglia_parametri[q, q] = 0.5

# Convertiamo la matrice in un array JAX pronto per il silicio
jax_batch = jnp.array(griglia_parametri, dtype=jnp.float64)

print("â³ Invio della matrice di ispezione al core hardware...")
t_calc_start = time.perf_counter()
statevectors_batch = sim.run_parametric_batch_jit(base_ops, jax_batch)
t_calc = time.perf_counter() - t_calc_start

risultati_ispezione = []

# Analizziamo lo spettro risultante direttamente dagli statevector
for local_qubit in range(N_Q):
    sv_nodo = statevectors_batch[local_qubit]
    
    # Calcoliamo le probabilitÃ  reali dal tracer risolto
    prob_local = jnp.abs(sv_nodo)[0]**2
    coerenza_residua = float(prob_local)
    
    print(f"Ispezione Nodo {local_qubit+1:02d}/{N_Q} | Coerenza Residua: {coerenza_residua*100:.4f}%")
    risultati_ispezione.append({
        "Nodo_Hardware": local_qubit,
        "Coerenza_Residua": coerenza_residua
    })

# Esportazione rapida dei dati
df = pd.DataFrame(risultati_ispezione)
df.to_csv("mappa_difetti_silicio.csv", index=False)

# Rendering grafico della mappa dei difetti strutturali
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Nodo_Hardware"], df["Coerenza_Residua"], marker='v', linestyle='-', color='#00FFFF', linewidth=2, label='Mappa Resilienza Silicio')
ax.fill_between(df["Nodo_Hardware"], df["Coerenza_Residua"], color='#00FFFF', alpha=0.1)
ax.axhline(0.50, color='#FF007F', linestyle='--', alpha=0.7, label='Soglia Minima di Rottura')

ax.set_title("Quantum Defect Mapping: Native JAX Parallel Batch Mode (18 Qubits)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Nodi / Coordinate dei Qubit Hardware", color='#888888')
ax.set_ylabel("Indice di StabilitÃ  Quantistica Modale", color='#888888')
ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("mappa_difetti_silicio.png", dpi=300)

latenza_totale = time.perf_counter() - t_start
print("\n============================================================")
print(f"âœ… ISPEZIONE QUANTISTICA EFFETTUATA CON SUCCESSO!")
print(f"ðŸ“Š Tempo puro di calcolo XLA: {t_calc:.4f} secondi!")
print(f"â³ Latenza totale di esecuzione: {latenza_totale:.2f} secondi!")
print("============================================================")

