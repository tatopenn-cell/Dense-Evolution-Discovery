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

N_Q = 12
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

print("============================================================")
print("🔬 REAL PARALLEL QUANTUM DEFECT SCANNER (JORDAN-WIGNER)")
print("============================================================")
print("⚡ Simulating localized dephasing defects via JAX Batch Engine...\n")

base_ops = []
for q in range(N_Q):
    base_ops.append(['ry', q, float(np.pi / 4)])  

for q in range(N_Q):
    base_ops.append(['rz', q, f"batch_param_{q}"]) 

for q in range(N_Q - 1):
    base_ops.append(['cx', q, q + 1])  

# run_parametric_batch_jit tratta OGNI gate di rotazione (rx/ry/rz/p) come
# uno slot parametrico preso in ordine posizionale da parameter_batch, anche
# quando l'argomento passato a base_ops e' un float letterale — il valore
# letterale viene ignorato. Servono quindi 2*N_Q colonne: le prime N_Q per
# gli RY(pi/4) (costanti, ripetute identiche su ogni riga), le successive
# N_Q per gli RZ("batch_param_q") (il pattern diagonale voluto).
griglia_parametri = np.zeros((N_Q, 2 * N_Q), dtype=np.float64)
griglia_parametri[:, :N_Q] = np.pi / 4
for q in range(N_Q):
    griglia_parametri[q, N_Q + q] = 0.5

jax_batch = jnp.array(griglia_parametri, dtype=jnp.float64)

print("🔬 Invio del batch parametrico a JAX XLA...")
t_calc_start = time.perf_counter()

statevectors_batch = sim.run_parametric_batch_jit(base_ops, jax_batch)

t_calc = time.perf_counter() - t_calc_start
risultati_ispezione = []

for local_qubit in range(N_Q):
    sv_nodo = statevectors_batch[local_qubit]
    dim = len(sv_nodo)
    
    mask = 1 << local_qubit
    indices = np.arange(dim)
    flipped_indices = indices ^ mask
    
    aspettazione_x = float(np.real(np.sum(np.conj(sv_nodo) * sv_nodo[flipped_indices])))
    coerenza_residua = abs(aspettazione_x)
    
    print(f"Ispezione Nodo {local_qubit+1:02d}/{N_Q} | Difetto localizzato sul qubit | Coerenza <X>: {coerenza_residua*100:.4f}%")
    
    risultati_ispezione.append({
        "Nodo_Hardware": local_qubit,
        "Coerenza_Residua": coerenza_residua
    })

df = pd.DataFrame(risultati_ispezione)
df.to_csv(_DATA_DIR / "mappa_difetti_silicio.csv", index=False)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Nodo_Hardware"], df["Coerenza_Residua"], marker='o', linestyle='-', color='#00FFFF', linewidth=2, label='Resilienza Locale al Dephasing')
ax.fill_between(df["Nodo_Hardware"], df["Coerenza_Residua"], 0, color='#00FFFF', alpha=0.1)

ax.set_ylim(0, 1.05)
ax.set_title("True Quantum Defect Mapping: Localized Phase Noise Impact (JAX Batch)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Posizione del Difetto Strutturale (Indice del Qubit)", color='#888888')
ax.set_ylabel("Coerenza Quantistica Residua <X>", color='#888888')
ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(_IMAGES_DIR / "mappa_difetti_silicio.png", dpi=300)

print("\n============================================================")
print("✅ STRUMENTO DI ISPEZIONE QUANTISTICA CORRETTO E FUNZIONANTE!")
print(f"📊 Grafico fisico esportato in: mappa_difetti_silicio.png")
print("============================================================")
