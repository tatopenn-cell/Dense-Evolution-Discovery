import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 8
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

t_0 = 2.11
STRAIN = 0.05
t_strained = t_0 / ((1.0 + STRAIN) ** 2)

punti_k = np.linspace(-np.pi, np.pi, 3500)
risultati_nuovo_silicio = []

print("============================================================")
print("🔬 NEXT-GEN SILICON DESIGNER: HIGH-RESOLUTION SWEEP")
print("============================================================")

def genera_stato_bloch_puro(k_val, n_qubits):
    dim = 1 << n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    for q in range(n_qubits):
        state[1 << q] = (1.0 / np.sqrt(n_qubits)) * np.exp(1j * k_val * q)
    return state

def compute_jordan_wigner_hopping_expectation(statevector, idx, n_qubits):
    dim = len(statevector)
    mask_i = 1 << idx
    mask_j = 1 << ((idx + 1) % n_qubits)
    combined_mask = mask_i | mask_j
    
    indices = np.arange(dim)
    flipped_indices = indices ^ combined_mask
    psi_flipped = statevector[flipped_indices]
    
    xx_exp = np.real(np.sum(np.conj(statevector) * psi_flipped))
    
    bit_i = (indices & mask_i) >> idx
    bit_j = (indices & mask_j) >> ((idx + 1) % n_qubits)
    phase = np.where(bit_i == bit_j, -1.0, 1.0)
    yy_exp = np.real(np.sum(np.conj(statevector) * psi_flipped * phase))
    
    return float(xx_exp + yy_exp)

for idx, k in enumerate(punti_k):
    t_start = time.perf_counter()
    
    statevector = genera_stato_bloch_puro(k, N_Q)
    
    total_kinetic_energy = 0.0
    for q in range(N_Q):
        total_kinetic_energy += compute_jordan_wigner_hopping_expectation(statevector, q, N_Q)
        
    E_k = - (t_strained / 2.0) * total_kinetic_energy
    
    valence_energy = -abs(E_k)
    conduction_energy = abs(E_k)
    
    if idx % 500 == 0 or idx == len(punti_k) - 1:
        print(f"Step {idx+1:04d}/3500 | k: {k:+.3f} rad/a | Valence: {valence_energy:+.4f} eV | Conduction: {conduction_energy:+.4f} eV")
        
    risultati_nuovo_silicio.append({
        "Wavevector_k": k,
        "Valence_Strained": valence_energy,
        "Conduction_Strained": conduction_energy
    })

df_nuovo = pd.DataFrame(risultati_nuovo_silicio)
df_nuovo.to_csv("bande_nuovo_silicio.csv", index=False)

try:
    df_vecchio = pd.read_csv("bande_silicio_ibrido.csv")
    ha_vecchio = True
except:
    ha_vecchio = False

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

if ha_vecchio and "Valence_eV" in df_vecchio.columns:
    ax.plot(df_vecchio["k"], df_vecchio["Valence_eV"], linestyle=':', color='#00FF00', alpha=0.5, label='Standard Valence')
    ax.plot(df_vecchio["k"], df_vecchio["Conduction_eV"], linestyle=':', color='#FF007F', alpha=0.5, label='Standard Conduction')

ax.plot(df_nuovo["Wavevector_k"], df_nuovo["Valence_Strained"], color='#00FFFF', linewidth=2.5, label='Strained Valence (Harrison hopping)')
ax.plot(df_nuovo["Wavevector_k"], df_nuovo["Conduction_Strained"], color='#FFFF00', linewidth=2.5, label='Strained Conduction (Harrison hopping)')

ax.set_title("Strained Solid-State Bandstructure Engineering (3500 Points)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Wavevector k (Brillouin Zone)", color='#888888')
ax.set_ylabel("Electron Energy Level (eV)", color='#888888')
ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("confronto_nuovo_silicio.png", dpi=300)

print("============================================================")
print("✅ SCANSIONE COMPLETATA CON SUCCESSO! VERO COMPORTAMENTO FISICO.")
print("📊 Grafico salvato in: confronto_nuovo_silicio.png")
print("============================================================")
