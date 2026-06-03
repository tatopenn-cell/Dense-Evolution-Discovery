import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
t_hopping = 2.11

def calcola_aspettazione_da_sv(statevector):
    dim = len(statevector)
    indices = np.arange(dim)
    total_kinetic = 0.0
    
    for q in range(N_Q):
        q_next = (q + 1) % N_Q
        mask = (1 << q) | (1 << q_next)
        psi_flipped = statevector[indices ^ mask]
        
        xx_exp = np.real(np.sum(np.conj(statevector) * psi_flipped))
        bit_i = (indices & (1 << q)) >> q
        bit_j = (indices & (1 << q_next)) >> q_next
        phase = np.where(bit_i == bit_j, -1.0, 1.0)
        yy_exp = np.real(np.sum(np.conj(statevector) * psi_flipped * phase))
        
        total_kinetic += float(xx_exp + yy_exp)
        
    return - (t_hopping / 2.0) * total_kinetic

punti_theta = np.linspace(0.0, 2 * np.pi, 3500)
N_STEPS = len(punti_theta)

base_ops = []
base_ops.append(['x', 0])
for q in range(N_Q - 1):
    base_ops.append(['cx', q + 1, q])
    base_ops.append(['ry', q + 1, f"param_vqe"])
    base_ops.append(['cx', q, q + 1])
    base_ops.append(['ry', q + 1, f"param_vqe_inv"])
    base_ops.append(['cx', q + 1, q])

print("============================================================")
print("🚀 MASSIVE JAX BATCH: COMPILING 10,500 PARALLEL INSTANCES...")
print("============================================================")

griglia_globale = np.zeros((N_STEPS * 3, 2), dtype=np.float64)

for idx, theta in enumerate(punti_theta):
    theta_plus = float(theta + np.pi / 2)
    theta_minus = float(theta - np.pi / 2)
    
    griglia_globale[idx * 3]     = [theta, -theta]
    griglia_globale[idx * 3 + 1] = [theta_plus, -theta_plus]
    griglia_globale[idx * 3 + 2] = [theta_minus, -theta_minus]

jax_batch = jnp.array(griglia_globale, dtype=jnp.float64)

t_global_start = time.perf_counter()

statevectors_batch = sim.run_parametric_batch_jit(base_ops, jax_batch)

print("📊 Unpacking statevectors and measuring observables...")
dati_parameter_shift = []

for idx, theta in enumerate(punti_theta):
    sv_curr = statevectors_batch[idx * 3]
    sv_plus = statevectors_batch[idx * 3 + 1]
    sv_minus = statevectors_batch[idx * 3 + 2]
    
    E_curr = calcola_aspettazione_da_sv(sv_curr)
    E_plus = calcola_aspettazione_da_sv(sv_plus)
    E_minus = calcola_aspettazione_da_sv(sv_minus)
    
    gradiente_psr = 0.5 * (E_plus - E_minus)
    
    if (idx + 1) % 250 == 0 or idx == 0 or idx == N_STEPS - 1:
        print(f"Step {idx+1:04d}/3500 | θ: {theta:.3f} rad | E(θ): {E_curr:+.4f} eV | PSR Gradient: {gradiente_psr:+.6f}")
        
    dati_parameter_shift.append({
        "Theta": theta,
        "Energia": E_curr,
        "Gradiente_PSR": gradiente_psr
    })

df = pd.DataFrame(dati_parameter_shift)
df.to_csv("vqe_jax_gradient.csv", index=False)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(df["Theta"], df["Energia"], color='#00FF00', linewidth=2.5, label='VQE Energy Surface E(θ)')
ax1.set_ylabel("Energy (eV)", color='#888888')
ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
ax1.legend(loc="upper right")
ax1.set_title("Exact Parameter-Shift Rule Gradients (10,500 Parallel JAX Tracks)", fontsize=11, fontweight='bold', pad=15)

ax2.plot(df["Theta"], df["Gradiente_PSR"], color='#FF007F', linewidth=2, label='Analytic PSR Gradient (dE/dθ)')
ax2.axhline(0.0, color='#888888', linestyle=':', alpha=0.5)
ax2.set_xlabel("Variational Parameter θ (radians)", color='#888888')
ax2.set_ylabel("Gradient Magnitude", color='#888888')
ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("vqe_jax_gradient.png", dpi=300)

tempo_totale = time.perf_counter() - t_global_start
print("============================================================")
print(f"⚡ VMAP COMPILER SUCCESS: 10,500 TRACKS COMPLETATE IN {tempo_totale:.2f} s")
print("============================================================")
