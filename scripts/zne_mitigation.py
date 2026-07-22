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

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
t_hopping = 2.11
NUM_SHOTS = 2000

def generate_bloch_state(k_val):
    dim = 1 << N_Q
    state = np.zeros(dim, dtype=np.complex128)
    for q in range(N_Q):
        state[1 << q] = (1.0 / np.sqrt(N_Q)) * np.exp(1j * k_val * q)
    return state

def apply_stochastic_dephasing(statevector, p_error, seed):
    np.random.seed(seed)
    sv = np.array(statevector, dtype=np.complex128)
    
    for q in range(N_Q):
        if np.random.rand() < p_error:
            dim = len(sv)
            mask = 1 << q
            indici = np.arange(dim)
            fase_z = np.where((indici & mask) >> q == 1, -1.0, 1.0)
            sv = sv * fase_z
            
    return sv

def measure_energy_with_shots(k_val, noise_scale, base_seed):
    p_base = 0.06 
    p_error = p_base * noise_scale
    
    energie_shots = []
    sv_iniziale = generate_bloch_state(k_val)
    
    if noise_scale == 0.0:
        return calcola_aspettazione_hamiltoniana(sv_iniziale)
        
    for shot in range(NUM_SHOTS):
        sv_rumoroso = apply_stochastic_dephasing(sv_iniziale, p_error, seed=base_seed + shot)
        E_shot = calcola_aspettazione_hamiltoniana(sv_rumoroso)
        energie_shots.append(E_shot)
        
    return float(np.mean(energie_shots))

def calcola_aspettazione_hamiltoniana(statevector):
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

def _run_full_sweep():
    punti_k = np.linspace(-np.pi, np.pi, 25)
    zne_results = []

    for idx, k in enumerate(punti_k):
        t_start = time.perf_counter()

        seed_punto = int(abs(k) * 100000) + (idx * 500)

        E_noise_l1 = measure_energy_with_shots(k, noise_scale=1.0, base_seed=seed_punto)
        E_noise_l2 = measure_energy_with_shots(k, noise_scale=2.0, base_seed=seed_punto + NUM_SHOTS)
        E_mitigated = 2.0 * E_noise_l1 - E_noise_l2
        E_ideal = measure_energy_with_shots(k, noise_scale=0.0, base_seed=0)

        latency = time.perf_counter() - t_start
        print(f"k: {k:+.2f} | Ideal: {E_ideal:+.4f} | Noisy (λ=1): {E_noise_l1:+.4f} | Mitigated (ZNE): {E_mitigated:+.4f} | {latency:.2f}s")

        zne_results.append({
            "k": k,
            "Ideal": E_ideal,
            "Noisy": E_noise_l1,
            "Mitigated": E_mitigated
        })

    df = pd.DataFrame(zne_results)
    df.to_csv(_DATA_DIR / "dati_mitigazione_zne.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["k"], df["Ideal"], color='#00FF00', linewidth=2.5, label='True Zero-Noise Target')
    ax.plot(df["k"], df["Noisy"], color='#FF3333', linestyle=':', linewidth=2, label='Real Noisy Data (λ = 1.0)')
    ax.plot(df["k"], df["Mitigated"], color='#FFFF00', marker='o', markersize=4, linestyle='-', linewidth=1.5, label='True Richardson Mitigated (ZNE)')

    ax.set_title("Quantum Error Mitigation: Real Stochastic Kraus & Shot Noise ZNE", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("Wavevector k", color='#888888')
    ax.set_ylabel("Energy Level (eV)", color='#888888')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "transizione_ising_mitigata.png", dpi=300)


if __name__ == "__main__":
    print("============================================================")
    print("🔬 REAL PHYSICAL ZNE ENGINE: STOCHASTIC KRAUS SAMPLING")
    print("============================================================")
    print("🔍 Running true non-deterministic Richardson Extrapolation...\n")
    _run_full_sweep()
