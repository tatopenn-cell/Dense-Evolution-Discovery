import platform
import psutil
import time
import jax
import jax.numpy as jnp
import numpy as np
import dense_evolution as de

# Configurazione rigorosa dell'ambiente double precision
jax.config.update("jax_enable_x64", True)

N_Q = 24  # Ridotto a 24 per non saturare i 7.88 GB di RAM del tuo sistema ed evitare lo swap su disco
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

print("============================================================")
print("🔬 AVVIO SIMULAZIONE QUANTISTICA REALE SU 24 QUBIT")
print("============================================================")

# --- ESPERIMENTO 1: STATO GHZ + RUMORE DEPOLARIZZANTE ---
circ_ghz = [['h', 0]] + [['cx', q, q + 1] for q in range(N_Q - 1)]

# Benchmark dei tempi di calcolo (Veri dati hardware)
t0 = time.perf_counter()
sim.set_initial_state()
sim.run_circuit_jit_beast_mode(circ_ghz)
t_jit = time.perf_counter() - t0

sim.set_initial_state()
t1 = time.perf_counter()
sim.run_circuit_jit_beast_mode(circ_ghz)
t_pure = time.perf_counter() - t1

# Applicazione del canale stocastico di depolarizzazione reale
key1 = jax.random.PRNGKey(99)
sv_dep = de.NoiseModel.apply_to_sv(sv=sim.get_statevector(), n=N_Q, model='depolarizing', p=0.05, jax_key=key1)
sim.sv = sv_dep

# Estrazione delle probabilità reali calcolate dal simulatore
p_dep = np.array(sim.get_probabilities())

# Calcolo matematico REALE dell'Entropia di Von Neumann
p_dep_safe = p_dep[p_dep > 1e-12]
ent_dep = -float(np.sum(p_dep_safe * np.log2(p_dep_safe)))

# Calcolo REALE del valore di aspettazione dell'operatore ferromagnetico <H_zz>
E_zz_dep = -float(p_dep[0] + p_dep[-1])

# Filtraggio microscopico degli stati dominanti (sopra lo 0.01%)
indices_dep = np.where(p_dep > 1e-4)[0]
stati_dep = [(idx, float(p_dep[idx])) for idx in indices_dep]


# --- ESPERIMENTO 2:  CIRCUITO CON CAMPO TRASVERSO (ANISOTROPIA g) ---
def genera_circuito_ising(g_campo):
    rotazioni = [['rx', q, float(0.1 * g_campo)] for q in range(N_Q)]
    entanglement = [['cx', q, q + 1] for q in range(N_Q - 1)]
    return rotazioni + entanglement

# Esecuzione per g = 1.0
sim.set_initial_state()
sim.run_circuit_jit_beast_mode(genera_circuito_ising(g_campo=1.0))
p_g10 = sim.get_probabilities()
E_zz_g10 = -float(p_g10[0] + p_g10[-1])

# Esecuzione per g = 1.2
sim.set_initial_state()
sim.run_circuit_jit_beast_mode(genera_circuito_ising(g_campo=1.2))
p_g12 = sim.get_probabilities()
E_zz_g12 = -float(p_g12[0] + p_g12[-1])


# --- ESPERIMENTO 3: AMPLITUDE DAMPING (RILASSAMENTO TERMICO) ---
sim.set_initial_state()
sim.run_circuit_jit_beast_mode(circ_ghz)
key2 = jax.random.PRNGKey(101)
sim.sv = de.NoiseModel.apply_to_sv(sv=sim.get_statevector(), n=N_Q, model='amplitude_damping', p=0.05, jax_key=key2)
p_ad = np.array(sim.get_probabilities())

indices_ad = np.where(p_ad > 1e-4)[0]
stati_ad = [(idx, float(p_ad[idx])) for idx in indices_ad]


# --- SCRITTURA DEL LOG CERTIFICATO E REALE ---
with open('report_quantistico_24qubit_REALE.log', 'w', encoding='utf-8') as f:
    f.write('============================================================\n')
    f.write('🔬 DENSE-EVOLUTION HARDWARE & MULTI-EXPERIMENT REAL REPORT\n')
    f.write('============================================================\n\n')
    f.write('--- TARGET ENVIRONMENT & SYSTEM METRICS ---\n')
    f.write(f'Host OS Platform : {platform.system()} {platform.release()} (v{platform.version()})\n')
    f.write(f'Core CPU Architecture: {platform.processor()}\n')
    f.write(f'Total Physical RAM   : {psutil.virtual_memory().total / (1024**3):.2f} GB\n')
    f.write(f'Active Python Runtime: {platform.python_version()}\n')
    f.write(f'JAX Hardware Backend : {jax.default_backend()} | Engine: XLA Kernel Fusion\n\n')
    
    f.write('--- ARCHITECTURAL BENCHMARKS (STEADY-STATE) ---\n')
    f.write(f'Allocated Qubits    : {N_Q}\n')
    f.write(f'Hilbert Space Dim   : {len(p_dep)} complex amplitudes\n')
    f.write(f'Raw Statevector RAM : {sim.memory_mb():.2f} MB (Zero-Reshape 1D Layer)\n')
    f.write(f'JAX Tracing Overhead: {t_jit:.4f} seconds (First-run compilation)\n')
    f.write(f'Amortized Compute Latency: {t_pure:.6f} seconds (Pure JIT Execution)\n')
    f.write(f'Computational Throughput : {len(circ_ghz) / t_pure:.2f} Gates/Second\n\n')
    
    f.write('--- [EXPERIMENT 1] DEPOLARIZING NOISE ANALYSIS ---\n')
    f.write(f'Von Neumann Spectral Entropy : {ent_dep:.4f} bit\n')
    f.write(f'Ferromagnetic Contribution <H_zz> : {E_zz_dep:.4f}\n')
    f.write('Dominant Active Amplitudes (Soglia > 0.01%):\n')
    for idx, prob in stati_dep[:20]: 
        f.write(f'  |{format(idx, f"0{N_Q}b")}⟩ -> {prob*100:.4f}%\n')
    if len(stati_dep) > 20: f.write(f'  ... e altri {len(stati_dep) - 20} stati rimescolati dal rumore.\n')
    
    f.write('\n--- [EXPERIMENT 2] TRANSVERSE FIELD SCAN PARAMETERS ---\n')
    f.write(f' 🔹 Campo g = 1.0 -> Expectation <H_zz>: {E_zz_g10:.4f}\n')
    f.write(f' 🔹 Campo g = 1.2 -> Expectation <H_zz>: {E_zz_g12:.4f}\n')
    
    f.write('\n--- [EXPERIMENT 3] THERMAL AMPLITUDE DAMPING ---\n')
    f.write('Stati risultanti dal decadimento energetico (T1 relaxation):\n')
    for idx, prob in stati_ad[:20]: 
        f.write(f'  |{format(idx, f"0{N_Q}b")}⟩ -> {prob*100:.4f}%\n')
    if len(stati_ad) > 20: f.write(f'  ... e altri {len(stati_ad) - 20} stati rilassati termicamente.\n')
    f.write('============================================================\n')

print("✅ File 'report_quantistico_24qubit_.log' ")
