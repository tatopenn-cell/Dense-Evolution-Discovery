import pathlib
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import dense_evolution as de

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)

N_Q = 12
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

def esegui_circuito_ising_reale(g_campo):
    ansatz_circuit = []
    
    for q in range(N_Q - 1):
        ansatz_circuit.append(['cx', q, q + 1])
        ansatz_circuit.append(['rz', q + 1, float(1.2)])
        ansatz_circuit.append(['cx', q, q + 1])
        
    for q in range(N_Q):
        ansatz_circuit.append(['rx', q, float(g_campo * 0.6)])
        
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ansatz_circuit)
    return sim.get_probabilities()

def calcola_vera_correlazione_zz(prob_array):
    dim = len(prob_array)
    indici = np.arange(dim)
    somma_zz = 0.0
    
    for q in range(N_Q - 1):
        bit_i = (indici & (1 << q)) >> q
        bit_j = (indici & (1 << (q + 1))) >> (q + 1)
        parita = np.where(bit_i == bit_j, 1.0, -1.0)
        somma_zz += float(np.sum(prob_array * parita))
        
    return somma_zz / (N_Q - 1)

def _run_full_sweep():
    punti_g = np.linspace(0.0, 2.5, 3500)
    risultati = []

    t_global_start = time.perf_counter()

    for idx, g in enumerate(punti_g):
        prob = esegui_circuito_ising_reale(g)
        E_zz = calcola_vera_correlazione_zz(prob)

        if (idx + 1) % 250 == 0 or idx == 0 or idx == len(punti_g) - 1:
            print(f"Step {idx+1:04d}/3500 | Campo g: {g:.3f} | Correlazione <H_zz>: {E_zz:+.6f}")

        risultati.append({
            "Campo_g": g,
            "Expectation_H_zz": E_zz
        })

    df = pd.DataFrame(risultati)
    df.to_csv(_DATA_DIR / "transizione_fase_ising.csv", index=False)

    tempo_totale = time.perf_counter() - t_global_start
    print("============================================================")
    print(f"✅ SCANSIONE ULTRA-RISOLTA COMPLETATA IN {tempo_totale:.2f} s")
    print("============================================================")


if __name__ == "__main__":
    print("============================================================")
    print("🔬 ULTRA-HIGH RESOLUTION QUANTUM ISING SCAN: 3500 STEPS")
    print("============================================================")
    _run_full_sweep()

