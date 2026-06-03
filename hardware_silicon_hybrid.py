import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import dense_evolution as de

# Forza la precisione a 64 bit per garantire la convergenza della simulazione
jax.config.update("jax_enable_x64", True)

print("============================================================")
print("⚛️ TRUE VARIATIONAL QUANTUM EIGENSOLVER (VQE) SIMULATION")
print("============================================================")
print("🔍 Finding the true ground state energy of a 1D Fermionic Chain...\n")

N_Q = 6  # Ridotto a 6 qubit per garantire un'ottimizzazione VQE rapida e stabile
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

# Parametro fisico reale di hopping (eV)
t_hopping = 2.11  

def cost_function(theta_params):
    """
    Funzione di costo del VQE: calcola il valore di aspettazione dell'Hamiltoniana
    <psi(theta)| H |psi(theta)> usando il simulatore quantistico.
    """
    ansatz_circuit = []
    
    # 1. Prepariamo lo stato di Fock ad 1 elettrone: |100000>
    ansatz_circuit.append(['x', 0])
    
    # 2. Hardware-Efficient Excitation-Preserving Ansatz (Rotazioni di Givens)
    # Questo blocco distribuisce l'eccitazione preservando il settore a 1 fermione.
    # Usiamo porte controllate e rotazioni parametrizzate dagli angoli theta.
    param_idx = 0
    for q in range(N_Q - 1):
        # Implementazione di una rotazione orbitale fermionica coerente (Givens gate)
        # Sfrutta accoppiamenti controllati per non violare lo spazio delle particelle
        ansatz_circuit.append(['cx', q + 1, q])
        ansatz_circuit.append(['ry', q + 1, float(theta_params[param_idx])])
        ansatz_circuit.append(['cx', q, q + 1])
        ansatz_circuit.append(['ry', q + 1, -float(theta_params[param_idx])])
        ansatz_circuit.append(['cx', q + 1, q])
        param_idx += 1

    # Esecuzione del circuito nel simulatore quantistico
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ansatz_circuit)
    statevector = sim.get_statevector()
    
    # 3. Misurazione dei valori di aspettazione di Jordan-Wigner (H = -t/2 * sum(XX + YY))
    dim = len(statevector)
    indices = np.arange(dim)
    total_kinetic_energy = 0.0
    
    for q in range(N_Q):
        q_next = (q + 1) % N_Q  # Condizioni al contorno periodiche (PBC)
        mask_i = 1 << q
        mask_j = 1 << q_next
        combined_mask = mask_i | mask_j
        
        flipped_indices = indices ^ combined_mask
        psi_flipped = statevector[flipped_indices]
        
        # Termine XX
        xx_exp = np.real(np.sum(np.conj(statevector) * psi_flipped))
        
        # Termine YY
        bit_i = (indices & mask_i) >> q
        bit_j = (indices & mask_j) >> q_next
        phase = np.where(bit_i == bit_j, -1.0, 1.0)
        yy_exp = np.real(np.sum(np.conj(statevector) * psi_flipped * phase))
        
        total_kinetic_energy += float(xx_exp + yy_exp)
        
    # Calcolo dell'energia totale
    E_total = - (t_hopping / 2.0) * total_kinetic_energy
    return E_total

# --- ESECUZIONE DELL'OTTIMIZZAZIONE CO-ASSISTITA (VQE) ---
# Inizializziamo i parametri theta in modo casuale
num_parameters = N_Q - 1
initial_thetas = np.random.uniform(0, 2 * np.pi, num_parameters)

print("🚀 Avvio dell'ottimizzatore classico (COBYLA)...")
t_start = time.perf_counter()

res = minimize(cost_function, initial_thetas, method='COBYLA', options={'maxiter': 200})

vqe_duration = time.perf_counter() - t_start
exact_ground_state = -2 * t_hopping  # Valore teorico esatto per k=0

print("\n============================================================")
print("✅ OTTIMIZZAZIONE VQE COMPLETATA CON SUCCESSO!")
print("============================================================")
print(f"⏱️ Tempo impiegato dall'algoritmo: {vqe_duration:.3f} secondi")
print(f"📊 Energia minima trovata dal VQE: {res.fun:.6f} eV")
print(f"🎯 Valore teorico analitico esatto: {exact_ground_state:.6f} eV")
print(f"📉 Errore residuo assoluto: {abs(res.fun - exact_ground_state):.6e} eV")
print("============================================================")

