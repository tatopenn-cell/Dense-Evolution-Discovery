import pytest, numpy as np, jax, jax.numpy as jnp, dense_evolution as de
from dense_evolution.healing import calculate_delta_preemp
jax.config.update("jax_enable_x64", True)

N_Q, T_HOP, ATOL_MITIGATED, TARGET_SIGMA_IDEALE = 20, 2.11, 1e-3, 10.0
_sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)

def _bloch_state(k, n):
    sv = np.zeros(1 << n, dtype=np.complex128)
    for q in range(n): sv[1 << q] = (1.0 / np.sqrt(n)) * np.exp(1j * k * q)
    return sv

def _de_xy_energy(sv):
    DIM = len(sv)
    idx = np.arange(DIM, dtype=np.int32)
    E = 0.0
    for q in range(N_Q - 1):
        m = (1 << q) | (1 << (q + 1))
        pf = sv[idx ^ m]
        bi = (idx & (1 << q)) >> q
        bj = (idx & (1 << (q + 1))) >> (q + 1)
        E += np.real(np.sum(np.conj(sv) * pf)) + np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
    return -(T_HOP / 2.0) * E

def _simulate_noisy_chain_with_sigma(sv, noise_factor):
    ie = _de_xy_energy(sv)
    att = np.exp(-0.05 * noise_factor)
    return ie * att, TARGET_SIGMA_IDEALE * att

def _static_richardson(e_l1, e_l2, e_l3):
    return 3.0 * e_l1 - 3.0 * e_l2 + 1.0 * e_l3

def _adaptive_healing_richardson(e_l1, e_l2, e_l3, delta_p):
    c1, c2, c3 = 3.0 - 0.01 * delta_p, -3.0 + 0.02 * delta_p, 1.0 - 0.01 * delta_p
    return (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)

@pytest.mark.parametrize("k", [0.0, np.pi/2])
def test_zne_predictive_healing_efficiency(k):
    sv = _bloch_state(k, N_Q)
    ei = _de_xy_energy(sv)
    el1, sl1 = _simulate_noisy_chain_with_sigma(sv, 1.0)
    el2, _ = _simulate_noisy_chain_with_sigma(sv, 2.0)
    el3, _ = _simulate_noisy_chain_with_sigma(sv, 3.0)
    dp = float(calculate_delta_preemp(jnp.array(sl1), TARGET_SIGMA_IDEALE))
    em = _adaptive_healing_richardson(el1, el2, el3, dp)
    assert abs(em - ei) < abs(el1 - ei) and abs(em - ei) < ATOL_MITIGATED

if __name__ == "__main__":
    print("====================================================================================")
    print(f"PREDICTIVE QUANTUM ERROR MITIGATION: STATIC ZNE vs HEALING CORE ENGINE ({N_Q} QUBITS)")
    print("====================================================================================")
    sv = _bloch_state(0.0, N_Q)
    vi = _de_xy_energy(sv)
    vl1, sl1 = _simulate_noisy_chain_with_sigma(sv, 1.0)
    vl2, _ = _simulate_noisy_chain_with_sigma(sv, 2.0)
    vl3, _ = _simulate_noisy_chain_with_sigma(sv, 3.0)
    dp = float(calculate_delta_preemp(jnp.array(sl1), TARGET_SIGMA_IDEALE))
    vm_static = _static_richardson(vl1, vl2, vl3)
    vm_healed = _adaptive_healing_richardson(vl1, vl2, vl3, dp)
    err_unmit = abs(vl1 - vi)
    err_static = abs(vm_static - vi)
    err_healed = abs(vm_healed - vi)
    print(f"Ideal      : {vi:+.8f} eV\nNoisy L1   : {vl1:+.8f} eV | Err: {err_unmit:.6f}")
    print(f"------------------------------------------------------------------------------------")
    print(f"Static ZNE : {vm_static:+.8f} eV | Err: {err_static:.8f} | Fid: {((err_unmit-err_static)/err_unmit)*100:.4f}%")
    print(f"Healed ZNE : {vm_healed:+.8f} eV | Err: {err_healed:.8f} | Fid: {((err_unmit-err_healed)/err_unmit)*100:.4f}%")
    print(f"------------------------------------------------------------------------------------")
    print(f"Delta_Pre  : {dp:.6f}\nNet Repair : {abs(vm_static-vm_healed)*1e6:.4f} micro-eV")
