"""
End-to-end integration check for the bucketed-SVD MPS optimization
(see mps_bucketed_svd_correctness.py for the single-gate math, already
verified across 23 (chi_l, chi_r) combinations x 4 precision/jsd_budget
configurations there -- including a real finding that the padded/exact
approach picks up floating-point "ghost" singular values (~1e-8) from its
own zero-padding in complex64, which a sensitive sqrt-scaled JSD metric
can amplify into a spurious budget-violation; the bucketed approach,
never padding before the SVD, doesn't have that artifact).

This script tests the part the single-gate check couldn't: running a real
MULTI-GATE circuit through a jax.lax.scan, where the carry must track a
real (not max_bond-padded) bond-size array across gates, and each 2-qubit
gate's SVD size is chosen via jax.lax.switch over a small fixed bucket
set -- the same JIT-compatible dispatch mechanism dense_evolution's own
_mps_1q_matrix/_mps_2q_matrix already use for gate-ID branching (verified
by reading that code this session, not assumed).

Reference: MPSSimulator's own EAGER path (apply_gate_1q/apply_gate_2q),
which already operates at real (non-padded) tensor shapes -- confirmed by
reading _svd_truncate/apply_gate_2q directly (theta_mat built from the
gamma tensors' own real shapes, not max_bond). This is the correct ground
truth to compare against, not run_circuit_jit (which is the thing being
proposed to change).

Scope: adjacent 2-qubit gates only (matches the TFIM circuit used
throughout this session's MPS benchmarking) -- non-adjacent gates need
run_circuit_jit's own SWAP-chain expansion (_apply_nonlocal_2q), not
reimplemented here.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dense_evolution.backends.mps import (
    MPSSimulator,
    _mps_1q_matrix,
    _mps_2q_matrix,
    _vectorized_chi_search_jax,
    _pad_gamma,
    _pad_lambda,
)

BUCKETS = (2, 4, 8, 16, 32, 64, 128)


def _bucketed_runner(n_qubits, max_bond, eps, jsd_budget, dtype, ops):
    buckets = tuple(b for b in BUCKETS if b <= max_bond)
    if buckets[-1] != max_bond:
        buckets = buckets + (max_bond,)
    bucket_arr = jnp.array(buckets)

    def step(carry, row):
        gammas, lambdas, real_chi = carry
        g_id = row[0].astype(jnp.int32)
        q1 = row[1].astype(jnp.int32)
        q2 = row[2].astype(jnp.int32)
        param = row[3]
        transpose_flag = row[4] > 0.5

        def branch_1q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_1q = _mps_1q_matrix(g_id, param, dtype)
            new_g = jnp.einsum("ij,ljr->lir", gate_1q, gammas_[q1])
            return gammas_.at[q1].set(new_g), lambdas_, real_chi_

        def branch_2q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_2q = _mps_2q_matrix(g_id, param, dtype)
            gate_2q = jnp.where(transpose_flag, jnp.transpose(gate_2q, (1, 0, 3, 2)), gate_2q)

            chi_l_real = real_chi_[q1]
            chi_r_real = real_chi_[q2 + 1]
            bound = jnp.minimum(chi_l_real * 2, chi_r_real * 2)
            ge_mask = bucket_arr >= bound
            bucket_idx = jnp.where(jnp.any(ge_mask), jnp.argmax(ge_mask), len(buckets) - 1)

            def make_branch(B):
                def branch_fn(_):
                    g1 = gammas_[q1][:B, :, :B]
                    g2 = gammas_[q2][:B, :, :B]
                    lam_l = lambdas_[q1][:B]
                    lam_m = lambdas_[q2][:B]
                    lam_r = lambdas_[q2 + 1][:B]
                    theta = jnp.einsum("l,lik,k,kjr,r->lijr", lam_l, g1, lam_m, g2, lam_r)
                    theta_new = jnp.einsum("abcd,ecdf->eabf", gate_2q, theta)
                    theta_mat = theta_new.reshape(B * 2, 2 * B)
                    U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
                    chi_new, jsd_val = _vectorized_chi_search_jax(S, eps, jsd_budget, min(B, max_bond))
                    col_mask = jnp.arange(B) < chi_new

                    norm_full = jnp.sqrt(jnp.sum(S ** 2) + 1e-30)
                    S_norm = S / (norm_full + 1e-30)

                    S_kept_masked = jnp.where(col_mask, S[:B], 0.0)
                    kept_norm = jnp.sqrt(jnp.sum(S_kept_masked ** 2) + 1e-30)
                    S_fixed = jnp.where(col_mask, S_kept_masked / (kept_norm + 1e-30), 0.0)

                    lam_l_inv = jnp.where(lam_l > eps, 1.0 / lam_l, 0.0)
                    lam_r_inv = jnp.where(lam_r > eps, 1.0 / lam_r, 0.0)

                    U_masked = jnp.where(col_mask[None, :], U[:, :B], 0.0)
                    Vh_masked = jnp.where(col_mask[:, None], Vh[:B, :], 0.0)

                    new_g1 = jnp.einsum("l,lir->lir", lam_l_inv, U_masked.reshape(B, 2, B))
                    new_g2 = jnp.einsum("ljr,r->ljr", Vh_masked.reshape(B, 2, B), lam_r_inv)

                    return _pad_gamma(new_g1, max_bond), _pad_gamma(new_g2, max_bond), _pad_lambda(S_fixed, max_bond), chi_new

                return branch_fn

            branches = [make_branch(B) for B in buckets]
            new_g1_p, new_g2_p, S_fixed_p, chi_new = jax.lax.switch(bucket_idx, branches, operand=None)

            new_gammas = gammas_.at[q1].set(new_g1_p).at[q2].set(new_g2_p)
            new_lambdas = lambdas_.at[q2].set(S_fixed_p)
            new_real_chi = real_chi_.at[q2].set(chi_new)
            return new_gammas, new_lambdas, new_real_chi

        is_2q = g_id >= 20
        new_carry = jax.lax.cond(is_2q, branch_2q, branch_1q, carry)
        return new_carry, None

    gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=dtype).at[0, 0, 0].set(1.0), max_bond) for _ in range(n_qubits)])
    lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=_real_dtype(dtype)), max_bond) for _ in range(n_qubits + 1)])
    real_chi0 = jnp.ones(n_qubits + 1, dtype=jnp.int32)

    ops_arr = jnp.asarray(ops, dtype=jnp.float64)
    (gammas_f, lambdas_f, real_chi_f), _ = jax.lax.scan(step, (gammas0, lambdas0, real_chi0), ops_arr)
    return gammas_f, lambdas_f, real_chi_f


def _real_dtype(dtype):
    return jnp.float64 if dtype == jnp.complex128 else jnp.float32


def _reference_run(n_qubits, max_bond, eps, jsd_budget, dtype, gate_list):
    sim = MPSSimulator(n_qubits, max_bond=max_bond, svd_cutoff=eps, jsd_budget=jsd_budget,
                        use_float32=(dtype == jnp.complex64))
    for g_id, q1, q2, param in gate_list:
        if g_id < 20:
            sim.apply_gate_1q(_mps_1q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype), q1)
        else:
            sim.apply_gate_2q(_mps_2q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype), q1, q2)
    return sim


def _tfim_gate_list(n, dt, steps, J, g):
    theta_zz, theta_x = -2.0 * dt * J, -2.0 * dt * g
    gates = []
    for _ in range(steps):
        for i in range(n - 1):
            gates.append((20, i, i + 1, 0.0))       # CX
            gates.append((11, i + 1, 0, theta_zz))  # RZ on i+1
            gates.append((20, i, i + 1, 0.0))       # CX
        for i in range(n):
            gates.append((9, i, 0, theta_x))        # RX on i
    return gates


def _to_ops_rows(gate_list):
    rows = []
    for g_id, q1, q2, param in gate_list:
        rows.append((float(g_id), float(q1), float(q2), float(param), 0.0))
    return rows


def compare(n_qubits, max_bond, eps, jsd_budget, dtype, label):
    gate_list = _tfim_gate_list(n_qubits, 0.1, 3, 1.0, 1.0)
    ops = _to_ops_rows(gate_list)

    ref_sim = _reference_run(n_qubits, max_bond, eps, jsd_budget, dtype, gate_list)
    ref_sv = np.asarray(ref_sim.contract_to_statevector())

    gammas_f, lambdas_f, real_chi_f = _bucketed_runner(n_qubits, max_bond, eps, jsd_budget, dtype, ops)

    bucketed_sim = MPSSimulator(n_qubits, max_bond=max_bond, svd_cutoff=eps, jsd_budget=jsd_budget,
                                 use_float32=(dtype == jnp.complex64))
    bucketed_sim.gammas = [gammas_f[i, : int(real_chi_f[i]), :, : int(real_chi_f[i + 1])] for i in range(n_qubits)]
    bucketed_sim.lambdas = [lambdas_f[i, : int(real_chi_f[i])] for i in range(n_qubits + 1)]
    bucketed_sv = np.asarray(bucketed_sim.contract_to_statevector())

    fidelity = float(np.abs(np.vdot(ref_sv, bucketed_sv)) ** 2)
    max_bond_ref = ref_sim.max_bond_used()
    max_bond_bucketed = int(np.max(real_chi_f))
    print(f"[{label}] n_qubits={n_qubits} max_bond={max_bond} n_gates={len(ops)}")
    print(f"[{label}] fidelity(ref, bucketed) = {fidelity:.12f}")
    print(f"[{label}] max_bond_used: ref={max_bond_ref} bucketed={max_bond_bucketed}")
    print(f"[{label}] ref truncation_error={ref_sim.total_truncation_error():.2e}")
    return fidelity


if __name__ == "__main__":
    f1 = compare(6, 16, 1e-12, 1e-5, jnp.complex128, "complex128,N=6,max_bond=16")
    f2 = compare(8, 32, 1e-12, 1e-5, jnp.complex128, "complex128,N=8,max_bond=32")
    f3 = compare(6, 16, 1e-6, 1e-5, jnp.complex64, "complex64,N=6,max_bond=16")

    all_ok = all(f > 1 - 1e-8 for f in (f1, f2)) and f3 > 1 - 1e-4
    print(f"\nALL FIDELITIES ACCEPTABLE: {all_ok}")
    assert all_ok, "bucketed circuit run disagrees with the reference eager simulator"
