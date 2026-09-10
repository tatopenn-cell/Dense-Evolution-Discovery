"""
The COMPLETE, promotion-ready run_circuit_jit(ops, fuse_gates=True), fully
written and tested here in Discovery -- not just the internal runner, the
entire method body including the MPSSimulator wiring (self.gammas,
self._real_chi, bookkeeping). Attached to the REAL installed MPSSimulator
class via monkey-patching (Python does not enforce class boundaries), not
copied into dense_evolution's own source -- so this script tests the exact
code that would be promoted, not a standalone approximation of it.

Promotion into dense_evolution/backends/mps.py must be a literal copy of
the functions below (_embed_1q_matrix, _fuse_compiled_rows,
_fused_entries_to_arrays, _build_fused_mps_runner, _record_diag_bookkeeping,
run_circuit_jit_fused), not a rewrite -- that is the whole point of
finishing the adaptation here first.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import dense_evolution as de
from dense_evolution.backends.mps import (
    MPSSimulator,
    _compile_mps_ops,
    _mps_1q_matrix,
    _mps_2q_matrix,
    _vectorized_chi_search_jax,
    _pad_gamma,
    _pad_lambda,
    _bucket_sizes,
    _real_dtype_for,
)


def _embed_1q_matrix(mat1, qubit, pair):
    eye2 = np.eye(2, dtype=mat1.dtype)
    a, _ = pair
    return np.kron(mat1, eye2) if qubit == a else np.kron(eye2, mat1)


def _fuse_compiled_rows(rows, dtype):
    def row_matrix(row):
        g_id, q1, q2, param, transpose_flag = row
        g_id_arr = jnp.asarray(g_id).astype(jnp.int32)
        if g_id >= 20:
            mat4 = np.asarray(_mps_2q_matrix(g_id_arr, jnp.asarray(param), dtype))
            if transpose_flag > 0.5:
                mat4 = np.transpose(mat4, (1, 0, 3, 2))
            return mat4.reshape(4, 4), (int(q1), int(q2))
        return np.asarray(_mps_1q_matrix(g_id_arr, jnp.asarray(param), dtype)), (int(q1),)

    fused = []
    i = 0
    n = len(rows)
    while i < n:
        mat, qubits = row_matrix(rows[i])
        pair = qubits if len(qubits) == 2 else None
        active_qubit = qubits[0]
        j = i + 1
        while j < n:
            nmat, nqubits = row_matrix(rows[j])
            n_is_2q = len(nqubits) == 2
            if pair is None:
                if not n_is_2q:
                    if nqubits[0] != active_qubit:
                        break
                    mat = nmat @ mat
                    j += 1
                    continue
                if active_qubit not in nqubits:
                    break
                pair = nqubits
                mat = _embed_1q_matrix(mat, active_qubit, pair)
                mat = nmat @ mat
                j += 1
                continue
            if n_is_2q:
                if nqubits != pair:
                    break
            else:
                if nqubits[0] not in pair:
                    break
                nmat = _embed_1q_matrix(nmat, nqubits[0], pair)
            mat = nmat @ mat
            j += 1
        if pair is not None:
            fused.append(('2q', pair[0], pair[1], mat.reshape(2, 2, 2, 2)))
        else:
            fused.append(('1q', active_qubit, mat))
        i = j
    return fused


def _fused_entries_to_arrays(fused, dtype):
    is_2q, q1s, q2s, mats2q, mats1q = [], [], [], [], []
    eye2 = np.eye(2, dtype=dtype)
    eye4 = np.eye(4, dtype=dtype).reshape(2, 2, 2, 2)
    for entry in fused:
        if entry[0] == '2q':
            _, a, b, mat = entry
            is_2q.append(True); q1s.append(a); q2s.append(b); mats2q.append(mat); mats1q.append(eye2)
        else:
            _, a, mat = entry
            is_2q.append(False); q1s.append(a); q2s.append(0); mats2q.append(eye4); mats1q.append(mat)
    return (jnp.asarray(is_2q), jnp.asarray(q1s, dtype=jnp.int32), jnp.asarray(q2s, dtype=jnp.int32),
            jnp.asarray(np.stack(mats2q)), jnp.asarray(np.stack(mats1q)))


def _build_fused_mps_runner(n_qubits, max_bond, eps, jsd_budget):
    buckets = _bucket_sizes(max_bond)
    bucket_arr = jnp.array(buckets)

    def step(carry, xs):
        gammas, lambdas, real_chi = carry
        dtype = gammas.dtype
        is_2q, q1, q2, mat2q, mat1q = xs
        q1 = q1.astype(jnp.int32)
        q2 = q2.astype(jnp.int32)
        real_dtype = _real_dtype_for(dtype)

        def branch_1q(c):
            gammas_, lambdas_, real_chi_ = c
            new_g = jnp.einsum('ij,ljr->lir', mat1q, gammas_[q1])
            new_carry = (gammas_.at[q1].set(new_g), lambdas_, real_chi_)
            diag = (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0, dtype=real_dtype),
                    jnp.asarray(0.0, dtype=real_dtype), jnp.asarray(0.0, dtype=real_dtype))
            return new_carry, diag

        def branch_2q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_2q = mat2q
            chi_l_real = real_chi_[q1]
            chi_m_real = real_chi_[q2]
            chi_r_real = real_chi_[q2 + 1]
            input_min = jnp.maximum(jnp.maximum(chi_l_real, chi_r_real), chi_m_real)
            output_bound = jnp.minimum(chi_l_real * 2, chi_r_real * 2)
            bound = jnp.maximum(input_min, output_bound)
            ge_mask = bucket_arr >= bound
            bucket_idx = jnp.where(jnp.any(ge_mask), jnp.argmax(ge_mask), len(buckets) - 1)

            def make_branch(B):
                def branch_fn(_):
                    g1 = gammas_[q1][:B, :, :B]
                    g2 = gammas_[q2][:B, :, :B]
                    lam_l = lambdas_[q1][:B]
                    lam_m = lambdas_[q2][:B]
                    lam_r = lambdas_[q2 + 1][:B]
                    theta = jnp.einsum('l,lik,k,kjr,r->lijr', lam_l, g1, lam_m, g2, lam_r)
                    theta_new = jnp.einsum('abcd,ecdf->eabf', gate_2q, theta)
                    theta_mat = theta_new.reshape(B * 2, 2 * B)

                    U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
                    chi_new, jsd_val = _vectorized_chi_search_jax(S, eps, jsd_budget, min(B, max_bond))
                    col_mask = jnp.arange(B) < chi_new

                    norm_full = jnp.sqrt(jnp.sum(S ** 2) + 1e-30)
                    S_norm_full = S / (norm_full + 1e-30)
                    trunc_err = jnp.sqrt(jnp.sum(jnp.where(jnp.arange(2 * B) >= chi_new, S_norm_full ** 2, 0.0)))

                    S_kept_masked = jnp.where(col_mask, S[:B], 0.0)
                    kept_norm = jnp.sqrt(jnp.sum(S_kept_masked ** 2) + 1e-30)
                    S_fixed = jnp.where(col_mask, S_kept_masked / (kept_norm + 1e-30), 0.0)

                    lam_l_inv = jnp.where(lam_l > eps, 1.0 / lam_l, 0.0)
                    lam_r_inv = jnp.where(lam_r > eps, 1.0 / lam_r, 0.0)

                    U_masked = jnp.where(col_mask[None, :], U[:, :B], 0.0)
                    Vh_masked = jnp.where(col_mask[:, None], Vh[:B, :], 0.0)

                    new_g1 = jnp.einsum('l,lir->lir', lam_l_inv, U_masked.reshape(B, 2, B))
                    new_g2 = jnp.einsum('ljr,r->ljr', Vh_masked.reshape(B, 2, B), lam_r_inv)

                    p_dist = S_fixed ** 2
                    ee = -jnp.sum(jnp.where(p_dist > 1e-20, p_dist * jnp.log2(jnp.where(p_dist > 1e-20, p_dist, 1.0)), 0.0))

                    return (_pad_gamma(new_g1, max_bond), _pad_gamma(new_g2, max_bond),
                            _pad_lambda(S_fixed, max_bond), chi_new.astype(jnp.int32),
                            jsd_val.astype(real_dtype), trunc_err.astype(real_dtype), ee.astype(real_dtype))

                return branch_fn

            branches = [make_branch(B) for B in buckets]
            new_g1_p, new_g2_p, S_fixed_p, chi_new, jsd_val, trunc_err, ee = jax.lax.switch(
                bucket_idx, branches, operand=None)

            new_gammas = gammas_.at[q1].set(new_g1_p).at[q2].set(new_g2_p)
            new_lambdas = lambdas_.at[q2].set(S_fixed_p)
            new_real_chi = real_chi_.at[q2].set(chi_new)

            new_carry = (new_gammas, new_lambdas, new_real_chi)
            diag = (chi_new, jsd_val, trunc_err, ee)
            return new_carry, diag

        new_carry, diag = jax.lax.cond(is_2q, branch_2q, branch_1q, carry)
        return new_carry, diag

    @jax.jit
    def run(gammas, lambdas, real_chi, xs):
        (final_gammas, final_lambdas, final_real_chi), diag = jax.lax.scan(
            step, (gammas, lambdas, real_chi), xs)
        return final_gammas, final_lambdas, final_real_chi, diag

    return run


def _record_diag_bookkeeping(self, diag, q1_ids, is_2q_mask):
    chi_history, jsd_history, trunc_err_history, entropy_history = (
        np.asarray(diag[0]), np.asarray(diag[1]), np.asarray(diag[2]), np.asarray(diag[3]))
    for i in np.nonzero(is_2q_mask)[0]:
        chi_new = int(chi_history[i])
        jsd_val = float(jsd_history[i])
        self._bond_history.append(chi_new)
        self.jsd_per_bond.append(jsd_val)
        self.truncation_errors.append(float(trunc_err_history[i]))
        q1 = q1_ids[i]
        if q1 < len(self.entanglement_entropy):
            self.entanglement_entropy[q1] = float(entropy_history[i])
        if jsd_val > self.jsd_budget:
            if self.budget_violations == 0:
                warnings.warn(
                    f"MPSSimulator: bond dimension capped at max_bond={self.chi}, "
                    f"jsd_budget={self.jsd_budget:.1e} not honored "
                    f"(jsd={jsd_val:.2e}) -- results may be unreliable, "
                    f"consider raising max_bond.",
                    UserWarning,
                    stacklevel=2,
                )
            self.budget_violations += 1


def run_circuit_jit_fused(self, ops, fuse_gates=True):
    dtype = self.gammas[0].dtype
    lambda_dtype = self.lambdas[0].dtype

    compiled_rows = _compile_mps_ops(ops, self.n)
    fused = _fuse_compiled_rows(compiled_rows, dtype) if compiled_rows else []

    if getattr(self, "_fused_mps_runner", None) is None:
        self._fused_mps_runner = _build_fused_mps_runner(self.n, self.chi, self.eps, self.jsd_budget)

    gammas_padded = jnp.stack([_pad_gamma(g, self.chi).astype(dtype) for g in self.gammas])
    lambdas_padded = jnp.stack([_pad_lambda(l, self.chi).astype(lambda_dtype) for l in self.lambdas])
    real_chi_initial = jnp.asarray(self._real_chi, dtype=jnp.int32)

    if fused:
        xs = _fused_entries_to_arrays(fused, dtype)
        final_gammas, final_lambdas, final_real_chi, diag = self._fused_mps_runner(
            gammas_padded, lambdas_padded, real_chi_initial, xs)

        self.gammas = [final_gammas[i] for i in range(self.n)]
        self.lambdas = [final_lambdas[i] for i in range(self.n + 1)]
        self._real_chi = np.asarray(final_real_chi)

        q1_ids = np.asarray([entry[1] for entry in fused])
        is_2q_mask = np.asarray([entry[0] == '2q' for entry in fused])
        _record_diag_bookkeeping(self, diag, q1_ids, is_2q_mask)


MPSSimulator.run_circuit_jit_fused = run_circuit_jit_fused


def _tfim_ops(n, dt, steps, J, g):
    theta_zz, theta_x = -2.0 * dt * J, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [("cx", i, i + 1), ("rz", i + 1, theta_zz), ("cx", i, i + 1)]
        for i in range(n):
            ops.append(("rx", i, theta_x))
    return ops


def compare(n_qubits, max_bond, ops, label):
    ref = MPSSimulator(n_qubits, max_bond=max_bond)
    ref.run_circuit_jit(ops)
    ref_sv = np.asarray(ref.contract_to_statevector())

    fused = MPSSimulator(n_qubits, max_bond=max_bond)
    fused.run_circuit_jit_fused(ops, fuse_gates=True)
    fused_sv = np.asarray(fused.contract_to_statevector())

    fidelity = float(np.abs(np.vdot(ref_sv, fused_sv)) ** 2)
    print(f"[{label}] fidelity(ref, fused_via_monkeypatched_method) = {fidelity:.12f}")
    print(f"[{label}] bond_history entries: ref={len(ref._bond_history)} fused={len(fused._bond_history)}")
    return fidelity


if __name__ == "__main__":
    f1 = compare(6, 16, _tfim_ops(6, 0.1, 3, 1.0, 1.0), "TFIM, N=6")
    f2 = compare(5, 16, [["h", 0], ["h", 2], ["cx", 4, 0], ["cx", 3, 1], ["cx", 0, 4], ["rz", 2, 0.7]], "non-adjacent")
    f3 = compare(4, 16, [["x", 0], ["x", 1], ["ccx", 0, 1, 2], ["h", 3], ["cx", 2, 3]], "CCX")

    all_ok = all(f > 1 - 1e-8 for f in (f1, f2, f3))
    print(f"\nALL FIDELITIES ACCEPTABLE (via the exact monkey-patched run_circuit_jit_fused method): {all_ok}")
    assert all_ok
