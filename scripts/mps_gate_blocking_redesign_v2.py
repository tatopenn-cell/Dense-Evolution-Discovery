"""
Redesign for promoting gate blocking into dense_evolution proper, closing
three real gaps found in the first version (mps_gate_blocking_experiment.py)
before it could be promoted:

1. Row encoding: the shipped run_circuit_jit's rows are
   [g_id, q1, q2, param, transpose_flag] -- reconstructed into a gate
   matrix INSIDE the compiled kernel via _mps_1q_matrix/_mps_2q_matrix.
   A fused gate is an arbitrary matrix, not one of gates.py's known IDs,
   so it cannot be expressed in that encoding at all. This version
   applies fusion AFTER _compile_mps_ops has already run (reusing the
   real, already-correct production function -- CCX decomposition and
   SWAP-chain expansion for non-adjacent gates happen there, unchanged),
   converting its g_id rows into concrete matrices via the same real
   _mps_1q_matrix/_mps_2q_matrix production helpers, fusing those, then
   feeding the compiled kernel raw matrices directly -- removing the
   gate-ID switch from inside the kernel entirely, not just adding to it.

2. Untested interactions: the first version only ever exercised adjacent
   gates on a simple TFIM circuit. Fusion operates on _compile_mps_ops's
   OWN output rows, which are already SWAP-chain-expanded for
   non-adjacent gates and CCX-decomposed -- so fusion should compose
   correctly by construction (SWAP is just another 2-qubit gate,
   fusable like any other), but "should" isn't good enough here. Tested
   below on a circuit with a genuine non-adjacent gate and a CCX.

3. Bookkeeping granularity: fusing multiple original 2-qubit gates into
   one step means self._bond_history/jsd_per_bond/truncation_errors/
   entanglement_entropy get FEWER entries than the eager path produces
   for the same circuit -- a real, observable behavior change, not just
   an internal speed optimization. Decided: opt-in via a flag (e.g.
   run_circuit_jit(ops, fuse_gates=True), default False) -- the current
   per-original-gate bookkeeping stays the default, unchanged behavior;
   fuse_gates=True trades finer-grained diagnostics for speed, with
   bookkeeping populated at fused-step granularity instead (still real,
   still meaningful -- fewer, coarser entries, not missing ones).
   build_fused_matrix_runner below returns the full (chi_new, jsd_val,
   trunc_err, entanglement_entropy) diag tuple per fused step, same
   shape of information run_circuit_jit already produces, just one
   entry per fused block instead of per original gate.

Does NOT touch dense_evolution -- reuses the real installed package's
_compile_mps_ops, _mps_1q_matrix, _mps_2q_matrix, _vectorized_chi_search_jax,
_pad_gamma, _pad_lambda, _bucket_sizes, and MPSSimulator directly.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dense_evolution.backends.mps import (
    MPSSimulator,
    _compile_mps_ops,
    _mps_1q_matrix,
    _mps_2q_matrix,
    _vectorized_chi_search_jax,
    _pad_gamma,
    _pad_lambda,
    _bucket_sizes,
)


def _real_dtype(dtype):
    return jnp.float64 if dtype == jnp.complex128 else jnp.float32


def _embed_1q(mat1, qubit, pair):
    """Embeds a 2x2 single-qubit matrix into the 4x4 space of a 2-qubit
    pair (a, b), acting as identity on the other qubit."""
    eye2 = np.eye(2, dtype=mat1.dtype)
    a, b = pair
    return np.kron(mat1, eye2) if qubit == a else np.kron(eye2, mat1)


def compile_and_fuse(ops, n_qubits, dtype):
    """Runs the REAL, unmodified _compile_mps_ops first (CCX decomposition,
    SWAP-chain expansion for non-adjacent gates, GATE_IDS lookup -- all
    exactly as run_circuit_jit already does it), then fuses the resulting
    rows: a chain of consecutive gates acting on the same, or a growing,
    qubit pair gets folded into one matrix via host-side matrix
    multiplication (exact). transpose_flag is resolved into the matrix
    itself here (transposing _mps_2q_matrix's output before fusing), so
    the fused representation never needs it downstream.

    Returns a list of ('1q', q, matrix_2x2) / ('2q', q1, q2, matrix_4x4)
    entries, same tagged format as the first gate-blocking prototype."""
    rows = _compile_mps_ops(ops, n_qubits)
    fused = []
    i = 0
    n = len(rows)

    def row_matrix(row):
        g_id, q1, q2, param, transpose_flag = row
        g_id_int = jnp.asarray(g_id).astype(jnp.int32)
        if g_id >= 20:
            mat4 = np.asarray(_mps_2q_matrix(g_id_int, jnp.asarray(param), dtype))
            if transpose_flag > 0.5:
                mat4 = np.transpose(mat4, (1, 0, 3, 2))
            return mat4.reshape(4, 4), (int(q1), int(q2))
        return np.asarray(_mps_1q_matrix(g_id_int, jnp.asarray(param), dtype)), (int(q1),)

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
                mat = _embed_1q(mat, active_qubit, pair)
                mat = nmat @ mat
                j += 1
                continue
            if n_is_2q:
                if nqubits != pair:
                    break
            else:
                if nqubits[0] not in pair:
                    break
                nmat = _embed_1q(nmat, nqubits[0], pair)
            mat = nmat @ mat
            j += 1
        if pair is not None:
            fused.append(('2q', pair[0], pair[1], mat.reshape(2, 2, 2, 2)))
        else:
            fused.append(('1q', active_qubit, mat))
        i = j
    return fused


def _fused_to_arrays(fused, dtype):
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


def build_fused_matrix_runner(n_qubits, max_bond, eps, jsd_budget, dtype):
    """No gate-ID switch inside the compiled kernel at all -- every step
    receives its matrix directly (precomputed and fused on the host by
    compile_and_fuse), removing _mps_1q_matrix/_mps_2q_matrix calls from
    the traced program entirely, not just adding a bucket-size switch on
    top of them."""
    buckets = _bucket_sizes(max_bond)
    bucket_arr = jnp.array(buckets)

    def step(carry, xs):
        gammas, lambdas, real_chi = carry
        is_2q, q1, q2, mat2q, mat1q = xs
        q1 = q1.astype(jnp.int32)
        q2 = q2.astype(jnp.int32)

        real_dtype = _real_dtype(dtype)

        def branch_1q(c):
            gammas_, lambdas_, real_chi_ = c
            new_g = jnp.einsum('ij,ljr->lir', mat1q, gammas_[q1])
            new_carry = gammas_.at[q1].set(new_g), lambdas_, real_chi_
            diag = (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0, dtype=real_dtype),
                    jnp.asarray(0.0, dtype=real_dtype), jnp.asarray(0.0, dtype=real_dtype))
            return new_carry, diag

        def branch_2q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_2q = mat2q
            chi_l_real = real_chi_[q1]
            chi_m_real = real_chi_[q2]
            chi_r_real = real_chi_[q2 + 1]
            bound = jnp.maximum(
                jnp.maximum(jnp.maximum(chi_l_real, chi_r_real), chi_m_real),
                jnp.minimum(chi_l_real * 2, chi_r_real * 2),
            )
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
                    S_kept_masked = jnp.where(col_mask, S[:B], 0.0)
                    kept_norm = jnp.sqrt(jnp.sum(S_kept_masked ** 2) + 1e-30)
                    S_fixed = jnp.where(col_mask, S_kept_masked / (kept_norm + 1e-30), 0.0)
                    lam_l_inv = jnp.where(lam_l > eps, 1.0 / lam_l, 0.0)
                    lam_r_inv = jnp.where(lam_r > eps, 1.0 / lam_r, 0.0)
                    U_masked = jnp.where(col_mask[None, :], U[:, :B], 0.0)
                    Vh_masked = jnp.where(col_mask[:, None], Vh[:B, :], 0.0)
                    new_g1 = jnp.einsum('l,lir->lir', lam_l_inv, U_masked.reshape(B, 2, B))
                    new_g2 = jnp.einsum('ljr,r->ljr', Vh_masked.reshape(B, 2, B), lam_r_inv)

                    norm_full = jnp.sqrt(jnp.sum(S ** 2) + 1e-30)
                    S_norm_full = S / (norm_full + 1e-30)
                    trunc_err = jnp.sqrt(jnp.sum(jnp.where(jnp.arange(2 * B) >= chi_new, S_norm_full ** 2, 0.0)))
                    p_dist = S_fixed ** 2
                    ee = -jnp.sum(jnp.where(p_dist > 1e-20, p_dist * jnp.log2(jnp.where(p_dist > 1e-20, p_dist, 1.0)), 0.0))

                    return (_pad_gamma(new_g1, max_bond), _pad_gamma(new_g2, max_bond),
                            _pad_lambda(S_fixed, max_bond), chi_new.astype(jnp.int32),
                            jsd_val.astype(real_dtype), trunc_err.astype(real_dtype), ee.astype(real_dtype))
                return branch_fn

            branches = [make_branch(B) for B in buckets]
            new_g1_p, new_g2_p, S_fixed_p, chi_new, jsd_val, trunc_err, ee = jax.lax.switch(bucket_idx, branches, operand=None)
            new_gammas = gammas_.at[q1].set(new_g1_p).at[q2].set(new_g2_p)
            new_lambdas = lambdas_.at[q2].set(S_fixed_p)
            new_real_chi = real_chi_.at[q2].set(chi_new)
            new_carry = new_gammas, new_lambdas, new_real_chi
            diag = (chi_new, jsd_val, trunc_err, ee)
            return new_carry, diag

        new_carry, diag = jax.lax.cond(is_2q, branch_2q, branch_1q, carry)
        return new_carry, diag

    @jax.jit
    def run(gammas, lambdas, real_chi, xs):
        (fg, fl, fc), diag_hist = jax.lax.scan(step, (gammas, lambdas, real_chi), xs)
        return fg, fl, fc, diag_hist

    return run


def compare(n_qubits, max_bond, eps, jsd_budget, dtype, ops, label):
    """Compares against the real eager MPSSimulator, driven by run_circuit_jit
    would-be-equivalent ops (name-based tuples), for genuine end-to-end
    coverage including CCX/non-adjacent expansion, not the hand-built
    (g_id, q1, q2, param) tuples the first prototype used."""
    ref_sim = MPSSimulator(n_qubits, max_bond=max_bond, svd_cutoff=eps, jsd_budget=jsd_budget,
                            use_float32=(dtype == jnp.complex64))
    ref_sim.run_circuit_jit(ops)
    ref_sv = np.asarray(ref_sim.contract_to_statevector())

    fused = compile_and_fuse(ops, n_qubits, dtype)
    xs = _fused_to_arrays(fused, dtype)
    run = build_fused_matrix_runner(n_qubits, max_bond, eps, jsd_budget, dtype)
    gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=dtype).at[0, 0, 0].set(1.0), max_bond) for _ in range(n_qubits)])
    lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=_real_dtype(dtype)), max_bond) for _ in range(n_qubits + 1)])
    real_chi0 = jnp.ones(n_qubits + 1, dtype=jnp.int32)
    gammas_f, lambdas_f, real_chi_f, diag_hist = run(gammas0, lambdas0, real_chi0, xs)

    blocked_sim = MPSSimulator(n_qubits, max_bond=max_bond, svd_cutoff=eps, jsd_budget=jsd_budget,
                                use_float32=(dtype == jnp.complex64))
    blocked_sim.gammas = [gammas_f[i, : int(real_chi_f[i]), :, : int(real_chi_f[i + 1])] for i in range(n_qubits)]
    blocked_sim.lambdas = [lambdas_f[i, : int(real_chi_f[i])] for i in range(n_qubits + 1)]
    blocked_sv = np.asarray(blocked_sim.contract_to_statevector())

    fidelity = float(np.abs(np.vdot(ref_sv, blocked_sv)) ** 2)
    print(f"[{label}] n_gates_compiled={len(_compile_mps_ops(ops, n_qubits))} n_fused_steps={len(fused)}")
    print(f"[{label}] fidelity(ref, blocked) = {fidelity:.12f}")
    print(f"[{label}] max_bond_used: ref={ref_sim.max_bond_used()} blocked={int(np.max(real_chi_f))}")
    return fidelity


def _tfim_ops(n, dt, steps, J, g):
    theta_zz, theta_x = -2.0 * dt * J, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [("cx", i, i + 1), ("rz", i + 1, theta_zz), ("cx", i, i + 1)]
        for i in range(n):
            ops.append(("rx", i, theta_x))
    return ops


if __name__ == "__main__":
    f1 = compare(6, 16, 1e-12, 1e-5, jnp.complex128, _tfim_ops(6, 0.1, 3, 1.0, 1.0), "TFIM regression, N=6")

    nonadjacent_ops = [("h", 0), ("h", 2), ("cx", 4, 0), ("cx", 3, 1), ("cx", 0, 4), ("rz", 2, 0.7)]
    f2 = compare(5, 16, 1e-12, 1e-5, jnp.complex128, nonadjacent_ops, "non-adjacent CX (SWAP-chain expansion)")

    ccx_ops = [("x", 0), ("x", 1), ("ccx", 0, 1, 2), ("h", 3), ("cx", 2, 3)]
    f3 = compare(4, 16, 1e-12, 1e-5, jnp.complex128, ccx_ops, "CCX decomposition")

    all_ok = all(f > 1 - 1e-8 for f in (f1, f2, f3))
    print(f"\nALL FIDELITIES ACCEPTABLE: {all_ok}")
    assert all_ok, "fused-matrix runner (via real _compile_mps_ops) disagrees with the eager reference"
