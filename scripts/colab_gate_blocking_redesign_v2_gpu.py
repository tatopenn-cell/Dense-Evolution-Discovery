# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install --no-deps -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# GPU re-verification of the redesigned gate-blocking pipeline
# (mps_gate_blocking_redesign_v2.py): fusion now runs AFTER the real
# _compile_mps_ops (so it inherits correct CCX/SWAP-chain handling for
# free, verified on CPU already), feeds the compiled kernel raw fused
# matrices directly (no gate-ID switch inside the traced program at
# all), and carries the full (chi_new, jsd_val, trunc_err, entropy) diag
# tuple per fused step -- the shape of information run_circuit_jit
# already produces, just coarser-grained when fusion is used (decided:
# this will be an opt-in flag, not the default, when promoted).
#
# Same rigorous apples-to-apples methodology already established: all
# three variants (original fixed-size SVD, bucketed-only, bucketed+
# fused) measured through the identical internal-scan code path, so the
# resulting ratios are genuinely comparable -- not mixing run_circuit_jit
# overhead into only one side of the comparison.
#
# The circuit, in plain terms: a Trotterized Transverse Field Ising
# Model (TFIM) chain -- N qubits, each entangled with its neighbor (cx),
# given a small phase kick (rz) sized by J, then a rotation (rx) sized
# by g, repeated STEPS times.
PRECISION = "complex128"  # or "complex64"

import time
import jax
import jax.numpy as jnp
import numpy as np

if PRECISION == "complex128":
    jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

from dense_evolution.backends.mps import (
    _compile_mps_ops, _mps_1q_matrix, _mps_2q_matrix, _vectorized_chi_search_jax,
    _pad_gamma, _pad_lambda, _bucket_sizes,
)

N, DT, STEPS, J, G, MAX_BOND = 50, 0.1, 5, 1.0, 1.0, 64
DTYPE = jnp.complex128 if PRECISION == "complex128" else jnp.complex64
REAL_DTYPE = jnp.float64 if PRECISION == "complex128" else jnp.float32
EPS = 1e-12 if PRECISION == "complex128" else 1e-6
JSD_BUDGET = 1e-5
print(f"PRECISION={PRECISION} N={N} STEPS={STEPS} max_bond={MAX_BOND}")

theta_zz = -2.0 * DT * J
theta_x = -2.0 * DT * G
ops = []
for _ in range(STEPS):
    for i in range(N - 1):
        ops += [("cx", i, i + 1), ("rz", i + 1, theta_zz), ("cx", i, i + 1)]
    for i in range(N):
        ops.append(("rx", i, theta_x))

rows = _compile_mps_ops(ops, N)
print(f"n_gates_compiled={len(rows)}")


def _embed_1q(mat1, qubit, pair):
    eye2 = np.eye(2, dtype=mat1.dtype)
    a, b = pair
    return np.kron(mat1, eye2) if qubit == a else np.kron(eye2, mat1)


_MATRIX_CACHE = {}


def row_matrix(row, dtype):
    g_id, q1, q2, param, transpose_flag = row
    key = (id(dtype), g_id, param, transpose_flag)
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached, (int(q1), int(q2)) if g_id >= 20 else (int(q1),)
    g_id_int = jnp.asarray(g_id).astype(jnp.int32)
    if g_id >= 20:
        mat4 = np.asarray(_mps_2q_matrix(g_id_int, jnp.asarray(param), dtype))
        if transpose_flag > 0.5:
            mat4 = np.transpose(mat4, (1, 0, 3, 2))
        mat = mat4.reshape(4, 4)
        qubits = (int(q1), int(q2))
    else:
        mat = np.asarray(_mps_1q_matrix(g_id_int, jnp.asarray(param), dtype))
        qubits = (int(q1),)
    _MATRIX_CACHE[key] = mat
    return mat, qubits


def compile_and_fuse(rows, dtype):
    fused = []
    i = 0
    n = len(rows)
    while i < n:
        mat, qubits = row_matrix(rows[i], dtype)
        pair = qubits if len(qubits) == 2 else None
        active_qubit = qubits[0]
        j = i + 1
        while j < n:
            nmat, nqubits = row_matrix(rows[j], dtype)
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


def build_original_runner(n_qubits, max_bond, eps, jsd_budget, dtype):
    def step(carry, row):
        gammas, lambdas = carry
        g_id = row[0].astype(jnp.int32); q1 = row[1].astype(jnp.int32); q2 = row[2].astype(jnp.int32); param = row[3]

        def branch_1q(c):
            gammas_, lambdas_ = c
            gate_1q = _mps_1q_matrix(g_id, param, dtype)
            new_g = jnp.einsum('ij,ljr->lir', gate_1q, gammas_[q1])
            return gammas_.at[q1].set(new_g), lambdas_

        def branch_2q(c):
            gammas_, lambdas_ = c
            gate_2q = _mps_2q_matrix(g_id, param, dtype)
            g1 = gammas_[q1]; g2 = gammas_[q2]
            lam_l = lambdas_[q1]; lam_m = lambdas_[q2]; lam_r = lambdas_[q2 + 1]
            theta = jnp.einsum('l,lik,k,kjr,r->lijr', lam_l, g1, lam_m, g2, lam_r)
            theta_new = jnp.einsum('abcd,ecdf->eabf', gate_2q, theta)
            theta_mat = theta_new.reshape(max_bond * 2, 2 * max_bond)
            U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
            chi_new, jsd_val = _vectorized_chi_search_jax(S, eps, jsd_budget, max_bond)
            col_mask = jnp.arange(max_bond) < chi_new
            S_kept_masked = jnp.where(col_mask, S[:max_bond], 0.0)
            kept_norm = jnp.sqrt(jnp.sum(S_kept_masked ** 2) + 1e-30)
            S_fixed = jnp.where(col_mask, S_kept_masked / (kept_norm + 1e-30), 0.0)
            lam_l_inv = jnp.where(lam_l > eps, 1.0 / lam_l, 0.0)
            lam_r_inv = jnp.where(lam_r > eps, 1.0 / lam_r, 0.0)
            U_masked = jnp.where(col_mask[None, :], U[:, :max_bond], 0.0)
            Vh_masked = jnp.where(col_mask[:, None], Vh[:max_bond, :], 0.0)
            new_g1 = jnp.einsum('l,lir->lir', lam_l_inv, U_masked.reshape(max_bond, 2, max_bond))
            new_g2 = jnp.einsum('ljr,r->ljr', Vh_masked.reshape(max_bond, 2, max_bond), lam_r_inv)
            return gammas_.at[q1].set(new_g1).at[q2].set(new_g2), lambdas_.at[q2].set(S_fixed)

        is_2q = g_id >= 20
        return jax.lax.cond(is_2q, branch_2q, branch_1q, carry), None

    @jax.jit
    def run(gammas, lambdas, ops_arr):
        (fg, fl), _ = jax.lax.scan(step, (gammas, lambdas), ops_arr)
        return fg, fl

    return run


def build_fused_matrix_runner(n_qubits, max_bond, eps, jsd_budget, dtype):
    buckets = _bucket_sizes(max_bond)
    bucket_arr = jnp.array(buckets)
    real_dtype = REAL_DTYPE

    def step(carry, xs):
        gammas, lambdas, real_chi = carry
        is_2q, q1, q2, mat2q, mat1q = xs
        q1 = q1.astype(jnp.int32); q2 = q2.astype(jnp.int32)

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
            chi_l_real = real_chi_[q1]; chi_m_real = real_chi_[q2]; chi_r_real = real_chi_[q2 + 1]
            bound = jnp.maximum(jnp.maximum(jnp.maximum(chi_l_real, chi_r_real), chi_m_real),
                                 jnp.minimum(chi_l_real * 2, chi_r_real * 2))
            ge_mask = bucket_arr >= bound
            bucket_idx = jnp.where(jnp.any(ge_mask), jnp.argmax(ge_mask), len(buckets) - 1)

            def make_branch(B):
                def branch_fn(_):
                    g1 = gammas_[q1][:B, :, :B]; g2 = gammas_[q2][:B, :, :B]
                    lam_l = lambdas_[q1][:B]; lam_m = lambdas_[q2][:B]; lam_r = lambdas_[q2 + 1][:B]
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
                    return (_pad_gamma(new_g1, max_bond), _pad_gamma(new_g2, max_bond), _pad_lambda(S_fixed, max_bond),
                            chi_new.astype(jnp.int32), jsd_val.astype(real_dtype), trunc_err.astype(real_dtype), ee.astype(real_dtype))
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


fused = compile_and_fuse(rows, DTYPE)
print(f"n_fused_steps={len(fused)} (reduction {len(rows)/len(fused):.2f}x)")

ops_rows = jnp.asarray([(r[0], r[1], r[2], r[3]) for r in rows], dtype=jnp.float64)
unfused_entries = []
for r in rows:
    mat, qubits = row_matrix(r, DTYPE)
    if len(qubits) == 2:
        unfused_entries.append(('2q', qubits[0], qubits[1], mat.reshape(2, 2, 2, 2)))
    else:
        unfused_entries.append(('1q', qubits[0], mat))
xs_unfused = _fused_to_arrays(unfused_entries, DTYPE)
xs_fused = _fused_to_arrays(fused, DTYPE)

gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=DTYPE).at[0, 0, 0].set(1.0), MAX_BOND) for _ in range(N)])
lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=REAL_DTYPE), MAX_BOND) for _ in range(N + 1)])
real_chi0 = jnp.ones(N + 1, dtype=jnp.int32)

run_orig = build_original_runner(N, MAX_BOND, EPS, JSD_BUDGET, DTYPE)
jax.block_until_ready(run_orig(gammas0, lambdas0, ops_rows))
t0 = time.perf_counter(); out = run_orig(gammas0, lambdas0, ops_rows); jax.block_until_ready(out)
t_orig = time.perf_counter() - t0
print(f"1. ORIGINAL (always fixed-size SVD): warm={t_orig:.3f}s")

run_bucketed = build_fused_matrix_runner(N, MAX_BOND, EPS, JSD_BUDGET, DTYPE)
jax.block_until_ready(run_bucketed(gammas0, lambdas0, real_chi0, xs_unfused))
t0 = time.perf_counter(); out_b = run_bucketed(gammas0, lambdas0, real_chi0, xs_unfused); jax.block_until_ready(out_b)
t_bucketed = time.perf_counter() - t0
print(f"2. BUCKETED ONLY (matrix-based runner, no fusion): warm={t_bucketed:.3f}s max_bond_used={int(out_b[2].max())}")

jax.block_until_ready(run_bucketed(gammas0, lambdas0, real_chi0, xs_fused))
t0 = time.perf_counter(); out_f = run_bucketed(gammas0, lambdas0, real_chi0, xs_fused); jax.block_until_ready(out_f)
t_fused = time.perf_counter() - t0
print(f"3. BUCKETED + GATE BLOCKED (redesign v2, real _compile_mps_ops + full diag): warm={t_fused:.3f}s max_bond_used={int(out_f[2].max())}")

print(f"\n--- Same-methodology ratios ---")
print(f"bucketed-only vs original:           {t_orig/t_bucketed:.2f}x")
print(f"gate-blocking alone (over bucketed): {t_bucketed/t_fused:.2f}x")
print(f"TOTAL (fused vs original):           {t_orig/t_fused:.2f}x  <-- vs the 2x target")
