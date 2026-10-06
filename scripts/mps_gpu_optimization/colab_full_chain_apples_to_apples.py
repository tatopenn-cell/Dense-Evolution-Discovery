# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install --no-deps -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# Fixes a real rigor mistake: an earlier "total speedup" claim divided
# the ORIGINAL baseline measured via the real run_circuit_jit API
# (colab_gpu_mps_fair_comparison_old_vs_new.py, 3.746s/3.820s) by the
# gate-blocked result measured via a DIFFERENT code path (the internal
# bucketed-switch reimplementation, colab_gate_blocking_gpu_benchmark.py,
# 1.185s) -- two different measurement methodologies stitched together,
# not a valid ratio. This script measures all THREE variants (original
# fixed-size SVD, bucketed-only, bucketed+gate-blocked) through the
# EXACT SAME internal-scan methodology (bypassing run_circuit_jit
# entirely for all three), so the resulting ratios are actually
# apples-to-apples.
#
# The circuit, in plain terms: a Trotterized Transverse Field Ising
# Model (TFIM) chain -- N qubits, each entangled with its neighbor (cx),
# given a small phase kick (rz) sized by the interaction strength J,
# then a small rotation (rx) sized by the transverse field g, repeated
# STEPS times. Same circuit family used throughout this repo.
#
# CPU sanity check already run (not GPU, no quota spent): original
# (fixed-size SVD, internal-scan) = 3.641s, bucketed-only (from
# mps_gate_blocking_experiment.py's own timing_check) = 0.065s,
# bucketed+fused = 0.071s (slightly slower, expected -- CPU doesn't pay
# per-step GPU launch overhead, so fusion has nothing to amortize there).
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
    _mps_1q_matrix, _mps_2q_matrix, _vectorized_chi_search_jax,
    _pad_gamma, _pad_lambda, _bucket_sizes,
)

N, DT, STEPS, J, G, MAX_BOND = 50, 0.1, 5, 1.0, 1.0, 64
DTYPE = jnp.complex128 if PRECISION == "complex128" else jnp.complex64
REAL_DTYPE = jnp.float64 if PRECISION == "complex128" else jnp.float32
EPS = 1e-12 if PRECISION == "complex128" else 1e-6
JSD_BUDGET = 1e-5
print(f"PRECISION={PRECISION} N={N} STEPS={STEPS} max_bond={MAX_BOND}")


def _tfim_gate_list(n, dt, steps, J, g):
    tzz, tx = -2.0 * dt * J, -2.0 * dt * g
    gates = []
    for _ in range(steps):
        for i in range(n - 1):
            gates.append((20, i, i + 1, 0.0))
            gates.append((11, i + 1, 0, tzz))
            gates.append((20, i, i + 1, 0.0))
        for i in range(n):
            gates.append((9, i, 0, tx))
    return gates


# ---------------------------------------------------------------------
# Variant 1: ORIGINAL -- always a full SVD at fixed max_bond*2 size,
# matching 8.1.75's own (pre-optimization) approach exactly, but reached
# here via the same bare internal-scan style as the other two variants
# (no run_circuit_jit, no QASM) so the comparison stays consistent.
# ---------------------------------------------------------------------
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
    def run(gammas, lambdas, ops):
        (fg, fl), _ = jax.lax.scan(step, (gammas, lambdas), ops)
        return fg, fl

    return run


# ---------------------------------------------------------------------
# Variants 2 & 3: bucketed-only and bucketed+gate-blocked, identical to
# colab_gate_blocking_gpu_benchmark.py.
# ---------------------------------------------------------------------
_MATRIX_CACHE = {}


def _matrix_1q(g_id, param, dtype):
    key = (id(dtype), g_id, param)
    if key not in _MATRIX_CACHE:
        _MATRIX_CACHE[key] = np.asarray(_mps_1q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype))
    return _MATRIX_CACHE[key]


def _matrix_2q(g_id, param, dtype):
    key = (id(dtype), g_id, param, "2q")
    if key not in _MATRIX_CACHE:
        _MATRIX_CACHE[key] = np.asarray(_mps_2q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype)).reshape(4, 4)
    return _MATRIX_CACHE[key]


def _embed_1q(mat1, qubit, pair):
    eye2 = np.eye(2, dtype=mat1.dtype)
    a, b = pair
    return np.kron(mat1, eye2) if qubit == a else np.kron(eye2, mat1)


def fuse_gate_list(gate_list, dtype):
    fused = []
    i = 0
    n = len(gate_list)
    while i < n:
        g_id, q1, q2, param = gate_list[i]
        is_2q = g_id >= 20
        if is_2q:
            mat = _matrix_2q(g_id, param, dtype); pair = (q1, q2)
        else:
            mat = _matrix_1q(g_id, param, dtype); pair = None
        active_qubit = q1
        j = i + 1
        while j < n:
            ng_id, nq1, nq2, nparam = gate_list[j]
            n_is_2q = ng_id >= 20
            if pair is None:
                if not n_is_2q:
                    if nq1 != active_qubit:
                        break
                    mat = _matrix_1q(ng_id, nparam, dtype) @ mat
                    j += 1
                    continue
                if active_qubit not in (nq1, nq2):
                    break
                pair = (nq1, nq2)
                mat = _embed_1q(mat, active_qubit, pair)
                mat = _matrix_2q(ng_id, nparam, dtype) @ mat
                j += 1
                continue
            if n_is_2q:
                if (nq1, nq2) != pair:
                    break
                nmat = _matrix_2q(ng_id, nparam, dtype)
            else:
                if nq1 not in pair:
                    break
                nmat = _embed_1q(_matrix_1q(ng_id, nparam, dtype), nq1, pair)
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


def build_blocked_runner(n_qubits, max_bond, eps, jsd_budget, dtype):
    buckets = _bucket_sizes(max_bond)
    bucket_arr = jnp.array(buckets)

    def step(carry, xs):
        gammas, lambdas, real_chi = carry
        is_2q, q1, q2, mat2q, mat1q = xs
        q1 = q1.astype(jnp.int32); q2 = q2.astype(jnp.int32)

        def branch_1q(c):
            gammas_, lambdas_, real_chi_ = c
            new_g = jnp.einsum('ij,ljr->lir', mat1q, gammas_[q1])
            return gammas_.at[q1].set(new_g), lambdas_, real_chi_

        def branch_2q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_2q = mat2q
            chi_l_real = real_chi_[q1]; chi_m_real = real_chi_[q2]; chi_r_real = real_chi_[q2 + 1]
            input_min = jnp.maximum(jnp.maximum(chi_l_real, chi_r_real), chi_m_real)
            output_bound = jnp.minimum(chi_l_real * 2, chi_r_real * 2)
            bound = jnp.maximum(input_min, output_bound)
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
                    return _pad_gamma(new_g1, max_bond), _pad_gamma(new_g2, max_bond), _pad_lambda(S_fixed, max_bond), chi_new.astype(jnp.int32)
                return branch_fn

            branches = [make_branch(B) for B in buckets]
            new_g1_p, new_g2_p, S_fixed_p, chi_new = jax.lax.switch(bucket_idx, branches, operand=None)
            new_gammas = gammas_.at[q1].set(new_g1_p).at[q2].set(new_g2_p)
            new_lambdas = lambdas_.at[q2].set(S_fixed_p)
            new_real_chi = real_chi_.at[q2].set(chi_new)
            return new_gammas, new_lambdas, new_real_chi

        new_carry = jax.lax.cond(is_2q, branch_2q, branch_1q, carry)
        return new_carry, None

    @jax.jit
    def run(gammas, lambdas, real_chi, xs):
        (fg, fl, fc), _ = jax.lax.scan(step, (gammas, lambdas, real_chi), xs)
        return fg, fl, fc

    return run


gate_list = _tfim_gate_list(N, DT, STEPS, J, G)
fused = fuse_gate_list(gate_list, DTYPE)
print(f"n_gates original=unfused={len(gate_list)} fused={len(fused)}")

ops_rows = jnp.asarray([(float(g), float(q1), float(q2), float(p)) for g, q1, q2, p in gate_list], dtype=jnp.float64)
xs_unfused = _fused_to_arrays(
    [('2q', q1, q2, _matrix_2q(g_id, param, DTYPE).reshape(2, 2, 2, 2)) if g_id >= 20 else
     ('1q', q1, _matrix_1q(g_id, param, DTYPE)) for g_id, q1, q2, param in gate_list], DTYPE)
xs_fused = _fused_to_arrays(fused, DTYPE)

gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=DTYPE).at[0, 0, 0].set(1.0), MAX_BOND) for _ in range(N)])
lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=REAL_DTYPE), MAX_BOND) for _ in range(N + 1)])
real_chi0 = jnp.ones(N + 1, dtype=jnp.int32)

run_orig = build_original_runner(N, MAX_BOND, EPS, JSD_BUDGET, DTYPE)
jax.block_until_ready(run_orig(gammas0, lambdas0, ops_rows))
t0 = time.perf_counter(); out = run_orig(gammas0, lambdas0, ops_rows); jax.block_until_ready(out)
t_orig = time.perf_counter() - t0
print(f"1. ORIGINAL (always fixed-size SVD, internal-scan): warm={t_orig:.3f}s")

run_bucketed = build_blocked_runner(N, MAX_BOND, EPS, JSD_BUDGET, DTYPE)
jax.block_until_ready(run_bucketed(gammas0, lambdas0, real_chi0, xs_unfused))
t0 = time.perf_counter(); out_b = run_bucketed(gammas0, lambdas0, real_chi0, xs_unfused); jax.block_until_ready(out_b)
t_bucketed = time.perf_counter() - t0
print(f"2. BUCKETED ONLY (internal-scan): warm={t_bucketed:.3f}s max_bond_used={int(out_b[2].max())}")

jax.block_until_ready(run_bucketed(gammas0, lambdas0, real_chi0, xs_fused))
t0 = time.perf_counter(); out_f = run_bucketed(gammas0, lambdas0, real_chi0, xs_fused); jax.block_until_ready(out_f)
t_fused = time.perf_counter() - t0
print(f"3. BUCKETED + GATE BLOCKED (internal-scan): warm={t_fused:.3f}s max_bond_used={int(out_f[2].max())}")

print(f"\n--- Same-methodology ratios (all three via internal-scan, no run_circuit_jit) ---")
print(f"bucketed-only vs original:        {t_orig/t_bucketed:.2f}x")
print(f"gate-blocking alone (over bucketed): {t_bucketed/t_fused:.2f}x")
print(f"TOTAL (fused vs original):         {t_orig/t_fused:.2f}x  <-- the real, honest number vs the 2x target")
