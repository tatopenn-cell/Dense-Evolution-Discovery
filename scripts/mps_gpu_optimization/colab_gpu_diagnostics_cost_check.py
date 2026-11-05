# Run on Google Colab with a GPU runtime selected BEFORE running anything:
# Runtime -> Change runtime type -> T4 GPU (free tier).
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# Tests one specific hypothesis for why run_circuit_jit's real GPU kernel
# looked much slower than the isolated bucketed-runner benchmark that
# validated this optimization: the real per-branch SVD step also
# computes trunc_err and entanglement entropy every 2-qubit gate (to
# populate self.truncation_errors/self.entanglement_entropy, existing,
# tested public bookkeeping) -- extra work the isolated benchmark's
# simplified reimplementation never computed at all. Result: REFUTED.
# With and without those diagnostics returned as real jax.lax.scan
# outputs (not just computed-and-discarded, which the first attempt at
# this check got wrong and had to be redone), timing was statistically
# identical (~2.44-2.47s both ways) -- see
# mps_bucketed_svd_gpu_timing_followup.md for the real cause (a flawed
# "fresh instance per timing" benchmark methodology, not this).
PRECISION = "complex128"  # or "complex64"

import jax
import jax.numpy as jnp

if PRECISION == "complex128":
    jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

import dense_evolution as de
from dense_evolution.backends.mps import (
    _mps_1q_matrix, _mps_2q_matrix, _vectorized_chi_search_jax,
    _pad_gamma, _pad_lambda, _bucket_sizes,
)

print("dense_evolution version:", de.__version__)

N, DT, STEPS, J, G, MAX_BOND = 50, 0.1, 5, 1.0, 1.0, 64
DTYPE = jnp.complex128 if PRECISION == "complex128" else jnp.complex64
REAL_DTYPE = jnp.float64 if PRECISION == "complex128" else jnp.float32
EPS = 1e-12 if PRECISION == "complex128" else 1e-6
JSD_BUDGET = 1e-5

buckets = _bucket_sizes(MAX_BOND)
bucket_arr = jnp.array(buckets)


def make_runner(with_diagnostics):
    def step(carry, row):
        gammas, lambdas, real_chi = carry
        g_id = row[0].astype(jnp.int32)
        q1 = row[1].astype(jnp.int32)
        q2 = row[2].astype(jnp.int32)
        param = row[3]
        transpose_flag = row[4] > 0.5

        def branch_1q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_1q = _mps_1q_matrix(g_id, param, DTYPE)
            new_g = jnp.einsum('ij,ljr->lir', gate_1q, gammas_[q1])
            new_carry = (gammas_.at[q1].set(new_g), lambdas_, real_chi_)
            if with_diagnostics:
                diag = (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0, dtype=REAL_DTYPE),
                        jnp.asarray(0.0, dtype=REAL_DTYPE), jnp.asarray(0.0, dtype=REAL_DTYPE))
            else:
                diag = jnp.asarray(0, dtype=jnp.int32)
            return new_carry, diag

        def branch_2q(c):
            gammas_, lambdas_, real_chi_ = c
            gate_2q = _mps_2q_matrix(g_id, param, DTYPE)
            gate_2q = jnp.where(transpose_flag, jnp.transpose(gate_2q, (1, 0, 3, 2)), gate_2q)
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
                    chi_new, jsd_val = _vectorized_chi_search_jax(S, EPS, JSD_BUDGET, min(B, MAX_BOND))
                    col_mask = jnp.arange(B) < chi_new
                    S_kept_masked = jnp.where(col_mask, S[:B], 0.0)
                    kept_norm = jnp.sqrt(jnp.sum(S_kept_masked ** 2) + 1e-30)
                    S_fixed = jnp.where(col_mask, S_kept_masked / (kept_norm + 1e-30), 0.0)
                    lam_l_inv = jnp.where(lam_l > EPS, 1.0 / lam_l, 0.0)
                    lam_r_inv = jnp.where(lam_r > EPS, 1.0 / lam_r, 0.0)
                    U_masked = jnp.where(col_mask[None, :], U[:, :B], 0.0)
                    Vh_masked = jnp.where(col_mask[:, None], Vh[:B, :], 0.0)
                    new_g1 = jnp.einsum('l,lir->lir', lam_l_inv, U_masked.reshape(B, 2, B))
                    new_g2 = jnp.einsum('ljr,r->ljr', Vh_masked.reshape(B, 2, B), lam_r_inv)

                    if with_diagnostics:
                        norm_full = jnp.sqrt(jnp.sum(S ** 2) + 1e-30)
                        S_norm_full = S / (norm_full + 1e-30)
                        trunc_err = jnp.sqrt(jnp.sum(jnp.where(jnp.arange(2 * B) >= chi_new, S_norm_full ** 2, 0.0)))
                        p_dist = S_fixed ** 2
                        ee = -jnp.sum(jnp.where(p_dist > 1e-20, p_dist * jnp.log2(jnp.where(p_dist > 1e-20, p_dist, 1.0)), 0.0))
                        return (_pad_gamma(new_g1, MAX_BOND), _pad_gamma(new_g2, MAX_BOND), _pad_lambda(S_fixed, MAX_BOND),
                                chi_new.astype(jnp.int32), jsd_val.astype(REAL_DTYPE),
                                trunc_err.astype(REAL_DTYPE), ee.astype(REAL_DTYPE))

                    return _pad_gamma(new_g1, MAX_BOND), _pad_gamma(new_g2, MAX_BOND), _pad_lambda(S_fixed, MAX_BOND), chi_new.astype(jnp.int32)

                return branch_fn

            branches = [make_branch(B) for B in buckets]
            if with_diagnostics:
                new_g1_p, new_g2_p, S_fixed_p, chi_new, jsd_val, trunc_err, ee = jax.lax.switch(bucket_idx, branches, operand=None)
            else:
                new_g1_p, new_g2_p, S_fixed_p, chi_new = jax.lax.switch(bucket_idx, branches, operand=None)
            new_gammas = gammas_.at[q1].set(new_g1_p).at[q2].set(new_g2_p)
            new_lambdas = lambdas_.at[q2].set(S_fixed_p)
            new_real_chi = real_chi_.at[q2].set(chi_new)
            new_carry = (new_gammas, new_lambdas, new_real_chi)
            diag = (chi_new, jsd_val, trunc_err, ee) if with_diagnostics else chi_new
            return new_carry, diag

        is_2q = g_id >= 20
        new_carry, diag = jax.lax.cond(is_2q, branch_2q, branch_1q, carry)
        return new_carry, diag

    @jax.jit
    def run(gammas, lambdas, real_chi, ops):
        (fg, fl, fc), diag_hist = jax.lax.scan(step, (gammas, lambdas, real_chi), ops)
        return fg, fl, fc, diag_hist

    return run


def trotter_ops_de(n, dt, steps, j, g):
    theta_zz, theta_x = -2.0 * dt * j, -2.0 * dt * g
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops += [(20.0, float(i), float(i + 1), 0.0, 0.0),
                    (11.0, float(i + 1), float(i + 1), theta_zz, 0.0),
                    (20.0, float(i), float(i + 1), 0.0, 0.0)]
        for i in range(n):
            ops.append((9.0, float(i), float(i), theta_x, 0.0))
    return ops


ops_rows = trotter_ops_de(N, DT, STEPS, J, G)
ops_jnp = jnp.asarray(ops_rows, dtype=jnp.float64)

gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=DTYPE).at[0, 0, 0].set(1.0), MAX_BOND) for _ in range(N)])
lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=REAL_DTYPE), MAX_BOND) for _ in range(N + 1)])
real_chi0 = jnp.ones(N + 1, dtype=jnp.int32)

for with_diag in (False, True):
    run = make_runner(with_diag)
    out = run(gammas0, lambdas0, real_chi0, ops_jnp)
    jax.block_until_ready(out)
    import time
    t0 = time.perf_counter()
    out = run(gammas0, lambdas0, real_chi0, ops_jnp)
    jax.block_until_ready(out)
    t_warm = time.perf_counter() - t0
    label = "WITH trunc_err+entropy (matches production)" if with_diag else "WITHOUT them (matches original Discovery benchmark)"
    print(f"{label}: warm={t_warm:.3f}s max_bond_used={int(out[2].max())}")
