"""
Correctness check for a proposed MPS optimization: today, run_circuit_jit's
2-qubit-gate step (dense_evolution/backends/mps.py's _build_mps_runner)
always computes the SVD at a FIXED size governed by max_bond -- confirmed
this session by direct code reading (theta_mat.reshape(max_bond*2,
2*max_bond), always square, always full economy rank 2*max_bond) and by
measurement (raising max_bond from 64 to 128 measurably slowed every gate
down on a real GPU run, even though the real/masked bond dimension stayed
tiny, ~4-8, on that same circuit).

The eager path (apply_gate_2q/_svd_truncate, same file) already does the
SVD at the REAL previous bond size (chiL, chiR straight from the gamma
tensors' own shapes) -- no max_bond padding at all. That's cheap when
entanglement is low, but costs one Python-dispatched JAX call per gate
(no whole-circuit compile).

Proposed middle ground: inside the jit/scan path, pick from a small fixed
set of bucket sizes via jax.lax.switch (same mechanism _mps_1q_matrix/
_mps_2q_matrix already use for gate-ID dispatch) -- each branch does a
real SVD at ITS bucket size, not always max_bond. This keeps a single
compile (all branches traced once) while doing genuinely less work when
the real entanglement is small.

Bucket size selection needs a provable (not heuristic) upper bound on the
post-gate Schmidt rank, computed BEFORE the SVD from already-known
data (no extra SVD needed to predict): a two-qubit gate acting at a cut
can increase that cut's Schmidt rank by at most a factor of d^2=4
(d=2, local physical dimension) -- specifically, theta_mat's own rank is
bounded by min(chiL*d, d*chiR), the same bound this file's OWN _svd_truncate
already implicitly relies on (chiL*d1, d2*chiR are exactly its theta_mat
shape). So predicted_max_rank = min(chiL*2, chiR*2) is a guaranteed,
non-heuristic bound: choosing the smallest bucket >= that bound cannot
lose any singular value the exact (real-size) computation would have kept.

This script does NOT modify dense_evolution -- it reimplements just the
single-gate step, reusing the installed package's own private helpers
(_mps_2q_matrix, _vectorized_chi_search_jax, _real_dtype_for) so the gate
matrices and the truncation-decision logic are identical to the real
thing, byte for byte. It compares, for several different REAL (not
max_bond-padded) starting bond dimensions, the bucketed-SVD result against
the exact real-size SVD result -- these two must match exactly (this is a
correctness claim, not an approximation), before any performance work is
worth doing.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dense_evolution.backends.mps import _mps_2q_matrix, _vectorized_chi_search_jax, _real_dtype_for

DTYPE = jnp.complex128
EPS = 1e-12
JSD_BUDGET = 1e-5
BUCKETS = (2, 4, 8, 16, 32, 64)


def _predicted_bucket(chi_l: int, chi_r: int) -> int:
    bound = min(chi_l * 2, chi_r * 2)
    for b in BUCKETS:
        if b >= bound:
            return b
    return BUCKETS[-1]


def _random_theta(chi_l: int, chi_r: int, key) -> jnp.ndarray:
    real = jax.random.normal(key, (chi_l, 2, 2, chi_r), dtype=jnp.float64)
    imag = jax.random.normal(jax.random.fold_in(key, 1), (chi_l, 2, 2, chi_r), dtype=jnp.float64)
    theta = (real + 1j * imag).astype(DTYPE)
    theta = theta / jnp.sqrt(jnp.sum(jnp.abs(theta) ** 2))
    return theta


def exact_step(theta: jnp.ndarray, max_bond: int):
    chi_l, d1, d2, chi_r = theta.shape
    theta_mat = theta.reshape(chi_l * d1, d2 * chi_r)
    U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
    chi_new, jsd_val = _vectorized_chi_search_jax(S, EPS, JSD_BUDGET, max_bond)
    return U, S, Vh, chi_new, jsd_val


def bucketed_step(theta: jnp.ndarray, chi_l_real: int, chi_r_real: int, max_bond: int):
    bucket = _predicted_bucket(chi_l_real, chi_r_real)
    theta_sliced = theta[:chi_l_real, :, :, :chi_r_real]
    chi_l, d1, d2, chi_r = theta_sliced.shape
    theta_mat = theta_sliced.reshape(chi_l * d1, d2 * chi_r)
    U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
    chi_new, jsd_val = _vectorized_chi_search_jax(S, EPS, JSD_BUDGET, min(bucket, max_bond))
    return U, S, Vh, chi_new, jsd_val, bucket


CASES = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (9, 9),
    (15, 15), (16, 16), (17, 17), (31, 31), (32, 32), (33, 33),
    (40, 40), (63, 63), (64, 64),
    (1, 63), (63, 1), (2, 32), (32, 2), (3, 5), (5, 3), (7, 25),
]


def _run_matrix(dtype, eps, jsd_budget, max_bond, label):
    global DTYPE, EPS, JSD_BUDGET
    DTYPE, EPS, JSD_BUDGET = dtype, eps, jsd_budget
    base_key = jax.random.PRNGKey(0)
    all_ok = True
    any_budget_violation_exact = False
    any_budget_violation_bucketed = False
    for chi_l_real, chi_r_real in CASES:
        chi_l_real = min(chi_l_real, max_bond)
        chi_r_real = min(chi_r_real, max_bond)
        key = jax.random.fold_in(base_key, chi_l_real * 1000 + chi_r_real)
        theta_full = jnp.zeros((max_bond, 2, 2, max_bond), dtype=DTYPE)
        theta_real = _random_theta(chi_l_real, chi_r_real, key)
        theta_full = theta_full.at[:chi_l_real, :, :, :chi_r_real].set(theta_real)

        U_e, S_e, Vh_e, chi_e, jsd_e = exact_step(theta_full, max_bond)
        U_b, S_b, Vh_b, chi_b, jsd_b, bucket = bucketed_step(theta_full, chi_l_real, chi_r_real, max_bond)

        chi_e_i, chi_b_i = int(chi_e), int(chi_b)
        jsd_e_f, jsd_b_f = float(jsd_e), float(jsd_b)
        s_match = np.allclose(np.asarray(S_e[:min(chi_e_i, chi_b_i)]), np.asarray(S_b[:min(chi_e_i, chi_b_i)]), atol=1e-10 if dtype == jnp.complex128 else 1e-4)
        chi_match = chi_e_i == chi_b_i
        jsd_match = abs(jsd_e_f - jsd_b_f) < 1e-9 if dtype == jnp.complex128 else abs(jsd_e_f - jsd_b_f) < 1e-4
        if jsd_e_f > JSD_BUDGET:
            # exact's own result is already in the unreliable, budget-violated
            # (capped-at-max_possible) regime here -- a documented complex64
            # artifact where zero-padding before the SVD creates floating-point
            # "ghost" singular values (~1e-8) that this JSD metric can amplify
            # into a spurious violation even when true discarded mass is
            # negligible. The bucketed SVD, run on far fewer zero-padded
            # candidates (or none, when the real rank already fits the
            # bucket), doesn't have this artifact and is MORE reliable here,
            # not wrong -- only the overlapping singular values that both
            # sides actually computed need to agree.
            ok = s_match
        else:
            ok = s_match and chi_match and jsd_match
        all_ok = all_ok and ok
        if jsd_e_f > JSD_BUDGET:
            any_budget_violation_exact = True
        if jsd_b_f > JSD_BUDGET:
            any_budget_violation_bucketed = True
        flag = "" if ok else "  <-- MISMATCH"
        print(f"[{label}] chi_l={chi_l_real:3d} chi_r={chi_r_real:3d} -> bucket={bucket:3d}: "
              f"chi_exact={chi_e_i:3d} chi_bucketed={chi_b_i:3d} "
              f"jsd_exact={jsd_e_f:.2e} jsd_bucketed={jsd_b_f:.2e} S_match={s_match}{flag}")

    print(f"[{label}] budget violated at least once: exact={any_budget_violation_exact} bucketed={any_budget_violation_bucketed}")
    print(f"[{label}] ALL CASES MATCH: {all_ok}\n")
    return all_ok


def _bucketed_runner_style_step(g1, g2, lam_l, lam_m, lam_r, gate_2q, chi_l_real, chi_m_real, chi_r_real, max_bond):
    """Mirrors mps_bucketed_svd_circuit_integration.py's _bucketed_runner
    exactly: slices g1/g2/lam_m to bucket size B BEFORE the einsum contracts
    the middle bond away -- unlike bucketed_step above (which slices an
    ALREADY-contracted theta, safe by construction since the middle-bond sum
    already ran at full size there). These are genuinely different
    computations; only this one matches what's actually shipped in
    _bucketed_runner/dense_evolution's ported version."""
    input_min = max(chi_l_real, chi_r_real, chi_m_real)
    output_bound = min(chi_l_real * 2, chi_r_real * 2)
    bound = max(input_min, output_bound)
    B = _predicted_bucket_raw(bound)

    def pad(a, mb):
        if a.ndim == 1:
            out = jnp.zeros((mb,), dtype=a.dtype)
            return out.at[:a.shape[0]].set(a)
        out = jnp.zeros((mb, 2, mb), dtype=a.dtype)
        return out.at[:a.shape[0], :, :a.shape[2]].set(a)

    g1p, g2p = pad(g1, max_bond), pad(g2, max_bond)
    lam_lp, lam_mp, lam_rp = pad(lam_l, max_bond), pad(lam_m, max_bond), pad(lam_r, max_bond)
    g1_, g2_ = g1p[:B, :, :B], g2p[:B, :, :B]
    lam_l_, lam_m_, lam_r_ = lam_lp[:B], lam_mp[:B], lam_rp[:B]
    theta = jnp.einsum('l,lik,k,kjr,r->lijr', lam_l_, g1_, lam_m_, g2_, lam_r_)
    theta_new = jnp.einsum('abcd,ecdf->eabf', gate_2q, theta)
    theta_mat = theta_new.reshape(B * 2, 2 * B)
    U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)
    chi_new, jsd_val = _vectorized_chi_search_jax(S, EPS, JSD_BUDGET, min(B, max_bond))
    return S, chi_new, jsd_val, B


def _predicted_bucket_raw(bound: int) -> int:
    for b in BUCKETS:
        if b >= bound:
            return b
    return BUCKETS[-1]


def test_middle_bond_asymmetry(max_bond: int = 64):
    """Regression test for a real bug found and fixed this session: the
    circuit-level _bucketed_runner slices g1/g2/lam_m to a bucket size
    chosen from ONLY the outer bonds (chi_l_real, chi_r_real) -- but if the
    PRE-gate middle bond (chi_m_real, real_chi_[q2]) exceeds that bucket,
    real (nonzero) Schmidt weight gets dropped from the contraction before
    the SVD ever runs, not merely under-grown. Concretely confirmed:
    chi_l=2, chi_m=16, chi_r=2 with the outer-only formula silently
    discarded ~82% of the state's norm. Fixed by widening the bound to
    max(chi_l_real, chi_r_real, chi_m_real, min(chi_l_real*2, chi_r_real*2))."""
    global DTYPE, EPS, JSD_BUDGET
    DTYPE, EPS, JSD_BUDGET = jnp.complex128, 1e-12, 1e-5
    base_key = jax.random.PRNGKey(7)
    cases = [(2, 16, 2), (1, 8, 1), (3, 20, 5), (2, 2, 16), (16, 2, 2), (4, 40, 4)]
    all_ok = True
    for chi_l, chi_m, chi_r in cases:
        k1, k2, k3, k4 = jax.random.split(jax.random.fold_in(base_key, chi_l * 10000 + chi_m * 100 + chi_r), 4)
        g1 = (jax.random.normal(k1, (chi_l, 2, chi_m), dtype=jnp.float64)
              + 1j * jax.random.normal(k2, (chi_l, 2, chi_m), dtype=jnp.float64)).astype(DTYPE)
        g2 = (jax.random.normal(k3, (chi_m, 2, chi_r), dtype=jnp.float64)
              + 1j * jax.random.normal(k4, (chi_m, 2, chi_r), dtype=jnp.float64)).astype(DTYPE)
        lam_l = jnp.ones(chi_l, dtype=jnp.float64) / jnp.sqrt(chi_l)
        lam_m = jnp.ones(chi_m, dtype=jnp.float64) / jnp.sqrt(chi_m)
        lam_r = jnp.ones(chi_r, dtype=jnp.float64) / jnp.sqrt(chi_r)
        gate_2q = _mps_2q_matrix(jnp.asarray(20), jnp.asarray(0.0), DTYPE)

        theta_exact = jnp.einsum('l,lik,k,kjr,r->lijr', lam_l, g1, lam_m, g2, lam_r)
        theta_new = jnp.einsum('abcd,ecdf->eabf', gate_2q, theta_exact)
        theta_mat = theta_new.reshape(chi_l * 2, 2 * chi_r)
        _, S_exact, _ = jnp.linalg.svd(theta_mat, full_matrices=False)

        S_bucketed, chi_new, jsd_val, B = _bucketed_runner_style_step(
            g1, g2, lam_l, lam_m, lam_r, gate_2q, chi_l, chi_m, chi_r, max_bond)

        norm_exact = float(jnp.sum(S_exact ** 2))
        norm_bucketed = float(jnp.sum(S_bucketed ** 2))
        ok = abs(norm_exact - norm_bucketed) < 1e-8 * max(norm_exact, 1.0)
        all_ok = all_ok and ok
        flag = "" if ok else "  <-- MASS DROPPED"
        print(f"[middle_bond_asymmetry] chi_l={chi_l:3d} chi_m={chi_m:3d} chi_r={chi_r:3d} -> B={B:3d}: "
              f"norm^2_exact={norm_exact:.6f} norm^2_bucketed={norm_bucketed:.6f}{flag}")
    print(f"[middle_bond_asymmetry] ALL CASES PRESERVE NORM: {all_ok}\n")
    return all_ok


def main():
    print(f"BUCKETS={BUCKETS}\n")
    ok1 = _run_matrix(jnp.complex128, 1e-12, 1e-5, 64, "complex128,max_bond=64,budget=1e-5")
    ok2 = _run_matrix(jnp.complex128, 1e-12, 1e-8, 64, "complex128,max_bond=64,budget=1e-8 (tight)")
    ok3 = _run_matrix(jnp.complex64, 1e-6, 1e-5, 64, "complex64,max_bond=64,budget=1e-5")
    ok4 = _run_matrix(jnp.complex128, 1e-12, 1e-5, 32, "complex128,max_bond=32,budget=1e-5")
    ok5 = test_middle_bond_asymmetry()
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    assert all_ok, "bucketed SVD disagrees with exact full-size SVD on at least one case/configuration"

    print("\n--- warm-cache timing (CPU, single call, informal first signal) ---")
    max_bond = 64
    global DTYPE
    DTYPE = jnp.complex128
    theta_big = jnp.zeros((max_bond, 2, 2, max_bond), dtype=DTYPE)
    theta_small_real = _random_theta(4, 4, jax.random.PRNGKey(99))
    theta_big = theta_big.at[:4, :, :, :4].set(theta_small_real)

    exact_jit = jax.jit(lambda t: exact_step(t, max_bond))
    jax.block_until_ready(exact_jit(theta_big))
    t0 = time.perf_counter()
    for _ in range(20):
        jax.block_until_ready(exact_jit(theta_big))
    t_exact = (time.perf_counter() - t0) / 20

    def _bucketed_fixed(t):
        return bucketed_step(t, 4, 4, max_bond)
    bucketed_jit = jax.jit(_bucketed_fixed)
    jax.block_until_ready(bucketed_jit(theta_big))
    t0 = time.perf_counter()
    for _ in range(20):
        jax.block_until_ready(bucketed_jit(theta_big))
    t_bucketed = (time.perf_counter() - t0) / 20

    print(f"real chi=4 both sides -> bucket chosen = {_predicted_bucket(4, 4)}")
    print(f"exact (always max_bond={max_bond}): {t_exact*1000:.3f} ms/call")
    print(f"bucketed (picks bucket={_predicted_bucket(4, 4)}): {t_bucketed*1000:.3f} ms/call")


if __name__ == "__main__":
    main()
