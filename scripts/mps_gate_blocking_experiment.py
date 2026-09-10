"""
Gate blocking (gate fusion) for the bucketed-SVD MPS runner -- attempting
to close the gap between the confirmed real GPU speedup (~1.37-1.40x,
see mps_bucketed_svd_gpu_timing_followup.md, measured correctly via a
fresh |0...0> instance sharing an already-compiled runner) and the 2x
target.

Two papers found via quantumrag's tensor_networks collection motivate
this specific technique (not a random guess):

  - arXiv:2212.09782 (QR-based fast TEBD): SVD truncation is empirically
    SLOWER on GPU than CPU in some regimes -- cuSOLVER's SVD appears to
    carry disproportionate per-call overhead on GPU, independent of size,
    consistent with this session's own HLO finding that per-step kernel-
    launch overhead (not the switch/dispatch mechanism itself) dominates.
  - arXiv:2511.23438 (2D TFIM MPS heuristic, same physical circuit family
    as this repo's own TFIM benchmark): explicitly uses "site blocking" --
    grouping sites/gates together BEFORE truncation reduces the NUMBER of
    SVDs needed per time step, calling this out as a real efficiency gain
    for TEBD.

This experiment fuses each CX-RZ-CX triple (three consecutive gates that
all act on the exact same 2-qubit pair, with nothing else touching that
pair in between) into ONE 4x4 unitary matrix, computed once on the host
via plain matrix multiplication (exact -- unitary composition, not an
approximation) BEFORE the JIT-compiled scan ever runs. For this repo's
own N=50/STEPS=5 TFIM Trotter benchmark, this cuts the 2-qubit-gate
scan-step count from 3 gates/bond to 1 gate/bond -- 985 total gate/scan
steps down to 495 (49 fused 2q-gates + 50 1q RX gates, per layer, x5).

Does NOT touch dense_evolution -- reuses the real installed package's own
private helpers (_mps_1q_matrix, _mps_2q_matrix, _vectorized_chi_search_jax,
_pad_gamma, _pad_lambda, _bucket_sizes) for the gate matrices and the
truncation logic, same discipline as every other experiment in this
directory.
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
    _bucket_sizes,
)


def _real_dtype(dtype):
    return jnp.float64 if dtype == jnp.complex128 else jnp.float32


def _tfim_gate_list(n, dt, steps, J, g):
    """Same convention as mps_bucketed_svd_circuit_integration.py's own
    _tfim_gate_list: (g_id, q1, q2, param) tuples, g_id>=20 for 2-qubit."""
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


def _embed_1q(mat1, qubit, pair):
    """Embeds a 2x2 single-qubit matrix into the 4x4 space of a 2-qubit
    pair (a, b), acting as identity on the other qubit -- kron(mat1, I2)
    if qubit is the pair's first (more significant) qubit, kron(I2, mat1)
    if it's the second, matching _mps_2q_matrix's own convention (its
    (4,4) matrices, e.g. CX, are built with qubit0 as the more
    significant index before reshaping to (2,2,2,2))."""
    eye2 = np.eye(2, dtype=mat1.dtype)
    a, b = pair
    return np.kron(mat1, eye2) if qubit == a else np.kron(eye2, mat1)


def fuse_gate_list(gate_list, dtype):
    """Host-side (pure Python/NumPy), exact -- fuses a chain of
    consecutive gates into one matrix via matrix multiplication,
    GROWING the active qubit pair as needed: a 1-qubit gate on a qubit
    already inside the current chain's pair gets embedded into that same
    4x4 space (kron with identity on the other qubit) rather than
    breaking the chain -- this is what actually fuses CX-RZ-CX (RZ acts
    on only one of CX's two qubits, so a naive "exact qubit set match"
    rule never fuses anything, as first discovered by measuring a 1.00x
    reduction here). A chain starting on a single qubit that is later
    joined by a 2-qubit gate touching it also grows into the 2-qubit
    space the same way. Any gate touching a qubit OUTSIDE the current
    chain's pair ends the chain. Returns a list of tagged forms:
      ('1q', q, matrix_2x2)
      ('2q', q1, q2, matrix_4x4_as_2x2x2x2)
    """
    fused = []
    i = 0
    n = len(gate_list)
    while i < n:
        g_id, q1, q2, param = gate_list[i]
        is_2q = g_id >= 20
        if is_2q:
            mat = np.asarray(_mps_2q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype)).reshape(4, 4)
            pair = (q1, q2)
        else:
            mat = np.asarray(_mps_1q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype))
            pair = None  # still a 1-qubit chain, no pair fixed yet
        active_qubit = q1  # meaningful only while pair is None

        j = i + 1
        while j < n:
            ng_id, nq1, nq2, nparam = gate_list[j]
            n_is_2q = ng_id >= 20

            if pair is None:
                if not n_is_2q:
                    if nq1 != active_qubit:
                        break
                    nmat = np.asarray(_mps_1q_matrix(jnp.asarray(ng_id), jnp.asarray(nparam), dtype))
                    mat = nmat @ mat
                    j += 1
                    continue
                if active_qubit not in (nq1, nq2):
                    break
                # grow into the 2-qubit space: embed the 1q matrix so far,
                # then left-multiply by the new 2-qubit gate.
                pair = (nq1, nq2)
                mat = _embed_1q(mat, active_qubit, pair)
                nmat = np.asarray(_mps_2q_matrix(jnp.asarray(ng_id), jnp.asarray(nparam), dtype)).reshape(4, 4)
                mat = nmat @ mat
                j += 1
                continue

            # pair is fixed (a, b) -- fuse anything acting only within it.
            if n_is_2q:
                if (nq1, nq2) != pair:
                    break
                nmat = np.asarray(_mps_2q_matrix(jnp.asarray(ng_id), jnp.asarray(nparam), dtype)).reshape(4, 4)
            else:
                if nq1 not in pair:
                    break
                nmat = _embed_1q(
                    np.asarray(_mps_1q_matrix(jnp.asarray(ng_id), jnp.asarray(nparam), dtype)),
                    nq1, pair,
                )
            mat = nmat @ mat
            j += 1

        if pair is not None:
            fused.append(('2q', pair[0], pair[1], mat.reshape(2, 2, 2, 2)))
        else:
            fused.append(('1q', active_qubit, mat))
        i = j
    return fused


def _build_blocked_runner(n_qubits, max_bond, eps, jsd_budget, dtype):
    """Same bucketed-SVD dispatch as the already-validated (and
    middle-bond-bug-fixed) _bucketed_runner in
    mps_bucketed_svd_circuit_integration.py, EXCEPT the 2-qubit gate
    matrix is taken directly from a pre-fused matrix stream instead of
    being reconstructed inside the JIT via _mps_2q_matrix(g_id, param) --
    fusion already happened on the host, once, before compiling."""
    buckets = _bucket_sizes(max_bond)
    bucket_arr = jnp.array(buckets)
    real_dtype = _real_dtype(dtype)

    def step(carry, xs):
        gammas, lambdas, real_chi = carry
        is_2q, q1, q2, mat2q, mat1q = xs
        q1 = q1.astype(jnp.int32)
        q2 = q2.astype(jnp.int32)

        def branch_1q(c):
            gammas_, lambdas_, real_chi_ = c
            new_g = jnp.einsum('ij,ljr->lir', mat1q, gammas_[q1])
            return gammas_.at[q1].set(new_g), lambdas_, real_chi_

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


def _fused_to_arrays(fused, n_qubits, dtype):
    """Host-side: turns fuse_gate_list's output into the parallel arrays
    jax.lax.scan needs -- every step carries BOTH a 1q and a 2q matrix
    slot (one is an unused identity placeholder) so shapes stay uniform
    across the whole scanned sequence, same principle _mps_1q_matrix/
    _mps_2q_matrix's own switch already relies on (all branches traced,
    only one path's real work executed)."""
    is_2q, q1s, q2s, mats2q, mats1q = [], [], [], [], []
    eye2 = np.eye(2, dtype=dtype)
    eye4 = np.eye(4, dtype=dtype).reshape(2, 2, 2, 2)
    for entry in fused:
        if entry[0] == '2q':
            _, a, b, mat = entry
            is_2q.append(True)
            q1s.append(a)
            q2s.append(b)
            mats2q.append(mat)
            mats1q.append(eye2)
        else:
            _, a, mat = entry
            is_2q.append(False)
            q1s.append(a)
            q2s.append(0)
            mats2q.append(eye4)
            mats1q.append(mat)
    return (jnp.asarray(is_2q), jnp.asarray(q1s, dtype=jnp.int32), jnp.asarray(q2s, dtype=jnp.int32),
            jnp.asarray(np.stack(mats2q)), jnp.asarray(np.stack(mats1q)))


def _reference_run(n_qubits, max_bond, eps, jsd_budget, dtype, gate_list):
    sim = MPSSimulator(n_qubits, max_bond=max_bond, svd_cutoff=eps, jsd_budget=jsd_budget,
                        use_float32=(dtype == jnp.complex64))
    for g_id, q1, q2, param in gate_list:
        if g_id < 20:
            sim.apply_gate_1q(_mps_1q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype), q1)
        else:
            sim.apply_gate_2q(_mps_2q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype), q1, q2)
    return sim


def compare(n_qubits, max_bond, eps, jsd_budget, dtype, steps, label):
    gate_list = _tfim_gate_list(n_qubits, 0.1, steps, 1.0, 1.0)
    fused = fuse_gate_list(gate_list, dtype)

    ref_sim = _reference_run(n_qubits, max_bond, eps, jsd_budget, dtype, gate_list)
    ref_sv = np.asarray(ref_sim.contract_to_statevector())

    xs = _fused_to_arrays(fused, n_qubits, dtype)
    run = _build_blocked_runner(n_qubits, max_bond, eps, jsd_budget, dtype)
    gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=dtype).at[0, 0, 0].set(1.0), max_bond) for _ in range(n_qubits)])
    lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=_real_dtype(dtype)), max_bond) for _ in range(n_qubits + 1)])
    real_chi0 = jnp.ones(n_qubits + 1, dtype=jnp.int32)
    gammas_f, lambdas_f, real_chi_f = run(gammas0, lambdas0, real_chi0, xs)

    blocked_sim = MPSSimulator(n_qubits, max_bond=max_bond, svd_cutoff=eps, jsd_budget=jsd_budget,
                                use_float32=(dtype == jnp.complex64))
    blocked_sim.gammas = [gammas_f[i, : int(real_chi_f[i]), :, : int(real_chi_f[i + 1])] for i in range(n_qubits)]
    blocked_sim.lambdas = [lambdas_f[i, : int(real_chi_f[i])] for i in range(n_qubits + 1)]
    blocked_sv = np.asarray(blocked_sim.contract_to_statevector())

    fidelity = float(np.abs(np.vdot(ref_sv, blocked_sv)) ** 2)
    print(f"[{label}] n_qubits={n_qubits} max_bond={max_bond} n_gates_original={len(gate_list)} "
          f"n_steps_fused={len(fused)} (reduction {len(gate_list)/len(fused):.2f}x)")
    print(f"[{label}] fidelity(ref, blocked) = {fidelity:.12f}")
    print(f"[{label}] max_bond_used: ref={ref_sim.max_bond_used()} blocked={int(np.max(real_chi_f))}")
    return fidelity


def timing_check(n_qubits, max_bond, eps, jsd_budget, dtype, steps):
    """CPU-only, informal (this sandbox has no GPU, same caveat every
    other timing check in this repo states) -- just confirms the fused
    step count actually translates into a real wall-clock difference
    before spending a Colab GPU round-trip on it."""
    import time

    gate_list = _tfim_gate_list(n_qubits, 0.1, steps, 1.0, 1.0)
    fused = fuse_gate_list(gate_list, dtype)
    xs_fused = _fused_to_arrays(fused, n_qubits, dtype)

    unfused_as_fused = [
        ('2q', q1, q2, np.asarray(_mps_2q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype)))
        if g_id >= 20 else
        ('1q', q1, np.asarray(_mps_1q_matrix(jnp.asarray(g_id), jnp.asarray(param), dtype)))
        for g_id, q1, q2, param in gate_list
    ]
    xs_unfused = _fused_to_arrays(unfused_as_fused, n_qubits, dtype)

    gammas0 = jnp.stack([_pad_gamma(jnp.zeros((1, 2, 1), dtype=dtype).at[0, 0, 0].set(1.0), max_bond) for _ in range(n_qubits)])
    lambdas0 = jnp.stack([_pad_lambda(jnp.ones(1, dtype=_real_dtype(dtype)), max_bond) for _ in range(n_qubits + 1)])
    real_chi0 = jnp.ones(n_qubits + 1, dtype=jnp.int32)
    run = _build_blocked_runner(n_qubits, max_bond, eps, jsd_budget, dtype)

    jax.block_until_ready(run(gammas0, lambdas0, real_chi0, xs_unfused))
    t0 = time.perf_counter()
    out_unfused = run(gammas0, lambdas0, real_chi0, xs_unfused)
    jax.block_until_ready(out_unfused)
    t_unfused = time.perf_counter() - t0

    jax.block_until_ready(run(gammas0, lambdas0, real_chi0, xs_fused))
    t0 = time.perf_counter()
    out_fused = run(gammas0, lambdas0, real_chi0, xs_fused)
    jax.block_until_ready(out_fused)
    t_fused = time.perf_counter() - t0

    print(f"\nN={n_qubits} max_bond={max_bond} steps={steps}: "
          f"unfused n_steps={len(gate_list)} warm={t_unfused*1000:.2f}ms | "
          f"fused n_steps={len(fused)} warm={t_fused*1000:.2f}ms | "
          f"speedup={t_unfused/t_fused:.2f}x")


if __name__ == "__main__":
    f1 = compare(6, 16, 1e-12, 1e-5, jnp.complex128, 3, "complex128,N=6,max_bond=16")
    f2 = compare(8, 32, 1e-12, 1e-5, jnp.complex128, 3, "complex128,N=8,max_bond=32")
    f3 = compare(6, 16, 1e-6, 1e-5, jnp.complex64, 3, "complex64,N=6,max_bond=16")

    all_ok = all(f > 1 - 1e-8 for f in (f1, f2)) and f3 > 1 - 1e-4
    print(f"\nALL FIDELITIES ACCEPTABLE: {all_ok}")
    assert all_ok, "gate-blocked bucketed run disagrees with the reference eager simulator"

    timing_check(50, 64, 1e-12, 1e-5, jnp.complex128, 5)
