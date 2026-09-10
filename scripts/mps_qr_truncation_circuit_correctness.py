"""
Full-circuit correctness check for the QR-truncation prototype (see
mps_qr_truncation_approach_b_prototype.py for the single-gate math and
the bug found/fixed there). Runs a small TFIM Trotter circuit through a
simple left-canonical MPS chain (left-to-right sweep, standard TEBD --
not yet the production Gamma-Lambda representation) two ways -- SVD
truncation (reference) and QR truncation (warm-started from the current
right tensor, iterated) -- and compares both against a real dense
statevector simulation. Only once this matches to acceptable fidelity
does CPU/GPU timing become a meaningful next question.
"""
import numpy as np

d = 2


def apply_1q_all(state_vec, n, gate1, rng=None):
    for q in range(n):
        state_vec = apply_1q_dense(state_vec, n, gate1, q)
    return state_vec


def apply_1q_dense(state_vec, n, gate1, q):
    v = state_vec.reshape([2] * n)
    v = np.tensordot(gate1, v, axes=([1], [q]))
    v = np.moveaxis(v, 0, q)
    return v.reshape(-1)


def apply_2q_dense(state_vec, n, gate4, q1, q2):
    v = state_vec.reshape([2] * n)
    v = np.tensordot(gate4.reshape(2, 2, 2, 2), v, axes=([2, 3], [q1, q2]))
    v = np.moveaxis(v, [0, 1], [q1, q2])
    return v.reshape(-1)


def tfim_gates(n, dt, steps, J, g):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    i2 = np.eye(2, dtype=complex)
    zz = np.kron(z, z).reshape(2, 2, 2, 2)
    rx = _expm_herm(x, -2.0 * dt * g / 2.0)
    czz = _expm_herm(np.kron(z, z), -2.0 * dt * J / 2.0).reshape(2, 2, 2, 2)
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops.append(("2q", czz, i, i + 1))
        for i in range(n):
            ops.append(("1q", rx, i))
    return ops


def _expm_herm(h, coeff):
    evals, evecs = np.linalg.eigh(h)
    return evecs @ np.diag(np.exp(1j * coeff * evals)) @ evecs.conj().T


def dense_reference(n, ops):
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for op in ops:
        if op[0] == "1q":
            _, gate, q = op
            state = apply_1q_dense(state, n, gate, q)
        else:
            _, gate, q1, q2 = op
            state = apply_2q_dense(state, n, gate, q1, q2)
    return state


class MPSChain:
    """Simple left-canonical chain: tensors[i] has shape (chi_l,d,chi_r),
    left-isometric for i < n-1, tensors[-1] carries the global norm."""

    def __init__(self, n):
        self.n = n
        self.tensors = [np.zeros((1, d, 1), dtype=complex) for _ in range(n)]
        for t in self.tensors:
            t[0, 0, 0] = 1.0

    def apply_1q(self, gate, q):
        self.tensors[q] = np.einsum("ij,ajb->aib", gate, self.tensors[q], optimize=True)

    def apply_2q_svd(self, gate4, q1, q2, chi_max):
        theta = np.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4, optimize=True)
        chi_l, d1, d2, chi_r = theta.shape
        mat = theta.reshape(chi_l * d1, d2 * chi_r)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        chi_new = min(chi_max, len(s))
        u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
        self.tensors[q1] = u.reshape(chi_l, d1, chi_new)
        self.tensors[q2] = np.einsum("k,kjb->kjb", s, vh.reshape(chi_new, d2, chi_r), optimize=True)

    def apply_2q_qr(self, gate4, q1, q2, chi_max, n_iter, rng):
        theta = np.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4, optimize=True)
        chi_l, d1, d2, chi_r = theta.shape
        b_old = self.tensors[q2]
        chi_old = b_old.shape[0]

        b_mat = b_old.reshape(chi_old, d2 * chi_r).conj().T
        eta = min(chi_max, d2 * chi_r, chi_l * d1)
        extra = eta - min(chi_old, eta)
        if extra > 0 and eta > chi_old:
            pad = rng.normal(size=(d2 * chi_r, extra)) + 1j * rng.normal(size=(d2 * chi_r, extra))
            proj = np.eye(d2 * chi_r) - b_mat[:, :min(chi_old, eta)] @ b_mat[:, :min(chi_old, eta)].conj().T
            pad = proj @ pad
            y0 = np.concatenate([b_mat[:, :min(chi_old, eta)], pad], axis=1)
        else:
            y0 = b_mat[:, :eta]
        q0, _ = np.linalg.qr(y0)
        eta_eff = q0.shape[1]
        b_cur = q0.conj().T.reshape(eta_eff, d2, chi_r)

        for _ in range(n_iter):
            x = np.einsum("aijb,mjb->aim", theta, b_cur.conj(), optimize=True)
            x_mat = x.reshape(chi_l * d1, eta_eff)
            nu = min(x_mat.shape)
            q_m, _ = np.linalg.qr(x_mat)
            q_left = q_m[:, :nu].reshape(chi_l, d1, nu)

            y = np.einsum("ain,aijb->njb", q_left.conj(), theta, optimize=True)
            y_mat = y.reshape(nu, d2 * chi_r)
            nu2 = min(y_mat.shape)
            qy, ry = np.linalg.qr(y_mat.conj().T)
            l_mat = ry[:nu2, :].conj().T
            b_cur = qy[:, :nu2].conj().T.reshape(nu2, d2, chi_r)
            eta_eff = nu2

        p_mat, s_full, vh_mat = np.linalg.svd(l_mat, full_matrices=False)
        chi_new = min(chi_max, len(s_full), int(np.sum(s_full > 1e-13)))
        chi_new = max(chi_new, 1)
        p_trunc, s_trunc, vh_trunc = p_mat[:, :chi_new], s_full[:chi_new], vh_mat[:chi_new, :]

        self.tensors[q1] = np.einsum("ain,nk->aik", q_left, p_trunc, optimize=True)
        self.tensors[q2] = np.einsum("k,kn,njb->kjb", s_trunc, vh_trunc, b_cur, optimize=True)

    def to_statevector(self):
        v = self.tensors[0]
        for t in self.tensors[1:]:
            v = np.einsum("...a,ajb->...jb", v, t, optimize=True)
        return v.reshape(-1)


def run(n, dt, steps, chi_max, qr_n_iter, seed):
    ops = tfim_gates(n, dt, steps, J=1.0, g=1.0)
    ref_state = dense_reference(n, ops)

    chain_svd = MPSChain(n)
    for op in ops:
        if op[0] == "1q":
            chain_svd.apply_1q(op[1], op[2])
        else:
            chain_svd.apply_2q_svd(op[1], op[2], op[3], chi_max)
    svd_state = chain_svd.to_statevector()
    svd_fidelity = abs(np.vdot(ref_state, svd_state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(svd_state, svd_state).real)

    rng = np.random.default_rng(seed)
    chain_qr = MPSChain(n)
    for op in ops:
        if op[0] == "1q":
            chain_qr.apply_1q(op[1], op[2])
        else:
            chain_qr.apply_2q_qr(op[1], op[2], op[3], chi_max, qr_n_iter, rng)
    qr_state = chain_qr.to_statevector()
    qr_fidelity = abs(np.vdot(ref_state, qr_state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(qr_state, qr_state).real)

    bonds = [t.shape[0] for t in chain_svd.tensors[1:]]
    print(f"N={n} dt={dt} steps={steps} chi_max={chi_max} qr_iter={qr_n_iter} "
          f"final_bonds={bonds}: SVD fidelity={svd_fidelity:.12f}  QR fidelity={qr_fidelity:.12f}")


def cpu_timing(n, dt, steps, chi_max, qr_n_iter, seed):
    import time
    ops = tfim_gates(n, dt, steps, J=1.0, g=1.0)

    chain_svd = MPSChain(n)
    t0 = time.perf_counter()
    for op in ops:
        if op[0] == "1q":
            chain_svd.apply_1q(op[1], op[2])
        else:
            chain_svd.apply_2q_svd(op[1], op[2], op[3], chi_max)
    t_svd = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    chain_qr = MPSChain(n)
    t0 = time.perf_counter()
    for op in ops:
        if op[0] == "1q":
            chain_qr.apply_1q(op[1], op[2])
        else:
            chain_qr.apply_2q_qr(op[1], op[2], op[3], chi_max, qr_n_iter, rng)
    t_qr = time.perf_counter() - t0

    print(f"N={n} steps={steps} chi_max={chi_max} qr_iter={qr_n_iter}: "
          f"SVD={t_svd:.3f}s  QR={t_qr:.3f}s  speedup={t_svd/t_qr:.2f}x")


if __name__ == "__main__":
    for chi_max in (2, 3, 4):
        for qr_n_iter in (1, 3, 5, 10):
            run(n=8, dt=0.1, steps=6, chi_max=chi_max, qr_n_iter=qr_n_iter, seed=1)

    print("\n--- CPU timing, N=50, max_bond=64 (production-scale) ---")
    for qr_n_iter in (1, 2):
        cpu_timing(n=50, dt=0.1, steps=5, chi_max=64, qr_n_iter=qr_n_iter, seed=1)
