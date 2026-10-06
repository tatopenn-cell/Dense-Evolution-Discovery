"""
JAX port of the QR-truncation two-site update (see
mps_qr_truncation_approach_b_prototype.py for the math/bug-fix history,
mps_qr_truncation_circuit_correctness.py for the NumPy full-circuit
fidelity check that this reproduces). CPU-only correctness check here;
mirrors NumPy op-for-op so a Kaggle GPU timing script built on top of
this is testing the same algorithm, not a reimplementation.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

d = 2


def _expm_herm(h, coeff):
    evals, evecs = np.linalg.eigh(h)
    return evecs @ np.diag(np.exp(1j * coeff * evals)) @ evecs.conj().T


def tfim_gates(n, dt, steps, J, g):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    rx = _expm_herm(x, -2.0 * dt * g / 2.0)
    czz = _expm_herm(np.kron(z, z), -2.0 * dt * J / 2.0).reshape(2, 2, 2, 2)
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops.append(("2q", jnp.asarray(czz), i, i + 1))
        for i in range(n):
            ops.append(("1q", jnp.asarray(rx), i))
    return ops


class MPSChainJax:
    def __init__(self, n):
        self.n = n
        self.tensors = []
        for _ in range(n):
            t = jnp.zeros((1, d, 1), dtype=jnp.complex128)
            t = t.at[0, 0, 0].set(1.0)
            self.tensors.append(t)

    def apply_1q(self, gate, q):
        self.tensors[q] = jnp.einsum("ij,ajb->aib", gate, self.tensors[q])

    def apply_2q_svd(self, gate4, q1, q2, chi_max):
        theta = jnp.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4)
        chi_l, d1, d2, chi_r = theta.shape
        mat = theta.reshape(chi_l * d1, d2 * chi_r)
        u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
        chi_new = min(chi_max, s.shape[0])
        u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
        self.tensors[q1] = u.reshape(chi_l, d1, chi_new)
        self.tensors[q2] = jnp.einsum("k,kjb->kjb", s, vh.reshape(chi_new, d2, chi_r))

    def apply_2q_qr(self, gate4, q1, q2, chi_max, n_iter, rng):
        theta = jnp.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4)
        chi_l, d1, d2, chi_r = theta.shape
        b_old = self.tensors[q2]
        chi_old = b_old.shape[0]

        b_mat = np.asarray(b_old.reshape(chi_old, d2 * chi_r)).conj().T
        eta = min(chi_max, d2 * chi_r, chi_l * d1)
        keep = min(chi_old, eta)
        extra = eta - keep
        if extra > 0:
            pad = rng.normal(size=(d2 * chi_r, extra)) + 1j * rng.normal(size=(d2 * chi_r, extra))
            proj = np.eye(d2 * chi_r) - b_mat[:, :keep] @ b_mat[:, :keep].conj().T
            pad = proj @ pad
            y0 = np.concatenate([b_mat[:, :keep], pad], axis=1)
        else:
            y0 = b_mat[:, :eta]
        q0, _ = jnp.linalg.qr(jnp.asarray(y0))
        eta_eff = q0.shape[1]
        b_cur = q0.conj().T.reshape(eta_eff, d2, chi_r)

        for _ in range(n_iter):
            x = jnp.einsum("aijb,mjb->aim", theta, b_cur.conj())
            x_mat = x.reshape(chi_l * d1, eta_eff)
            nu = min(x_mat.shape)
            q_m, _ = jnp.linalg.qr(x_mat)
            q_left = q_m[:, :nu].reshape(chi_l, d1, nu)

            y = jnp.einsum("ain,aijb->njb", q_left.conj(), theta)
            y_mat = y.reshape(nu, d2 * chi_r)
            nu2 = min(y_mat.shape)
            qy, ry = jnp.linalg.qr(y_mat.conj().T)
            l_mat = ry[:nu2, :].conj().T
            b_cur = qy[:, :nu2].conj().T.reshape(nu2, d2, chi_r)
            eta_eff = nu2

        p_mat, s_full, vh_mat = jnp.linalg.svd(l_mat, full_matrices=False)
        chi_new = min(chi_max, s_full.shape[0])
        p_trunc, s_trunc, vh_trunc = p_mat[:, :chi_new], s_full[:chi_new], vh_mat[:chi_new, :]

        self.tensors[q1] = jnp.einsum("ain,nk->aik", q_left, p_trunc)
        self.tensors[q2] = jnp.einsum("k,kn,njb->kjb", s_trunc, vh_trunc, b_cur)

    def to_statevector(self):
        v = self.tensors[0]
        for t in self.tensors[1:]:
            v = jnp.einsum("...a,ajb->...jb", v, t)
        return v.reshape(-1)


def apply_1q_dense(state_vec, n, gate1, q):
    v = state_vec.reshape([2] * n)
    v = np.tensordot(gate1, v, axes=([1], [q]))
    v = np.moveaxis(v, 0, q)
    return v.reshape(-1)


def apply_2q_dense(state_vec, n, gate4, q1, q2):
    v = state_vec.reshape([2] * n)
    v = np.tensordot(np.asarray(gate4).reshape(2, 2, 2, 2), v, axes=([2, 3], [q1, q2]))
    v = np.moveaxis(v, [0, 1], [q1, q2])
    return v.reshape(-1)


def dense_reference(n, ops):
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for op in ops:
        if op[0] == "1q":
            state = apply_1q_dense(state, n, np.asarray(op[1]), op[2])
        else:
            state = apply_2q_dense(state, n, op[1], op[2], op[3])
    return state


def correctness_check():
    n, dt, steps = 8, 0.1, 6
    ops = tfim_gates(n, dt, steps, J=1.0, g=1.0)
    ref_state = dense_reference(n, ops)

    for chi_max in (2, 3, 4):
        chain_svd = MPSChainJax(n)
        for op in ops:
            if op[0] == "1q":
                chain_svd.apply_1q(op[1], op[2])
            else:
                chain_svd.apply_2q_svd(op[1], op[2], op[3], chi_max)
        svd_state = np.asarray(chain_svd.to_statevector())
        svd_fid = abs(np.vdot(ref_state, svd_state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(svd_state, svd_state).real)

        rng = np.random.default_rng(1)
        chain_qr = MPSChainJax(n)
        for op in ops:
            if op[0] == "1q":
                chain_qr.apply_1q(op[1], op[2])
            else:
                chain_qr.apply_2q_qr(op[1], op[2], op[3], chi_max, n_iter=1, rng=rng)
        qr_state = np.asarray(chain_qr.to_statevector())
        qr_fid = abs(np.vdot(ref_state, qr_state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(qr_state, qr_state).real)

        print(f"chi_max={chi_max}: SVD fidelity={svd_fid:.12f}  QR(1 iter) fidelity={qr_fid:.12f}")


if __name__ == "__main__":
    correctness_check()
