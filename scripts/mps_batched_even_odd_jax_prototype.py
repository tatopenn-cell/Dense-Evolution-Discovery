"""
JAX port of the batched even-odd gate application (see
mps_batched_even_odd_prototype.py for the math and the NumPy fidelity
check this reproduces, exactly, since batching is mathematically exact
-- no approximation, unlike QR truncation). Uses jax.vmap over the SVD
so all bonds of one parity dispatch as ONE batched call instead of a
Python loop of individual calls.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

d = 2


def _expm_herm(h, coeff):
    evals, evecs = np.linalg.eigh(h)
    return evecs @ np.diag(np.exp(1j * coeff * evals)) @ evecs.conj().T


def even_odd_tfim_layers(n, dt, steps, J, g):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    rx_half = jnp.asarray(_expm_herm(x, -2.0 * (dt / 2) * g / 2.0))
    czz = jnp.asarray(_expm_herm(np.kron(z, z), -2.0 * dt * J / 2.0).reshape(2, 2, 2, 2))
    even_bonds = list(range(0, n - 1, 2))
    odd_bonds = list(range(1, n - 1, 2))
    layers = []
    for _ in range(steps):
        layers.append(("1q_all", rx_half))
        layers.append(("2q_batch", czz, even_bonds))
        layers.append(("2q_batch", czz, odd_bonds))
        layers.append(("1q_all", rx_half))
    return layers


def sequential_layers(n, dt, steps, J, g):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    rx = jnp.asarray(_expm_herm(x, -2.0 * dt * g / 2.0))
    czz = jnp.asarray(_expm_herm(np.kron(z, z), -2.0 * dt * J / 2.0).reshape(2, 2, 2, 2))
    ops = []
    for _ in range(steps):
        for i in range(n - 1):
            ops.append(("2q", czz, i, i + 1))
        for i in range(n):
            ops.append(("1q", rx, i))
    return ops


class MPSChainJax:
    """Tensors are kept at their TRUE current shape (like the NumPy
    prototype), not permanently padded to max_bond -- padding is used
    only as a temporary scratch buffer inside apply_2q_batch so bonds
    of different current sizes can be stacked for one vmapped SVD call,
    and the output is always trimmed back down to each bond's own real
    size immediately after. Keeping tensors permanently padded (an
    earlier version of this prototype) hit a real, reproducible
    correctness failure once a cut's real bond dimension reached its
    own exact theoretical max (no slack left at all) -- not yet fully
    root-caused, sidestepped here by matching the proven-correct NumPy
    prototype's storage discipline instead of chasing it further."""

    def __init__(self, n, max_bond):
        self.n = n
        self.max_bond = max_bond
        self.tensors = [jnp.zeros((1, d, 1), dtype=jnp.complex128).at[0, 0, 0].set(1.0) for _ in range(n)]

    def apply_1q(self, gate, q):
        self.tensors[q] = jnp.einsum("ij,ajb->aib", gate, self.tensors[q])

    def apply_2q_sequential(self, gate4, q1, q2):
        max_bond = self.max_bond
        theta = jnp.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4)
        chi_l, d1, d2, chi_r = theta.shape
        theta_mat = theta.reshape(chi_l * d1, d2 * chi_r)
        u, s, vh = jnp.linalg.svd(theta_mat, full_matrices=False)
        chi_new = min(max_bond, s.shape[0])
        u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
        self.tensors[q1] = u.reshape(chi_l, d1, chi_new)
        self.tensors[q2] = jnp.einsum("k,kjb->kjb", s, vh.reshape(chi_new, d2, chi_r))

    def apply_2q_batch(self, gate4, bonds):
        max_bond = self.max_bond
        if not bonds:
            return
        real_shapes = [(self.tensors[i].shape[0], self.tensors[i + 1].shape[2]) for i in bonds]

        padded_left, padded_right = [], []
        for i in bonds:
            lt, rt = self.tensors[i], self.tensors[i + 1]
            pl = jnp.zeros((max_bond, d, max_bond), dtype=jnp.complex128).at[:lt.shape[0], :, :lt.shape[2]].set(lt)
            pr = jnp.zeros((max_bond, d, max_bond), dtype=jnp.complex128).at[:rt.shape[0], :, :rt.shape[2]].set(rt)
            padded_left.append(pl)
            padded_right.append(pr)
        left = jnp.stack(padded_left)
        right = jnp.stack(padded_right)
        theta = jnp.einsum("naim,nmjb,ijkl->naklb", left, right, gate4)
        theta_mat = theta.reshape(len(bonds), max_bond * d, d * max_bond)

        u, s, vh = jax.vmap(lambda m: jnp.linalg.svd(m, full_matrices=False))(theta_mat)
        chi_new_max = min(max_bond, s.shape[1])
        u, s, vh = u[:, :, :chi_new_max], s[:, :chi_new_max], vh[:, :chi_new_max, :]

        u = u.reshape(len(bonds), max_bond, d, chi_new_max)
        new_right = jnp.einsum("nk,nkjb->nkjb", s, vh.reshape(len(bonds), chi_new_max, d, max_bond))
        for idx2, i in enumerate(bonds):
            chi_l, chi_r = real_shapes[idx2]
            real_rank = min(chi_l * d, chi_r * d, max_bond)
            self.tensors[i] = u[idx2, :chi_l, :, :real_rank]
            self.tensors[i + 1] = new_right[idx2, :real_rank, :, :chi_r]

    def to_statevector_trimmed(self):
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
    v = np.tensordot(np.asarray(gate4), v, axes=([2, 3], [q1, q2]))
    v = np.moveaxis(v, [0, 1], [q1, q2])
    return v.reshape(-1)


def dense_reference_even_odd(n, layers):
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for layer in layers:
        if layer[0] == "1q_all":
            gate = np.asarray(layer[1])
            for q in range(n):
                state = apply_1q_dense(state, n, gate, q)
        else:
            _, gate, bonds = layer
            for i in bonds:
                state = apply_2q_dense(state, n, gate, i, i + 1)
    return state


def correctness_check():
    n, dt, steps = 8, 0.1, 6
    layers = even_odd_tfim_layers(n, dt, steps, J=1.0, g=1.0)
    ref_state = dense_reference_even_odd(n, layers)

    for max_bond in (2, 4, 8):
        chain = MPSChainJax(n, max_bond)
        for layer in layers:
            if layer[0] == "1q_all":
                for q in range(n):
                    chain.apply_1q(layer[1], q)
            else:
                _, gate, bonds = layer
                chain.apply_2q_batch(gate, bonds)
        state = np.asarray(chain.to_statevector_trimmed())
        fid = abs(np.vdot(ref_state, state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(state, state).real)

        chain_seq = MPSChainJax(n, max_bond)
        for layer in layers:
            if layer[0] == "1q_all":
                for q in range(n):
                    chain_seq.apply_1q(layer[1], q)
            else:
                _, gate, bonds = layer
                for i in bonds:
                    chain_seq.apply_2q_sequential(gate, i, i + 1)
        state_seq = np.asarray(chain_seq.to_statevector_trimmed())
        fid_seq = abs(np.vdot(ref_state, state_seq)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(state_seq, state_seq).real)

        print(f"max_bond={max_bond}: batched fidelity={fid:.12f}  sequential(JAX) fidelity={fid_seq:.12f}")


if __name__ == "__main__":
    correctness_check()
