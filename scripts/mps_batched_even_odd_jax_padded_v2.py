"""
Approach C, v2: same batched even-odd gate application as
mps_batched_even_odd_jax_prototype.py, but with tensors kept
PERMANENTLY padded to max_bond (matching dense_evolution's own
production convention) instead of trimmed to their true current shape
after every gate. The v1 prototype's real-shape-tracking approach
required a Python-level padding loop before every batched SVD call and
a Python-level trimming loop after -- both are per-bond eager
operations, exactly the "many small eager dispatches" anti-pattern this
whole investigation already found and fixed once in production
(_pad_gamma/_pad_lambda). Real GPU test of v1 confirmed this actually
made things WORSE than sequential (0.21x on GPU, worse than CPU's
0.36x) -- the padding/trimming bookkeeping cost more than the batched
SVD saved.

With permanent max_bond padding, jnp.stack needs no padding loop at all
(every tensor is already the same shape), and the real-rank truncation
after the batched SVD is one vectorized jnp.where mask, not a Python
loop of dynamic slices. The earlier real bug found in v1's permanently-
padded ancestor (shared middle axis not padded to a common size before
stacking) is fixed here by construction: every tensor's every axis is
always exactly max_bond, no exceptions, so there is no "sometimes
smaller, sometimes not" axis to get wrong.
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


class MPSChainPadded:
    def __init__(self, n, max_bond):
        self.n = n
        self.max_bond = max_bond
        self.real_chi = [1] * (n + 1)
        self.tensors = []
        for _ in range(n):
            t = jnp.zeros((max_bond, d, max_bond), dtype=jnp.complex128)
            t = t.at[0, 0, 0].set(1.0)
            self.tensors.append(t)

    def apply_1q(self, gate, q):
        self.tensors[q] = jnp.einsum("ij,ajb->aib", gate, self.tensors[q])

    def apply_2q_sequential(self, gate4, q1, q2):
        max_bond = self.max_bond
        theta = jnp.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4)
        theta_mat = theta.reshape(max_bond * d, d * max_bond)
        u, s, vh = jnp.linalg.svd(theta_mat, full_matrices=False)
        u, s, vh = u[:, :max_bond], s[:max_bond], vh[:max_bond, :]
        real_rank = min(self.real_chi[q1] * d, self.real_chi[q2 + 1] * d, max_bond)
        s = jnp.where(jnp.arange(s.shape[0]) < real_rank, s, 0.0)
        self.tensors[q1] = u.reshape(max_bond, d, max_bond)
        self.tensors[q2] = jnp.einsum("k,kjb->kjb", s, vh.reshape(max_bond, d, max_bond))
        self.real_chi[q2] = int(real_rank)

    def apply_2q_batch(self, gate4, bonds):
        max_bond = self.max_bond
        if not bonds:
            return
        left = jnp.stack([self.tensors[i] for i in bonds])
        right = jnp.stack([self.tensors[i + 1] for i in bonds])
        theta = jnp.einsum("naim,nmjb,ijkl->naklb", left, right, gate4)
        theta_mat = theta.reshape(len(bonds), max_bond * d, d * max_bond)

        u, s, vh = jax.vmap(lambda m: jnp.linalg.svd(m, full_matrices=False))(theta_mat)
        u, s, vh = u[:, :, :max_bond], s[:, :max_bond], vh[:, :max_bond, :]

        real_ranks = [min(self.real_chi[i] * d, self.real_chi[i + 2] * d, max_bond) for i in bonds]
        real_ranks_arr = jnp.array(real_ranks)
        mask = jnp.arange(s.shape[1])[None, :] < real_ranks_arr[:, None]
        s = jnp.where(mask, s, 0.0)

        u = u.reshape(len(bonds), max_bond, d, max_bond)
        new_right = jnp.einsum("nk,nkjb->nkjb", s, vh.reshape(len(bonds), max_bond, d, max_bond))

        for idx2, i in enumerate(bonds):
            self.tensors[i] = u[idx2]
            self.tensors[i + 1] = new_right[idx2]
            self.real_chi[i + 1] = real_ranks[idx2]

    def to_statevector(self):
        v = self.tensors[0][:1]
        for t in self.tensors[1:]:
            v = jnp.einsum("...a,ajb->...jb", v, t)
        return v[..., :1].reshape(-1)


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


def dense_reference(n, layers):
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
    ref_state = dense_reference(n, layers)

    for max_bond in (2, 4, 5, 6, 7, 8, 9, 16, 32):
        chain = MPSChainPadded(n, max_bond)
        for layer in layers:
            if layer[0] == "1q_all":
                for q in range(n):
                    chain.apply_1q(layer[1], q)
            else:
                _, gate, bonds = layer
                chain.apply_2q_batch(gate, bonds)
        state = np.asarray(chain.to_statevector())
        fid = abs(np.vdot(ref_state, state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(state, state).real)
        print(f"max_bond={max_bond}: batched (permanently-padded) fidelity={fid:.12f}")


if __name__ == "__main__":
    correctness_check()
