"""
Approach C: batched even-odd gate application. Every GPU speedup tried
so far in this investigation (bucketed SVD, gate blocking, QR
truncation) attacked the cost of ONE two-site update. This one attacks
the NUMBER of sequential dispatches directly, a different axis: for a
1D nearest-neighbor circuit, bonds (0,1),(2,3),(4,5),... ("even" bonds)
never share a qubit with each other, and neither do bonds
(1,2),(3,4),...  ("odd" bonds) -- so within one Trotter layer, all even
bonds can be updated in one batched jax.vmap dispatch, then all odd
bonds in a second, instead of walking every bond in a sequential
Python loop (n-1 separate dispatches per layer).

This requires restructuring the Trotter decomposition itself into the
standard even-odd (2nd-order Suzuki-Trotter) form -- physically
equivalent to the naive left-to-right sweep this repo's benchmark
circuit has used throughout (same Hamiltonian terms, same dt, just
grouped by parity instead of walked in index order), not a new
approximation.

Since every bond's tensors are already padded to a uniform max_bond
(exactly like the rest of this investigation's production code), all
even (or all odd) bonds' theta tensors have the SAME shape and can be
stacked into one (num_bonds, max_bond, d, d, max_bond) batch and SVD'd
via a single vmapped call.

This script validates correctness only (dense statevector reference,
small N) -- CPU/GPU timing is a separate, later question.
"""
import numpy as np

d = 2


def _expm_herm(h, coeff):
    evals, evecs = np.linalg.eigh(h)
    return evecs @ np.diag(np.exp(1j * coeff * evals)) @ evecs.conj().T


def even_odd_tfim_gates(n, dt, steps, J, g):
    """2nd-order Suzuki-Trotter, even-odd split: for each step, apply
    RX/2 on all sites, then all EVEN-bond ZZ gates (batchable), then all
    ODD-bond ZZ gates (batchable), then RX/2 again -- standard
    symmetric Trotterization, same physics as the naive sweep."""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    rx_half = _expm_herm(x, -2.0 * (dt / 2) * g / 2.0)
    czz = _expm_herm(np.kron(z, z), -2.0 * dt * J / 2.0).reshape(2, 2, 2, 2)

    even_bonds = list(range(0, n - 1, 2))
    odd_bonds = list(range(1, n - 1, 2))

    layers = []
    for _ in range(steps):
        layers.append(("1q_all", rx_half))
        layers.append(("2q_batch", czz, even_bonds))
        layers.append(("2q_batch", czz, odd_bonds))
        layers.append(("1q_all", rx_half))
    return layers


def apply_1q_dense(state_vec, n, gate1, q):
    v = state_vec.reshape([2] * n)
    v = np.tensordot(gate1, v, axes=([1], [q]))
    v = np.moveaxis(v, 0, q)
    return v.reshape(-1)


def apply_2q_dense(state_vec, n, gate4, q1, q2):
    v = state_vec.reshape([2] * n)
    v = np.tensordot(gate4, v, axes=([2, 3], [q1, q2]))
    v = np.moveaxis(v, [0, 1], [q1, q2])
    return v.reshape(-1)


def dense_reference(n, layers):
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for layer in layers:
        if layer[0] == "1q_all":
            _, gate = layer
            for q in range(n):
                state = apply_1q_dense(state, n, gate, q)
        else:
            _, gate, bonds = layer
            for i in bonds:
                state = apply_2q_dense(state, n, gate, i, i + 1)
    return state


class MPSChain:
    def __init__(self, n):
        self.n = n
        self.tensors = [np.zeros((1, d, 1), dtype=complex) for _ in range(n)]
        for t in self.tensors:
            t[0, 0, 0] = 1.0

    def apply_1q(self, gate, q):
        self.tensors[q] = np.einsum("ij,ajb->aib", gate, self.tensors[q], optimize=True)

    def _pad(self, t, max_bond):
        chi_l, dd, chi_r = t.shape
        out = np.zeros((max_bond, dd, max_bond), dtype=complex)
        out[:chi_l, :, :chi_r] = t
        return out

    def apply_2q_bond_sequential(self, gate4, q1, q2, chi_max):
        theta = np.einsum("aim,mjb,ijkl->aklb", self.tensors[q1], self.tensors[q2], gate4, optimize=True)
        chi_l, d1, d2, chi_r = theta.shape
        mat = theta.reshape(chi_l * d1, d2 * chi_r)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        chi_new = min(chi_max, len(s))
        u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
        self.tensors[q1] = u.reshape(chi_l, d1, chi_new)
        self.tensors[q2] = np.einsum("k,kjb->kjb", s, vh.reshape(chi_new, d2, chi_r), optimize=True)

    def apply_2q_batch(self, gate4, bonds, chi_max):
        """Reference NumPy version of the batched update: pads every
        bond's theta to a COMMON (chi_max*d, d*chi_max) shape only so
        they can be stacked into one array for the batched SVD call --
        the real JAX version replaces np.linalg.svd's per-item loop
        with a single jax.vmap(jnp.linalg.svd) call, dispatched once.
        Each bond's own true (possibly smaller, e.g. chi=1 at the chain
        boundary) left/right dimension is tracked separately and the
        output is trimmed back to it -- the padding is a batching
        convenience for the SVD call, not a change to any tensor's real
        shape, exactly like this project's own production padding for
        run_circuit_jit."""
        real_shapes = []
        thetas = []
        for i in bonds:
            theta = np.einsum("aim,mjb,ijkl->aklb", self.tensors[i], self.tensors[i + 1], gate4, optimize=True)
            chi_l, d1, d2, chi_r = theta.shape
            real_shapes.append((chi_l, chi_r))
            padded = np.zeros((chi_max, d, d, chi_max), dtype=complex)
            padded[:chi_l, :, :, :chi_r] = theta
            thetas.append(padded)
        batch = np.stack(thetas, axis=0).reshape(len(bonds), chi_max * d, d * chi_max)

        for idx, i in enumerate(bonds):
            chi_l, chi_r = real_shapes[idx]
            u, s, vh = np.linalg.svd(batch[idx], full_matrices=False)
            real_rank = min(chi_l * d, chi_r * d, chi_max)
            chi_new = min(real_rank, len(s))
            u, s, vh = u[:, :chi_new], s[:chi_new], vh[:chi_new, :]
            self.tensors[i] = u.reshape(chi_max, d, chi_new)[:chi_l]
            self.tensors[i + 1] = np.einsum("k,kjb->kjb", s, vh.reshape(chi_new, d, chi_max), optimize=True)[:, :, :chi_r]

    def to_statevector(self):
        v = self.tensors[0]
        for t in self.tensors[1:]:
            v = np.einsum("...a,ajb->...jb", v, t, optimize=True)
        return v.reshape(-1)


def run(n, dt, steps, chi_max):
    layers_dense = even_odd_tfim_gates(n, dt, steps, J=1.0, g=1.0)
    czz_gate = layers_dense[1][1]
    rx_half_gate = layers_dense[0][1]

    ref_state = dense_reference(n, layers_dense)

    chain_seq = MPSChain(n)
    for layer in layers_dense:
        if layer[0] == "1q_all":
            for q in range(n):
                chain_seq.apply_1q(rx_half_gate, q)
        else:
            _, gate, bonds = layer
            for i in bonds:
                chain_seq.apply_2q_bond_sequential(gate, i, i + 1, chi_max)
    seq_state = chain_seq.to_statevector()
    seq_fid = abs(np.vdot(ref_state, seq_state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(seq_state, seq_state).real)

    chain_batch = MPSChain(n)
    for layer in layers_dense:
        if layer[0] == "1q_all":
            for q in range(n):
                chain_batch.apply_1q(rx_half_gate, q)
        else:
            _, gate, bonds = layer
            chain_batch.apply_2q_batch(gate, bonds, chi_max)
    batch_state = chain_batch.to_statevector()
    batch_fid = abs(np.vdot(ref_state, batch_state)) ** 2 / (np.vdot(ref_state, ref_state).real * np.vdot(batch_state, batch_state).real)

    print(f"N={n} steps={steps} chi_max={chi_max}: "
          f"sequential-SVD fidelity={seq_fid:.12f}  batched-even-odd fidelity={batch_fid:.12f}")


if __name__ == "__main__":
    for chi_max in (2, 4, 8, 16):
        run(n=8, dt=0.1, steps=6, chi_max=chi_max)
