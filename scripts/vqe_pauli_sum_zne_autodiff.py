"""
The "killer example" from prog.txt Sezione 4.3: ZNE + a Pauli-sum
Hamiltonian + autodiff VQE on a real molecule, in the same JAX trace --
something the dense-statevector-only textbook VQE recipe (energy =
<psi| H_dense |psi>, autodiff through H_dense) structurally cannot scale
to, because the DENSE Hamiltonian matrix, not the statevector, is what
runs out of memory first.

WHY THE DENSE HAMILTONIAN IS THE REAL WALL, NOT THE STATEVECTOR:
a statevector is O(2**n) -- 2**20 complex128 = 16MB, no problem at all
even at 20 qubits. A dense Hamiltonian is O(4**n) -- 2**24 complex128
(12 qubits, this script's system) is already 268MB; 2**28 (14 qubits) is
4GB, the practical ceiling on an 8GB laptop; 2**32 (16 qubits) is 64GB,
impossible on any laptop. circuit_to_energy_fn's own energy_fn(theta,
h_matrix, ...) computes ``h_matrix @ statevector`` and needs h_matrix
densely materialized -- so autodiff VQE through it hits that wall well
before the statevector itself would.

THE FIX, promoted to dense_evolution as part of this experiment:
PauliSumOperator (dense_evolution.physics.observables) wraps a Pauli-sum
Hamiltonian (the same terms format pauli_hamiltonian_to_matrix accepts)
behind ``__matmul__``, backed by the new pauli_sum_matvec_jax -- a pure-jnp,
jax.grad/jax.jit-traceable H @ vector that never builds the dense matrix.
Passed as circuit_to_energy_fn's h_matrix argument, it is a drop-in
replacement (h_matrix is only ever used as ``h_matrix @ statevector``) that
turns "differentiable VQE" and "avoid the O(4**n) Hamiltonian" from a
tradeoff into a combination.

WHY 12 QUBITS, NOT A LARGER NUMBER: this experiment targets a real chain
of 10 hydrogen atoms (H10, STO-3G, the standard VQE-scaling benchmark in
the literature), which needs exactly 20 qubits at full active space --
verified directly (`dashboard_core.hamiltonians.get_molecule_n_qubits`).
At full active space that Hamiltonian has 7151 Pauli terms; the first
jax.jit COMPILE of that many unrolled term-blocks measured ~4.3GB peak
RAM on this project's own 8GB development machine (dominated by XLA
graph-building overhead, not the actual O(dim) per-term compute) --
workable on a bigger machine, but not the "runs comfortably on a modest
laptop or a free Colab instance" bar this demo is meant to clear. Cutting
the active space to 6 of H10's 10 spatial orbitals (`active_electrons=10,
active_orbitals=6`) keeps the same molecule and mapping but gives 12
qubits / 919 terms, measured to compile in ~52s and run in ~0.3s per
already-compiled gradient step -- comfortably light. The physics point
(dense Hamiltonian already impossible, PauliSumOperator unaffected) holds
at 12 qubits too: 2**24 complex128 = 268MB is already the largest dense
Hamiltonian this demo's own reference diagonalization below is willing to
build, and the wall only gets steeper from there.

WHAT ZNE ADDS ON TOP: the noiseless-optimized ansatz is evaluated at 3
depolarizing-noise scales (1x, 2x, 3x) and Richardson-extrapolated
(`zero_noise_extrapolation`) back to an estimated zero-noise energy --
the standard error-mitigation pattern, run here on the exact same
PauliSumOperator-based energy function used for the noiseless optimization,
not a separate code path.

Produces `data/vqe_pauli_sum_zne_autodiff.csv` (VQE convergence trace),
`images/vqe_pauli_sum_zne_autodiff.png` (convergence + ZNE panel).

    python scripts/vqe_pauli_sum_zne_autodiff.py
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax

import dense_evolution as de
from dense_evolution.physics.observables import pauli_hamiltonian_to_matrix, pauli_sum_matvec_jax

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_CSV = _REPO_ROOT / "data" / "vqe_pauli_sum_zne_autodiff.csv"
OUT_PNG = _REPO_ROOT / "images" / "vqe_pauli_sum_zne_autodiff.png"

N_QUBITS = 12
N_LAYERS = 2
N_STEPS = 80
LEARNING_RATE = 0.1
NOISE_FACTORS = (1.0, 2.0, 3.0)
BASE_DEPOLARIZING_P = 0.05
N_NOISE_SHOTS = 40


def selftest_pauli_sum_operator_matches_dense(seed: int = 0) -> None:
    """PauliSumOperator's whole point is agreeing with the dense
    h_matrix @ statevector path it replaces -- verified here on THIS
    script's own ansatz/Hamiltonian shape (n_qubits=4 subset, cheap),
    not just trusted from dense_evolution's own unit tests."""
    rng = np.random.default_rng(seed)
    n_qubits = 4
    letters = ["I", "X", "Y", "Z"]
    terms = [
        (float(rng.normal()), "".join(rng.choice(letters) for _ in range(n_qubits)))
        for _ in range(10)
    ]
    h_dense = pauli_hamiltonian_to_matrix(terms, n_qubits)
    h_op = de.PauliSumOperator(terms, n_qubits)

    v = jnp.asarray(rng.normal(size=2 ** n_qubits) + 1j * rng.normal(size=2 ** n_qubits))
    expected = h_dense @ np.asarray(v)
    actual = np.asarray(h_op @ v)
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff < 1e-9, f"PauliSumOperator selftest: max_diff={max_diff:.2e}"
    print(f"selftest_pauli_sum_operator_matches_dense: OK (max_diff={max_diff:.2e})")


def hardware_efficient_ansatz_qasm(n_qubits: int, n_layers: int) -> str:
    """RY rotation layer + linear CX entangling layer, repeated
    n_layers times -- the standard hardware-efficient VQE ansatz shape.
    Every ry parameter is a sentinel (circuit_to_energy_fn discards the
    literal value written here and injects theta by order of appearance
    -- see _build_template's docstring), so the literal 0.0 below is
    just a placeholder, never actually used."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n_qubits}];"]
    for _ in range(n_layers):
        for q in range(n_qubits):
            lines.append(f"ry(0.0) q[{q}];")
        for q in range(n_qubits - 1):
            lines.append(f"cx q[{q}],q[{q + 1}];")
    return "\n".join(lines)


def build_h10_active_space_terms(active_orbitals: int = 6):
    """H10 (linear chain of 10 hydrogens, STO-3G, bond length 0.74
    Angstrom -- the standard literature VQE-scaling benchmark), reduced
    to `active_orbitals` of its 10 spatial orbitals (2*active_orbitals
    qubits) via PennyLane's frozen-core active-space selection. Returns
    (terms, n_qubits) in this package's own Pauli-term format, ready for
    PauliSumOperator -- never a dense Hamiltonian matrix at any point."""
    import dashboard_core.hamiltonians as dh

    geometry = dh.linear_chain_geometry(10, 0.74)
    H, n_qubits = dh._get_hamiltonian(
        ["H"] * 10, geometry, 0, "jordan_wigner", 10, active_orbitals
    )
    terms = dh._pennylane_hamiltonian_to_pauli_terms(H, n_qubits)
    return terms, n_qubits


def exact_ground_state_energy(terms, n_qubits: int) -> float:
    """Reference only -- dense diagonalization, deliberately never used
    in the VQE/ZNE path itself (that stays PauliSumOperator-based
    throughout). dim=2**12=4096 makes this a tiny, fast dense matrix
    (268MB) -- fine as a one-off correctness check, exactly the kind of
    matrix this script's whole point is to NOT need for the actual
    optimization."""
    H_dense = pauli_hamiltonian_to_matrix(terms, n_qubits)
    eigvals = np.linalg.eigvalsh(H_dense)
    return float(np.min(np.real(eigvals)))


def main():
    # Real quantum chemistry needs float64 -- dense_evolution normally
    # enables this lazily (the first DenseSVSimulator/circuit_to_energy_fn
    # construction), but this script's own selftest calls
    # PauliSumOperator/pauli_sum_matvec_jax BEFORE any of those run, so
    # it would otherwise silently execute at JAX's float32 default
    # (verified directly: without this line, the selftest below fails at
    # max_diff=7.10e-07 -- well above float64 tolerance, exactly float32
    # relative precision -- instead of the ~1e-16 machine-precision
    # agreement float64 gives).
    de.set_precision(True)

    selftest_pauli_sum_operator_matches_dense()

    print(f"Building H10 Hamiltonian (active_orbitals=6 -> {N_QUBITS} qubits)...")
    terms, n_qubits = build_h10_active_space_terms(active_orbitals=6)
    assert n_qubits == N_QUBITS
    print(f"n_qubits={n_qubits}, n_pauli_terms={len(terms)}")

    print("Exact ground-state energy (dense reference, one-off)...")
    e_exact = exact_ground_state_energy(terms, n_qubits)
    print(f"E_exact = {e_exact:.6f} Ha")

    qasm = hardware_efficient_ansatz_qasm(n_qubits, N_LAYERS)
    circuit = de.QASMParser().parse(qasm)
    energy_fn, n_params = de.circuit_to_energy_fn(circuit, n_qubits)
    h_op = de.PauliSumOperator(terms, n_qubits)
    print(f"ansatz: {N_LAYERS} layers, n_params={n_params}")

    @jax.jit
    def loss(theta):
        energy, _ = energy_fn(theta, h_op)
        return energy

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    @jax.jit
    def noisy_energy(theta, p, key):
        # Same reason the VQE loss above is jax.jit-wrapped: energy_fn's
        # PauliSumOperator path loops over all 919 Pauli terms, and that
        # loop only unrolls into one fast fused XLA kernel under jit --
        # called eagerly (as bare energy_fn(...) would be) it re-dispatches
        # each term's jnp ops one at a time, exactly the slow path already
        # measured earlier in this project at 20 qubits/7151 terms. With
        # N_NOISE_SHOTS * len(NOISE_FACTORS) calls needed for the ZNE
        # average below, jitting this once and reusing it matters here too.
        noise = de.NoiseSpec(model="depolarizing", p=p, jax_key=key)
        energy, _ = energy_fn(theta, h_op, noise=noise)
        return energy

    key = jax.random.PRNGKey(0)
    theta = 0.1 * jax.random.normal(key, (n_params,))
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(theta)

    print(f"VQE: {N_STEPS} Adam steps, all through PauliSumOperator (no dense H at any point)...")
    trace = []
    for step in range(N_STEPS):
        energy, grad = value_and_grad(theta)
        updates, opt_state = optimizer.update(grad, opt_state)
        theta = optax.apply_updates(theta, updates)
        trace.append((step, float(energy)))
        if step % 20 == 0 or step == N_STEPS - 1:
            print(f"  step {step:3d}: E = {float(energy):.6f} Ha")

    e_vqe = trace[-1][1]
    print(f"E_vqe (noiseless, converged) = {e_vqe:.6f} Ha")
    print(f"|E_vqe - E_exact| = {abs(e_vqe - e_exact):.6f} Ha "
          f"(hardware-efficient ansatz, not expected to reach chemical accuracy)")

    print(f"ZNE: evaluating the converged ansatz at noise scales {NOISE_FACTORS} "
          f"({N_NOISE_SHOTS} trajectories per scale, averaged)...")
    # NoiseModel.apply_to_sv is a single-shot stochastic Kraus sample (one
    # random fire/no-fire draw per qubit per call), not an ensemble average
    # -- its own docstring says so directly ("Run it many times and average
    # ... to see what a real noisy device's *typical* output looks like, not
    # just one random draw"). A single draw at p=0.02-0.06 across only 12
    # qubits has a real chance of landing on "no error fired at all" (~78%
    # per-qubit survival at p=0.02, compounding), which is exactly what a
    # first version of this script did -- all 3 scales returned bit-identical
    # energies, an artifact of 3 lucky no-error draws, not evidence that
    # noise had no effect. Averaging over many trajectories per scale is the
    # fix, not a larger p (matches this project's own zne_density_matrix
    # precedent of K=200-400 trajectories per noise scale).
    noisy_energies = []
    for scale in NOISE_FACTORS:
        p = BASE_DEPOLARIZING_P * scale
        shot_energies = []
        for shot in range(N_NOISE_SHOTS):
            key = jax.random.PRNGKey(int(scale * 100000) + shot)
            e_noisy = noisy_energy(theta, p, key)
            shot_energies.append(float(e_noisy))
        mean_e = float(np.mean(shot_energies))
        std_e = float(np.std(shot_energies))
        noisy_energies.append(mean_e)
        print(f"  scale={scale:.1f}x (p={p:.4f}): E = {mean_e:.6f} +/- {std_e:.6f} Ha "
              f"({N_NOISE_SHOTS} shots)")

    e_zne = float(de.zero_noise_extrapolation(jnp.array(noisy_energies), jnp.array(NOISE_FACTORS)))
    print(f"E_zne (Richardson-extrapolated to zero noise) = {e_zne:.6f} Ha")
    print(f"raw base-noise error  = {abs(noisy_energies[0] - e_vqe):.6f} Ha")
    print(f"ZNE-corrected error   = {abs(e_zne - e_vqe):.6f} Ha")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy_ha"])
        writer.writerows(trace)
        writer.writerow([])
        writer.writerow(["quantity", "value_ha"])
        writer.writerow(["e_exact", e_exact])
        writer.writerow(["e_vqe_noiseless", e_vqe])
        for scale, e_noisy in zip(NOISE_FACTORS, noisy_energies):
            writer.writerow([f"e_noisy_scale_{scale}", e_noisy])
        writer.writerow(["e_zne", e_zne])
    print(f"Saved {OUT_CSV}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    steps = [t[0] for t in trace]
    energies = [t[1] for t in trace]
    ax1.plot(steps, energies, color="#1f77b4", linewidth=1.6, label="VQE energy")
    ax1.axhline(e_exact, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"exact ({e_exact:.4f} Ha)")
    ax1.set_xlabel("Adam step")
    ax1.set_ylabel("Energy (Ha)")
    ax1.set_title(f"VQE convergence, H10 active space ({n_qubits} qubits)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(NOISE_FACTORS, noisy_energies, "o-", color="#d62728", label="noisy (measured)")
    ax2.scatter([0.0], [e_zne], color="#9467bd", zorder=5, s=60, label=f"ZNE ({e_zne:.4f} Ha)")
    ax2.axhline(e_vqe, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"noiseless ({e_vqe:.4f} Ha)")
    ax2.set_xlabel("Noise scale factor")
    ax2.set_ylabel("Energy (Ha)")
    ax2.set_title("Zero-Noise Extrapolation")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
