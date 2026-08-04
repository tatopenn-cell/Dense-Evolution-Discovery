"""
Topological Mott Isolator ("New Silicon") material design: VQE
ground-state optimization, validated against exact diagonalization.

A Haldane-Hubbard-flavored toy Hamiltonian (on-site Mott repulsion U +
nearest-neighbor hopping t1 + complex Haldane-phase next-nearest-
neighbor hopping t2) is swept over U. For every U, this script minimizes
E(theta) = <psi(theta)| H(U) |psi(theta)> over a hardware-efficient
RY-CX-RZ ansatz via gradient descent (Adam, exact JAX autodiff through
`dense_evolution.circuit_to_energy_fn`).

Every U point uses `N_STARTS` independent random initializations,
optimized in parallel via a single `jax.vmap`'d Adam loop (one JIT'd step
per epoch for the whole (U, start) grid at once), keeping the
lowest-energy result per U -- a real, standard defense against landing in
a bad local minimum, not a cosmetic detail (see the honest expressivity
gap reported below).

Correctness gate: the variational principle guarantees
E_vqe(U) >= E_exact_ground(U) for every U, where E_exact_ground comes
from direct dense diagonalization of the same (small, 2**N_QUBITS x
2**N_QUBITS) Hamiltonian matrix -- an independent reference the ansatz
never sees. main() and the test suite both assert this holds (up to
numerical tolerance) before anything downstream is trusted.

Honest result: the RY-CX-RZ ansatz used here reaches the exact ground
state at U=0 but its gap to the true ground energy grows with U (roughly
0.02 eV at U=0.55 up to ~0.5 eV at U=6.0) -- a real expressivity
limitation of this fixed circuit in the strongly-correlated regime, not
an optimizer failure (multi-start restarts consistently converge to the
same plateau at high U).

In addition to the arbitrary-unit U sweep, run_real_gaas_point() runs the
same pipeline at a physically-grounded point: t1 comes from
dft_gaas_valence_parameters.py's converged, wavefunction-stability-
confirmed PBE/STO-3G calculation at the real Ga-As nearest-neighbor bond
length (2.44 A, the actual zinc-blende distance). U is that same
calculation's on-site Coulomb integral, converted from its gas-phase
value to an approximate solid-state value by dividing by GaAs's real
static dielectric constant (epsilon_r = 12.9, a textbook-cited value,
e.g. Sze) -- the standard way to account for the screening a solid's
surrounding crystal provides that an isolated gas-phase pair cannot. GaAs
is not conventionally modeled as a Hubbard material, so there is no
literature-tabulated U to compare against directly; the derived U/t
ratio landing well below 1 is itself the cross-check (see README Section
19). t2/phi (the Haldane-phase term) have no independent real-GaAs
source at all -- no measured "topological phase" for this material --
and are kept at the same T2/T1 ratio as the toy sweep, an explicit
illustrative choice, not a physical parameter.

Produces `data/vqe_tmi_material_design.csv`,
`data/vqe_tmi_material_design_gaas_real.csv`, and
`images/vqe_tmi_material_design.png`.

    python scripts/vqe_tmi_material_design.py
"""
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de

jax.config.update("jax_enable_x64", True)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 4
T1 = 1.0
T2 = 0.3
PHI = np.pi / 3
U_RANGE = np.linspace(0.0, 6.0, 12)
N_STARTS = 8
N_EPOCHS = 400
LEARNING_RATE = 0.1

# Real GaAs point (see dft_gaas_valence_parameters.py): converged,
# wavefunction-stability-confirmed PBE/STO-3G SCF at the real Ga-As
# nearest-neighbor bond length (2.44 A). U_GAAS_BARE_EV is the raw
# gas-phase (unscreened) on-site Coulomb integral; U_GAAS_SCREENED_EV
# divides it by GaAs's real static dielectric constant (Sze) to
# approximate the solid-state value. T2_GAAS_EV has no independent real
# source and is kept at the toy sweep's T2/T1 ratio.
T1_GAAS_DFT_EV = 7.9170
U_GAAS_BARE_EV = 38.3847
GAAS_EPSILON_R = 12.9
U_GAAS_SCREENED_EV = U_GAAS_BARE_EV / GAAS_EPSILON_R
T2_GAAS_EV = 0.3 * T1_GAAS_DFT_EV


def build_tmi_hamiltonian(u_val, n_qubits=N_QUBITS, t1=T1, t2=T2, phi=PHI):
    """On-site Mott repulsion U (per 2-qubit "site", A = qubits[:n/2],
    B = qubits[n/2:]) + real nearest-neighbor hopping t1 + complex
    Haldane-phase next-nearest-neighbor hopping t2. Built by touching each
    unordered basis-state pair exactly once and setting both conjugate
    entries together, so the result is Hermitian by construction (see
    test_hamiltonian_is_hermitian)."""
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    half = n_qubits // 2
    for i in range(dim):
        bits = [(i >> q) & 1 for q in range(n_qubits)]
        n_A, n_B = sum(bits[:half]), sum(bits[half:])
        H[i, i] = u_val * (n_A * (n_A - 1) + n_B * (n_B - 1)) / 2.0

    for q in range(n_qubits - 1):
        mask1 = (1 << q) | (1 << (q + 1))
        for i in range(dim):
            j = i ^ mask1
            if j > i:
                H[i, j] += -t1
                H[j, i] += -t1

    phase = t2 * np.exp(1j * phi)
    for q in range(n_qubits - 2):
        mask2 = (1 << q) | (1 << (q + 2))
        for i in range(dim):
            j = i ^ mask2
            if j > i:
                H[i, j] += -phase
                H[j, i] += -np.conj(phase)
    return H


def exact_ground_energy(u_val, n_qubits=N_QUBITS, t1=T1, t2=T2, phi=PHI):
    """Independent reference: direct dense diagonalization, not seen by
    the VQE ansatz at any point."""
    H = build_tmi_hamiltonian(u_val, n_qubits, t1, t2, phi)
    return float(np.linalg.eigvalsh(H)[0])


def build_ansatz_energy_fn(n_qubits=N_QUBITS):
    """RY-CX-RZ hardware-efficient ansatz. The literal angles written in
    the QASM text (0.5, 0.2) are placeholders -- circuit_to_energy_fn
    overrides every parametric-gate slot positionally with theta, in
    circuit order (see its docstring), so only the gate COUNT and ORDER
    here matter, not these numbers."""
    qasm_header = f'OPENQASM 2.0; include "qelib1.inc"; qreg q[{n_qubits}]; creg c[{n_qubits}];'
    gates = (
        [f'ry(0.5) q[{i}];' for i in range(n_qubits)]
        + [f'cx q[{i}],q[{i + 1}];' for i in range(0, n_qubits - 1, 2)]
        + [f'rz(0.2) q[{i}];' for i in range(n_qubits)]
    )
    qasm = f"{qasm_header} " + " ".join(gates) + " " + " ".join(
        f"measure q[{i}] -> c[{i}];" for i in range(n_qubits)
    )
    circ = de.QASMParser().parse(qasm)
    return de.circuit_to_energy_fn(circ, circ.n_qubits)


def _adam_optimize(h_list, n_starts=N_STARTS, n_epochs=N_EPOCHS, lr=LEARNING_RATE, seed=0):
    """Core batched Adam loop shared by the U sweep and the real GaAs
    point: optimizes theta, minimizing E(theta, H) via exact JAX autodiff
    (jax.value_and_grad through circuit_to_energy_fn), for every
    (Hamiltonian, random start) pair at once via jax.vmap. Returns
    (initial_energy[point, start], final_energy[point, start],
    final_grad_norm[point, start])."""
    energy_fn, n_params = build_ansatz_energy_fn()
    n_points = len(h_list)

    h_single = jnp.stack([jnp.asarray(h, dtype=jnp.complex128) for h in h_list])
    dim = h_single.shape[-1]
    h_batch = jnp.repeat(h_single[:, None, :, :], n_starts, axis=1).reshape(n_points * n_starts, dim, dim)

    rng = np.random.default_rng(seed)
    theta_batch = jnp.asarray(rng.uniform(-np.pi, np.pi, (n_points * n_starts, n_params)))

    def single(theta, h):
        return energy_fn(theta, h)[0]

    value_and_grad_batched = jax.jit(jax.vmap(jax.value_and_grad(single), in_axes=(0, 0)))

    e_init, _ = value_and_grad_batched(theta_batch, h_batch)

    th = theta_batch
    m = jnp.zeros_like(th)
    v = jnp.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, n_epochs + 1):
        e, g = value_and_grad_batched(th, h_batch)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g ** 2)
        m_hat = m / (1 - b1 ** epoch)
        v_hat = v / (1 - b2 ** epoch)
        th = th - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    e_final, g_final = value_and_grad_batched(th, h_batch)

    reshape = lambda x: np.asarray(x).reshape(n_points, n_starts)
    grad_norm_final = np.linalg.norm(np.asarray(g_final).reshape(n_points, n_starts, n_params), axis=2)
    return reshape(e_init), reshape(e_final), grad_norm_final


def _best_over_starts(e_final, grad_norm):
    """Per-point minimum energy across random restarts, and the gradient
    norm at that winning restart."""
    idx = np.argmin(e_final, axis=1)
    rows = np.arange(e_final.shape[0])
    return e_final[rows, idx], grad_norm[rows, idx]


def vqe_optimize_batch(u_range, n_starts=N_STARTS, n_epochs=N_EPOCHS, lr=LEARNING_RATE, seed=0):
    """Real Adam optimization at every U in u_range (toy-unit sweep,
    default t1/t2/phi). See _adam_optimize for return shapes."""
    h_list = [build_tmi_hamiltonian(u) for u in u_range]
    return _adam_optimize(h_list, n_starts, n_epochs, lr, seed)


def run_experiment(u_range=U_RANGE, n_starts=N_STARTS, n_epochs=N_EPOCHS, lr=LEARNING_RATE, seed=0):
    """Returns a dict of per-U arrays: exact ground energy, best VQE
    energy across starts, an unoptimized single-random-theta baseline
    energy, and the final gradient norm at the winning start."""
    e_init, e_final, grad_norm = vqe_optimize_batch(u_range, n_starts, n_epochs, lr, seed)
    e_vqe_best, grad_norm_best = _best_over_starts(e_final, grad_norm)
    e_random_baseline = e_init[:, 0]
    e_exact = np.array([exact_ground_energy(u) for u in u_range])
    return {
        "U": np.asarray(u_range),
        "E_exact_ground": e_exact,
        "E_vqe_optimized": e_vqe_best,
        "E_random_baseline": e_random_baseline,
        "grad_norm_final": grad_norm_best,
    }


def run_real_gaas_point(n_starts=N_STARTS, n_epochs=N_EPOCHS, lr=LEARNING_RATE, seed=1):
    """Same VQE-vs-exact-diagonalization pipeline as run_experiment, at
    the real GaAs point: t1 = T1_GAAS_DFT_EV (real Ga-As bond length, DFT)
    and U = U_GAAS_SCREENED_EV (the DFT on-site integral, dielectrically
    screened by GaAs's real static dielectric constant)."""
    h_screened = build_tmi_hamiltonian(U_GAAS_SCREENED_EV, t1=T1_GAAS_DFT_EV, t2=T2_GAAS_EV, phi=PHI)

    e_init, e_final, grad_norm = _adam_optimize([h_screened], n_starts, n_epochs, lr, seed)
    e_vqe_best, grad_norm_best = _best_over_starts(e_final, grad_norm)
    e_exact = np.array([float(np.linalg.eigvalsh(h_screened)[0])])
    return {
        "label": np.array(["gaas"]),
        "U": np.array([U_GAAS_SCREENED_EV]),
        "t1": np.array([T1_GAAS_DFT_EV]),
        "E_exact_ground": e_exact,
        "E_vqe_optimized": e_vqe_best,
        "grad_norm_final": grad_norm_best,
    }


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    print("============================================================")
    print("Topological Mott Isolator: real VQE optimization vs. exact diagonalization")
    print("============================================================")
    result = run_experiment()

    gap = result["E_vqe_optimized"] - result["E_exact_ground"]
    assert np.all(gap > -1e-6), (
        f"Variational principle violated: VQE found an energy below the exact "
        f"ground state at U={result['U'][np.argmin(gap)]:.2f} (gap={gap.min():.6f})"
    )
    print("OK -- E_vqe(U) >= E_exact_ground(U) for every U (variational principle holds).\n")

    for u, e_ex, e_vqe, e_rand, g in zip(
        result["U"], result["E_exact_ground"], result["E_vqe_optimized"],
        result["E_random_baseline"], result["grad_norm_final"],
    ):
        print(f"U={u:5.2f} eV | exact={e_ex:+.4f} | VQE-optimized={e_vqe:+.4f} "
              f"| random baseline={e_rand:+.4f} | gap={e_vqe - e_ex:+.4f} | |grad|_final={g:.2e}")

    df = pd.DataFrame(result)
    df.to_csv(_DATA_DIR / "vqe_tmi_material_design.csv", index=False)

    print("\n============================================================")
    print("Real GaAs point: DFT-derived hopping, dielectrically-screened U")
    print("============================================================")
    print(f"t1 (DFT, real Ga-As bond 2.44 A): {T1_GAAS_DFT_EV:.4f} eV")
    print(f"U (DFT integral / epsilon_r={GAAS_EPSILON_R}):   {U_GAAS_SCREENED_EV:.4f} eV")

    gaas_result = run_real_gaas_point()
    gaas_gap = gaas_result["E_vqe_optimized"] - gaas_result["E_exact_ground"]
    assert np.all(gaas_gap > -1e-6), (
        f"Variational principle violated at the real GaAs point: gap={gaas_gap.min():.6f}"
    )
    print(f"exact={gaas_result['E_exact_ground'][0]:+.4f} eV | "
          f"VQE-optimized={gaas_result['E_vqe_optimized'][0]:+.4f} eV | "
          f"gap={gaas_gap[0]:+.4f} eV")

    u_t_ratio = U_GAAS_SCREENED_EV / T1_GAAS_DFT_EV
    print(f"\nU/t1 ratio: {u_t_ratio:.3f} -- deep in the weakly-correlated regime (U/t << 1), "
          f"consistent with GaAs being a conventional band semiconductor rather than a Mott insulator.")

    pd.DataFrame(gaas_result).to_csv(_DATA_DIR / "vqe_tmi_material_design_gaas_real.csv", index=False)

    # Two panels: the arbitrary-unit sweep (U in [0, 6] eV) and the real GaAs
    # point (U ~ 3 eV but t1 ~ 8 eV, a different absolute energy scale) don't
    # share a readable x-axis -- plotting them together crushes the sweep
    # into an unreadable sliver. Left panel: the sweep alone. Right panel:
    # exact vs. VQE-optimized energy at the real GaAs point.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(result["U"], result["E_exact_ground"], "g-", lw=2, label="Exact ground state (diagonalization)")
    ax1.plot(result["U"], result["E_vqe_optimized"], "b-o", lw=2, label=f"VQE-optimized ({N_STARTS} restarts, Adam)")
    ax1.plot(result["U"], result["E_random_baseline"], "r--", lw=1.5, label="Unoptimized theta (random)")
    ax1.set_xlabel("Mott repulsion U (eV)")
    ax1.set_ylabel("Ground-state energy (eV)")
    ax1.set_title(f"Arbitrary-unit U sweep (0-{U_RANGE[-1]:.0f} eV)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    x = np.arange(1)
    width = 0.35
    ax2.bar(x - width / 2, gaas_result["E_exact_ground"], width, color="#2ecc71", label="Exact ground state")
    ax2.bar(x + width / 2, gaas_result["E_vqe_optimized"], width, color="#3498db", label="VQE-optimized")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["GaAs"])
    ax2.set_xlim(-1, 1)
    ax2.set_ylabel("Ground-state energy (eV)")
    ax2.set_title(f"Real GaAs point (t1={T1_GAAS_DFT_EV:.3f} eV, U={U_GAAS_SCREENED_EV:.3f} eV)")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Topological Mott Isolator: VQE ground-state optimization")
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "vqe_tmi_material_design.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'vqe_tmi_material_design.png'}")
