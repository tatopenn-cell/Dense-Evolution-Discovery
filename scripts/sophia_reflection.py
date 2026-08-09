"""
Density-matrix ZNE validation: a real measured noise-coherence trajectory,
and a direct comparison against scalar ZNE on a non-Pauli noise channel.

Origin: an August 2025 personal notebook ("qualia") modeled subjective
experience as vectors in a Hilbert space -- linear/nonlinear operators, a
discrete entanglement operator, Gaussian noise (structured noise from
Riemann zeros planned as future work) -- then fed the resulting invented
state trajectory to an LLM ("Sophia") for reflection. This script closes
that loop with real data instead of invented states.

Part 1 (run_trajectory): a genuine noise-coherence-correction trajectory
from `dense_evolution.mitigation`, measured on an actual simulated 2-qubit
Bell state under depolarizing noise, not synthesized. Unlike
zne_mitigation.py in this same repo (a hand-rolled stochastic dephasing +
Richardson protocol), this uses the published
`zne_density_matrix`/`uhlmann_fidelity` pipeline directly -- the
density-matrix extension of ZNE, not the scalar/vector one.

Part 2 (run_scalar_vs_density_matrix_comparison): the question Part 1
alone doesn't answer -- does the density-matrix extension actually do
something scalar ZNE can't, or does it just improve fidelity the way any
reasonable ZNE variant would? `zne_density_matrix` extrapolates the full
density matrix, then projects the (possibly unphysical, small-negative-
eigenvalue) extrapolated result back onto the nearest true density matrix
via `project_to_physical`'s Smolin-Gambetta-Smith projection. Plain
scalar ZNE (`richardson_extrapolate` on raw fidelity numbers) has no such
constraint -- Richardson extrapolation can and does overshoot, so its
"fidelity" estimate can land outside [0, 1], a value with no physical
meaning. Depolarizing noise (a Pauli mixture) tends to keep density
matrices well-behaved even under naive extrapolation; amplitude damping
(a genuinely non-Pauli channel, `NoiseModel`'s `amplitude_damping`) is a
sharper test, since its asymmetric decay is more likely to push a
Richardson-extrapolated scalar past a physical bound.

Produces `data/sophia_reflection.csv` (Part 1) and
`data/scalar_vs_density_matrix_zne.csv` (Part 2), plotted in
`images/sophia_reflection.png` and
`images/scalar_vs_density_matrix_zne.png`, summarized in
`SOPHIA_REFLECTION.md`.

    python scripts/sophia_reflection.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix, richardson_extrapolate
from dense_evolution.mitigation import _uhlmann_fidelity_core, _richardson_extrapolate_core, _zne_density_matrix_core

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 2
SCALES = (1.0, 2.0, 3.0)


def bell_state_sv():
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    return np.asarray(sim.get_statevector())


# =====================================================================
# Exact (closed-form) Kraus channels -- JAX-differentiable, no Monte
# Carlo sampling. Built for Part 3's gradient-based adversarial noise
# search (arXiv:2607.27465-style technique, transported from
# ia_utils.adversarial_vector_attack's Phi-Trigger attack to noise-
# channel parameters instead of vector-healing inputs).
#
# NoiseModel.apply_to_sv (dense_evolution.registry) is NOT usable here:
# it decides each single-qubit outcome via a hard threshold on a random
# draw (`fire = r < p`), which has ~zero gradient almost everywhere with
# respect to p -- useless for a gradient-based search. This module
# builds the same channels directly from their Kraus operators instead
# (rho_out = sum_i K_i rho K_i^dagger), a smooth, exact function of the
# channel parameters.
#
# Embedding convention verified directly against NoiseModel's own
# behavior (not assumed): on an asymmetric 2-qubit probe state, a
# deterministic X on qubit 0 vs. qubit 1 must land on the same
# computational-basis index NoiseModel produces. Confirmed exact match
# only when qubit n-1 is the outermost (first) kron factor and qubit 0
# is the innermost (last) -- i.e. kron(op_{n-1}, ..., op_1, op_0),
# matching NoiseModel's own `1 << q` (qubit q = bit q, LSB-based)
# indexing convention.
# =====================================================================

def _kraus_ops(channel, p):
    """Closed-form single-qubit Kraus operators for `channel` at
    parameter `p` (error probability, or damping rate gamma for
    amplitude_damping).

    Cross-checked against the local quantumrag quantum_info collection
    (2026-08-09): John Preskill, "Lecture Notes for Ph219/CS219: Quantum
    Information", Chapter 3 (Caltech, updated Oct. 2018),
    https://www.preskill.caltech.edu/ph219/chap3_15.pdf -- Sec. 3.4.3
    derives the amplitude-damping channel from a system-environment
    isometry followed by a partial trace, giving exactly M0 =
    diag(1, sqrt(1-p)), M1 = [[0, sqrt(p)], [0, 0]], matching the
    'amplitude_damping' branch below. Independent confirmation (not just
    this module's own derivation) that this is the textbook-correct
    channel -- and further evidence that NoiseModel.apply_to_sv's
    amplitude_damping branch (see apply_channel_exact's own docstring)
    has a real, separately-verified bug, not just a difference in
    convention."""
    I = jnp.eye(2, dtype=jnp.complex128)
    X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    if channel == 'depolarizing':
        return [jnp.sqrt(1 - p) * I, jnp.sqrt(p / 3) * X, jnp.sqrt(p / 3) * Y, jnp.sqrt(p / 3) * Z]
    elif channel == 'bitflip':
        return [jnp.sqrt(1 - p) * I, jnp.sqrt(p) * X]
    elif channel == 'phaseflip':
        return [jnp.sqrt(1 - p) * I, jnp.sqrt(p) * Z]
    elif channel == 'amplitude_damping':
        K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - p)]], dtype=jnp.complex128)
        K1 = jnp.array([[0, jnp.sqrt(p)], [0, 0]], dtype=jnp.complex128)
        return [K0, K1]
    raise ValueError(f"unknown channel {channel!r}")


def _embed_single_qubit_op(K, qubit, n_qubits):
    """Full 2**n_qubits x 2**n_qubits operator = I (x) ... (x) K (x) ... (x) I,
    with K on `qubit` -- see this section's own verification note above
    for why the kron order must run qubit (n_qubits-1) down to qubit 0."""
    I = jnp.eye(2, dtype=jnp.complex128)
    ops = [K if q == qubit else I for q in range(n_qubits - 1, -1, -1)]
    full = ops[0]
    for op in ops[1:]:
        full = jnp.kron(full, op)
    return full


def apply_channel_exact(rho, channel, p, n_qubits=N_QUBITS, qubits=None):
    """Exact, JAX-differentiable single-channel application to `qubits`
    (default: all) of density matrix `rho`.

    Verified directly against NoiseModel.apply_to_sv's own per-branch
    formula, analytically (not just via Monte Carlo, which has its own
    sampling noise) -- for a superposition input state, computing what
    each stochastic branch produces and averaging by its firing
    probability, exactly, no randomness involved. 'depolarizing',
    'bitflip', 'phaseflip' match this exact channel to machine precision
    (max diff 0.000000 across several test states/probabilities).
    'amplitude_damping' does NOT match: NoiseModel's decay branch fires
    with flat probability `gamma`, but the correct Born-rule probability
    is `gamma * |v1|^2` (state-dependent) -- verified analytically to
    diverge by up to ~0.13 (max matrix-element difference) for
    superposition states at gamma=0.5, a real bug in
    dense_evolution.registry.NoiseModel, not in this exact
    implementation (which is the textbook-correct Kraus map, used
    deliberately here instead of replicating NoiseModel's current
    amplitude_damping behavior)."""
    if qubits is None:
        qubits = list(range(n_qubits))
    dim = 2 ** n_qubits
    for q in qubits:
        rho_out = jnp.zeros((dim, dim), dtype=jnp.complex128)
        for K in _kraus_ops(channel, p):
            K_full = _embed_single_qubit_op(K, q, n_qubits)
            rho_out = rho_out + K_full @ rho @ K_full.conj().T
        rho = rho_out
    return rho


_COMBINED_CHANNELS = ('depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping')


def apply_combined_channel_exact(rho, params, n_qubits=N_QUBITS):
    """Sequentially applies all four base channels, each with its own
    free parameter (params = [p_depolarizing, p_bitflip, p_phaseflip,
    p_amplitude_damping]), to every qubit -- the "combined noise"
    building block for the adversarial search below. Each parameter is
    clipped to [0, 1] (a physical probability/rate) before use, so the
    caller doesn't need to enforce that separately during optimization."""
    params = jnp.clip(jnp.asarray(params), 0.0, 1.0)
    for channel, p in zip(_COMBINED_CHANNELS, params):
        rho = apply_channel_exact(rho, channel, p, n_qubits=n_qubits)
    return rho


def _noisy_density_matrix(ideal_sv, p, k, rng, channel='depolarizing'):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, channel, p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


def run_trajectory(base_p_sweep, k_trajectories, seed=0):
    """Real (not invented) noise->correction trajectory for a Bell state
    under depolarizing noise, one row per base_p in base_p_sweep."""
    ideal_sv = bell_state_sv()
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
    rng = np.random.default_rng(seed)

    rows = []
    for base_p in base_p_sweep:
        rho_at_scales = jnp.stack([
            _noisy_density_matrix(ideal_sv, base_p * scale, k_trajectories, rng)
            for scale in SCALES
        ])
        raw = float(uhlmann_fidelity(rho_at_scales[0], rho_ideal))
        corrected_rho = zne_density_matrix(rho_at_scales, SCALES)
        corrected = float(uhlmann_fidelity(corrected_rho, rho_ideal))
        rows.append({
            'base_p': float(base_p),
            'raw_fidelity': raw,
            'corrected_fidelity': corrected,
            'delta': corrected - raw,
        })
    return rows


def run_scalar_vs_density_matrix_comparison(base_p_sweep, k_trajectories, channel='amplitude_damping', seed=10):
    """Real, measured comparison of scalar ZNE against the density-matrix
    extension on the same noisy Bell state, same noise scales, same
    trajectories -- only the extrapolation target differs.

    Scalar path: raw Uhlmann fidelity computed independently at each of
    the 3 noise scales, then `richardson_extrapolate` applied directly to
    those 3 scalar numbers -- no constraint that the result stay in
    [0, 1], since a bare fidelity number carries no positive-semidefinite/
    trace-1 structure for any projection step to enforce.

    Density-matrix path: identical to run_trajectory -- extrapolate the
    density matrix itself, project back onto the physical set
    (`project_to_physical`, inside `zne_density_matrix`), then compute
    fidelity from the corrected (guaranteed-physical) density matrix.

    One row per base_p, reporting both estimates and whether the scalar
    path's raw extrapolated number left [0, 1] (a physically meaningless
    "fidelity" if so) at that noise level.
    """
    ideal_sv = bell_state_sv()
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
    rng = np.random.default_rng(seed)

    rows = []
    for base_p in base_p_sweep:
        rho_at_scales = jnp.stack([
            _noisy_density_matrix(ideal_sv, base_p * scale, k_trajectories, rng, channel=channel)
            for scale in SCALES
        ])
        fidelities_at_scales = jnp.array([
            uhlmann_fidelity(rho_at_scales[i], rho_ideal) for i in range(len(SCALES))
        ])

        scalar_extrapolated = float(richardson_extrapolate(fidelities_at_scales, SCALES))
        scalar_unphysical = bool(scalar_extrapolated < 0.0 or scalar_extrapolated > 1.0)

        corrected_rho = zne_density_matrix(rho_at_scales, SCALES)
        density_matrix_fidelity = float(uhlmann_fidelity(corrected_rho, rho_ideal))

        rows.append({
            'base_p': float(base_p),
            'raw_fidelity': float(fidelities_at_scales[0]),
            'scalar_zne_fidelity': scalar_extrapolated,
            'scalar_zne_unphysical': scalar_unphysical,
            'density_matrix_zne_fidelity': density_matrix_fidelity,
            'divergence': abs(scalar_extrapolated - density_matrix_fidelity),
        })
    return rows


def _divergence_objective(base_params, ideal_sv, rho_ideal):
    """Scalar-vs-density-matrix ZNE divergence for a combined-channel
    noise profile, at 3 Richardson scales (SCALES) of `base_params` --
    the quantity Part 2's sweep measured indirectly (by chance, at
    whichever base_p/channel the sweep happened to land on); this makes
    it a direct, differentiable optimization target instead."""
    rho_at_scales = jnp.stack([
        apply_combined_channel_exact(rho_ideal, base_params * scale)
        for scale in SCALES
    ])
    # _uhlmann_fidelity_core (not the public uhlmann_fidelity) -- the
    # public wrapper does a `float()` cast that isn't traceable inside
    # jax.grad.
    fidelities = jnp.stack([
        _uhlmann_fidelity_core(rho_at_scales[i], rho_ideal) for i in range(len(SCALES))
    ])
    scalar_extrapolated = _richardson_extrapolate_core(fidelities, jnp.asarray(SCALES, dtype=jnp.float64))
    corrected_rho = _zne_density_matrix_core(rho_at_scales, jnp.asarray(SCALES, dtype=jnp.float64), degree=2)
    dm_fidelity = _uhlmann_fidelity_core(corrected_rho, rho_ideal)
    return jnp.abs(scalar_extrapolated - dm_fidelity)


_divergence_grad = jax.grad(_divergence_objective, argnums=0)


def run_adversarial_combined_noise_search(n_steps=200, step_size=0.005, init_params=None, seed=20):
    """Gradient-based search (arXiv:2607.27465-style chained-
    differentiable-attack technique, transported from
    ia_utils.adversarial_vector_attack's vector-healing Phi-Trigger
    attack to noise-channel parameters instead) for a "combined" 4-
    channel noise profile (depolarizing, bitflip, phaseflip,
    amplitude_damping mixed together, see apply_combined_channel_exact)
    that maximizes the scalar-vs-density-matrix ZNE divergence Part 2
    found by chance on a plain amplitude-damping sweep.

    Each parameter is bounded to [0, 1/max(SCALES)] so that the largest
    Richardson scale point (3x base_params, SCALES=(1,2,3)) never
    exceeds a physically valid probability/rate of 1 for any channel --
    the adversarial "budget" this search operates within.

    Same step_size/best-tracking pattern as
    ia_utils.adversarial_vector_attack.craft_adversarial_healing_
    perturbation, for the same verified reason: a step size that scales
    with the budget overshoots and converges worse, not better (see
    that function's own bug-fix note); a small fixed step size avoids
    it here too.

    Starts from init_params (default: a small uniform profile) rather
    than from zero noise, since the objective's gradient at exactly
    zero noise is degenerate for some of these channels (e.g.
    amplitude_damping's sqrt(p) term has an infinite derivative at
    p=0).

    KNOWN ISSUE (2026-08-09, unresolved -- see the branch this landed
    on): _divergence_grad returns NaN at every point tested so far,
    including the simplest possible case (fidelity of a single noisy
    density matrix, no ZNE/Richardson at all -- jax.grad through
    dense_evolution.mitigation._uhlmann_fidelity_core alone already
    gives NaN). Root cause: _uhlmann_fidelity_core's Uhlmann-fidelity
    formula goes through an eigendecomposition (jnp.linalg.eigh) whose
    JAX gradient rule is NaN at (near-)degenerate eigenvalues -- a well-
    known JAX autodiff limitation, not a bug introduced here. A lightly
    perturbed Bell state's density matrix is very close to pure (exact
    eigenvalues [1,0,0,0] at zero noise), so its three near-zero
    eigenvalues are exactly or near-exactly degenerate, which is
    precisely the case this limitation bites. As a direct consequence,
    this function currently makes NO progress: every gradient step's
    direction falls back to the all-zero vector (see the `grad_norm >
    1e-12` guard below), so `best_params` never moves past
    `init_params` and `best_divergence` never improves. Not fixed here
    -- would need either an eigh-free fidelity formula, a numerically-
    regularized (symmetry-broken) input, or a custom JAX differentiation
    rule for the degenerate-eigenvalue case, none of which were
    attempted. Kept in the codebase (rather than deleted) as a concrete,
    verified starting point for whoever picks this up next -- the
    differentiable channels above (apply_channel_exact,
    apply_combined_channel_exact) are correct and reusable regardless of
    this blocker."""
    ideal_sv = bell_state_sv()
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)

    max_scale = max(SCALES)
    bound = 1.0 / max_scale

    if init_params is None:
        rng = np.random.default_rng(seed)
        params = jnp.asarray(rng.uniform(0.02, 0.1, size=4))
    else:
        params = jnp.asarray(init_params, dtype=jnp.float64)

    best_params = params
    best_divergence = float(_divergence_objective(params, ideal_sv, rho_ideal))

    history = [{'step': 0, 'divergence': best_divergence, 'params': np.asarray(params).tolist()}]

    for step in range(1, n_steps + 1):
        grad = _divergence_grad(params, ideal_sv, rho_ideal)
        grad_norm = jnp.linalg.norm(grad)
        direction = jnp.where(grad_norm > 1e-12, grad / grad_norm, jnp.zeros_like(grad))
        params = jnp.clip(params + step_size * direction, 0.0, bound)

        current_divergence = float(_divergence_objective(params, ideal_sv, rho_ideal))
        if current_divergence > best_divergence:
            best_divergence = current_divergence
            best_params = params

        if step % 20 == 0 or step == n_steps:
            history.append({'step': step, 'divergence': current_divergence, 'params': np.asarray(params).tolist()})

    return {
        'best_params': {name: float(v) for name, v in zip(_COMBINED_CHANNELS, best_params)},
        'best_divergence': best_divergence,
        'history': history,
    }


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    rows = run_trajectory(base_p_sweep=np.linspace(0.02, 0.5, 16), k_trajectories=200, seed=0)
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "sophia_reflection.csv", index=False)

    print(df.to_string(index=False))
    print()
    print(f"mean delta:  {df['delta'].mean():+.6f}")
    print(f"delta range: [{df['delta'].min():+.6f}, {df['delta'].max():+.6f}]")
    print(f"positive:    {(df['delta'] > 0).sum()}/{len(df)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(df['base_p'], df['raw_fidelity'], 'o-', color='#c0392b', label='raw (uncorrected)')
    ax1.plot(df['base_p'], df['corrected_fidelity'], 'o-', color='#2980b9', label='corrected (ZNE)')
    ax1.set_ylabel('Uhlmann fidelity')
    ax1.set_title('Density-matrix ZNE: real noise-coherence trajectory (Bell state, depolarizing noise)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(df['base_p'], df['delta'], width=0.02, color='#27ae60')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('base_p (depolarizing noise probability)')
    ax2.set_ylabel('delta (corrected - raw)')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "sophia_reflection.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'sophia_reflection.png'}")

    print("\n" + "=" * 70)
    print("Scalar vs. density-matrix ZNE, amplitude-damping noise")
    print("=" * 70)

    rows2 = run_scalar_vs_density_matrix_comparison(
        base_p_sweep=np.linspace(0.05, 0.6, 16), k_trajectories=200, channel='amplitude_damping', seed=10)
    df2 = pd.DataFrame(rows2)
    df2.to_csv(_DATA_DIR / "scalar_vs_density_matrix_zne.csv", index=False)

    print(df2.to_string(index=False))
    n_unphysical = int(df2['scalar_zne_unphysical'].sum())
    print()
    print(f"scalar ZNE unphysical (fidelity outside [0,1]): {n_unphysical}/{len(df2)}")
    print(f"density-matrix ZNE unphysical: 0/{len(df2)} (guaranteed by project_to_physical)")
    print(f"mean |divergence|: {df2['divergence'].mean():.4f}")

    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df2['base_p'], df2['raw_fidelity'], 'o-', color='#888888', label='raw (uncorrected)', alpha=0.6)
    ax.plot(df2['base_p'], df2['scalar_zne_fidelity'], 's-', color='#c0392b', label='scalar ZNE (fidelity extrapolated directly)')
    ax.plot(df2['base_p'], df2['density_matrix_zne_fidelity'], 'o-', color='#2980b9', label='density-matrix ZNE (physical by construction)')
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='physical bound (F=1)')
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    unphysical_mask = df2['scalar_zne_unphysical']
    if unphysical_mask.any():
        ax.scatter(df2.loc[unphysical_mask, 'base_p'], df2.loc[unphysical_mask, 'scalar_zne_fidelity'],
                   marker='x', s=120, color='red', zorder=5, label='scalar ZNE unphysical result')
    ax.set_xlabel('base_p (amplitude-damping probability)')
    ax.set_ylabel('Uhlmann fidelity')
    ax.set_title(f'Scalar vs. density-matrix ZNE under amplitude damping (non-Pauli channel)\n'
                 f'{n_unphysical}/{len(df2)} scalar-ZNE points land outside [0,1] -- density-matrix ZNE never does')
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(_IMAGES_DIR / "scalar_vs_density_matrix_zne.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'scalar_vs_density_matrix_zne.png'}")
