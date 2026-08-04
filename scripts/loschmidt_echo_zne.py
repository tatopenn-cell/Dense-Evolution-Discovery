"""
Loschmidt echo: does dense_evolution's ZNE recover the return fidelity of a
chaotic forward/backward ("kicked Ising") circuit under noise?

A genuine "kicked Ising" forward circuit is run, then its exact inverse
backward, with noise injected as a Kraus channel between every layer
(forward and backward) and reinjected into the simulator via
set_initial_state so each subsequent layer of unitary evolution acts on
the actually-noisy state.

Physical model: fixed transverse "kick" (RX) + per-step random
longitudinal disorder field (RZ) + nearest-neighbor CX coupling -- a
standard toy model for quantum chaos (kicked Ising chain). U^-1 for this
circuit is exact and cheap to build: RZ(theta)^-1 = RZ(-theta),
RX(theta)^-1 = RX(-theta), CX^-1 = CX, applied to the same gates in
reverse order.

Correctness gate: noiseless_echo_fidelity() must return exactly 1.0 (up
to floating point) -- if forward-then-backward doesn't return to the
initial state with zero noise, invert_layer() is wrong and nothing
downstream (the noisy fidelities, the ZNE correction) can be trusted.
main() asserts this before running anything noisy; the test suite
(tests/test_loschmidt_echo_zne.py) asserts it independently too.

Produces `data/loschmidt_echo_zne.csv` and `images/loschmidt_echo_zne.png`.

    python scripts/loschmidt_echo_zne.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import zne_density_matrix, uhlmann_fidelity

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 4
STEPS = 4
KICK_ANGLE = np.pi / 4
SCALES = (1.0, 1.5, 2.0)
K_TRAJECTORIES = 300
BASE_P = 0.015
NOISE_CHANNEL = "amplitude_damping"


def initial_state(sim=None):
    """|0101> (X on qubits 0 and 2) under a fresh or reused simulator."""
    if sim is None:
        sim = de.DenseSVSimulator(N_QUBITS)
    sim.set_initial_state(None)
    sim.run_circuit([("x", 0), ("x", 2)])
    return sim


def build_forward_layers(rng):
    """One list of gate-tuples per Trotter step (not flattened), so noise
    can be injected between steps."""
    h_fields = rng.uniform(-2.0, 2.0, (STEPS, N_QUBITS))
    layers = []
    for step in range(STEPS):
        layer = []
        for i in range(N_QUBITS):
            layer.append(("rz", i, float(h_fields[step, i])))
        for i in range(N_QUBITS):
            layer.append(("rx", i, KICK_ANGLE))
        for i in range(0, N_QUBITS - 1, 2):
            layer.append(("cx", i, i + 1))
        layers.append(layer)
    return layers


def invert_layer(layer):
    """U^-1 for one forward layer: same gates, reverse order, negated
    rotation angles (CX is its own inverse)."""
    inverted = []
    for op in reversed(layer):
        name = op[0]
        if name in ("rz", "rx"):
            inverted.append((name, op[1], -op[2]))
        else:  # cx
            inverted.append(op)
    return inverted


def run_echo(forward_layers, p, seed, sim=None, channel=NOISE_CHANNEL):
    """Real forward evolution, then real backward (inverse) evolution, with
    a Kraus channel injected between every single layer. p=0.0 must
    reproduce the initial state exactly -- see noiseless_echo_fidelity()."""
    rng = np.random.default_rng(seed)
    sim = initial_state(sim)

    for layer in forward_layers:
        sim.run_circuit(layer, transpile=False)
        if p > 0:
            sv = NoiseModel.apply_to_sv(np.asarray(sim.get_statevector()), N_QUBITS, channel, p, rng=rng)
            sim.set_initial_state(sv)

    for layer in reversed(forward_layers):
        sim.run_circuit(invert_layer(layer), transpile=False)
        if p > 0:
            sv = NoiseModel.apply_to_sv(np.asarray(sim.get_statevector()), N_QUBITS, channel, p, rng=rng)
            sim.set_initial_state(sv)

    return np.asarray(sim.get_statevector())


def noiseless_echo_fidelity(seed=0):
    """p=0 forward+backward return fidelity -- must be 1.0 (see module
    docstring). Real correctness gate, not a formality."""
    rng = np.random.default_rng(seed)
    forward_layers = build_forward_layers(rng)
    sim = initial_state()
    ideal_init_sv = np.asarray(sim.get_statevector())

    final_sv = run_echo(forward_layers, p=0.0, seed=seed, sim=sim)
    return float(np.abs(np.vdot(ideal_init_sv, final_sv)) ** 2)


def sample_return_density_matrix(forward_layers, p, k_trajectories, seed_offset=0):
    """Monte Carlo estimate of the noisy return density matrix at noise
    strength p, averaged over k_trajectories independent Kraus
    realizations."""
    dim = 2 ** N_QUBITS
    rho = np.zeros((dim, dim), dtype=np.complex128)
    sim = de.DenseSVSimulator(N_QUBITS)
    for t in range(k_trajectories):
        sv_noisy = run_echo(forward_layers, p, seed=seed_offset + t, sim=sim)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    return rho / k_trajectories


def run_experiment(seed=42, scales=SCALES, k_trajectories=K_TRAJECTORIES, base_p=BASE_P):
    """Returns (raw_fidelity, corrected_fidelity, rho_at_scales, rho_target)."""
    rng = np.random.default_rng(seed)
    forward_layers = build_forward_layers(rng)

    sim_ideal = initial_state()
    ideal_init_sv = np.asarray(sim_ideal.get_statevector())
    rho_target = jnp.asarray(np.outer(ideal_init_sv, ideal_init_sv.conj()), dtype=jnp.complex128)

    rho_at_scales = jnp.stack([
        jnp.asarray(sample_return_density_matrix(forward_layers, base_p * scale, k_trajectories), dtype=jnp.complex128)
        for scale in scales
    ])

    raw_fidelity = float(uhlmann_fidelity(rho_at_scales[0], rho_target))
    rho_corrected = zne_density_matrix(rho_at_scales, scales)
    corrected_fidelity = float(uhlmann_fidelity(rho_corrected, rho_target))
    return raw_fidelity, corrected_fidelity, rho_at_scales, rho_target


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    print("Self-check: noiseless echo must be exact...")
    fidelity0 = noiseless_echo_fidelity()
    print(f"[self-check] noiseless echo fidelity = {fidelity0:.12f} (must be 1.0)")
    assert abs(fidelity0 - 1.0) < 1e-9, "Inverse circuit is wrong -- echo does not return to the initial state"
    print("OK -- the forward/backward circuit is a genuine, verified echo.\n")

    print("Sampling the noisy echo at each noise scale...")
    raw_fidelity, corrected_fidelity, rho_at_scales, rho_target = run_experiment()

    print("\n--- LOSCHMIDT ECHO: kicked-Ising forward/backward circuit + ZNE ---")
    print(f"Raw noisy return fidelity:       {raw_fidelity:.4f}")
    print(f"ZNE-corrected return fidelity:   {corrected_fidelity:.4f}")
    print(f"Net gain from mitigation:        {corrected_fidelity - raw_fidelity:+.4f}")

    df = pd.DataFrame([{
        "noise_scales": str(SCALES), "base_p": BASE_P, "k_trajectories": K_TRAJECTORIES,
        "raw_fidelity": raw_fidelity, "zne_corrected_fidelity": corrected_fidelity,
        "net_gain": corrected_fidelity - raw_fidelity,
    }])
    df.to_csv(_DATA_DIR / "loschmidt_echo_zne.csv", index=False)

    # Per-scale fidelities plus the actual linear Richardson extrapolation
    # back to zero noise -- the standard way to visualize a ZNE result,
    # not just a before/after bar pair.
    scale_fidelities = [float(uhlmann_fidelity(rho_at_scales[i], rho_target)) for i in range(len(SCALES))]
    fit_slope, fit_intercept = np.polyfit(SCALES, scale_fidelities, 1)
    fit_x = np.linspace(0.0, max(SCALES) * 1.05, 50)
    fit_y = fit_intercept + fit_slope * fit_x

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(1.0, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Ideal echo (F=1.0)")
    ax.plot(fit_x, fit_y, color="#95a5a6", linestyle="--", linewidth=1.5, zorder=1,
            label="Linear extrapolation to zero noise")
    ax.scatter(SCALES, scale_fidelities, s=90, color="#c0392b", zorder=3, label="Measured (noisy)")
    ax.scatter([0.0], [corrected_fidelity], s=220, color="#2980b9", marker="*", zorder=4,
               label="ZNE-corrected (extrapolated)")
    for scale, fid in zip(SCALES, scale_fidelities):
        ax.annotate(f"{fid:.3f}", (scale, fid), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#c0392b")
    ax.annotate(f"{corrected_fidelity:.3f}", (0.0, corrected_fidelity), textcoords="offset points",
                xytext=(12, -4), ha="left", fontsize=10, color="#2980b9", fontweight="bold")
    ax.set_xlabel("Noise scale $\\lambda$ (base $p$ = 0.015)")
    ax.set_ylabel("Return fidelity")
    ax.set_xlim(-0.15, max(SCALES) * 1.15)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Loschmidt echo, {N_QUBITS}Q kicked Ising, {STEPS} steps, K={K_TRAJECTORIES}")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "loschmidt_echo_zne.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'loschmidt_echo_zne.png'}")
