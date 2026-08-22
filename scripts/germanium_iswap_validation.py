"""
Validates arXiv:2608.16716 (Massai et al., IBM Research Europe - Zurich,
17-18 Aug 2026, "Engineering two-qubit gates via anisotropic exchange in
germanium spin qubits") against dense_evolution's own tools: reproduces
their real single-pulse baseband iSWAP gate (56 ns), their SPAM-limited
gate-fidelity estimate (FiSWAP~87% from FQPT=60%), and extends the
analysis with four follow-up checks the paper itself either explicitly
invites or leaves unaddressed (randomized benchmarking, per-state SPAM,
coherent-vs-stochastic error, and the general off-resonance regime where
Trotter decomposition finally has real, non-zero error to converge).

WHAT THIS TESTS: dense_evolution.circuits.trotter (pauli_rotation_ops)
applied to a genuinely time-dependent pulse for the first time (previously
only exercised with static Hamiltonians), cross-checked against the native
`iswap` gate and an exact JAX reference evolution, then combined with the
paper's own exact 2-qubit global depolarizing SPAM channel scored via
dense_evolution.uhlmann_fidelity.

WHAT THIS DOES NOT TEST: dense_evolution.solvers.vhd_tb / harrison_tb
("tight-binding"). Those compute bulk-crystal band structure (sp3s* basis,
no spin, no confinement) -- a different physics regime from this paper's
confined two-spin exchange qubits. Already has real germanium parameters,
but answers a completely different question.

Produces images/germanium_iswap_reference_circuit.png,
images/germanium_iswap_trotter_slice.png, and
images/germanium_iswap_pulse_dynamics.png.
See docs/germanium_iswap_validation.md for the full write-up.
"""
import pathlib

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.optimize import curve_fit, brentq

jax.config.update("jax_enable_x64", True)

from dense_evolution import DenseSVSimulator, pauli_rotation_ops, statevector_fidelity, uhlmann_fidelity

_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_IMAGES_DIR.mkdir(exist_ok=True)

# ============================================================================
# Utility: draws a dense_evolution circuit (same tuple format run_circuit
# accepts) as a Quirk-style box diagram -- wires + gate boxes, auto-sized to
# the label text, red labels for readability against the cyan/green borders.
# ============================================================================
def draw_circuit(circuit, n_qubits, title=None, figsize=None):
    """Draw a circuit in dense_evolution's native tuple format as a
    Quirk-style box diagram.

    circuit : list[tuple]
        E.g. [('x', 1), ('cx', 0, 1), ('iswap', 0, 1), ('rz', 0, 0.5)].
        Integers after the gate name are qubit indices, floats are
        parameters (angles) -- no gate table to keep in sync by hand.
    n_qubits : int
    """
    WIRE_COLOR = "#4a5266"
    BG = "#0a0a0d"
    TEXT = "#9aa3b2"
    LABEL_COLOR = "#ff5c5c"
    G1_FILL, G1_EDGE = "#0f2a30", "#00e5ff"   # single-qubit gate
    G2_FILL, G2_EDGE = "#0f2a22", "#00ff9d"   # multi-qubit gate
    CHAR_W = 0.145
    MIN_BOX_W, BOX_H, GAP = 0.62, 0.5, 0.18

    def box_width_for(label):
        return max(MIN_BOX_W, len(label) * CHAR_W + 0.22)

    next_free_x = [0.5] * n_qubits
    boxes = []
    for op in circuit:
        name = op[0]
        qubits = [a for a in op[1:] if isinstance(a, int) and not isinstance(a, bool)]
        params = [a for a in op[1:] if isinstance(a, float)]
        if not qubits:
            continue
        label = name.upper() if not params else f"{name.upper()}({params[0]:.2f})"
        w = box_width_for(label)
        x = max(next_free_x[q] for q in qubits) + w / 2
        boxes.append((x, qubits, label, w))
        for q in qubits:
            next_free_x[q] = x + w / 2 + GAP

    total_w = max(next_free_x) + 0.3
    fig_w = figsize[0] if figsize else max(4.0, total_w * 0.9)
    fig_h = figsize[1] if figsize else 0.9 * n_qubits + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot([0, total_w], [y, y], color=WIRE_COLOR, lw=1.5, zorder=1)
        ax.text(-0.15, y, f"q{q}", ha="right", va="center", color=TEXT,
                 fontsize=11, family="monospace")

    for x, qubits, label, box_w in boxes:
        ys = [n_qubits - 1 - q for q in qubits]
        y_lo, y_hi = min(ys), max(ys)
        is_multi = len(qubits) > 1
        fill, edge = (G2_FILL, G2_EDGE) if is_multi else (G1_FILL, G1_EDGE)
        if is_multi:
            ax.plot([x, x], [y_lo, y_hi], color=edge, lw=1.3, zorder=2)
            for y in ys:
                ax.add_patch(Circle((x, y), 0.045, color=edge, zorder=4))
            rect = Rectangle((x - box_w / 2, y_lo - BOX_H / 2),
                              box_w, (y_hi - y_lo) + BOX_H,
                              facecolor=fill, edgecolor=edge, lw=1.4, zorder=3)
            ax.add_patch(rect)
            ax.text(x, (y_lo + y_hi) / 2, label, ha="center", va="center",
                     color=LABEL_COLOR, fontsize=10.5, family="monospace", fontweight="bold", zorder=5)
        else:
            y = ys[0]
            rect = Rectangle((x - box_w / 2, y - BOX_H / 2), box_w, BOX_H,
                              facecolor=fill, edgecolor=edge, lw=1.4, zorder=3)
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", color=LABEL_COLOR,
                     fontsize=10.5, family="monospace", fontweight="bold", zorder=4)

    ax.set_xlim(-0.6, total_w)
    ax.set_ylim(-0.6, n_qubits - 0.4)
    ax.axis("off")
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, family="monospace", pad=12)
    plt.tight_layout()
    return fig

# ============================================================================
# Real paper parameters (Fig. 4 table, Section VI, Supp. Fig. 13)
# ============================================================================
J0_MHz = 26.4
T_RAMP_NS = 16.0
T_FLAT_NS = 24.0
T_GATE_NS = 2 * T_RAMP_NS + T_FLAT_NS   # 56 ns, the paper's real calibrated duration

MHZ_TO_RAD_PER_NS = 2.0 * jnp.pi / 1000.0

def pulse_envelope(t):
    """Normalized [0, 1] envelope: raised-cosine rise, flat top, raised-cosine fall."""
    rise = 0.5 * (1.0 - jnp.cos(jnp.pi * t / T_RAMP_NS))
    fall = 0.5 * (1.0 - jnp.cos(jnp.pi * (T_GATE_NS - t) / T_RAMP_NS))
    return jnp.where(
        t < 0.0, 0.0,
        jnp.where(
            t < T_RAMP_NS, rise,
            jnp.where(
                t < (T_RAMP_NS + T_FLAT_NS), 1.0,
                jnp.where(t <= T_GATE_NS, fall, 0.0)
            )
        )
    )

# Analytic peak-amplitude calibration: in the {|01>,|10>} subspace, (XX+YY)
# acts as 2*sigma_x. With H_perp(t) = (J_perp(t)/4)*(XX+YY), the accumulated
# rotation angle is theta(T) = integral (J_perp(t)/2) dt. A clean iSWAP needs
# theta = pi/2. The raised-cosine envelope integrates to T_RAMP_NS + T_FLAT_NS
# = 40 ns over [0, T_GATE_NS], so A * 40 = pi -> A = pi/40 rad/ns.
INTEGRAL_ENVELOPE_NS = T_RAMP_NS + T_FLAT_NS
A_CALIBRATED_RAD_PER_NS = jnp.pi / INTEGRAL_ENVELOPE_NS
A_CALIBRATED_MHZ = A_CALIBRATED_RAD_PER_NS / MHZ_TO_RAD_PER_NS
J_PERP_THEORY_MHZ = J0_MHz / 2.0   # Eq. 2b estimate with J_parallel = 0

# ============================================================================
# Exact reference evolution (ground truth, plain JAX)
# ============================================================================
X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
Y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex128)
Z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
XX_plus_YY = jnp.kron(X, X) + jnp.kron(Y, Y)

psi_initial = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex128)   # |01>

dt = 0.05
n_steps_exact = int(T_GATE_NS / dt)
time_axis = jnp.linspace(0.0, T_GATE_NS, n_steps_exact)

def exact_final_state(peak_amplitude_rad_per_ns):
    envelope = pulse_envelope(time_axis)
    J_perp_t = peak_amplitude_rad_per_ns * envelope

    def step(psi, J_inst):
        H_t = 0.25 * J_inst * XX_plus_YY   # correct 1/4 coefficient, not 1/2
        U_step = expm(-1j * H_t * dt)
        return jnp.dot(U_step, psi), None

    final_psi, _ = jax.lax.scan(step, psi_initial, J_perp_t)
    return final_psi

psi_exact = exact_final_state(A_CALIBRATED_RAD_PER_NS)
prob_10_exact = float(jnp.abs(psi_exact[2]) ** 2)
prob_01_exact = float(jnp.abs(psi_exact[1]) ** 2)

# ============================================================================
# Native iSWAP reference via dense_evolution
# ============================================================================
sim_ideal = DenseSVSimulator(2)
circuit_ideal = [('x', 1), ('iswap', 0, 1)]
sim_ideal.run_circuit(circuit_ideal)
psi_ideal_native = sim_ideal.sv

draw_circuit(circuit_ideal, n_qubits=2, title="Reference: native iSWAP")
plt.savefig(_IMAGES_DIR / "germanium_iswap_reference_circuit.png", dpi=200,
            facecolor=plt.gcf().get_facecolor())
plt.close()

# ============================================================================
# Trotterized pulse circuit via dense_evolution.circuits.trotter.pauli_rotation_ops
# (not trotter_evolve_ops, since the coefficient is time-dependent -- each
# slice has its own instantaneous amplitude)
# ============================================================================
def build_pulse_circuit(peak_amplitude_rad_per_ns, n_slices):
    dt_slice = T_GATE_NS / n_slices
    t_mid = (jnp.arange(n_slices) + 0.5) * dt_slice
    envelope = pulse_envelope(t_mid)
    ops = [('x', 1)]
    for env_val in envelope:
        coeff = 0.25 * peak_amplitude_rad_per_ns * float(env_val)
        angle = coeff * dt_slice
        # XX and YY always commute (Bell basis) -> no Trotter error from
        # splitting them, regardless of slice count
        ops.extend(pauli_rotation_ops({0: 'X', 1: 'X'}, angle))
        ops.extend(pauli_rotation_ops({0: 'Y', 1: 'Y'}, angle))
    return ops

draw_circuit(build_pulse_circuit(A_CALIBRATED_RAD_PER_NS, 1), n_qubits=2,
             title="One slice of the Trotterized pulse (Rxx+Ryy, illustrative)")
plt.savefig(_IMAGES_DIR / "germanium_iswap_trotter_slice.png", dpi=200,
            facecolor=plt.gcf().get_facecolor())
plt.close()

trotter_convergence = []
for n_slices in (4, 8, 16, 32, 64):
    circuit = build_pulse_circuit(A_CALIBRATED_RAD_PER_NS, n_slices)
    sim = DenseSVSimulator(2)
    sim.run_circuit(circuit)
    fidelity = float(statevector_fidelity(sim.sv, psi_ideal_native))
    trotter_convergence.append((n_slices, len(circuit), fidelity))

# ============================================================================
# SPAM noise -- the paper's own exact 2-qubit global depolarizing channel
# (Section XVIII.C: "Approximating each SPAM operation as a depolarising
# channel D_p(rho) = (1-p)*rho + (p/4)*I")
#
# NOTE: dense_evolution.circuits.registry.NoiseModel['depolarizing'] is a
# PER-QUBIT local channel -- physically different from the paper's GLOBAL
# 2-qubit model (d=4). Reproducing their model honestly means implementing
# their exact channel here and scoring it with dense_evolution's own
# uhlmann_fidelity -- this is the part of dense_evolution.mitigation this
# experiment actually exercises.
# ============================================================================
def depolarize_2q(rho, p):
    """Global 2-qubit depolarizing channel, D_p(rho) = (1-p)*rho + (p/4)*I_4,
    exactly as defined in the paper (Section XVIII.C, d=4)."""
    identity_4 = jnp.eye(4, dtype=jnp.complex128)
    return (1.0 - p) * rho + (p / 4.0) * identity_4

psi_ideal_col = psi_ideal_native.reshape(4, 1)
rho_ideal = psi_ideal_col @ psi_ideal_col.conj().T

DIAG_PSPAM_MEASURED = 0.69   # paper's real measured <diag(PSPAM)>
P_SPAM_SELF_CONSISTENT = float((1.0 - jnp.sqrt(DIAG_PSPAM_MEASURED)) * 4.0 / 3.0)
P_SPAM_PAPER_REPORTED = 0.21   # paper's own rounded reported value

spam_roundtrip_results = {}
for label, p_spam in (("self_consistent", P_SPAM_SELF_CONSISTENT),
                       ("paper_reported", P_SPAM_PAPER_REPORTED)):
    rho_after_prep = depolarize_2q(rho_ideal, p_spam)
    rho_after_meas = depolarize_2q(rho_after_prep, p_spam)
    f_roundtrip = float(uhlmann_fidelity(rho_after_meas, rho_ideal))
    # Exact sequential-composition closed form (hand-derived, matches
    # uhlmann_fidelity to machine precision): D_p(D_p(rho)) =
    # (1-p)^2*rho + (p/4)(2-p)*I, so <psi|D_p(D_p(rho))|psi> = (1-p)^2 + (p/4)(2-p)
    f_exact_composition = (1.0 - p_spam) ** 2 + (p_spam / 4.0) * (2.0 - p_spam)
    # The paper's own back-of-envelope approximation (their own words:
    # "an estimate... not a definitive figure of merit"), (1-3p/4)^2
    f_paper_approx = (1.0 - 0.75 * p_spam) ** 2
    spam_roundtrip_results[label] = dict(p=p_spam, uhlmann=f_roundtrip,
                                          exact=f_exact_composition, paper_approx=f_paper_approx)

FQPT_MEASURED = 0.60
FISWAP_INFERRED = FQPT_MEASURED / DIAG_PSPAM_MEASURED   # reproduces the paper's own 0.87

# ============================================================================
# Main pulse + population-dynamics figure
# ============================================================================
def _make_dynamics_figure():
    envelope_plot = pulse_envelope(time_axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 2.5]})

    ax1.plot(np.array(time_axis), np.array(envelope_plot) * A_CALIBRATED_MHZ,
             color='#4A148C', lw=2, label=r'$J_{\perp}(t)$ (16-24-16 ns raised cosine)')
    ax1.set_ylabel('Coupling (MHz)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_title('Single-pulse baseband iSWAP -- Ge spin qubit (arXiv:2608.16716)',
                   fontsize=12, fontweight='bold')

    def step_record(psi, J_inst):
        H_t = 0.25 * J_inst * XX_plus_YY
        U_step = expm(-1j * H_t * dt)
        next_psi = jnp.dot(U_step, psi)
        return next_psi, jnp.array([jnp.abs(next_psi[1]) ** 2, jnp.abs(next_psi[2]) ** 2])

    _, history = jax.lax.scan(step_record, psi_initial,
                               A_CALIBRATED_RAD_PER_NS * pulse_envelope(time_axis))

    ax2.plot(np.array(time_axis), np.array(history[:, 0]),
             label=r'$|01\rangle$', color='#A0522D', lw=2)
    ax2.plot(np.array(time_axis), np.array(history[:, 1]),
             label=r'$|10\rangle$', color='#FF1493', lw=2)
    ax2.set_xlabel('Time (ns)', fontsize=10)
    ax2.set_ylabel('Occupation probability', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center right')

    plt.xlim(0, T_GATE_NS)
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "germanium_iswap_pulse_dynamics.png", dpi=200)
    plt.close()

_make_dynamics_figure()

# ============================================================================
# Follow-up 1 -- simplified randomized benchmarking (Pauli frame)
# Directly answers Section VII: "A more rigorous fidelity estimation via
# randomised benchmarking is left to follow-up work." The noise channel here
# is KNOWN (our own depolarize_2q), so this validates that the RB protocol
# via dense_evolution correctly recovers the injected parameter -- not full
# Clifford RB (would need the 2-qubit Clifford table), but Pauli-frame RB
# (Pauli twirling), a real, valid protocol.
# ============================================================================
I2 = jnp.eye(2, dtype=jnp.complex128)
PAULI_1Q = [I2, X, Y, Z]

def rb_trial(key, m, p_noise):
    rho = jnp.zeros((4, 4), dtype=jnp.complex128).at[0, 0].set(1.0)   # |00><00|
    U0_total = I2
    U1_total = I2
    for _ in range(m):
        key, k0, k1 = jax.random.split(key, 3)
        u0 = PAULI_1Q[int(jax.random.randint(k0, (), 0, 4))]
        u1 = PAULI_1Q[int(jax.random.randint(k1, (), 0, 4))]
        U0_total = u0 @ U0_total
        U1_total = u1 @ U1_total
        U_layer = jnp.kron(u0, u1)
        rho = U_layer @ rho @ U_layer.conj().T
        rho = depolarize_2q(rho, p_noise)
    U_inv = jnp.kron(U0_total.conj().T, U1_total.conj().T)
    rho = U_inv @ rho @ U_inv.conj().T
    rho = depolarize_2q(rho, p_noise)
    return float(jnp.real(rho[0, 0]))

def run_randomized_benchmarking(n_trials=150, seq_lengths=(0, 2, 4, 8, 16, 32, 64),
                                 p_noise=P_SPAM_PAPER_REPORTED, seed=2026):
    rb_key = jax.random.PRNGKey(seed)
    survival_means = []
    for m in seq_lengths:
        vals = []
        for _ in range(n_trials):
            rb_key, sub_key = jax.random.split(rb_key)
            vals.append(rb_trial(sub_key, m, p_noise))
        survival_means.append(float(np.mean(vals)))

    n_apps = np.array(seq_lengths, dtype=float) + 1.0

    def rb_decay(n, A, r, B):
        return A * r ** n + B

    popt, _ = curve_fit(rb_decay, n_apps, survival_means, p0=[0.75, 0.8, 0.25], maxfev=5000)
    A_fit, r_fit, B_fit = popt
    r_expected = 1.0 - p_noise
    return dict(survival_means=survival_means, A_fit=A_fit, r_fit=r_fit,
                B_fit=B_fit, r_expected=r_expected)

rb_result = run_randomized_benchmarking()

# ============================================================================
# Follow-up 2 -- per-state SPAM profile from Fig. 5f, not a uniform average.
# Only 4 of 16 diagonal values are precisely readable from the figure (0.17,
# 0.40, 0.39, 0.37 -- the states requiring concatenated rotations on both
# qubits). We reconstruct an honest two-tier profile: these 4 real values,
# and the remaining 12 states solved to reproduce exactly the real reported
# mean <diag(PSPAM)>=0.69 -- we do NOT invent the 12 unreadable values, only
# constrain them to the known mean.
# ============================================================================
def p_from_diag(diag_val):
    return float((1.0 - jnp.sqrt(diag_val)) * 4.0 / 3.0)

LOW_SPAM_STATES = [0.17, 0.40, 0.39, 0.37]
N_LOW, N_TOTAL = len(LOW_SPAM_STATES), 16
mean_low = float(np.mean(LOW_SPAM_STATES))
p_high_states = (DIAG_PSPAM_MEASURED * N_TOTAL - mean_low * N_LOW) / (N_TOTAL - N_LOW)

roundtrip_fids_per_state = []
for v in LOW_SPAM_STATES + [p_high_states]:
    p_v = p_from_diag(v)
    rho1 = depolarize_2q(rho_ideal, p_v)
    rho2 = depolarize_2q(rho1, p_v)
    roundtrip_fids_per_state.append(float(uhlmann_fidelity(rho2, rho_ideal)))

# ============================================================================
# Follow-up 3 -- coherent vs. stochastic error (Section XVIII.C, caveat (i):
# "assumes depolarising SPAM, whereas actual errors contain coherent
# components... not addressed here"). Shows that a small coherent pulse-
# amplitude miscalibration explains the paper's inferred FiSWAP~87% just as
# well -- the same aggregate number does not distinguish the two mechanisms,
# exactly the limitation the paper itself states.
# ============================================================================
def fidelity_at_amplitude_error(eps):
    psi = exact_final_state(A_CALIBRATED_RAD_PER_NS * (1.0 + eps))
    return float(jnp.abs(jnp.vdot(psi_ideal_native, psi)) ** 2)

def _objective(eps):
    return fidelity_at_amplitude_error(eps) - FISWAP_INFERRED

eps_solution = brentq(_objective, 1e-6, 0.5)

# ============================================================================
# Follow-up 4 -- general off-resonance regime (E_delta_g != 0): here, and
# only here, Trotter decomposition is genuinely approximate. The single-
# qubit detuning term (Z0-Z1) does NOT commute with XX+YY (verified:
# [Z⊗I, X⊗X] = 2i*Y⊗X != 0), unlike XX/YY/ZZ, which always mutually commute
# (Bell basis) -- which is why neither the pure iSWAP point above nor a pure
# CPhase point (J_perp=0, diagonal Hamiltonian) ever show Trotter error.
# Compares order-1 vs. order-2 Suzuki-Trotter, already documented in
# dense_evolution.circuits.trotter, but for the first time on a genuinely
# time-dependent pulse instead of a static Hamiltonian.
# ============================================================================
ZI = jnp.kron(Z, I2)
IZ = jnp.kron(I2, Z)
E_DELTA_ANG = A_CALIBRATED_RAD_PER_NS   # chosen equal to the transverse term's peak

def exact_final_state_general(peak_amplitude_rad_per_ns, e_delta_ang):
    envelope = pulse_envelope(time_axis)
    J_perp_t = peak_amplitude_rad_per_ns * envelope

    def step(psi, J_inst):
        H_t = 0.25 * J_inst * XX_plus_YY + 0.25 * e_delta_ang * (ZI - IZ)
        U_step = expm(-1j * H_t * dt)
        return jnp.dot(U_step, psi), None

    final_psi, _ = jax.lax.scan(step, psi_initial, J_perp_t)
    return final_psi

psi_exact_general = exact_final_state_general(A_CALIBRATED_RAD_PER_NS, E_DELTA_ANG)

def build_general_pulse_circuit(peak_amplitude_rad_per_ns, e_delta_ang, n_slices, order=1):
    dt_slice = T_GATE_NS / n_slices
    t_mid = (jnp.arange(n_slices) + 0.5) * dt_slice
    envelope = pulse_envelope(t_mid)
    ops = [('x', 1)]
    z_angle_full = 0.25 * e_delta_ang * dt_slice
    for env_val in envelope:
        xy_coeff = 0.25 * peak_amplitude_rad_per_ns * float(env_val)
        xy_angle = xy_coeff * dt_slice
        if order == 1:
            ops.extend(pauli_rotation_ops({0: 'Z'}, z_angle_full))
            ops.extend(pauli_rotation_ops({1: 'Z'}, -z_angle_full))
            ops.extend(pauli_rotation_ops({0: 'X', 1: 'X'}, xy_angle))
            ops.extend(pauli_rotation_ops({0: 'Y', 1: 'Y'}, xy_angle))
        else:   # order 2, Strang splitting (XX/YY terms commute with each other)
            ops.extend(pauli_rotation_ops({0: 'Z'}, z_angle_full / 2))
            ops.extend(pauli_rotation_ops({1: 'Z'}, -z_angle_full / 2))
            ops.extend(pauli_rotation_ops({0: 'X', 1: 'X'}, xy_angle))
            ops.extend(pauli_rotation_ops({0: 'Y', 1: 'Y'}, xy_angle))
            ops.extend(pauli_rotation_ops({0: 'Z'}, z_angle_full / 2))
            ops.extend(pauli_rotation_ops({1: 'Z'}, -z_angle_full / 2))
    return ops

general_trotter_results = {1: [], 2: []}
for order in (1, 2):
    for n_slices in (4, 8, 16, 32, 64):
        circuit = build_general_pulse_circuit(A_CALIBRATED_RAD_PER_NS, E_DELTA_ANG, n_slices, order=order)
        sim_g = DenseSVSimulator(2)
        sim_g.run_circuit(circuit)
        fidelity_g = float(statevector_fidelity(sim_g.sv, psi_exact_general))
        general_trotter_results[order].append((n_slices, 1.0 - fidelity_g))

if __name__ == "__main__":
    print(f"J_perp theory (Eq. 2b, J_par=0): {J_PERP_THEORY_MHZ:.2f} MHz")
    print(f"J_perp calibrated (this script): {A_CALIBRATED_MHZ:.2f} MHz")
    print(f"Exact evolution: P(|10>)={prob_10_exact:.6f}, P(|01>)={prob_01_exact:.6f}")
    print(f"Trotter convergence vs. native iSWAP: {trotter_convergence}")
    print(f"SPAM round-trip results: {spam_roundtrip_results}")
    print(f"FiSWAP inferred (FQPT/<diag(PSPAM)>): {FISWAP_INFERRED:.4f}")
    print(f"RB fit: r={rb_result['r_fit']:.4f} vs expected r={rb_result['r_expected']:.4f}")
    print(f"Per-state round-trip fidelity spread: {roundtrip_fids_per_state}")
    print(f"Coherent-error-equivalent amplitude miscalibration: {eps_solution * 100:.2f}%")
    print(f"General-regime Trotter infidelity (order 1): {general_trotter_results[1]}")
    print(f"General-regime Trotter infidelity (order 2): {general_trotter_results[2]}")
