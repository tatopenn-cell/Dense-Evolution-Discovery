"""
Validates dense_evolution.continuous_dissipative_evolve (Dense-Evolution
v8.1.67+) against real measured data from
arXiv:2104.05219 (McEwen et al., "Resolving catastrophic error bursts from
cosmic rays in large arrays of superconducting qubits", published in Nature
Physics): a cosmic-ray/gamma-ray impact on a 26-qubit Google Sycamore chip
produces a burst of quasiparticles that transiently collapses the chip's
effective T1 -- simultaneous decay errors jump from a baseline ~4/26 qubits
to ~10/26 within ~10us, rise further to ~15/26 over ~1ms, then decay
exponentially back to baseline with a time constant tightly grouped in the
25-30ms range (fitted across 415 real extracted events).

WHAT THIS TESTS: dense_evolution.continuous_dissipative_evolve applied to a
genuinely time-dependent DISSIPATIVE process for the first time -- until
now only continuous_pulse_evolve (coherent/unitary) had been exercised, in
germanium_iswap_validation.py's reproduction of arXiv:2608.16716's baseband
iSWAP pulse. This experiment is the first real-data validation of the
dissipative counterpart: an amplitude-damping channel (matching the paper's
own reported asymmetry -- Section on quasiparticle-poisoning signature
found decay errors |1>->|0> but no excess excitation errors in a control
RReCS run) driven by a time-varying decay probability reproducing the
paper's measured event shape.

WHAT THIS DOES NOT TEST: the paper's own matched-filter event-detection
algorithm, the spatial hotspot/heatmap dynamics across the real 26-qubit
array (Fig. 3c), or its T-RReCS multi-timescale measurement protocol. This
is a single representative qubit, not a 26-qubit simulation -- the
baseline/intermediate/peak error FRACTIONS (out of 26, from Fig. 1d/Fig. 3)
are read here as if they were single-qubit decay probabilities in one
native 1us idling window. That is a reasonable ensemble-to-single-qubit
reading, not a literal number the paper itself reports.

THE RISE SHAPE IS AN APPROXIMATION, THE DECAY IS NOT: the paper states two
descriptive rise timescales (~10us, ~1ms) with precise error counts at each
(Fig. 3), but does not publish a closed-form fit for the rise -- modeled
here as two sequential saturating exponentials (tau1=3us, tau2=300us)
chosen only to pass through those two described points. The DECAY, in
contrast, IS the paper's own directly fitted quantity (single exponential,
tau=25ms as the central value of their reported 25-30ms range across 415
real events) -- not an approximation of ours.

BASELINE T1 IS AN ASSUMPTION, THE EVENT SCALING IS NOT: the paper's own
baseline error count ("~4/26") mixes true T1 decay with finite readout
infidelity (its own words -- "Finite T1 and readout fidelities will
produce errors"), so reading it directly as a single-qubit decay
probability would imply an unrealistically short T1 (~6us, far below real
transmon T1 of this device generation, typically tens of us). Instead, the
model below fixes a representative baseline T1 (20us, a plausible value
for this hardware generation -- NOT itself extracted from this paper) and
scales the instantaneous decay probability up by the paper's own real,
dimensionless RATIOS (peak/baseline=15/4=3.75x, intermediate/baseline=
10/4=2.5x). That ratio, and the 25ms recovery, are paper-grounded; only
the absolute baseline T1 is an assumption, isolated to one named constant
(T1_BASELINE_ASSUMED_US) so it is easy to swap for a measured value later.

Produces images/cosmic_ray_burst_profile.png and
images/cosmic_ray_burst_survival.png. See
docs/cosmic_ray_burst_validation.md for the full write-up.
"""
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from dense_evolution import continuous_dissipative_evolve

_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_IMAGES_DIR.mkdir(exist_ok=True)

# ============================================================================
# Real measured parameters, arXiv:2104.05219 (McEwen et al.)
# ============================================================================
N_QUBITS = 26
IDLE_WINDOW_US = 1.0          # paper's own idling/measurement window
BASELINE_ERRORS = 4.0         # Fig. 1d / Fig. 3 baseline
INTERMEDIATE_ERRORS = 10.0    # Fig. 3: reached within ~10us of impact
PEAK_ERRORS = 15.0            # Fig. 3: reached within ~1ms of impact
TAU_DECAY_MS = 25.0           # paper's own fitted central value (415 events, 25-30ms range)

RATIO_INTERMEDIATE = INTERMEDIATE_ERRORS / BASELINE_ERRORS   # 2.5x, paper-grounded
RATIO_PEAK = PEAK_ERRORS / BASELINE_ERRORS                   # 3.75x, paper-grounded

T1_BASELINE_ASSUMED_US = 20.0   # representative device value -- see docstring caveat

# Rise timescales -- chosen (not fitted by the paper) to pass through the
# two described (time, error-count) points via two sequential saturating
# exponentials; see module docstring.
TAU1_US = 3.0
TAU2_US = 300.0


def scaling_factor(t_us):
    """Dimensionless multiplier on the baseline decay probability, 1.0 at
    t=0 rising to RATIO_PEAK and decaying back to 1.0 -- the paper's own
    real ratios and 25ms recovery, independent of the assumed absolute
    baseline T1."""
    stage1 = (RATIO_INTERMEDIATE - 1.0) * (1.0 - jnp.exp(-t_us / TAU1_US))
    stage2 = (RATIO_PEAK - RATIO_INTERMEDIATE) * (1.0 - jnp.exp(-t_us / TAU2_US))
    decay = jnp.exp(-t_us / (TAU_DECAY_MS * 1000.0))
    return 1.0 + (stage1 + stage2) * decay


P_BASELINE_1US = 1.0 - jnp.exp(-IDLE_WINDOW_US / T1_BASELINE_ASSUMED_US)


def p_1us(t_us):
    """Instantaneous decay-error probability in a 1us window, at time t_us
    (us) after impact: the assumed baseline probability times the paper-
    grounded scaling_factor(t_us)."""
    return P_BASELINE_1US * scaling_factor(t_us)


def effective_t1_us(p):
    """T1 (us) implied by treating p as a memoryless decay probability over
    one IDLE_WINDOW_US window: p = 1 - exp(-IDLE_WINDOW_US/T1). See module
    docstring's caveat -- illustrative, not a measured device T1."""
    return -IDLE_WINDOW_US / jnp.log(1.0 - p)


# ============================================================================
# Amplitude-damping channel -- matches the paper's own reported asymmetry
# (decay errors only; their control RReCS run found no excess excitation
# errors, consistent with quasiparticle poisoning). NOT the same physical
# map as dense_evolution.global_depolarizing_channel (symmetric, mixes
# toward the fully-mixed state) -- this one only ever moves population
# |1>->|0>, one-directional, exactly like the real mechanism.
# ============================================================================
def amplitude_damping_channel(rho, gamma):
    e0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]], dtype=jnp.complex128)
    e1 = jnp.array([[0.0, jnp.sqrt(gamma)], [0.0, 0.0]], dtype=jnp.complex128)
    return e0 @ rho @ e0.conj().T + e1 @ rho @ e1.conj().T


# ============================================================================
# Time grid, per-slice decay probability with and without the event, and
# evolution of a qubit prepared in |1> through both -- via
# continuous_dissipative_evolve, not a Python loop (at DT_US=10 over 150ms
# this is 15000 slices; a Python-list-based approach would build 15000
# entries just for this one run, exactly the OOM-prone pattern this
# utility exists to avoid).
# ============================================================================
DT_US = 10.0
T_TOTAL_MS = 150.0   # 6x the decay time constant -- comfortably back to baseline
N_SLICES = int(T_TOTAL_MS * 1000.0 / DT_US)

time_us = jnp.arange(N_SLICES) * DT_US

p_event_1us = p_1us(time_us)
t1_event_us = effective_t1_us(p_event_1us)
gamma_event = 1.0 - jnp.exp(-DT_US / t1_event_us)

t1_baseline_us = effective_t1_us(P_BASELINE_1US)
gamma_baseline = 1.0 - jnp.exp(-DT_US / t1_baseline_us)
gamma_baseline_profile = jnp.full((N_SLICES,), gamma_baseline)

RHO1 = jnp.zeros((2, 2), dtype=jnp.complex128).at[1, 1].set(1.0)   # |1><1|

final_rho_event, traj_event = continuous_dissipative_evolve(
    RHO1, amplitude_damping_channel, gamma_event,
    observable_fn=lambda rho: jnp.real(rho[1, 1]),
)
final_rho_baseline, traj_baseline = continuous_dissipative_evolve(
    RHO1, amplitude_damping_channel, gamma_baseline_profile,
    observable_fn=lambda rho: jnp.real(rho[1, 1]),
)

# NOTE: with T1_BASELINE_ASSUMED_US=20 and DT_US=10, even the undisturbed
# baseline has decayed to numerical zero well before 1ms (1ms is 50 T1
# lifetimes) -- an UNRESET, continuously idle qubit is not meant to survive
# that long regardless of any cosmic-ray event (exactly the paper's own
# point about QEC needing to correct faster than T1). The event's real,
# checkable effect is on the EARLY-TIME decay rate, while both curves are
# still numerically distinguishable from zero -- hence the checkpoint below.
_IDX_CHECKPOINT = 2   # t = 20us
SURVIVAL_AT_CHECKPOINT_EVENT = float(traj_event[_IDX_CHECKPOINT])
SURVIVAL_AT_CHECKPOINT_BASELINE = float(traj_baseline[_IDX_CHECKPOINT])
P_PEAK_1US = P_BASELINE_1US * RATIO_PEAK
T1_DROP_FACTOR = float(t1_baseline_us / effective_t1_us(P_PEAK_1US))


# ============================================================================
# Figures
# ============================================================================
def _make_profile_figure():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    t_ms = np.array(time_us) / 1000.0

    ax1.plot(t_ms, np.array(scaling_factor(time_us)), color="#4A148C", lw=2, label="model")
    ax1.scatter([0.0, 0.010, 1.0], [1.0, RATIO_INTERMEDIATE, RATIO_PEAK],
                color="#DD3B3B", zorder=5, label="paper's real ratios (Fig. 1d / Fig. 3)")
    ax1.set_ylabel("decay-rate multiplier vs. baseline")
    ax1.set_title("Cosmic-ray-induced error burst -- arXiv:2104.05219")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(t_ms, np.array(t1_event_us), color="#00838F", lw=2, label=r"$T_{1,eff}(t)$")
    ax2.axhline(float(t1_baseline_us), color="gray", ls="--", label="baseline")
    ax2.set_ylabel(r"effective $T_1$ (us, illustrative)")
    ax2.set_xlabel("time (ms)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "cosmic_ray_burst_profile.png", dpi=150)
    plt.close(fig)


def _make_survival_figure():
    # Zoomed to the first 100us: both curves are numerically indistinguishable
    # from zero well before the 25ms recovery tail even becomes relevant
    # (see NOTE above SURVIVAL_AT_CHECKPOINT_*) -- this window is where the
    # event's acceleration of decay is actually visible.
    fig, ax = plt.subplots(figsize=(8, 5))
    n_zoom = 11   # 0 to 100us at DT_US=10
    t_us_zoom = np.array(time_us[:n_zoom])
    ax.plot(t_us_zoom, np.array(traj_event[:n_zoom]), "o-", color="#C0392B", lw=2,
            label="with cosmic-ray event")
    ax.plot(t_us_zoom, np.array(traj_baseline[:n_zoom]), "o-", color="#2980B9", lw=2,
            label="baseline only (no event)")
    ax.set_yscale("log")
    ax.set_xlabel("time (us)")
    ax.set_ylabel(r"$P(|1\rangle)$ survival")
    ax.set_title("Excited-state survival: with vs. without the burst (early-time zoom)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "cosmic_ray_burst_survival.png", dpi=150)
    plt.close(fig)


_make_profile_figure()
_make_survival_figure()


if __name__ == "__main__":
    print(f"P_baseline={float(P_BASELINE_1US):.4f}  P_peak={float(P_PEAK_1US):.4f}")
    print(f"T1_eff baseline={float(t1_baseline_us):.2f}us  T1_eff peak={float(effective_t1_us(P_PEAK_1US)):.2f}us"
          f"  (drop factor {T1_DROP_FACTOR:.1f}x)")
    print(f"Survival at t=20us: event={SURVIVAL_AT_CHECKPOINT_EVENT:.4f}  "
          f"baseline={SURVIVAL_AT_CHECKPOINT_BASELINE:.4f}  "
          f"(ratio {SURVIVAL_AT_CHECKPOINT_BASELINE / SURVIVAL_AT_CHECKPOINT_EVENT:.1f}x)")
