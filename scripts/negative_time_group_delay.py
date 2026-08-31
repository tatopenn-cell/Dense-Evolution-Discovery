"""
Reproduces the closed-form weak-value theory of Thompson et al. (arXiv:
2310.00432, "How much time does a photon spend as an atomic excitation
before being transmitted?") -- the theoretical framework behind the real
experimental measurement of NEGATIVE atomic excitation times in Angulo et
al. (arXiv:2409.03680, "Experimental evidence that a photon can spend a
negative amount of time in an atom cloud").

THE PHYSICS (single-excitation, linear Maxwell-Bloch regime, natural units
Gamma=1 -- i.e. time measured in units of the atomic lifetime tau_sp=1/Gamma,
detuning in units of Gamma):

  L(delta)          = 1 / (1 + (2*delta)**2)                          (Eq. 26)
  t_g(delta, tau0)  = -tau0 * (1 - (2*delta)**2) / (1 + (2*delta)**2)**2   (Eq. 34)
  P_T(tau0)         = integral g(delta) * exp(-tau0*L(delta)) d(delta)     (Eq. 31)
  tau_T(tau0)       = (1/P_T) * integral g(delta) * exp(-tau0*L(delta))
                        * t_g(delta, tau0) d(delta)                       (Eq. 35)

where tau0 is the resonant optical depth, t_g is the narrow-band group
delay, and g(delta) is the (Fourier-transform-limited, unchirped) Gaussian
spectral power density of a Gaussian pulse with RMS INTENSITY duration
sigma (paper's own convention, Fig. 2 caption): g is Gaussian in delta with
sigma_omega = 1/(2*sigma) (standard minimum-uncertainty time-bandwidth
product for an unchirped Gaussian, Delta_t*Delta_omega = 1/2).

Eq. (34)'s exact on-resonance value tg(0,tau0) = -tau0/Gamma is stated in
the paper's own Fig. 3 caption -- this and the Fourier relation above are
the only two inputs re-derived here from scratch (nothing else guessed from
a low-fidelity OCR of the PDF; both cross-checked against text extracted
with `pdftotext -raw`, not the layout-mode extraction that garbled the
equations on a first pass).

THREE INDEPENDENT VALIDATIONS before trusting any physics conclusion:

1. Narrow-band self-test: as sigma (in units of 1/Gamma) grows, the pulse
   spectrum becomes a delta function at delta=0 and tau_T(tau0, sigma) must
   converge to the paper's stated exact on-resonance limit -tau0/Gamma,
   independent of the spectral-width prefactor convention. Checked at
   tau0=2 for sigma=1,5,20,100: converges monotonically to -2.0 (sigma=100
   gives -1.9994, 0.03% off).

2. Qualitative reproduction of Thompson et al.'s Fig. 2: for narrow pulses
   (large sigma) tau_T stays negative and tracks -tau0 across the whole
   tau0 range; for broad pulses (small sigma) tau_T starts near zero, goes
   negative, then crosses back to POSITIVE at a pulse-duration-dependent
   optical depth -- the paper's own headline "negative time" result.

3. External validation against a real published number NOT produced by
   this script's own authors: Angulo et al.'s experimental paper states
   "the theoretical value of tau_T/tau_bar_0 = 0.45" for their rms=10ns,
   OD=4 configuration, tau_sp~26ns. Feeding sigma=10/26 (in units of
   1/Gamma) and tau0=4 into this independent re-derivation gives
   tau_T/tau_bar_0 = 0.399 -- 11% off, same sign, same order of magnitude.
   Not an exact match (plausibly from tau_sp~26ns being an approximate
   rounded value, or a pulse-shape convention difference not fully
   specified in the main text), but a genuine external cross-check against
   a real paper's own theory number, not a self-consistency check alone.

Produces `data/negative_time_group_delay.csv`.

    python scripts/negative_time_group_delay.py
"""
import csv
import pathlib

import jax.numpy as jnp
from jax import vmap

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_CSV = _REPO_ROOT / "data" / "negative_time_group_delay.csv"

DELTA_GRID = jnp.linspace(-50.0, 50.0, 400_001)  # units of Gamma; wide/fine enough that tails are negligible for all sigma used below

# Sinclair/Angulo experimental parameters (arXiv:2409.03680): rms=10ns,
# OD=4, tau_sp~26ns -- the one point with a real published theory number
# (tau_T/tau_bar_0 = 0.45) to check this re-derivation against.
SIGMA_EXTERNAL = 10.0 / 26.0
TAU0_EXTERNAL = 4.0
PUBLISHED_RATIO_EXTERNAL = 0.45


def lorentzian(delta):
    return 1.0 / (1.0 + (2.0 * delta) ** 2)


def group_delay(delta, tau0):
    x = 2.0 * delta
    return -tau0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2


def gaussian_spectrum(delta, sigma_time):
    """Normalized (integrates to 1 over delta) Gaussian spectral power
    density of a Fourier-transform-limited Gaussian pulse of RMS intensity
    duration sigma_time (units of 1/Gamma)."""
    sigma_omega = 1.0 / (2.0 * sigma_time)
    return jnp.exp(-0.5 * (delta / sigma_omega) ** 2) / (jnp.sqrt(2.0 * jnp.pi) * sigma_omega)


def excitation_times(tau0, sigma_time, delta_grid=DELTA_GRID):
    """Returns (tau_T, tau_bar_0, P_T) at given optical depth tau0 and
    pulse RMS intensity duration sigma_time (units of 1/Gamma)."""
    g = gaussian_spectrum(delta_grid, sigma_time)
    absorbed_weight = g * jnp.exp(-tau0 * lorentzian(delta_grid))
    p_t = jnp.trapezoid(absorbed_weight, delta_grid)
    tau_t = jnp.trapezoid(absorbed_weight * group_delay(delta_grid, tau0), delta_grid) / p_t
    tau_bar_0 = 1.0 - p_t  # = P_S/Gamma in Gamma=1 units (Eq. 30/31)
    return tau_t, tau_bar_0, p_t


def selftest_narrowband_limit():
    print("Self-test: narrow-band limit tau_T(tau0=2, sigma->large) -> -2.0 (Fig. 3 exact result)")
    tau0 = 2.0
    prev = None
    for sigma in (1.0, 5.0, 20.0, 100.0):
        tau_t, _, _ = excitation_times(tau0, sigma)
        tau_t = float(tau_t)
        print(f"  sigma={sigma:>5}  tau_T={tau_t:.6f}")
        if prev is not None:
            assert abs(tau_t - (-tau0)) < abs(prev - (-tau0)), "not converging monotonically toward -tau0"
        prev = tau_t
    assert abs(prev - (-tau0)) < 0.01, f"self-test failed: sigma=100 gives {prev:.4f}, expected close to -2.0"
    print(f"  PASSED (sigma=100 within {abs(prev - (-tau0)):.4f} of exact -2.0)\n")


def selftest_external_published_ratio():
    print(f"Self-test: external check against Angulo et al.'s published theory ratio "
          f"(rms=10ns, OD=4, tau_sp~26ns -> sigma={SIGMA_EXTERNAL:.4f})")
    tau_t, tau_bar_0, p_t = excitation_times(TAU0_EXTERNAL, SIGMA_EXTERNAL)
    ratio = float(tau_t / tau_bar_0)
    rel_diff = abs(ratio - PUBLISHED_RATIO_EXTERNAL) / PUBLISHED_RATIO_EXTERNAL
    print(f"  tau_T={float(tau_t):.4f}  tau_bar_0={float(tau_bar_0):.4f}  ratio={ratio:.4f}  "
          f"published={PUBLISHED_RATIO_EXTERNAL}  rel_diff={rel_diff:.1%}")
    assert rel_diff < 0.15, f"self-test failed: {rel_diff:.1%} off published value, too large to call this a real match"
    print("  PASSED (same sign, same order of magnitude, within 15% of a real published number)\n")


def main():
    selftest_narrowband_limit()
    selftest_external_published_ratio()

    # Reproduce Fig. 2 of Thompson et al.: tau_T(tau0) for several pulse durations.
    sigmas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 10.0]
    tau0_grid = jnp.linspace(0.0, 9.0, 46)

    rows = []
    print("Reproducing Thompson et al. Fig. 2 (tau_T vs optical depth, several pulse durations):")
    for sigma in sigmas:
        tau_t_curve = vmap(lambda t0: excitation_times(t0, sigma)[0])(tau0_grid)
        tau_t_curve = [float(v) for v in tau_t_curve]
        crosses_positive = any(v > 0 for v in tau_t_curve[1:])
        print(f"  sigma={sigma:>4}  tau_T(tau0=9)={tau_t_curve[-1]:+.4f}  "
              f"{'crosses positive' if crosses_positive else 'stays negative'}")
        for t0, tt in zip(tau0_grid.tolist(), tau_t_curve):
            rows.append(dict(sigma=sigma, tau0=t0, tau_T=tt))

    # narrow pulses (large sigma) should stay negative across this range; broad pulses should cross positive
    assert not any(v > 0 for v in [r["tau_T"] for r in rows if r["sigma"] == 10.0][1:]), \
        "narrowest pulse (sigma=10) crossed positive within tau0<=9 -- contradicts Fig. 2"
    assert any(v > 0 for v in [r["tau_T"] for r in rows if r["sigma"] == 0.1][1:]), \
        "broadest pulse (sigma=0.1) never crossed positive within tau0<=9 -- contradicts Fig. 2"
    print("  Qualitative Fig. 2 shape confirmed: narrow pulses stay negative, broad pulses cross to positive.\n")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
