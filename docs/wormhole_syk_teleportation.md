# Traversable-Wormhole-Inspired Quantum Teleportation (SYK Model)

!!! note
    The implementation lives in the main library:
    [`dashboard_core.wormhole`](https://tatopenn-cell.github.io/Dense-Evolution/),
    built on [`dense_evolution.fermions`/`.entropy`/`.trotter`](https://tatopenn-cell.github.io/Dense-Evolution/)
    (all shipped in `dense-evolution>=8.1.49`, with their own unit test
    suite in the main repo). This page is the experimental log for what
    this repo adds on top: real parameter scans, run against the
    published package, not the implementation itself.

Real reproduction of the Gao-Jafferis-Wall traversable-wormhole
teleportation protocol on a chaotic binary sparse Sachdev-Ye-Kitaev (SYK)
model, following **arXiv:2604.10090**, "Quantum simulation of
traversable-wormhole-inspired quantum teleportation in a chaotic binary
sparse SYK model" (2026) -- a real IBM-hardware reproduction of the
protocol. This repo runs the equivalent as an exact/near-exact simulation
(statevector access the paper's own hardware run didn't have) and adds
five parameter scans the paper itself didn't publish.

## Why this exists

An earlier, discarded `dashboard_core` circuit called "Traversable
Wormhole (BGQ)" used the right vocabulary (SYK scrambling, a phase
"kick") but wasn't real physics: it ran on a single qubit register,
scrambling forward then backward around a bare `RZ` "kick". Verified
directly, not assumed: applying that kick with either sign gave
**identical** results. That's not a tuning bug -- a single-register
readout structurally cannot show sign-dependent behavior, because a
qubit inside a Bell pair has a maximally-mixed marginal that no local
operation changes (the no-signaling theorem forbids it outright,
regardless of circuit design).

The real protocol needs:

1. **Two coupled chaotic systems** (L, R) -- here, an L and R copy of a
   binary sparse N=8 SYK model (K=10 of the C(8,4)=70 possible
   four-Majorana coupling terms, random `+-J/sqrt(K)` coefficients).
2. **A message injected via a separate reference pair** (P, Q Bell pair,
   Q swapped into the L register) -- not a bare bit-flip on a qubit
   that's already maximally entangled.
3. **A real bilinear L-R coupling** `exp(i*mu*V)`,
   `V = (1/4N) * sum_j chi_L^j chi_R^j`.
4. **The right readout**: mutual information between the reference qubit
   P and a qubit read out from R -- not a single-qubit expectation
   value, which the no-signaling theorem forbids from ever showing this.

All four pieces, plus the paper's own instance-selection criterion
(below), are implemented and unit-tested in `dashboard_core.wormhole` /
`dense_evolution.fermions`/`.entropy`/`.trotter` in the main repo.

## Instance selection: seed 61

A uniformly-random draw of which K=10 SYK terms to keep does **not**
reliably show the sign-dependent signal -- some seeds give a clean peak,
some the wrong sign for most of a sweep, some are flat noise. This
matches arXiv:2604.10090 directly: they didn't use an arbitrary instance
either, they picked one "selected for favorable commutation properties"
-- 34 commuting vs. 11 anticommuting pairs, among the C(10,2)=45 pairs of
their chosen K=10 terms.

`select_good_instance` reproduces that exact criterion: screen many
random seeds by their exact commuting/anticommuting pair count, keep the
one closest to the paper's ratio. Screened across 200 candidates for
N=8, **seed 61 matches exactly** (34 commuting / 11 anticommuting) and
is used for every experiment below.

## Experiment 1: t1 sweep -- the headline signature (run 2026-08-06)

`n_majorana=8, k_terms=10, J=sqrt(2), seed=61, t0=0.3, with_message=True`,
sweeping post-coupling evolution time `t1`, backend=exact.

| t1 | I(mu=+12) | I(mu=-12) | delta |
|---|---|---|---|
| 0.10 | 0.05259 | 0.05327 | +0.00068 |
| 0.30 | 0.02903 | 0.03116 | +0.00213 |
| 0.50 | 0.01670 | 0.02080 | +0.00410 |
| **0.60** | **0.01326** | **0.01793** | **+0.00468** |
| 0.70 | 0.01060 | 0.01524 | +0.00463 |
| 1.00 | 0.00401 | 0.00485 | +0.00084 |
| 1.20 | 0.00141 | 0.00065 | -0.00076 |
| 1.50 | 0.00096 | 0.00073 | -0.00023 |

Both mutual-information curves decay with `t1` (the message dilutes as
the state keeps evolving), but `I(mu=-12)` stays consistently above
`I(mu=+12)` from `t1=0.1` through `t1=1.0` -- a single smooth,
sign-dependent peak at `t1=0.60`, before the two curves cross near
`t1=1.2`, at which point the signal is too small to trust the sign of
the (now negative) delta.

![t1 sweep](assets/wormhole_syk_teleportation/wormhole_t1_sweep.png)

Full data: `data/wormhole_t1_sweep.csv`.

## Experiment 2: message-vs-no-message control (run 2026-08-06)

Same setup, `t1 in {0.10, 0.30, 0.60, 0.85, 1.20}`, but comparing
`with_message=True` (the real protocol, message swapped into L) against
`with_message=False` (P, Q left as an isolated Bell pair, never touching
L or R).

**Result: `I(P:R[0])` is exactly `0` (to float precision, ~1e-15,
numerical noise) at every point tested, for both signs of mu, when
`with_message=False`.**

This is expected from the circuit's own topology -- without the swap, P
and Q are never coupled to L or R by anything (the L-R coupling only
touches L and R), so P and R *must* remain in a product state, and
`I(P:R)=0` follows structurally, not from a numerical coincidence. It's
still a real, useful check: it confirms the implementation does exactly
what it's supposed to (all of the sign-dependent signal genuinely comes
from the message-injection pathway) and rules out a class of bugs where
correlation could leak through some other, unintended path.

![message control](assets/wormhole_syk_teleportation/wormhole_message_control.png)

Full data: `data/wormhole_message_control.csv`.

## Experiment 3: mu-magnitude scan (run 2026-08-06)

Fixed `t0=0.3, t1=0.60` (the Experiment 1 peak), sweeping `|mu|` from 4
to 20 (paper's own value: `mu=12`).

| mu | delta |
|---|---|
| 4 | +0.00068 |
| 8 | +0.00336 |
| 9 | +0.00402 |
| 10 | +0.00450 |
| **11** | **+0.00473** |
| 12 | +0.00468 |
| 13 | +0.00435 |
| 14 | +0.00383 |
| 16 | +0.00255 |
| 20 | +0.00009 |

The sign-dependent delta rises, peaks at `mu~11-12`, then falls --
**not** a monotonic function of coupling strength, even though the
*total* mutual information (`I(mu=+12)` and `I(mu=-12)` individually)
keeps growing with `|mu|`. The paper's own choice, `mu=12`, sits almost
exactly on this instance's peak (`11` vs `12` are within each other's
noise floor at this scan resolution).

![mu scan](assets/wormhole_syk_teleportation/wormhole_mu_scan.png)

Full data: `data/wormhole_mu_scan.csv`.

## Experiment 4: t0 (pre-coupling scrambling time) scan (run 2026-08-06)

Fixed `mu=+-12, t1=0.60`, sweeping the pre-coupling scrambling time
`t0` from 0.05 to 0.9 (paper's own value: `t0=0.3`).

| t0 | delta |
|---|---|
| 0.05 | +0.00018 |
| 0.10 | +0.00067 |
| 0.20 | +0.00235 |
| 0.30 | +0.00468 |
| 0.40 | +0.00724 |
| **0.60** | **+0.00972** |
| 0.90 | +0.00537 |

The signal grows monotonically from `t0=0.05` through `t0~0.60` --
**more than double** the delta at the paper's own `t0=0.3` -- then falls
by `t0=0.9`, where `I(mu=+12)` itself has nearly vanished (`0.00039`),
meaning the message has decohered before the coupling even applies. This
is consistent with the protocol's theoretical chaos requirement: the SYK
system needs enough time to scramble before the coupling can produce a
traversable-wormhole-like signature, but too much scrambling dilutes the
injected message before it matters.

![t0 scan](assets/wormhole_syk_teleportation/wormhole_t0_scan.png)

Full data: `data/wormhole_t0_scan.csv`.

## Experiment 5: 2D (t0, mu) joint grid search (run 2026-08-06)

Experiments 3 and 4 scanned `mu` and `t0` independently, each holding
the other fixed at a "reasonable default" -- a quick follow-up check at
the best `t0=0.60` found the mu-peak shifts higher there (`mu=16` beat
`mu=12`, still rising at the edge of that check's tested range), meaning
neither 1D scan alone finds the true joint optimum. This experiment
resolves that with a real grid: `t0` in `[0.05, 1.50]` (30 values,
step 0.05) x `mu` in `[2, 30]` (29 values, step 1) = 870 points, `t1`
held fixed at `0.60` (Experiment 1's own peak), `with_message=True`.

**Global maximum on the grid: `t0=0.65, mu=15.0`, `delta=+0.01167`** --
noticeably better than either 1D scan's own peak (`t0=0.60,mu=12` gave
`+0.00972`; `t0=0.30,mu=11` gave `+0.00473`).

| t0 | mu | delta |
|---|---|---|
| **0.65** | **15.0** | **+0.01167** |
| 0.60 | 15.0 | +0.01163 |
| 0.65 | 16.0 | +0.01161 |
| 0.60 | 16.0 | +0.01153 |
| 0.60 | 14.0 | +0.01134 |
| 0.65 | 14.0 | +0.01133 |

The top points cluster in a compact region (`t0` in `[0.60, 0.70]`, `mu`
in `[14, 17]`) rather than a single isolated pixel -- a broad, smooth
plateau, not a grid-resolution artifact. The heatmap below shows a
single well-defined hill with no secondary peaks, falling off smoothly
in every direction; far from the peak (`t0 > 1.2` and `mu > 25`) `delta`
turns negative, an observation not investigated further here.

**Performance note**: `run_wormhole_protocol` rebuilds the SYK/coupling
Hamiltonians and re-diagonalizes both on every call, even though neither
depends on `t0`/`mu`/`t1` for a fixed `(seed, n_majorana, k_terms)` --
measured directly at 4.3-6.4s, versus 0.022s/call for the actual
`t0`/`mu`-dependent evolution + readout. Precomputing the Hamiltonians
and their eigendecompositions once, then reusing them for all 870 grid
points, cut this experiment from an estimated ~2 hours down to **47.6
seconds** (~165x) -- confirmed by timing both approaches directly, not
assumed. No multiprocessing needed. See `scripts/wormhole_syk_teleportation.py`'s
`run_2d_grid_search` for the implementation.

![2D grid search](assets/wormhole_syk_teleportation/wormhole_2d_grid.png)

Full data: `data/wormhole_2d_grid.csv`.

## Experiment 6: t1 re-scan at the 2D optimum (run 2026-08-06)

Experiment 5 flagged its own gap: the 870-point grid held `t1` fixed at
`0.60` (Experiment 1's 1D peak), and noted that `t1`'s optimum could
plausibly shift once `t0`/`mu` are no longer at their original 1D-scan
defaults -- unverified. This experiment checks it directly: fix
`t0=0.65, mu=+-15.0` (Experiment 5's grid optimum), sweep `t1`.

A fine sweep, `t1` in `[0.05, 1.30]` (126 values, step 0.01), reusing
Experiment 5's precompute-once approach:

**New best point: `t0=0.65, mu=15.0, t1=0.41`, `delta=+0.01518`** -- about
30% above Experiment 5's headline value (`+0.01167`, `t1=0.60` held
fixed), on a smooth, single-peaked curve (no secondary bumps, monotonic
decay on both sides across the full swept range).

| t1 | delta |
|---|---|
| 0.10 | +0.00976 |
| 0.20 | +0.01197 |
| 0.30 | +0.01403 |
| 0.40 | +0.01518 |
| **0.41** | **+0.01518** |
| 0.50 | +0.01439 |
| 0.60 (Experiment 5's fixed value) | +0.01167 |
| 0.70 | +0.00811 |
| 0.90 | +0.00361 |
| 1.10 | +0.00307 |
| 1.30 | +0.00158 |

This confirms the caveat was correct, but only takes one
coordinate-ascent step, not a full 3D search: `t0` and `mu` were held at
Experiment 5's values throughout this scan, so whether *they* would
shift again now that `t1` has moved (the same pattern that motivated
Experiment 5 in the first place) is still open. A converged joint
optimum would need to iterate this, or a real 3D grid -- Experiment 7
does exactly that.

![t1 re-scan at the 2D optimum](assets/wormhole_syk_teleportation/wormhole_t1_rescan_optimum.png)

Uses Experiment 5's precompute-once helpers (same measured ~165x speedup
rationale), now folded into `scripts/wormhole_syk_teleportation.py`'s
`run_t1_rescan` -- reproduced by `python scripts/wormhole_syk_teleportation.py`
like every other experiment on this page. Full data:
`data/wormhole_t1_rescan_optimum.csv`.

## Experiment 7: iterated coordinate ascent toward the joint optimum (run 2026-08-07)

Experiment 6 moved `t1` once but never checked whether `t0`/`mu` would
shift *again* -- the same gap that motivated Experiment 5 over the
original 1D scans. This experiment iterates the fix instead of taking
one more single step: starting from Experiment 5's point (`t0=0.65,
mu=15.0, t1=0.60`), each round alternates a full `t1` scan (Experiment
6's 126-point, step-0.01 resolution, `t0`/`mu` held fixed) and a full
`(t0, mu)` grid (Experiment 5's 870-point, step-0.05/1.0 resolution,
`t1` held fixed) -- both at their original resolutions, so every round
stays directly comparable to Experiments 5 and 6, not a shortcut.

| Round | Stage | t0 | mu | t1 | delta |
|---|---|---|---|---|---|
| 0 | start (Experiment 5) | 0.65 | 15.0 | 0.60 | +0.01167 |
| 1 | t1 scan | 0.65 | 15.0 | 0.41 | +0.01518 |
| 1 | t0/mu grid | 0.70 | 17.0 | 0.41 | +0.01658 |
| 2 | t1 scan | 0.70 | 17.0 | 0.36 | +0.01688 |
| 2 | t0/mu grid | 0.70 | 17.0 | 0.36 | +0.01688 |
| 3 | t1 scan | 0.70 | 17.0 | 0.36 | +0.01688 |
| **3** | **t0/mu grid** | **0.70** | **17.0** | **0.36** | **+0.01688** |

**Converged after 3 rounds: `t0=0.70, mu=17.0, t1=0.36`, `delta=+0.01688`**
-- a genuine fixed point (round 3 reproduces round 2 exactly, both
sub-steps), not just a small step-to-step wobble. **+44.6% over
Experiment 5's original headline value** (`+0.01167`), and +11.2% over
Experiment 6's single-step result (`+0.01518`). Total compute: ~186s for
all 7 evaluation rounds combined (precompute-once still applies).

![Convergence of iterated coordinate ascent](assets/wormhole_syk_teleportation/wormhole_coordinate_ascent_3d.png)

This resolves the open question from Experiment 6 -- but only up to
this grid's resolution, not as a proof of global optimality. An ad hoc
finer local grid explored around the converged point (not part of the
reproducible script) suggested the true continuum optimum sits close to
`mu=17.5`, just off this grid's integer `mu` values -- consistent with,
not contradicting, the converged answer; a real continuous optimizer
would be needed to settle global optimality rather than a fixed-point
on this specific grid. Produced by
`scripts/wormhole_syk_teleportation.py`'s `run_coordinate_ascent_3d`.
Full data: `data/wormhole_coordinate_ascent_3d.csv`.

## Experiment 8: does the converged point generalize across SYK instances? (run 2026-08-07)

Every experiment so far used a single instance, seed=61. Before treating
`t0=0.70, mu=17.0, t1=0.36` as anything more than a fact about that one
random Hamiltonian, the obvious check: does the *same* coordinate-ascent
procedure, run independently on other instances that equally match the
paper's own selection criterion (34 commuting / 11 anticommuting pairs),
converge to a similar point?

`find_multiple_seeds` screens seeds for an *exact* match (not just the
closest one, like `find_seed` does) -- 6 found within the first 3000
seeds tried: `61, 448, 1944, 2166, 2835, 2907`.

| Seed | Converged t0 | Converged mu | Converged t1 | Converged delta | Baseline delta (Exp. 5's start) |
|---|---|---|---|---|---|
| 61 | 0.70 | 17.0 | 0.36 | +0.01688 | +0.01167 |
| 448 | 0.90 | 19.0 | **1.30 (grid edge)** | +0.00714 | +0.00018 |
| 1944 | 0.40 | 16.0 | 0.29 | +0.06215 | +0.02075 |
| 2166 | **1.50 (grid edge)** | 20.0 | 0.89 | +0.01930 | **-0.00273** |
| 2835 | 0.55 | 17.0 | 0.24 | +0.01254 | **-0.00379** |
| 2907 | 1.05 | 19.0 | 0.42 | +0.00498 | +0.00069 |

![Converged (t0, mu, t1) scattered across 6 SYK instances](assets/wormhole_syk_teleportation/wormhole_generality_check.png)

**Honest negative result: it does not generalize.**

- **No clustering.** Converged `t0` spans `[0.40, 1.50]` and `t1` spans
  `[0.24, 1.30]` -- close to the *entire* scanned range in both cases,
  not a tight cluster near seed=61's answer.
- **2 of 6 instances hit the edge of the scanned grid** (seed 448 at
  `t1=1.30`, the top of the scanned range; seed 2166 at `t0=1.50`, the
  top of *its* scanned range) -- for these, the procedure didn't find a
  real interior fixed point, it ran off the edge of what was searched.
  Their true optimum, if the range were extended, is unknown.
- **2 of 6 instances start with the *wrong sign*.** At Experiment 5's
  own starting point (`t0=0.65, mu=15.0, t1=0.60` -- not derived from
  these instances at all, just reused as a common starting point), seeds
  2166 and 2835 give a *negative* delta: the sign-dependent asymmetry
  the whole protocol is built to show points the wrong way at that point
  for those instances, before any optimization even starts.
- Percentage-improvement figures are deliberately not reported here --
  with a baseline near zero (seed 448: `+0.00018`) or negative (seeds
  2166, 2835), a percentage explodes into a meaningless number (e.g.
  seed 448's nominal improvement is `+3865%`) that describes the tiny
  denominator, not a real effect size.

**Conclusion**: `t0=0.70, mu=17.0, t1=0.36` is a property of seed=61's
specific random Hamiltonian, not a general finding about the
traversable-wormhole-inspired teleportation protocol. Produced by
`scripts/wormhole_syk_teleportation.py`'s `run_generality_check`. Full
data: `data/wormhole_generality_check.csv`.

## Experiment 9: does the signal survive realistic hardware noise? (run 2026-08-07)

Every experiment so far used the exact-evolution backend -- ideal, no
noise. Even setting aside Experiment 8's generalization problem: does
the sign-dependent signal survive at seed=61's own best point
(`t0=0.70, mu=17.0, t1=0.36`) under noise like a real NISQ device would
actually have?

This needs the Trotterized gate-circuit backend, since noise has to be
injected *mid-circuit* -- `run_wormhole_protocol_trotter` runs its whole
circuit in one call with no seam to interrupt, so `run_trotter_noise_scan`
reimplements the same three-phase construction (t0 evolution -> mu
coupling -> t1 evolution, each a real `trotter_evolve_ops` circuit) and
applies a real stochastic depolarizing Kraus channel
(`dense_evolution.registry.NoiseModel`) after each phase, not just once
at the end. Compared against the *noiseless* Trotter result specifically
(`+0.01728`, ~2% above the exact backend's `+0.01688` -- consistent with
the known Trotter-vs-exact discretization gap), not the exact backend,
so the effect of physical noise isn't conflated with Trotterization
error. Each noise level averaged over 6 independent stochastic trials
(`NoiseModel.apply_to_sv` is a single-shot Kraus draw, not an ensemble
average).

| Depolarizing p | Mean delta | Std dev |
|---|---|---|
| 0.000 | +0.01728 | 0.00000 |
| 0.005 | +0.00924 | 0.00704 |
| 0.010 | +0.00051 | 0.01203 |
| 0.020 | -0.00330 | 0.01448 |
| 0.050 | -0.00979 | 0.00624 |

![Sign-dependent signal vs. realistic depolarizing noise](assets/wormhole_syk_teleportation/wormhole_trotter_noise_scan.png)

**Second honest negative result: the signal does not survive realistic
noise.** It decays monotonically and crosses zero between `p=0.01` and
`p=0.02`. Already at `p=0.01` -- a depolarizing rate well within range
of current real superconducting-qubit hardware -- the mean signal
(`+0.00051`) is smaller than its own trial-to-trial standard deviation
(`0.01203`): statistically indistinguishable from zero, not a small but
real effect. Produced by `scripts/wormhole_syk_teleportation.py`'s
`run_trotter_noise_scan`. Full data:
`data/wormhole_trotter_noise_scan.csv`.

## Experiment 10: cross-check against the paper's own "Ensemble robustness" claim (run 2026-08-07)

arXiv:2604.10090 has its own ensemble-robustness section: from 100
disorder realizations at K=10, it reports that "other disorder
realizations in the same ensemble also exhibit qualitatively similar
mutual-information dynamics, and in particular retain the sign-dependent
asymmetry... This indicates that the mutual information behavior is a
generic feature of the ensemble rather than a peculiarity of the chosen
Hamiltonian." The paper is explicit that their chosen instance (seed=61
here) was selected for having an unusually *large* asymmetry, not an
unusually *signed* one.

Experiment 8's negative-baseline finding (2 of 6 instances) has a real
confound worth taking seriously: it evaluated all 6 instances at
Experiment 5's point (`t0=0.65, mu=15.0, t1=0.60`), which was itself
optimized on seed=61's own 1D scans -- an instance showing the "wrong"
sign there could simply be a bad evaluation point for it, not a
genuinely reversed signal. This experiment controls for that directly:
re-evaluate all 6 instances at the paper's own stated default parameters
(`t0=0.3, mu=12, t1=0.60`) instead.

| Seed | delta @ Exp. 5's point (0.65, 15, 0.60) | delta @ paper defaults (0.3, 12, 0.60) |
|---|---|---|
| 61 | +0.01167 | +0.00468 |
| 448 | +0.00018 | +0.00042 |
| 1944 | +0.02075 | +0.03690 |
| 2166 | -0.00273 | **-0.00125 (still wrong)** |
| 2835 | -0.00379 | **+0.00200 (fixed)** |
| 2907 | +0.00069 | **-0.00011 (now wrong)** |

![Sign-dependent asymmetry at the paper's own default parameters](assets/wormhole_syk_teleportation/wormhole_paper_defaults_comparison.png)

**Third honest result, and the most direct comparison to the source
paper in this whole write-up**: controlling for the evaluation-point
confound changes *which* instances look wrong-signed (seed 2835's
Experiment 8 reversal turns out to have been a point-choice artifact --
correctly signed at the paper's own defaults), but does not make the
problem go away. **2 of 6 instances (2166, 2907) are still wrong-signed
at the paper's own stated default parameters.** Seed 2166 is wrong-signed
at *both* evaluation points tested across this whole write-up -- not an
artifact of any one specific parameter choice. For this specific
34/11-selection-matched subset, the "generic feature of the ensemble"
claim does not hold up. Produced by
`scripts/wormhole_syk_teleportation.py`'s
`run_paper_defaults_comparison`. Full data:
`data/wormhole_paper_defaults_comparison.csv`.

## Experiment 11: large-sample (n=100) ensemble sign check (run 2026-08-07)

Experiment 10's finding is real but statistically thin: 2 of 6 is a
small sample, easy to dismiss as noise. This experiment repeats the
identical check -- delta at arXiv:2604.10090's own default parameters
(`t0=0.3, mu=12, t1=0.60`) -- across `n=100` instances that exactly
match the paper's own selection criterion, the same sample size the
paper itself reports for its "Ensemble robustness" section.
`find_multiple_seeds` had to screen 106,097 candidate seeds to find
100 exact 34/11 matches (roughly 1 in 1,000).

Alongside the sign check, two candidate structural explanations for
*why* the sign varies -- floated informally while investigating
Experiment 10, not yet part of any committed experiment before this
one -- are tested for real correlation, not just eyeballed:

- **Majorana mode-usage imbalance**: for each instance's 10 sparse
  coupling terms, how unevenly are the 8 Majorana modes used (some
  modes appearing in many terms, "hubs"; others in few, "isolated"),
  measured as the standard deviation of per-mode usage counts.
- **Spectral level-spacing r-statistic**: `r = min(d_n, d_n+1) /
  max(d_n, d_n+1)` averaged over adjacent eigenvalue gaps of the
  L-side SYK Hamiltonian -- a standard quantum-chaos diagnostic
  (Poisson/integrable ~0.386, GOE/chaotic ~0.530).

![n=100 ensemble sign check vs. structural/spectral correlates](assets/wormhole_syk_teleportation/wormhole_ensemble_sign_check.png)

**Fourth honest result, and the strongest of the whole write-up:
49 of 100 instances (49%) are wrong-signed at the paper's own default
parameters** -- essentially a coin flip, not "a generic feature of the
ensemble." This isn't a marginal statistical wobble; at this sample
size the effect is unambiguous.

Neither candidate structural explanation holds up either. Mode-usage
imbalance: `r=+0.171, p=0.090` -- not significant at the conventional
0.05 threshold. Level-spacing r-statistic: `r=+0.087, p=0.388` -- not
significant, and nowhere close. **This corrects an earlier impression**:
an ad hoc `n=6` look at mode-usage imbalance (using only Experiment
10's 6 instances) had suggested a strong correlation (`r=+0.87,
p=0.022`) -- that does not replicate at `n=100` and should be treated
as a small-sample artifact, not a real effect. Why the sign varies
across instances remains genuinely open. Produced by
`scripts/wormhole_syk_teleportation.py`'s `run_ensemble_sign_check`.
Full data: `data/wormhole_ensemble_sign_check.csv`.

## Experiment 12: size winding (run 2026-08-07)

Experiment 11 ruled out two candidate structural explanations for the
sign variance -- Majorana mode-usage imbalance and the spectral
level-spacing r-statistic -- but both were informal, ad hoc candidates,
not something the paper itself proposes as diagnostic. arXiv:2604.10090
does propose its own theory-motivated diagnostic for exactly this kind
of question: **size winding** (Sec. S6, Eqs. S18-S22), a signature of
how a Heisenberg-evolved operator's growth is organized in the basis of
Majorana strings.

This experiment computes it directly, not by analogy. A single L-side
Majorana operator `chi_j(t) = exp(iHt) chi_j exp(-iHt)` is expanded in
the basis of Majorana strings `Gamma_P` (one string per subset `P` of
the 8 Majorana modes, `Gamma_P = 2^(|P|/2) * i^(|P|(|P|-1)/2) *`
ordered product of `chi_j` for `j` in `P`, per the paper's Eq. S19).
The winding size distribution `q(l) = sum_{|P|=l} c_P(t)^2` and
ordinary size distribution `P(l) = sum_{|P|=l} |c_P(t)|^2` give two
quantities per size sector `l`: the phase coherence ratio
`R(l) = |q(l)|/P(l)` (1 means every term in that sector has aligned
phase; less than 1 means dephasing) and the phase itself,
`arg(q(l))` (the paper's "perfect size winding" ansatz predicts this
phase grows linearly with `l`).

One subtlety had to be resolved empirically, not taken from the paper's
text as-is: the PDF-extracted formula for the basis normalization,
`Tr(Gamma_P Gamma_Q^dagger) = 2N delta_PQ`, doesn't reproduce correctly
when implemented literally (most likely a lost exponent from PDF
extraction). Building the `chi_j` operators directly from
`dense_evolution.fermions.majorana_pauli_terms` and computing the trace
numerically shows the real normalization is
`Tr(Gamma_P Gamma_Q^dagger) = 2^|P| * dim * delta_PQ` (`dim = 2^n_qubits
= 16`) -- verified directly via Hermiticity and orthogonality checks on
the actual matrices, not assumed. This is the normalization used
throughout `run_size_winding_check`.

Computed for the same 6 instances used in Experiments 8 and 10, at 4
post-quench times each (`t = 0.3, 0.7, 1.2, 2.0`) -- spot-checked first
on 3 individual seeds spanning both a correctly-signed instance (61)
and a consistently wrong-signed one (2166) before running the full
6-instance sweep, per an explicit request to verify generality before
writing this up.

![Size winding: mean operator size across 6 instances, phase coherence trivial everywhere](assets/wormhole_syk_teleportation/wormhole_size_winding.png)

**Fifth honest result: the diagnostic is structurally trivial, uniformly
across every instance and every time tested.** `R(l) = 1.0000` and
`arg(q(l)) = 0.0000` exactly (to floating-point precision, `~1e-15`) for
every `(seed, t)` pair in the sweep -- no dephasing, no winding, no
instance-to-instance variation at all. This is not a partial or noisy
null result; it is exactly flat. Like mode-usage imbalance and the
level-spacing r-statistic before it, this diagnostic does not
distinguish correctly- from wrong-signed instances -- a third
independent, theoretically-motivated candidate ruled out. The mean
operator size `<l>(t)` itself, by contrast, does show real physics:
consistent chaos-like growth from `<l>(0)=1` up to a peak around
`t~1.2-2.0` (values range `2.1-3.7` across the 6 instances) followed by
a finite-size recurrence (an expected artifact of evolving a small,
`n_majorana=8` system rather than a true large-N limit) -- confirming
the underlying operator-growth dynamics are real even though this
particular phase diagnostic isn't what explains the sign variance.
Produced by `scripts/wormhole_syk_teleportation.py`'s
`run_size_winding_check`. Full data: `data/wormhole_size_winding.csv`.

## Experiment 13: mechanistic check -- message-mode participation & operator growth rate (run 2026-08-07)

Experiments 11 and 12 ruled out three candidate explanations for the
sign variance, but all three were either informal aggregate statistics
(mode-usage imbalance, level-spacing r-statistic) or a diagnostic that
turned out to carry no per-instance information at all (size winding's
phase/R, exactly flat everywhere). This experiment tries two more
candidates that are each grounded directly in the actual protocol
implementation rather than a generic aggregate statistic, and reuses
Experiment 11's own n=100 instance set and `delta_at_paper_defaults`
values (`data/wormhole_ensemble_sign_check.csv`) rather than
re-screening from scratch -- both features are cheap enough (under 6
seconds total for the full n=100 sweep) that a fresh screen wasn't
needed.

**Feature A -- message-mode participation.**
`dense_evolution.fermions.majorana_pauli_terms`'s Jordan-Wigner mapping
(`j = (mode_index - 1) // 2`) shows Majorana modes 1 and 2 map onto
qubit index 0 -- exactly the qubit the message is swapped into
(`L[0]`) and read out from (`R[0]`) in
`dashboard_core.wormhole._initial_state_ops`/`run_wormhole_protocol`.
Experiment 11's mode-usage-std treated all 8 modes as interchangeable;
this instead asks a sharper, protocol-specific question: do the K=10
SYK quads that happen to touch the *message qubit's own* modes,
specifically, predict the sign? For each seed, the exact same
quad-selection RNG sequence used internally by `build_sparse_syk_terms`
is replayed to recover which quads were chosen, and the count touching
mode 1 or 2 is recorded (purely combinatorial, no simulation).

**Feature B -- operator growth rate.** Experiment 12 already computes
real, non-trivial, instance-varying mean operator size `<l>(t)` as a
side effect of its (trivial) phase-winding computation, but that
experiment's only goal at the time was checking the phase diagnostic,
which came back flat -- growth rate itself was never correlated
against the sign. This reuses `run_size_winding_check` unmodified,
called on the same 100 seeds at two probe times in the growth region
Experiment 12 identified (`t=0.7`, `t=1.2`).

![Mechanistic check: message-mode participation and operator growth rate vs. sign, n=100](assets/wormhole_syk_teleportation/wormhole_mechanistic_check.png)

**Fourth honest negative result: neither correlates.** Message-mode
participation: `r=-0.012, p=0.90`. Operator growth rate at `t=1.2`:
`r=+0.126, p=0.21`. Both are far from the conventional 0.05 threshold
-- not a marginal case, a clean null on both counts. This rules out a
4th and 5th candidate explanation, on top of Experiment 11's two and
Experiment 12's structurally-uninformative phase diagnostic. Both
features were tested against the *same* n=100 sample Experiment 11
already used, a real multiple-comparisons concern for any candidate
that *did* come back significant -- since neither did here (and not
marginally), no holdout re-verification was needed this time, but the
concern is flagged for any future candidate that does show a hit.
Produced by `scripts/wormhole_syk_teleportation.py`'s
`run_mechanistic_check`. Full data:
`data/wormhole_mechanistic_check.csv`.

## What this does NOT show

- **Even Experiment 11's n=100 sample isn't a strict replication of
  arXiv:2604.10090's own ensemble study.** It matches their reported
  sample size, but their ensemble's exact selection methodology for
  those 100 draws isn't fully specified in the main text available
  here -- our 100 instances are filtered strictly to an *exact*
  34/11 commuting/anticommuting match (`find_multiple_seeds`), a
  narrower and possibly differently-distributed subset than whatever
  their 100 realizations were. "49/100 wrong-signed" is a real,
  verified result for this specific subset, not a claim about their
  exact reported statistics.
- **The structural/spectral/theoretical null results don't rule out
  every possible explanation, just the five tested.** Mode-usage
  imbalance, the level-spacing r-statistic, the paper's own size
  winding diagnostic (Experiment 12), message-mode participation, and
  operator growth rate (both Experiment 13) all fail to correlate with
  the sign; none of the five correlating doesn't mean no structural or
  theoretical property does. Experiment 12 also only checked
  `majorana_index=1` (one of the 8 possible starting operators) and 4
  discrete times, on the single-sided L-Hamiltonian only -- not a
  full time/index sweep, and not the combined L+R+P+Q system that the
  mutual-information readout actually uses; Experiment 13's growth-rate
  feature inherits that same scope limit, and its message-mode feature
  only checks modes 1/2 (the message qubit), not whether *any* single
  mode's participation predicts the sign for that mode's own readout
  qubit if the message were injected elsewhere. Also worth flagging
  explicitly: Experiment 13 is the 4th/5th candidate tested on the
  *same* n=100 sample as Experiment 11's 2 -- a real multiple-testing
  risk that would matter more if any candidate had come back
  significant (none has, cleanly, so far).
- **Experiment 7's fixed point is not proven globally optimal, and does
  not generalize to other instances (Experiment 8).** Coordinate ascent
  converging to a stable point on a given grid resolution is not the
  same as proving that point is the true continuous joint maximum for
  even a single instance -- and Experiment 8 shows directly that it
  isn't a property of the protocol at all, just of seed=61.
- **Experiment 8 is itself only a 6-instance sample**, and 2 of those 6
  hit the edge of the scanned range rather than a real interior fixed
  point -- a wider scan could resolve those into genuine (still likely
  instance-specific) answers, but wasn't run here (cost scales with
  range x resolution; 6 instances at the current range already took
  ~15 minutes).
- **Experiment 9 was only run for seed=61, not the other 5 instances
  from Experiment 8.** Given Experiment 8's own finding, there's no
  reason to expect this noise-robustness result generalizes either --
  it's one more seed=61-specific data point, not evidence about noise
  robustness for the protocol broadly. Its 6-trial-per-point average is
  also a small sample; the standard deviations shown are real but not
  tight error bars.
- **Experiments 1-8 use backend=exact (matrix exponentiation), not the
  Trotterized real-gate-circuit backend** -- both are implemented and
  cross-verified in the main repo's test suite to agree closely at the
  known peak (`I(mu=+12)=0.01301` Trotter vs `0.01326` exact,
  `I(mu=-12)=0.01821` vs `0.01793`) -- exact was used for those
  experiments purely for scan speed (~78 protocol calls at ~4-5s each).
  Experiment 9 is the one exception, by necessity.
- **No claim that `t0=0.65`, `mu=15`, or any other exact decimal value
  is a "special" number** in any deeper sense -- even Experiment 5's
  finer grid (step 0.05 for t0, step 1 for mu) is still a grid, not a
  continuous optimizer; the broad, smooth plateau around the max (six
  points within 0.0004 of the top value) means the *true* continuum
  optimum is somewhere in that neighborhood, not necessarily at exactly
  `(0.65, 15.0)`. Reading more into the specific digits than the grid
  resolution supports would be a mistake.

## Reproduce

```bash
python scripts/wormhole_syk_teleportation.py
```

Requires `dense-evolution>=8.1.49` (`pip install dense-evolution`).
Regenerates all five CSVs in `data/` and PNGs in `images/` from scratch
-- no pre-computed results are checked into this repo outside of
`docs/assets/` (see this repo's README for that convention). Experiments
1-4 take a few minutes (~78 protocol calls at ~4-5s each); Experiment 5's
870-point grid takes well under a minute thanks to its precompute-once
optimization.
