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

## What this does NOT show

- **Experiment 7's fixed point is not proven globally optimal.**
  Coordinate ascent converging to a stable point on a given grid
  resolution is not the same as proving that point is the true
  continuous joint maximum -- a different grid resolution, or a real
  continuous optimizer, could in principle find a nearby but distinct
  point. See Experiment 7's own caveat above.
- **All backend=exact (matrix exponentiation), not the Trotterized
  real-gate-circuit backend.** Both are implemented and cross-verified
  in the main repo's test suite to agree closely at the known peak
  (`I(mu=+12)=0.01301` Trotter vs `0.01326` exact, `I(mu=-12)=0.01821`
  vs `0.01793`) -- exact was used here purely for scan speed (~78 protocol
  calls at ~4-5s each).
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
