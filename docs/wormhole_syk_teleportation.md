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
four parameter scans the paper itself didn't publish.

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

## What this does NOT show

- **The 2D joint optimum over (t0, mu) has not been found.** Each scan
  above holds the other axis fixed at a "reasonable default" (`t0=0.3`
  for the mu scan, `mu=12` for the t0 scan). A quick follow-up check at
  the best `t0=0.60` found the mu-peak shifts higher (`mu=16` gave a
  larger delta than `mu=12` at that `t0`, and was still rising at the
  edge of the tested range) -- so `t0=0.60, mu~12` is not necessarily
  the true global maximum, only the best point *along each individual
  axis*. A real 2D grid scan is the obvious next step if the exact
  optimum matters.
- **All backend=exact (matrix exponentiation), not the Trotterized
  real-gate-circuit backend.** Both are implemented and cross-verified
  in the main repo's test suite to agree closely at the known peak
  (`I(mu=+12)=0.01301` Trotter vs `0.01326` exact, `I(mu=-12)=0.01821`
  vs `0.01793`) -- exact was used here purely for scan speed (~78 protocol
  calls at ~4-5s each).
- **No claim that `t0=0.60` or `mu=11` are "special" numbers** in any
  deeper sense -- they're the best points on a coarse grid (step sizes
  0.1-0.3 for t0, 1-4 for mu), not precisely-converged optima. Reading
  more into their exact decimal values than the grid resolution supports
  would be a mistake.

## Reproduce

```bash
python scripts/wormhole_syk_teleportation.py
```

Requires `dense-evolution>=8.1.49` (`pip install dense-evolution`).
Regenerates all four CSVs in `data/` and PNGs in `images/` from scratch
-- no pre-computed results are checked into this repo outside of
`docs/assets/` (see this repo's README for that convention).
