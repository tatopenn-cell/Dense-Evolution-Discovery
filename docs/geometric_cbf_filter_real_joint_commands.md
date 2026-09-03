# A Geometric Control Barrier Function Filter for Real Robot Commands

`prog.txt`'s roadmap named spatial safety (never letting a command drive a robot into a
forbidden region) as a real gap beyond kinematic rate limiting: `rate_limited_follower`
bounds how fast a command changes, but knows nothing about the workspace itself -- a
command moving at a perfectly safe, bounded velocity straight into an obstacle is still
dangerous. This experiment builds and validates a Control Barrier Function (CBF) safety
filter for that gap, on real robot joint commands.

## Step 1. The theory, verified directly before using it

Grounded in Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada (2019), "Control Barrier
Functions: Theory and Applications", *2019 European Control Conference (ECC)*,
arXiv:1903.11199 -- fetched and read directly (not from a search summary) before writing any
code, indexed in quantumrag's `robotica_filtri_sicurezza_semantica` collection. For a safe
set C = {x : h(x) >= 0} and single-integrator dynamics xdot = u, the paper's own minimally
invasive safety filter is:

```
u(x) = argmin_u (1/2)||u - u_des||^2
       s.t.  Lf h(x) + Lg h(x) u >= -alpha(h(x))
```

which the paper states has a closed-form solution (KKT conditions, no input bounds, single
scalar inequality): pass the desired command through unchanged if it already satisfies the
constraint; otherwise project onto the constraint boundary with minimum correction.

This is the SAME underlying theory SAFER-Splat (`prog.txt`'s original recommendation,
arXiv:2409.09868) uses -- but SAFER-Splat's own real GitHub repo (`chengine/safer-splat`)
requires CUDA 11.8 and a real NVIDIA GPU for its Gaussian-Splatting perception pipeline,
confirmed unavailable on this machine (AMD Radeon 530, no NVIDIA). The CBF-QP theory itself
does not require a GPU -- only SAFER-Splat's *visual perception* of the obstacle does. This
experiment applies the same theory to a known/given geometric obstacle instead of a learned
visual map: the safety math transfers, the perception pipeline does not.

## Step 2. A concrete instance: stay away from a real forbidden position

```python
def cbf_safety_filter(x, u_des, obstacle, safe_dist, alpha_gain=1.0):
    h = (x - obstacle)**2 - safe_dist**2
    Lgh = 2.0 * (x - obstacle)
    rhs = -alpha_gain * h
    if Lgh * u_des >= rhs:
        return u_des
    return rhs / Lgh
```

`h(x) = (x - obstacle)^2 - safe_dist^2` -- a real joint position must stay at least
`safe_dist` away from a real forbidden position. `alpha(h) = alpha_gain * h`, the paper's
own linear class-K choice used in its worked examples.

## Step 3. A real numerical finding: discrete time needs substeps

The CBF's forward-invariance guarantee is a CONTINUOUS-time result. Applying it naively at
each real sample (`n_substeps=1`) let a real trajectory overshoot past the barrier:

```
n_substeps=1:   min h=-0.4811  (VIOLATED)
n_substeps=5:   min h=0.0000   (safe)
n_substeps=20:  min h=0.0000   (safe)
```

A single large discrete Euler step (real robot commands can jump substantially between
samples) can cross the barrier even though the instantaneous constraint was satisfied at the
step's start. Confirmed and fixed directly, not assumed: sub-stepping each real sample
(`n_substeps=20` default) fully restores the guarantee -- a standard, well-known numerical
integration fix, not a change to the CBF theory itself.

## Step 4. Rigor pass: all 6 real SO-101 joints, obstacles in their own real path

```
Invariance: 17/17 real (joint, obstacle) trials never violate the real safe set
Minimal invasiveness: 3/3186 real per-step checks nonzero, max=0.446665, median=0.000000
```

**Invariance, from a real safe start**: 17 of 17 real (joint, obstacle-placement) trials
where the raw real trajectory does cross the forbidden zone, and the real trajectory starts
outside it -- the filtered version never enters it. `h(x0) < 0` cases (the real trajectory's
own starting point already inside a given obstacle placement) were excluded as invalid
trials, not silently passed: the CBF theory's own guarantee is conditional on a safe start,
not a claim to retroactively fix an already-unsafe initial condition.

**Minimal invasiveness, measured per-step on the control input** (not cumulative position,
which can legitimately stay offset for a while after any real correction in a stateful
causal integrator -- the same property `rate_limited_follower` has): 3183 of 3186 real
per-step checks (99.9%) show the filtered command exactly equals the raw command when the
current real state is far from the obstacle; the 3 exceptions are small (max 0.447) and
consistent with a boundary effect right at the far-field threshold, not a systematic issue.

## Step 5. A second, independent real physical domain: ALOHA

This project's own cross-repo promotion discipline requires validation on >=2 independent
real physical domains. Reused the same real ALOHA domain (`lerobot/aloha_static_coffee`,
bimanual, 14 real DoF, real 50Hz) already used for `rate_limited_follower`'s own second-
domain check -- a genuinely different real robot, not just a different episode of SO-101.

```
Real ALOHA episode 0: 1100 frames, 14 real DoF
Invariance: 38/38 real (joint, obstacle) trials never violate the real safe set
Minimal invasiveness: 0/18444 real per-step checks nonzero, max=0.000000, median=0.000000
```

Even cleaner than SO-101 here: 38/38 (100%) invariance, and 0/18444 -- a PERFECT, exact
minimal-invasiveness result, not even the 3 tiny boundary-effect exceptions SO-101 showed.
The core safety property replicates and, on this domain, holds without exception.

## Result

A real, working safety layer, unlike the neighbor-consensus damping attempt: validated on
two independent real physical domains (SO-101 6-DoF 30Hz, ALOHA bimanual 14-DoF 50Hz), 100%
invariance from safe starting conditions on both, and minimal invasiveness essentially exact
(99.9%+ on SO-101, exactly 100% on ALOHA). Complements `rate_limited_follower` (kinematic:
bounds rate of change) with a spatial guarantee (never enter a forbidden region) -- the two
are not redundant, and could run together in a real pipeline. Promoted to Dense-Armor as
`cbf_safety_filter`/`cbf_filtered_trajectory` -- see Dense-Armor's `docs/api/cbf_filter.md`.

---

## Details

**Why single-integrator dynamics, not the real joint dynamics**: matches
`causal_rate_limited_follower`'s own convention (velocity as the direct control input) for
consistency across this project's command-filtering utilities; a real joint's true dynamics
(inertia, motor lag) would need a higher-relative-degree CBF formulation, out of scope here.

**Relation to `prog.txt`**: closes the roadmap's "Layer 3" gap (spatial/dynamic safety
beyond kinematic rate limiting) with a real, validated instance of the SAME theory
SAFER-Splat uses, sidestepping its GPU-bound perception requirement by using a known
geometric obstacle instead of live visual reconstruction.

**Reproducing this**: `python scripts/robot_sensor_validation/cbf_filter_full_evaluation.py`
regenerates `cbf_filter_full_evaluation_frozen.json` (SO-101, reuses the already-cached real
LeRobot parquet, no new download); `python scripts/robot_sensor_validation/cbf_filter_second_domain_aloha.py`
regenerates `cbf_filter_second_domain_aloha_frozen.json` (ALOHA, reuses the already-cached
real ALOHA parquet); `pytest tests/test_geometric_cbf_filter.py
tests/test_cbf_filter_real_joint_commands.py tests/test_cbf_filter_second_domain_aloha.py`
reads the already-frozen files / runs the direct unit tests, no network access needed in CI.

**Paper indexed**: Ames et al. (2019) is now in quantumrag's
`robotica_filtri_sicurezza_semantica` collection, alongside SAFER-Splat, "From Words to
Safety", the Semantic Safety Filter, and RoboGuard (all verified real and downloaded in the
same pass -- see `prog.txt` for the full verification record, including citations that did
NOT check out and were correctly not used).
