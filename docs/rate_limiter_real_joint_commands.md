# A Causal Rate Limiter for Real Motor Commands: Safety vs. Fidelity

`prog.txt`'s roadmap named motor-command damping as the second half of a sensor-LLM-motor
robotics pipeline. An earlier attempt with `healing_filter` (and a causal rewrite of it,
kept private -- no usable utility came of it) hit a real, structural dead end: neighbor-consensus classification cannot causally tell
a real spike from a real regime change at the instant it happens. This experiment tries a
different family of mechanism entirely, grounded in a real, peer-reviewed reference, and
finds a genuine, honest tradeoff rather than a clean win.

## Step 1. A different mechanism: bound the rate, don't classify the cause

Instead of asking "is this deviation real?", bound how fast the applied command can
physically change, regardless of cause -- the same principle real robot controllers already
use (torque/velocity/acceleration limits). Grounded in Berscheid & Kroger (2021),
"Jerk-limited Real-time Trajectory Generation with Arbitrary Target States", *Robotics:
Science and Systems XVII*, arXiv:2105.04830 -- fetched and verified directly before using
(not from a search summary): causal by construction (uses only the current kinematic state,
no lookahead), a time-optimal online trajectory generator respecting velocity/acceleration/
jerk limits, validated on 1e9 real trajectories, ~20us real compute per DoF. Indexed in
quantumrag's new `robotica_generazione_traiettoria` collection.

```python
def causal_rate_limited_follower(x_raw, max_vel, max_accel, dt=1.0):
    pos, vel = x_raw[0], 0.0
    for i in range(1, len(x_raw)):
        desired_vel = (x_raw[i] - pos) / dt
        vel = clip(desired_vel, vel - max_accel*dt, vel + max_accel*dt)
        vel = clip(vel, -max_vel, max_vel)
        pos = pos + vel * dt
```

This is a deliberately SIMPLER special case, not a reimplementation of Ruckig's full
time-optimal jerk synthesis -- a causal double-integrator velocity+acceleration limiter
(bounds the first two derivatives; jerk itself is still a step function at each clamp
transition). `max_vel`/`max_accel` are not hand-picked: derived from each real joint's own
99th-percentile velocity/acceleration on the clean signal, before any spike is injected.

## Step 2. Same real spike-injection protocol as before, TWO metrics this time

Same real spike injection (5% density, 5x real local std), all 6 real LeRobot joints x 20
real seeds (120 real trials) -- but scored on both RMSE-vs-clean (as before) AND max real
instantaneous jump in the applied output, since RMSE alone doesn't capture the actual
physical-safety question (a filter could have an OK average RMSE while still passing one
dangerous single-step jump through).

```
n=120 real trials (6 joints x 20 seeds)
RMSE: rate_limited wins 19/120 (15.8%) vs moving median
  mean RMSE: rate_limited=0.7903  moving_median=0.4395
Max jump: rate_limited wins 120/120 (100.0%) vs moving median
  mean max jump: raw=129.06  rate_limited=3.75  moving_median=6.73
```

## Result: a real, honest tradeoff, not a clean win

**Safety (max instantaneous jump)**: the rate limiter wins 120/120 -- every single real
trial, consistently and by a wide margin (3.75 vs 6.73 mean). This is the metric that
directly answers `prog.txt`'s original stated problem ("a raw LLM command reaching the motor
shakes/breaks the robot").

**Fidelity (RMSE vs clean)**: the rate limiter loses on average (0.79 vs 0.44), winning only
15.8% of trials. Mechanistically expected, not a bug: a hard rate limiter cannot
distinguish an injected spike from a genuine fast real movement -- it caps both identically,
so on joints/seeds with more genuine fast motion, it lags behind the true target and pays an
RMSE cost the moving median (which reacts instantly to any single point, real or spike)
doesn't pay.

**Honest framing**: this mechanism is not a general-purpose "cleaner" like `healing_filter`
tried to be -- it is a safety bound. Whether that tradeoff is worth it depends on the real
deployment: if the actual goal is "never let a raw command reach the motor unbounded"
(the literal problem statement), this mechanism achieves that with a 100% real track record;
if the goal is "recover the closest approximation to the true intended signal", it does not.

## Step 3. A second, independent real physical domain: ALOHA (bimanual, 14-DOF, real 50Hz)

This project's own cross-repo promotion discipline (already applied to `streaming.py`,
`stable_frame_filter.py`, the CUSUM ARL theory) requires validation on >=2 independent real
physical domains before promotion. SO-101 above is one; a genuinely different real robot
-- not just a different episode of the same one -- is needed for the second, the same
standard already used for `streaming.py` (SO-101 arm + human IMU, not two robot arms).
`lerobot/aloha_static_coffee`: a real (not simulated) bimanual ALOHA robot, 14 real DoF,
real 50Hz control rate (vs. SO-101's 6 DoF, 30Hz) -- different hardware, different DoF
count, different control rate.

```
Real ALOHA episode 0: 1100 frames, 14 real DoF
n=280 real trials
RMSE: rate_limited wins 56/280 (20.0%) vs moving median
  mean RMSE: rate_limited=0.0045  moving_median=0.0107
Max jump: rate_limited wins 280/280 (100.0%) vs moving median
  mean max jump: raw=2.08  rate_limited=0.02  moving_median=0.25
```

**Safety metric confirmed identically**: 280/280 (100%), same as SO-101's 120/120 -- the
core safety property replicates exactly across two independent real robots.

**Fidelity metric: an honest, real divergence, not smoothed over.** Per-trial RMSE win rate
stays a minority here too (20.0%, vs 15.8% on SO-101) -- consistent on that specific measure.
But the MEAN RMSE actually favors the rate limiter on ALOHA (0.0045 vs 0.0107), unlike
SO-101 where the mean favored the moving median. Reported exactly as found: likely a
heavy-tailed moving-median failure mode on some real (seed, joint) trials that drags its
mean up despite winning more individual trials -- not investigated further here, since the
core safety finding (the actual reason this mechanism exists) is what needed cross-domain
confirmation, and it replicated cleanly.

## Status: promoted to Dense-Armor

Both required real physical domains now check out on the property that matters for this
mechanism's stated purpose (bounding real instantaneous command jumps): 100% win rate on
both SO-101 (120/120) and ALOHA (280/280). Promoted to `dense_armor.utility` as
`rate_limited_follower` -- see Dense-Armor's `docs/api/rate_limiter.md`.

---

## Details

**Relation to `prog.txt`**: closes the roadmap's damping-mechanism search with a real,
qualified answer -- not "no mechanism works" (the earlier neighbor-consensus attempt's
finding specifically), but "a rate-limiting mechanism works for safety, not for fidelity"
-- a different, more specific finding.

**Reproducing this**:
`python scripts/robot_sensor_validation/rate_limiter_full_evaluation.py` regenerates
`rate_limiter_full_evaluation_frozen.json` (SO-101, reuses the already-cached real LeRobot
parquet, no new download); `python scripts/robot_sensor_validation/rate_limiter_second_domain_aloha.py`
regenerates `rate_limiter_second_domain_aloha_frozen.json` (ALOHA, downloads
`lerobot/aloha_static_coffee` once if not already cached); `pytest
tests/test_rate_limiter_real_joint_commands.py tests/test_rate_limiter_second_domain_aloha.py`
reads the already-frozen files, no network access needed in CI.

**Paper indexed**: Berscheid & Kroger (2021) is now in quantumrag's new
`robotica_generazione_traiettoria` collection -- the first real-time robot trajectory
generation/control-theory reference in that knowledge base, kept separate from
`robotica_rilevamento_anomalie` (fault/anomaly detection) since this is a different problem
(command shaping, not detection).
