# Cross-Channel Correlation for Robot Joint Fault Detection

A real, published result -- Anonymous, "Unsupervised Anomaly Detection for Autonomous
Robots via Mahalanobis SVDD with Audio-IMU Fusion" (arXiv:2505.05811, checked directly
via WebFetch before citing, indexed in quantumrag's `robotica_rilevamento_anomalie`
collection) -- shows that a real robot fault (collision, mechanical failure) shows up as a
breakdown in the normal correlation between sensor channels that usually co-vary,
validated on a real mobile robot (2h normal operation, F1=92.3%). Their mechanism is deep
learning (Mahalanobis SVDD + cross-attention fusion). This experiment tests the same
insight with a much lighter, classical mechanism -- consistent with Dense-Armor's
no-retraining philosophy -- on real SO-101 arm data, and reports an honest negative
result.

## Step 1. The detector

```python
from cross_channel_correlation import cross_channel_correlation_detector
```

For each point, compute every channel's mean pairwise Pearson correlation with every
OTHER channel over a causal reference window (same `radius*ref_mult` convention as
`arbiter.py`/`cusum.py`). Flag a channel when its own correlation drops more than
`drop_threshold` below the window's mean cross-channel correlation -- it decorrelated from
the group, not just got noisier overall.

## Step 2. False-alarm rate on real, unmodified data

```python
import numpy as np, pandas as pd
# lerobot/svla_so101_pickplace, 50 episodes, 6 real joints, already cached from Experiment 43
```

```
False-alarm check: 50 real episodes, 11939 total frames
  points with >=1 channel flagged: 5302 (44.41%)
```

`drop_threshold=0.4` (a reasonable-looking starting value) flags **44.41%** of real,
unmodified frames. Swept up to `drop_threshold=0.9` (near the theoretical maximum -- a
near-total correlation collapse): still **8.74%**. Not a tuning gap -- a structural one:
real coordinated arm motion during a pick-and-place task naturally desynchronizes joints
across task phases (reach, grasp, lift, place each drive different joint subsets), so
"decorrelated from the group" happens constantly during entirely normal motion.

## Step 3. Fault injection -- and a real methodological trap caught before it mattered

Injected a physically meaningful fault: joint 2 frozen at its own value for 15 frames
mid-episode (a stuck servo / stale encoder reading), while the other 5 joints continue
their real recorded motion.

```
Fault injection: episode 0, joint 2 stuck at frames [100:115]
  clean episode, this window: any flag on joint 2? True
  injected episode, this window: any flag on joint 2? True
  detection rate within injected window: 1.000 (15/15 frames)
```

100% detection -- but checked directly rather than trusted: the SAME window in the
**unmodified** episode was ALSO already flagged, for the same reason as Step 2's high
false-alarm rate. The apparent "detection" was not a real signal -- it was the same
false-alarm noise the injection happened to land inside. A result this clean deserved
exactly this kind of check before being reported as a win.

## Step 4. A velocity-gated variant, reusing an already-validated number

Idea: only trust correlation computed from ACTIVE frames (real coordinated motion), since
correlation between near-constant, mostly-noise channels is itself unstable and not a
meaningful "normal co-variation" baseline. This is the same `velocity_gated_stable_mask`
idea used in reverse: there, keep only STABLE frames; here, keep only ACTIVE ones.
`vel_threshold=1.0` reuses Experiment 43's own already-validated constant, not a new
number invented for this check.

```
false_alarm_point_rate=14.35% (velocity-gated, drop_threshold=0.4)
clean_window_any_flag=False (0/15)  injected_window_detection_rate=0.000 (0/15)
```

Real improvement (44.41% -> 14.35%), and the injection-window false positive from Step 3
is gone -- but the real cost is that the injected fault is no longer detected at all
(0/15). Velocity-gating removed the noise that was masquerading as a detection, and with
it went the only detection this approach ever had.

## Honest conclusion

Two variants tested directly on real data, both rejected: instantaneous pairwise
correlation is too noisy to use as-is (44.41% false-alarm rate, and its apparent fault
detection was itself a false-alarm artifact, caught by checking the clean baseline
explicitly rather than trusting a clean-looking number); velocity-gating fixes the
false-alarm rate but removes the detector's only sensitivity along with it. The underlying
insight (arXiv:2505.05811) is real and grounded, but this project's own no-retraining,
classical-statistics implementation of it does not work on this real robot-arm case --
the paper's own deep-learning mechanism is likely doing real work a simple correlation
threshold cannot replace here. Not pursued further; see the write-up's own next-step
discussion for what was proposed instead (deadband/backlash filtering, evaluated but not
yet implemented).

## Reproducing this

`scripts/robot_sensor_validation/cross_channel_correlation.py` (the detector, with
`vel_threshold`/`min_active_frac` for the gated variant) and
`scripts/robot_sensor_validation/validate_cross_channel_correlation.py` (the validation
script -- reuses `lerobot/svla_so101_pickplace`, already cached from Experiment 43, no new
download).
