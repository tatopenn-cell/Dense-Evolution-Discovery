# Deadband/Backlash Gating for the LeRobot Spike/Regime Problem

Experiment 43's own addendum closed with seven independent, real-data-tested candidate
fixes for the spike/regime labeling gap, all rejected -- and a specific conclusion: the
signal being analyzed (one joint's leader-follower offset) probably doesn't contain the
information needed to solve this with a cleverer function of itself alone. This experiment
tests a genuinely different, physically-grounded idea instead of another function of the
same signal: real mechanical backlash/deadband (a well-documented phenomenon in geared
robot joints -- Lima, Machado & Crisóstomo, "Experimental backlash study in mechanical
manipulators", *Robotica* 29(2):211-219, 2011, DOI:10.1017/S0263574710000056, checked via
WebFetch before citing) as a confound that could be masquerading as some of the mislabeled
spikes.

## Step 1. Does the deadband signature exist in this real data?

Real robot joints briefly resist motion when the commanded direction reverses, until
static friction is overcome -- not the paper's own pseudo-phase-plane+wavelet method, a
much simpler classical signature check: real velocity right after a commanded reversal,
against the joint's own baseline.

```python
sign_change = np.sign(cmd_vel[1:]) != np.sign(cmd_vel[:-1])
post_rev_actual_vel = np.abs(act_vel[idx+1])  # right after each reversal
post_rev_actual_vel.mean() / np.abs(act_vel).mean()  # ratio vs. baseline
```

```
joint 0: ratio=0.313   joint 1: ratio=0.393   joint 2: ratio=0.157
joint 3: ratio=0.680   joint 4: ratio=0.347   joint 5: ratio=2.313
```

Real signature, on real data: 5 of 6 joints show real velocity dropping to 16-68% of
baseline immediately after a commanded reversal. Joint 5 (the gripper -- an open/close
mechanism, not a geared rotary joint) shows the opposite pattern, physically sensible, not
discarded as noise.

## Step 2. Do the flagged spikes correlate with deadband, across the whole dataset?

`deadband_gate.py`'s `deadband_mask` flags frames plausibly inside a real deadband event.
Checked whether `classify_segments`' `spike`-labeled points on Experiment 43's own signal
(stable-frame joint-2 offset) are enriched inside those windows, across all 50 real
episodes, not just the 2 hand-inspected ones:

```
50 episodes: total_points=4871, total_spike=394
spike points in deadband: 239 (60.7% of all spikes)
deadband base rate (all points): 34.6%
enrichment ratio: 1.75x
```

A real, generalized, 1.75x enrichment -- the first result in this whole line of
investigation (Experiment 43's seven rejected fixes, plus the cross-channel-correlation
experiment above) that holds up past the two hand-inspected episodes on first try.

## Step 3. Does gating deadband points out actually fix the known mislabeled cases?

Experiment 43's addendum had already identified two specific `spike`-labeled runs
confirmed (via net-displacement) to be real, sustained transitions: episode 0's
`run[90:95]` and episode 22's `run[24:29]`. Re-ran `classify_segments` with deadband
points removed from the signal first:

```
--- episode 22: 50/122 points gated out ---
  {'label': 'regime', 'start': 24, 'end': 28, 'median': -4.724}   <- WAS 'spike', NOW CORRECT
--- episode 0: 38/169 points gated out ---
  {'label': 'spike', 'start': 90, 'end': 94, 'median': -4.244}    <- STILL 'spike', unresolved
```

**1 of 2 known cases fixed** -- the first intervention in this entire investigation that
moved a known-mislabeled case in the right direction at all. Not smoothed over: the other
known case stayed wrong.

## Step 4. Does it generalize, or was this two more hand-picked episodes?

Reused the same net-displacement ground-truth check from Experiment 43's own addendum
(does a run's label match whether the signal settles at a new level or returns to
baseline), across all 50 episodes, before vs. after deadband gating:

```
BEFORE gating: 135/207 runs match net-displacement ground truth (65.2%)
AFTER  gating: 57/92 runs match net-displacement ground truth (62.0%)
```

**A small net regression** (65.2% -> 62.0%), the same pattern every other candidate fix in
Experiment 43 showed: real improvement on the 1-2 inspected cases, no improvement (here,
slightly negative) once checked against the full dataset. Gating also structurally reduces
the number of runs with valid clean neighbors on both sides (207 -> 92), which is itself
part of why the aggregate comparison is noisier here.

## Honest conclusion

The deadband/backlash confound is real, physically grounded (real citation, real
signature verified directly in this data), and genuinely explains a disproportionate share
of the spikes `classify_segments` flags (1.75x enrichment, generalized across all 50
episodes) -- a real, distinct finding from Experiment 43's own spike/regime duration
problem, not a restatement of it. But using it as a *fix* for that specific problem does
not work: it corrects one of the two known mislabeled cases while leaving the other wrong,
and makes the aggregate metric slightly worse, not better. The two problems (deadband
noise vs. genuine short-duration real transitions) coexist in the same signal but are not
the same problem, and fixing awareness of one does not fix the other. Not pursued further
as a fix for Experiment 43's open gap; the deadband/spike association itself is worth
keeping as a separate, real, disclosed finding.

## Reproducing this

`scripts/robot_sensor_validation/deadband_gate.py` (`deadband_mask`) -- reuses the same
cached `lerobot/svla_so101_pickplace` dataset as Experiments 43 and 46, no new download.
Real literature grounding this page cites is indexed in quantumrag's
`robotica_rilevamento_anomalie` collection alongside Experiment 46's paper.
