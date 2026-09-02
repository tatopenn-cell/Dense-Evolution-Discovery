# Dense-Armor on Real Teleoperated-Robot-Arm Data (LeRobot)

Experiments 41 and 42 validated Dense-Armor's runtime detectors on a real accelerometer
and a real lidar. This experiment looks for an actual, currently-open need rather than a
hypothetical one -- and finds a real, complementary gap in a real, active open-source
robotics project, not a clean textbook win.

## Step 1. A real need, found by reading real GitHub issues, not guessed

[`huggingface/lerobot`](https://github.com/huggingface/lerobot) (27k stars) has two real,
open feature requests from the same contributor -- [#3758](https://github.com/huggingface/lerobot/issues/3758)
("`lerobot-check-calibration`": detect a systematic offset between a teleoperation
leader arm and its follower) and [#3760](https://github.com/huggingface/lerobot/issues/3760)
("`lerobot-dataset-quality`": flag outlier episodes via per-episode IQR). Both already
have a working PR in flight ([#3762](https://github.com/huggingface/lerobot/pull/3762))
from the same author -- so this experiment does **not** propose to solve either literal
request; that would be redundant. Both proposed tools compute **one static aggregate
number** (a dataset-wide mean/std, or a per-episode outlier score). The real question:
is there structure neither approach can see?

## Step 2. Two hypotheses, tested directly, both wrong

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("lerobot/svla_so101_pickplace", repo_type="dataset",
                        filename="data/chunk-000/file-000.parquet")
df = pd.read_parquet(path)
df.columns.tolist(), df.episode_index.nunique()
```

```
(['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'], 50)
```

Real SO-101 arm teleoperation data (the exact hardware #3758's author reports a ~17°
calibration offset on) -- 50 real pick-and-place episodes, leader position (`action`)
and follower position (`observation.state`) per frame, 6 joints, ~30Hz.

**Hypothesis 1 -- a slow within-episode drift**: checked directly, not assumed.
Correlation between `|action - observation.state|` and frame index, averaged across all
50 episodes and 6 joints: **0.07**. Not real -- episodes last 8-15 seconds, too short for
a slow drift to show.

**Hypothesis 2 -- flag transient spikes in the raw offset**: episode 0's offset does spike
(from ~0.6 to ~23), but checking the leader's own velocity at those exact frames shows the
spike is fully explained by ordinary control-loop lag during fast leader motion, not a
fault. Applied naively, Dense-Armor would false-positive on every fast motion in a
pick-and-place task. #3758's author had already anticipated this exact trap -- their
proposal explicitly restricts to "stable frames" (velocity below a threshold) for exactly
this reason.

## Step 3. What survives: pose-dependent calibration offset

```python
import numpy as np

VEL_THRESHOLD = 1.0  # matches #3758's own stable-frame filter

sub = df[df.episode_index == 0].sort_values("frame_index")
action = np.stack(sub["action"].values)
state = np.stack(sub["observation.state"].values)
diff = action - state
vel = np.abs(np.diff(action, axis=0, prepend=action[:1]))
stable = np.all(vel < VEL_THRESHOLD, axis=1)

x = diff[stable, 2]  # joint 2 -- found to have the largest real spread of all 6 joints
np.round(x[::15], 2)
```

```
array([ 0.62,  0.62,  0.53,  0.53, -1.53, -3.56,  1.78,  0.62])
```

Restricted to the SAME stable-frame filter #3758 already proposes, joint 2's offset is
**not** one constant value within this single episode -- it visits several distinct real
plateaus (~0.6 → ~-1.5 → ~-3.8 → back to ~0.6) as the arm moves through real task poses
(reach, grasp, lift, place). This is consistent with a real, documented phenomenon in
robot manipulators: joint calibration error is configuration-dependent, not a single
constant offset (Lu, He, Julius & Wen, "Configuration-Dependent Robot Kinematics Model
and Calibration," [arXiv:2510.19962](https://arxiv.org/abs/2510.19962) -- a single
fixed-offset calibration is shown to be structurally insufficient, and pose-dependent
calibration cuts positioning error by over 50% on 6-DoF arms). A single dataset-wide mean
(#3758) or per-episode outlier score (#3760) cannot distinguish "this joint tracks worse
at this specific pose" from "the arm generally drifted" -- both pool exactly the
information this experiment finds structure in.

## Step 4. Does the detector actually catch the real transitions?

```python
from dense_armor.utility.arbiter import classify_segments

labels, deviation, uncertainty = classify_segments(x, radius=5, ref_mult=2, n_sigmas=3.0, spike_run_max=2)
# run-length summary: (label, start, end, median)
```

```
('clean',  0, 26,  0.62)
('regime', 27, 31,  0.53)
('clean',  32, 89,  0.53)
('spike',  90, 95, -3.84)
('clean',  96, 109, -1.48)
...
('regime', 128, 132, -4.55)
('clean',  133, 140, -3.56)
('regime', 141, 145,  1.79)
('clean',  146, 168, 0.62)
```

Yes -- every real transition boundary gets flagged as something (`spike` or `regime`),
never silently smoothed over. But this is a **real, honest, mixed result, not a clean
win**: the label choice is not always the intuitive one. The transition into the -3.8
plateau (frames 90-95) is labeled `spike`, not `regime`, even though the arm genuinely
settles there for a while -- because at 30Hz, that transition lasts only 6 frames before
`spike_run_max=2`'s run-length rule and the causal window's own adaptation kick in. A
second, more extreme episode (22) shows the same pattern: a real, 70-frame sustained
plateau at ~-3.8 is correctly left `clean` once the detector adapts (consistent with
every prior experiment's causal-window behavior), but getting there involves several
short `spike`-labeled runs rather than one clean `regime` label.

---

## Details

**Tried the cheap fix first, it made things worse**: before considering any new code, the
obvious cheap fix for the spike-vs-regime labeling above was tried directly --
`spike_run_max` (the run-length threshold separating the two labels) at 2, 5, 8, and 12.
Raising it did **not** help; it made every borderline transition MORE likely to be
labeled `spike`, not less (a run only becomes `regime` once its length exceeds
`spike_run_max`, so raising the threshold moves the boundary the wrong way for these
already-short, ~5-6-frame real transitions). No tested value produced better regime
labeling than the library default (2) already gives. This is a genuine, honestly
documented limitation of `classify_segments` for very short-duration real transitions at
this sample rate -- not a bug to hide, and not (yet) grounds for new library code: per
this project's own established discipline, a fix only gets promoted after proving itself
on more than one real case, and no working fix was found here at all.

**Why joint 2**: of the arm's 6 joints, joint 2's per-episode stable-frame median offset
has by far the largest spread across the 50 episodes (std 0.93) versus the other 5
(std 0.10-0.20) -- checked directly before focusing the write-up here, not chosen to make
the story look best.

**What a static approach reports for comparison**: pooling ALL 50 episodes' stable
frames into one number (what #3758's own proposed tool would compute) gives mean=0.115,
std=0.932 for joint 2 -- an unremarkable-looking number that hides the real ~4-5-unit
within-episode swings this experiment found.

**Honest scope**: this is not a finished contribution. It demonstrates that real,
non-redundant structure exists (verified directly, twice-wrong-before-getting-it-right,
not assumed) and that Dense-Armor's causal detector catches the real transition points --
but the spike/regime labeling precision at 30Hz with `spike_run_max=2` needs more work
before this would be a good PR to propose upstream. Not pursued further in this
experiment; see the maintainer's own notes for the decision on next steps.

**Reproducing this**: `python scripts/robot_sensor_validation/run_lerobot_calibration_regime_analysis.py`
re-downloads the dataset (~370KB, gitignored `lerobot_data/`) and regenerates
`lerobot_calibration_regime_frozen.json`; `pytest tests/test_lerobot_calibration_regime.py`
reads the already-frozen file, no download needed.

**The velocity-gated stable-frame filter is now a shared, tested helper**
(`scripts/robot_sensor_validation/stable_frame_filter.py`,
`velocity_gated_stable_mask`) -- extracted here so a future experiment can reuse it
without copy-pasting, but deliberately kept in Discovery, not promoted to `dense-armor`'s
own `utility/` package yet: this project's own rule (see how `one_sided_upper_filter` was
promoted only after proving itself, with a real ablation, on two independent real
datasets) is that a filter earns library-code status after a SECOND real case shows it
still helps, not from one experiment alone. Re-running the analysis after this extraction
reproduces byte-identical output to the version committed with the original experiment --
verified directly, not assumed.

**Update: a second real case found a real gap in the function, then confirmed it after
fixing it.** Applied naively to real IMU data (Experiment 41's UCI HAR dataset, gating
accelerometer-magnitude analysis by gyroscope magnitude), the original function was
WRONG -- it always differentiated the reference signal, correct for LeRobot's
position-like leader command, but wrong for a gyroscope, which is already a rate. Fixed
with an explicit `already_rate` parameter (`False` for position-like references,
`True` to threshold an already-rate-like reference directly) rather than silently
patched -- see `stable_frame_filter.py`'s own docstring for the full account. Re-validated
both real cases through the corrected, unified function:
- LeRobot (`already_rate=False`): still byte-identical to the original committed result.
- Real UCI HAR WALKING segment (`already_rate=True`, subject 17, gyroscope magnitude
  gating accelerometer magnitude): a real, modest, honestly-reported effect --
  accelerometer-magnitude std drops from 0.208 (full signal) to 0.169 (gated-stable
  subset, -19%), physically sensible (less real device rotation, less incidental
  translational-acceleration variance) but not dramatic. See
  `validate_stable_frame_filter_second_case.py` for the frozen, reproducible numbers.

Two independent real physical domains now correctly handled by one function -- the bar
this project sets before considering a Discovery helper for promotion to `dense-armor`.

**Update: promoted.** `velocity_gated_stable_mask` shipped in `dense-armor` v1.1.13
(`dense_armor.utility.stable_frame_filter`) after clearing that bar. The spike/regime
labeling gap below is unrelated to this filter and remains open.

## Addendum: seven candidate fixes for the spike/regime labeling gap, all tested on real
## data and rejected

Step 4 above left one real, disclosed gap: a genuine, sustained pose transition lasting
only 5-6 frames at 30Hz sometimes gets labeled `spike` instead of `regime`. Seven
candidate fixes were tried directly against the real data already collected (episodes 0
and 22, then the full 50-episode dataset) -- none worked. Reported honestly rather than
silently abandoned, per this project's own discipline.

**1. Raising `spike_run_max`** (already documented above in Details) -- tested at 2, 5, 8,
12; made labeling worse, not better, because raising the run-length threshold moves the
spike/regime boundary the wrong way for transitions that are already short.

**2. Net-displacement (does the signal return to its pre-event value, or settle at a new
one?)** -- the idea: compare the `clean` run's median right before an event to the
`clean` run's median right after it. A real, sustained transition should leave a lasting
net displacement; noise that just perturbs and reverts should not.

- On the two hand-inspected episodes, a fixed absolute threshold (~0.3) cleanly recovered
  both previously-flagged mislabeled `spike` runs (net displacement 2.01 and 4.43) without
  breaking any of the five correctly-labeled short spikes (net displacement all <0.11).
- Tested against all 50 episodes (207 non-`clean` runs, not just the 2 inspected by
  eye), the same fixed threshold did **not** generalize: 66/101 (65%) of `spike`-labeled
  runs exceeded it, which would mean two-thirds of all detected spikes are actually
  mislabeled real transitions -- implausible, and a clear sign of overfitting to 2
  examples.
- Normalizing by the local MAD scale (the same robust scale `classify_segments` computes
  internally, via `_robust_center_scale`) did not fix this: `spike` runs' normalized net
  displacement has median z=2.72 (n=66) against `regime` runs' median z=15.83 (n=54), but
  the distributions overlap too much for a workable threshold -- even at z=3.0 (the same
  `n_sigmas` the arbiter itself uses), 48% of spikes would still flip.
- Honest interpretation: in a continuously-moving arm signal at 30Hz, a `clean` run
  adjacent to a `spike` is often still mid-motion, not a true resting plateau -- so
  "returns to baseline" is not a reliable signature here.

**3. Kinematic feasibility (a Lagrangian-inspired, dynamics-free check)** -- the idea:
use only the actuator's *kinematic* upper limits (no mass, inertia, or torque model
needed) to test whether the follower's implied motion during an event is physically
achievable, or whether it looks like a sensor/encoder artifact. The Feetech STS3215
datasheet publishes a speed rating but not a fixed acceleration limit (acceleration on
these bus servos is a software-configured trapezoidal-profile register, not a hardware
constant, and the specific driver configuration used to record this dataset is unknown)
-- so a datasheet number would have been fabricated. Used instead a real, conservative
empirical ceiling: the largest `observation.state` acceleration this exact system ever
produced across all 50 episodes' full (not just stable-filtered) frames, joint 2:
2662.5 units/s².

Checked whether any `spike`- or `regime`-labeled event's implied follower acceleration
exceeds that ceiling: **0/101 spikes and 0/106 regimes do** (spike median 484.1,
regime median 403.4, both comfortably under the ceiling). This is not a failed test --
it is a real, clarifying negative result: every labeled event, `spike` or `regime`
alike, is physically genuine servo motion, not an artifact. The spike/regime ambiguity
was never a real-vs-fake-data problem, so a physical-plausibility filter has nothing to
discriminate on here -- it rules out sensor noise as the cause and confirms the actual
open question is purely one of duration classification, which is exactly what
`spike_run_max` already (unsuccessfully, per fix #1) tries to solve.

**4. Phase-space curvature (Menger curvature via Pythagorean triangle sides)** -- the
idea: embed the signal as `(diff[t], d(diff)/dt)` and compute, for every three
consecutive points, the curvature of the triangle they form (`κ = 4*Area /
(side1*side2*side3)`, sides via Euclidean/Pythagorean distance). Noise that loops back to
its starting value has to turn sharply somewhere (high curvature); a real transition
moving toward a new state should be smoother (low curvature).

- On the two hand-inspected episodes, mixed: episode 22 supported the hypothesis clearly
  (spike median curvature 0.98 vs regime median 0.00), episode 0 did not (spike median
  0.30 vs regime median 0.43, the wrong direction) -- the same overfitting warning sign as
  fix #2.
- Aggregated per run (mean/max/median curvature within each run) across all 50 episodes,
  every statistic overlapped heavily between `spike` and `regime`: e.g. run-mean median
  1.056 (spike) vs 0.920 (regime), run-max median 2.106 (spike) vs 2.292 (regime) --
  no usable separation.

**5. Distributional divergence (Jensen-Shannon, dynamic binning)** -- the idea: compare
the empirical distributions of the `clean` run right before an event and right after it;
a real regime change should look like two different distributions, noise should not.
Implemented with dynamically-sized histogram bins (adapted to each window's sample count)
and `scipy.spatial.distance.jensenshannon`.

- Across all 50 episodes (149 runs with usable neighbor windows), `spike` median JSD =
  0.871, `regime` median JSD = 1.000 (saturated at the maximum) -- heavy overlap; at every
  threshold tried (0.3 to 0.9), 46-92% of spikes still exceeded it.
- Root cause: with very few samples per window (often <10, sometimes 1), even a small
  positional difference saturates JSD to its maximum almost every time -- the same
  underlying problem as fix #2, dressed differently.

**6. Velocity sign-changes (a sample-efficient stand-in for "resonance")** -- true
frequency-domain analysis (FFT, resonant-mode detection) needs many oscillation cycles to
resolve a frequency; these events are 5-6 samples long, nowhere near enough -- computing
one anyway would have reported noise dressed as physics, so it was not attempted. Instead
counted zero-crossings of the discrete velocity `d(diff)/dt` within each event, as a
sample-efficient proxy for "oscillates and reverts" (spike) vs "moves monotonically to a
new state" (regime).

- Across all 50 episodes: `spike` runs had a *higher* median sign-change count than
  `regime` runs (1.00 vs 0.00), which is the expected direction, but the overlap was
  total -- at threshold "0 sign changes", 46% of spikes and 58% of regimes both qualify;
  at threshold "<=1", 80% vs 84%. No usable separation.

**7. `pressure_valve` (an existing, more mature library detector, not a from-scratch
attempt)** -- before writing any more one-off code, checked whether `dense_armor.utility`
already had something better. It does: `robust_filters.py`'s `pressure_valve` combines
four classical robust estimators (Chauvenet, Tukey, Hampel, sigma-clipping) via a
minimum-variance (BLUE) weighting, with an adaptive threshold that widens near genuine
distribution shifts via the module's own `_jensen_shannon` -- which, unlike the
hand-rolled version in fix #5, already uses adaptive bin counts *and* Laplace smoothing
specifically to avoid the saturation problem fix #5 ran into (documented in its own
docstring, verified independently before this addendum was written).

Applied directly to the same real event windows (episodes 0 and 22): it does not solve
this problem either, but for a different, clarifying reason -- it answers a different
question than the one asked here. `pressure_valve` measures how much a point deviates
from its local neighborhood, not whether that deviation persists afterward. The clearest
counter-example: episode 22's run[70:74], a run fix #2 had already confirmed to be a pure
transient (net displacement 0.109, returns almost exactly to its pre-event value), got
the *highest* `pressione` (12.58) of every event checked -- higher than every confirmed
real `regime` transition (1.69-3.18). Duration/persistence and instantaneous deviation
magnitude are simply not the same signal.

**Where this leaves the experiment, in plain terms**: the signal being analyzed (one
joint's leader-follower offset, 30 samples/second) sometimes jumps to a new level and
*stays* there (a real, lasting pose-dependent calibration change), and sometimes jumps and
*reverts* within the same handful of samples (a real but brief motor movement, not
sensor noise -- fix #3 already ruled that out). Both can last exactly 5-6 samples, so
duration alone cannot tell them apart, and unambiguously telling them apart requires
seeing what happens *after* the event -- which is exactly the information a short window
doesn't have yet. Seven independent, real-data-tested approaches to finding some other
early signature in the shape of this one signal (run-length, endpoint displacement,
phase-space curvature, distributional divergence -- twice, oscillation pattern,
neighborhood-deviation magnitude) all failed to generalize past the 2 examples they were
first tried on. That is not a proof that no signature exists, but seven independent
failures on the same question is real evidence that whatever would solve this is
probably not "a cleverer function of this same signal" -- it would more likely need
substantially more real episodes (50 was not enough to validate even the fixes that
looked clean on 2 examples), or genuine external information this dataset does not
provide (e.g., ground-truth timestamps for when the human operator's intent actually
changed). Neither has been attempted. Not pursued further here.

Reproducing the numbers above: `scripts/robot_sensor_validation/lerobot_calibration_regime_frozen.json`
has the frozen per-run labels and medians used for the net-displacement analysis; the
kinematic-feasibility, curvature, JSD, sign-change, and `pressure_valve` checks all read
`observation.state`/the stable-frame diff directly from the same cached parquet
(`scripts/robot_sensor_validation/lerobot_data/`, gitignored) and are not frozen to a
JSON file since they recompute cheaply from data already on disk.
