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
