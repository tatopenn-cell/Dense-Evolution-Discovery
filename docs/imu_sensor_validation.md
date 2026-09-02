# Dense-Armor on Real IMU Sensor Data (UCI HAR)

Dense-Armor's runtime detectors had only ever been validated on quantum-chemistry
simulation output (real physics, but not sensor hardware) and LLM agent latency (real
software, but not physical). This experiment is the first test on a real physical
sensor -- an actual accelerometer, not a simulation.

## Step 1. Real accelerometer data, not synthetic noise

```python
import numpy as np

x = np.loadtxt("scripts/robot_sensor_validation/data/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt")
x.shape
```

```
(7352, 128)
```

[UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(Anguita et al. 2013) recorded a real Samsung Galaxy S II's accelerometer and gyroscope,
worn on the waist by 30 real volunteers, at a real fixed 50 Hz, while they performed six
real activities. The file above is one 3-axis channel (`x`), pre-split into 7352
overlapping 128-sample windows (2.56s each, 50% overlap) -- the dataset's own packaging,
not something built here.

## Step 2. Reconstructing one continuous signal -- two real mistakes on the way

```python
subj = np.loadtxt(".../subject_train.txt").astype(int)
y = np.loadtxt(".../y_train.txt").astype(int)

mask = (subj == 1) & (y == 4)
np.where(mask)[0]
```

```
array([ 27, ..., 50, 202, ..., 212, 225, ..., 236])
```

The first attempt at rebuilding a continuous signal assumed every window sharing a
(subject, activity) pair was one recording. This output disproves that directly: subject
1's `STANDING` windows are **three separate index runs**, not one -- the volunteer stood
still at three different points in the session, not once continuously. Concatenating
them naively would have stitched three unrelated moments together, producing fake jumps
indistinguishable from real drift.

Fixed by requiring genuinely *contiguous* raw-file indices. That fix still wasn't enough:

```python
xa = np.loadtxt(".../total_acc_x_train.txt")
w = xa[4010:4057]  # 47 contiguous-index windows, all labeled SITTING
diffs = [np.max(np.abs(w[i][64:] - w[i+1][:64])) for i in range(len(w)-1)]
max(diffs), diffs.index(max(diffs))
```

```
(0.08109379999999988, 22)
```

Consecutive overlapping windows should share their overlapping samples **exactly** (this
is literal duplicated raw sensor data, not a statistic) -- every boundary here does,
except one. That single non-zero jump is a real recording seam sitting inside what
looked like one clean run: the dataset's own collection process paused and resumed
partway through, even though the index numbering stayed contiguous. A systematic search
of every boundary in the dataset further found that **no run ever crosses an
activity-label change without also breaking this exact-overlap identity** -- in this
dataset, changing activity always means a real recording pause, confirmed directly, not
assumed.

## Step 3. Four real conditions on a verified-clean segment

```python
from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

standing = ...  # subject 17, indices 3237-3265, verified exact overlap at every boundary
labels, _, _ = classify_segments(standing, radius=25, ref_mult=2)
flagged, _ = cusum_detector(standing, radius=25, ref_mult=2)
float((labels != "clean").mean() | flagged.mean())
```

```
0.0519
```

Using a genuinely clean 38.4-second real recording (subject 17 standing still), the
combined detector's false-positive rate is **5.2%** -- higher than the ~1.3% seen on
synthetic gaussian noise. A person standing "still" has real postural micro-sway; it
isn't gaussian.

```python
# B: a 2-sample IMU glitch injected on top of the SAME real recording
# C: a sustained +3.0 m/s^2 offset from a fixed point onward (bias/calibration drift)
# D: the SAME real STANDING recording, followed by a SEPARATE real WALKING
#    recording from the same subject -- a genuine behavioral transition
```

```
B: 100% of the injected region flagged
C: 50% flagged within 1s of the transition
D: 64% of the first second of real walking flagged
```

`B` and `C` reproduce the same pattern already seen on synthetic data and real LLM agent
telemetry: transient glitches are caught essentially every time, sustained shifts only
partially. `D` is the interesting real difference -- on LLM latency, a legitimate
behavior change was rarely over-flagged; here, a real `STANDING -> WALKING` transition
gets flagged **most of the time**, because gait is a large, genuinely oscillatory change
in acceleration magnitude, not a subtle one. Whether that is a false alarm or a useful
"motion started" signal depends entirely on what the monitor is *for* -- this experiment
only measures that it happens, not whether it's good or bad for a given use case.

---

## Details

**Why this dataset**: chosen for being real, public, small enough to download in one
session (61MB), and the closest practical stand-in for real robot IMU telemetry
(accelerometer + gyroscope, the same sensing modality) -- a person wore the sensor, not a
robot, and this is not a substitute for validating on genuine robot proprioceptive data.
A real lidar validation was considered and set aside: public lidar datasets are
GB-scale and mostly gated behind registration, not practical to pull into a single
session.

**Signal used**: total acceleration magnitude, `sqrt(x^2+y^2+z^2)`, from the raw
`total_acc_{x,y,z}` channels (accelerometer including gravity) -- one scalar channel,
the same shape as the LLM-latency signal in
[Experiment 40](agent_indirect_prompt_injection.md)/Benchmark v2, now a real physical
quantity from real hardware.

**Two-sided, not one-sided**: [Experiment 40](agent_indirect_prompt_injection.md)'s
sibling work found a one-sided upper-tail filter cut false positives sharply on LLM
latency, where only an increase is ever meaningful. That reasoning does not transfer
here -- an unusually *low* acceleration reading is itself a real fault signature for this
sensor (a frozen/flatlined sensor, or genuine free-fall), so this experiment deliberately
keeps `classify_segments`/`cusum_detector` two-sided.

**Parameters**: `radius=25, ref_mult=2` (a 1.0s causal window at the dataset's real 50Hz
sampling rate) and `cusum`'s `k=0.5, h=5.0` (unchanged library defaults) were declared
before this was run and never adjusted afterward.

**Reproducing this**: `python scripts/robot_sensor_validation/run_imu_validation.py`
re-downloads UCI HAR (61MB) into a gitignored `data/` folder and regenerates
`imu_validation_frozen.json`; `pytest tests/test_imu_sensor_validation.py` reads the
already-frozen file, no download needed.
