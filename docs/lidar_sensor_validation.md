# Dense-Armor on Real Lidar Sensor Data (Sydney Urban Objects)

[Experiment 41](imu_sensor_validation.md) validated Dense-Armor's runtime detectors on a
real accelerometer. This experiment does the same for a real rotating lidar -- the
sensing modality actually requested -- using a real Velodyne HDL-64E, not a simulator.

## Step 1. Real Velodyne returns, two binary formats, only one trusted

```python
import numpy as np

names = ["t", "intensity", "id", "x", "y", "z", "azimuth", "range", "pid"]
formats = ["int64", "uint8", "uint8", "float32", "float32", "float32", "float32", "float32", "int32"]
dtype = np.dtype(dict(names=names, formats=formats))

d = np.fromfile("scripts/robot_sensor_validation/lidar_data/sydney-urban-objects-dataset/objects/car.81.12346.bin", dtype)
d["t"][0], d["range"].mean(), d["intensity"].max()
```

```
(1320370833971539, 13.498847961425781, 207)
```

The [Sydney Urban Objects Dataset](https://www.acfr.usyd.edu.au/papers/data/sydney-urban-objects-dataset.tar.gz)
(Quadros, Underwood, Douillard 2013) contains 631 real objects -- cars, pedestrians,
trees, signs -- individually segmented from a real Velodyne HDL-64E mounted on a vehicle
driven through Sydney CBD on 2011-11-04. `t` above decodes to `2011-11-04 01:40:33` UTC --
matched exactly, to the microsecond, against the dataset's own separate CSV export of the
same object, before trusting this parse.

That cross-check mattered: the archive also ships a `scans/*.bin` folder in a different,
undocumented-in-detail raw Velodyne packet format ("8 byte timestamp + 1206 byte packet"
per the dataset's own README). A first attempt parsed those files with the *documented*
`objects/*.bin` layout above -- it ran without error and produced numbers, but the
decoded timestamps were impossible (values near the extremes of a 64-bit integer in the
same file). Caught only by checking the decoded value against a real calendar date, not
by any error the code itself raised. `scans/*.bin` is not used anywhere in this
experiment as a result -- reconstructing it correctly would mean implementing the real
Velodyne packet format from scratch, a separate, larger task.

## Step 2. One real signal from 631 irregularly-timed real events

```python
import glob, os

recs = []
for f in glob.glob(".../objects/*.bin"):
    obj = np.fromfile(f, dtype)
    recs.append((obj["t"].mean(), obj["range"].mean()))
recs.sort()
t = np.array([r[0] for r in recs])
gaps_s = np.diff(t) / 1e6
gaps_s.min(), np.median(gaps_s), gaps_s.max()
```

```
(0.0, 0.00147, 175.4)
```

Unlike Experiment 41's fixed-50Hz accelerometer, lidar objects arrive irregularly --
median gap 1.5ms (many objects detected in the same 360-degree sweep), up to a real 175
second gap. Verified this is one continuous real 21-minute session (01:30:15 to 01:51:23),
not several separate recording bouts the way the IMU dataset turned out to be -- no gap
here exceeds 175s. The signal used below is each object's mean range (meters), ordered by
its real mean timestamp: an irregularly-sampled real index sequence, the same treatment
Benchmark v2 gave a real LLM's per-turn latency.

## Step 3. Four real conditions

```python
from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector

ranges = np.array([r[1] for r in recs])  # 631 real mean-range values
labels, _, _ = classify_segments(ranges, radius=15, ref_mult=2)
flagged, _ = cusum_detector(ranges, radius=15, ref_mult=2)
float(((labels != "clean") | flagged).mean())
```

```
0.0466
```

**A (normal)**: 4.7% baseline false-positive rate across the real, unfiltered mix of
object classes and distances -- higher than synthetic gaussian noise's ~1.3%, similar
order of magnitude to Experiment 41's real IMU baseline (5.2%).

```python
# B: a 2-object +50m spike injected on the SAME real range sequence
# C: a sustained +10m offset from a fixed point onward (range-calibration bias drift)
# D: the SAME real sequence, split at its own largest real time gap (175s) -- no injection
```

```
B: 100% of the injected region flagged
C: 3.3% flagged within 30 objects of the transition
D: 6.7% flagged in the 30 objects after the real gap
```

`B` reproduces the by-now-consistent pattern: a sharp transient is caught essentially
every time, on every sensor modality tested so far. `C` is a real, honest negative
result, and a genuinely different one from Experiment 41's: there, a sustained shift on a
fixed-rate accelerometer was caught 50% of the time; here, only 3.3%. `D` stays quiet
(6.7%) across a genuine real 175-second pause in the session -- the detector does not
mistake "the vehicle stopped or paused" for an anomaly. Step 4 below decomposes exactly
why `C`'s number is so much lower than Experiment 41's, quantitatively rather than by
guessing.

## Step 4. Why only 3.3%? A quantitative decomposition, not a guess

A first draft of this page guessed the driver was *inter*-class range difference (a car,
a pedestrian, and a tree naturally sit at different distances). Checked directly instead
of assumed:

```python
uniq = np.unique(classes)
class_means = {c: ranges[classes == c].mean() for c in uniq}
inter_class_var = sum(counts[c] * (class_means[c] - ranges.mean())**2 for c in uniq) / n
intra_class_var = sum(counts[c] * ranges[classes == c].var() for c in uniq) / n  # weighted average
inter_class_var, intra_class_var, inter_class_var / ranges.var()
```

```
(7.86, 33.17, 0.192)
```

Wrong guess: **81% of the real variance is *within* a class** (a pedestrian alone ranges
from close to far, real std ~4.7m), only 19% comes from *between* classes. The dominant
real driver is a single class's own natural spread, not which classes get mixed
together.

The direct, mechanical explanation is simpler and more useful than either guess -- what
does the detector's own causal reference window actually see at the injection point?

```python
window = ranges[PERSISTENT_FROM - 30 : PERSISTENT_FROM]
med, mad = np.median(window), np.median(np.abs(window - med)) * 1.4826
PERSISTENT_OFFSET / mad
```

```
1.29
```

**The injected +10m offset sits at 1.29 sigma of the real local noise -- structurally
below the `n_sigmas=3.0` detection threshold.** That gap alone accounts for the 3.3%
result: most individual points in the shifted region simply never cross the same
threshold that a 100%-caught transient clears easily. This is the single most direct
explanation for `C`'s number, more so than any class-composition argument.

An external review (saved in the maintainer's own notes) proposed a specific, testable
fix before this was run: normalize each object's range against its own class's real
median first (treat class as an "operating regime"), rather than inventing new detector
math. Tested directly, not assumed:

```python
class_medians = {c: np.median(ranges[classes == c]) for c in uniq}
normalized = ranges - np.array([class_medians[c] for c in classes])
# same injection, same detector, same threshold, on the normalized signal
```

| variant | local MAD | offset/MAD (sigma) | detect rate (30 objects) |
|---|---:|---:|---:|
| raw range | 7.72m | 1.29 | 3.3% |
| class-normalized range | 5.88m | 1.70 | 6.7% |
| Experiment 41 IMU (for scale) | 0.0018g | 1624.1 | 50.0% |

Class normalization **helps but does not close the gap**: local noise drops, detection
rate doubles -- still nowhere close to IMU's 50%, because most of the real noise is
intra-class (a real property of the driving scene), not an artifact of comparing
different classes on one shared scale. The honest conclusion: no new detector math is
needed (the external review's instinct was right about that), but the real bottleneck
isn't primarily a normalization problem either -- an irregularly-sampled, per-object real
lidar signal has intrinsically worse signal-to-noise for a fixed-magnitude drift than a
fixed-rate physical channel like an accelerometer, independent of which feature
engineering is applied on top.

Reproducing this: `python scripts/robot_sensor_validation/analyze_lidar_persistent_gap.py`
(reuses `run_lidar_validation.py`'s dataset loading, no separate download);
`pytest tests/test_lidar_persistent_gap_analysis.py` reads the frozen result.

---

## Details

**Why this dataset, not KITTI/nuScenes**: those are the standard public lidar
benchmarks, but they are GB-scale and mostly gated behind a registration wall, not
practical to pull into a single working session. Sydney Urban Objects is real, public,
no login required, and small (79.5MB) -- the tradeoff is individually segmented objects
rather than full continuous 360-degree sweeps.

**Two-sided, not one-sided**: same reasoning as [Experiment 41](imu_sensor_validation.md)
-- for lidar range, an unusually *low* reading (a false near-return, an occlusion
artifact) is just as real a fault signature as an unusually high one (signal loss,
specular reflection), so `classify_segments`/`cusum_detector` are kept two-sided here too.

**Parameters**: `radius=15, ref_mult=2` (a 30-object causal window, sized for the real
~631-object sequence length) and `cusum`'s `k=0.5, h=5.0` (unchanged library defaults)
were declared before this was run and never adjusted afterward.

**Reproducing this**: `python scripts/robot_sensor_validation/run_lidar_validation.py`
re-downloads the dataset (79.5MB) into a gitignored `lidar_data/` folder and regenerates
`lidar_validation_frozen.json`; `pytest tests/test_lidar_sensor_validation.py` reads the
already-frozen file, no download needed.
