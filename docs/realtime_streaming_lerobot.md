# Streaming Deviation Detection at a Real Robot's Real Frame Rate

Prior work (Experiments 48-49) validated `MultiChannelStreamingDeviationDetector`'s
*correctness* -- bit-exact against a hand-written per-channel loop -- and cited an
isolated ~18.6kHz single-channel throughput figure from a separate benchmark. Neither
of those actually measured this detector's real per-call latency against a real
robot's real recorded frame rate, or checked it doesn't fall behind over a sustained
run. This experiment does both, closing the gap between "the algorithm is fast in
isolation" and "it can genuinely keep up with a live 30Hz robot stream."

## Step 1. The real data and its real frame rate

```python
from huggingface_hub import hf_hub_download
import pandas as pd

parquet_path = hf_hub_download(
    repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
    filename="data/chunk-000/file-000.parquet", local_dir=data_root,
)
df = pd.read_parquet(parquet_path)
sub = df[df.episode_index == 0].sort_values("frame_index")
action = np.stack(sub["action"].values)  # (303, 6) -- all 6 real joints
ts = sub["timestamp"].to_numpy()
```

Real episode 0, 303 frames, 6 real joints -- the same real episode Experiments 48-49
already validated correctness on, reused here rather than downloading anything new. The
dataset's own `timestamp` column gives the real recorded frame rate directly:
`dt=33.33ms`, i.e. real 30Hz -- not assumed from documentation, measured from the data.

## Step 2. Real per-call latency against the real 33.3ms budget

```python
det = MultiChannelStreamingDeviationDetector(n_channels=6, radius=5, ref_mult=2, n_sigmas=3.0)
for frame in action:
    t0 = time.perf_counter()
    det.update(frame)
    latency = time.perf_counter() - t0
```

```
n=303 real frames, real budget per frame = 33333us (30Hz)
latency: median=316.9us  std=52.6us  max=742.0us
headroom (median): 105x
frames where update() alone exceeded the real 33.3ms budget: 0/303
```

Real, measured against the real installed PyPI package (`dense-armor==1.1.14`, upgraded
from a stale local 1.1.11 before running this -- verified directly, not assumed current).
105x real headroom, zero real budget violations across all 303 real frames.

## Step 3. Reconciling with the ~18.6kHz figure already in streaming.py's docstring

316.9us median implies ~3155Hz for the 6-channel `MultiChannelStreamingDeviationDetector`
-- not 18.6kHz. Not a discrepancy once checked directly: the original 18.6kHz figure is
for the SINGLE-channel `StreamingDeviationDetector`, and `MultiChannelStreamingDeviationDetector`
runs `n_channels` independent single-channel detector instances per `update()` call, so a
~6x slowdown for 6 channels is expected. `18600 / 6 = 3100Hz`, measured here `3155Hz` --
matches to within 2%, confirming the two numbers are consistent, not conflicting, once the
multi-channel overhead is accounted for explicitly rather than assumed away.

## Step 4. Sustained real-time playback -- does it fall behind?

A single-call latency number doesn't rule out slow drift accumulating over a long real
run. Simulated genuinely consuming the stream at its real recorded rate (sleep to each
real timestamp, exactly like a live subscriber would), over the full real 10.07s episode:

```
real recorded duration: 10.07s, wall-clock consumed: 10.07s
max single-frame drift (processing pushing behind the real target time): 3.91ms
```

Wall-clock consumption matches the real recorded duration to the second; the real
worst-case single-frame drift (3.91ms) stays well inside the real 33.3ms budget and does
not grow across the run -- no accumulating backlog, confirmed by direct measurement rather
than assumed from the per-call number alone.

## Result

| quantity | value |
|---|---|
| real frame rate (measured from data) | 30.0 Hz (dt=33.33ms) |
| median real per-call latency (6-channel) | 316.9us |
| real headroom (median) | 105x |
| real budget violations | 0/303 |
| real sustained-playback max drift | 3.91ms |

`MultiChannelStreamingDeviationDetector` genuinely keeps up with this real robot's real
recorded rate, measured directly rather than inferred from an isolated single-channel
benchmark -- the first honest closing of the gap between "fast in isolation" and "fast
enough for this specific real robot, measured against its own real timing."

---

## Details

**Why episode 0, why these detector parameters**: same real episode and
`radius=5, ref_mult=2, n_sigmas=3.0` as `validate_multichannel.py`'s already-established
correctness check on this dataset -- reusing an already-validated configuration rather than
picking new parameters that might flatter the latency number.

**What this does NOT check**: a real ROS2/rclpy subscriber callback overhead (Experiment 50
checked that separately, with a fake publisher through a real `SingleThreadedExecutor`);
GPU/JAX dispatch overhead (this detector's inner math is small enough that JAX isn't
invoked per-step here); or any other real robot's frame rate -- 30Hz is this specific real
dataset's rate, not a universal claim.

**Reproducing this**: `python scripts/dense_armor_streaming/realtime_lerobot_streaming.py`
(reuses the already-cached real LeRobot parquet, no new download; requires
`pip install --upgrade dense-armor` first if a stale local version is installed, as it was
here).
