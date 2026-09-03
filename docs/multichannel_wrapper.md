# Native Multi-Channel Support for classify_segments and Streaming Detection

Second standard building block toward real robotics adoption. Every real robotics
experiment in this repo -- LeRobot's 6 joints (Experiments 43, 46, 47), the UCI HAR IMU's
multiple axes (Experiment 41) -- has needed a hand-written `for j in range(n_channels):`
loop around a function built for one 1D signal. Real robots always have more than one
channel. This is ergonomics, not a new algorithm: `classify_segments_multichannel` and
`MultiChannelStreamingDeviationDetector` apply the SAME, already-validated per-channel
logic across all channels at once.

## Correctness bar: identical to the hand-written loop, not "close enough"

```python
et_mc, dev_mc, unc_mc = classify_segments_multichannel(X, classify_segments, **ARBITER_KW)
# vs. calling classify_segments(X[:, j], **ARBITER_KW) for each j and stacking by hand
```

Verified on real, multi-channel data from two independent domains -- the same bar
`velocity_gated_stable_mask` and Experiment 48's streaming detector were promoted at:

```
LeRobot episode 0 (all 6 joints) (batch, n=303, c=6): all_channels_match=True
LeRobot episode 0 (all 6 joints) (streaming, n=303, c=6): all_channels_match=True
LeRobot episode 22 (all 6 joints) (batch, n=237, c=6): all_channels_match=True
LeRobot episode 22 (all 6 joints) (streaming, n=237, c=6): all_channels_match=True
IMU (real UCI HAR, 3 axes, n=3072) batch match: True
IMU streaming match: True
```

Zero mismatches, both the batch wrapper (`classify_segments_multichannel`) and the
streaming one (`MultiChannelStreamingDeviationDetector`, built on Experiment 48's
`StreamingDeviationDetector`), across a real 6-joint robot arm and a real 3-axis IMU.

## Design note: channels are independent by construction

Each channel gets its own reference window and its own baseline -- a robot's joints or an
IMU's axes are not assumed to share one statistical baseline (a fast-moving joint and a
near-stationary one have very different natural noise scales; forcing a shared baseline
would either desensitize the quiet channel or false-alarm on the active one). This is a
design choice stated explicitly, not an oversight: cross-channel relationships (does
channel A's anomaly correlate with channel B's) are a different question, tested and
rejected as a detection mechanism in Experiment 46 -- this module does not attempt that,
it only removes the need to hand-loop over independent per-channel detection.

## Reproducing this

`scripts/dense_armor_streaming/multichannel.py`
(`classify_segments_multichannel`, `MultiChannelStreamingDeviationDetector`) and
`scripts/dense_armor_streaming/validate_multichannel.py` -- reuses already-cached LeRobot
and UCI HAR data, no new downloads.
