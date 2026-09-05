# 2D Lidar Scan Matching via Modified Hausdorff Distance

Nazate-Burgos et al.'s real SLAM approach for GNSS-denied agricultural robots matches a new 2D
lidar scan against a reference (a previous scan or an accumulated local map) using the Modified
Hausdorff Distance (MHD, Dubuisson & Jain 1994) instead of feature extraction -- robust to
outliers and foliage, and cheap enough for a single 2D lidar.

## What this builds

A raw range array becomes a local-frame point cloud, then MHD scores how well one point set
covers another:

```python
from mhd_scan_matching import range_to_points, modified_hausdorff_distance, match_scan_to_reference

pts, mask = range_to_points(ranges)              # (N, 2) local-frame points, invalid points masked
d = modified_hausdorff_distance(pts_a, pts_b, mask_a, mask_b)
dx, dy, dtheta, cost = match_scan_to_reference(ref_pts, ref_mask, new_pts, new_mask,
                                                dx0=0.0, dy0=0.0, dtheta0=0.0)
```

`modified_hausdorff_distance(A, B)` is `max(directed(A,B), directed(B,A))`, where
`directed(A,B)` is the mean over `A` of each point's distance to its nearest neighbor in `B` --
the "modified" (average-based) Hausdorff distance, not the classic (and far more outlier-prone)
max-max version. `match_scan_to_reference` grid-searches the rigid transform `(dx, dy, dtheta)`
around an initial guess that minimizes this cost -- MHD has no useful closed-form gradient (a
`min` over a discrete point set), so this is a real, deliberate grid search, not a differentiable
relaxation.

## Real data, not synthetic

Validated first on a synthetic case (a known rigid transform applied to a real scan, recovered
by grid search to within one grid step) before touching real logs. Then on the real Intel
Research Lab and MIT CSAIL 2D-lidar SLAM datasets (Radish repository, CARMEN log format,
`x`/`y`/`theta` there is the format's own *corrected* pose field, not raw odometry) -- re-packaged
as JSON by `github.com/YoloPopo/SLAM-Scan-Matching` (MIT licensed), no ROS or rosbag needed.

---

## Details

**Why not the original paper's own dataset.** Nazate-Burgos et al.'s real Pullally orchard
dataset (`RAL-UC/Pullally_Dataset`) is real and freely documented, but its actual download link
requires logging in with a `uc.cl` university account -- not truly public despite the README
calling it a "download link". Intel Research Lab / MIT CSAIL are a real, freely downloadable
substitute good enough to validate the matching *algorithm* itself (real 2D lidar scans with a
real corrected-pose reference) -- the arboreal-specific domain was the paper's own motivation for
MHD, not a requirement of MHD itself.

**A real, disclosed limitation, not a bug.** On a near-pure-rotation case (index 100, Intel:
ground truth `dtheta=0.572` rad, no translation), matching recovers the pose to within one grid
step (`dtheta` error `-0.029` rad). On cases with larger translation between scans, the MHD cost
surface has a real local minimum *below* the cost at the true pose -- e.g. at index 34 (ground
truth `(-1.049, 0.149, -0.074)`), cost at the true pose is `0.654`, but the grid search finds a
different pose with cost `0.540`. Accumulating a 5-scan local map (each scan composed into a
common frame via the logged poses, verified by a self-composition identity check: transforming a
scan into its own frame reproduces it exactly, 0 diff) did not fix this -- cost at the true pose
rose to `1.02`, worse, not better, on the same case, and the same pattern held on a second,
independently-chosen "calm" segment (index 733, dominant translation, minimal rotation): cost at
ground truth `0.992` vs. a found minimum of `0.461`.

The most likely real cause: these classic datasets log scans far apart (median per-step motion
here is `~0.66` m / `~24` degrees -- not a densely-sampled continuous stream), so even a 5-scan
local map may not add much genuinely new overlapping structure. Pure Hausdorff-based matching
without a tight, trustworthy prior or a multi-resolution/ICP-style local refinement is known in
the literature to be exactly this vulnerable to spurious lower-cost alignments when scan overlap
is limited -- a real, disclosed limit of the method as implemented here, not something papered
over.

**Reproducing this**: `python scripts/slam_mhd/mhd_scan_matching.py` functions, exercised
directly against `scripts/slam_mhd/data/intel.json` (no test suite yet -- exploratory).
