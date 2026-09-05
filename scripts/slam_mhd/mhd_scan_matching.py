import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def load_scans(path):
    """
    Real Intel Research Lab / MIT CSAIL 2D-lidar SLAM logs (Radish repository,
    Dirk Hahnel / Cyrill Stachniss), re-packaged as JSON by
    github.com/YoloPopo/SLAM-Scan-Matching. Each entry has a "range" array
    (raw lidar readings) and the real ground-truth "x"/"y"/"theta" pose at
    that scan, ordered by timestamp key.
    """
    with open(path) as f:
        raw = json.load(f)["map"]
    keys = sorted(raw.keys(), key=float)
    ranges = np.array([raw[k]["range"] for k in keys])
    poses = np.array([[raw[k]["x"], raw[k]["y"], raw[k]["theta"]] for k in keys])
    return ranges, poses


def range_to_points(ranges, fov=jnp.pi, max_range=10.0):
    """
    Raw range readings -> local-frame (x, y) points, robot forward = +x.
    FOV=pi (180 degrees), max_range=10m matches the real sensor geometry
    the reference implementation (og_params.lidar_fov, lidar_max_range)
    uses for these exact datasets -- a SICK LMS200-class scanner.
    Out-of-range ("no return") readings are dropped, not left as ghost
    points at the sensor's max range.
    """
    n = ranges.shape[0]
    angles = -fov / 2.0 + jnp.arange(n) * (fov / (n - 1))
    x = ranges * jnp.cos(angles)
    y = ranges * jnp.sin(angles)
    valid = ranges < (max_range - 1e-6)
    return jnp.stack([x, y], axis=-1), valid


def transform_points(points, dx, dy, dtheta):
    c, s = jnp.cos(dtheta), jnp.sin(dtheta)
    rot = jnp.array([[c, -s], [s, c]])
    return points @ rot.T + jnp.array([dx, dy])


def _directed_hausdorff(a, b, mask_a, mask_b):
    """Mean over a in A of min over b in B of ||a-b|| (Dubuisson & Jain 1994's
    "modified" -- average, not max-max -- directed distance), masking out
    invalid (out-of-range) points from both sets."""
    d = jnp.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    d = jnp.where(mask_b[None, :], d, jnp.inf)
    min_d = jnp.min(d, axis=1)
    min_d = jnp.where(mask_a, min_d, 0.0)
    return jnp.sum(min_d) / jnp.maximum(jnp.sum(mask_a), 1)


def modified_hausdorff_distance(a, b, mask_a, mask_b):
    """MHD(A,B) = max(directed(A,B), directed(B,A))."""
    return jnp.maximum(_directed_hausdorff(a, b, mask_a, mask_b),
                        _directed_hausdorff(b, a, mask_b, mask_a))


@partial(jax.jit, static_argnames=("n_dx", "n_dy", "n_dtheta"))
def _mhd_grid(ref_pts, ref_mask, new_pts, new_mask,
              dx0, dy0, dtheta0, r_xy, r_theta, n_dx, n_dy, n_dtheta):
    dxs = dx0 + jnp.linspace(-r_xy, r_xy, n_dx)
    dys = dy0 + jnp.linspace(-r_xy, r_xy, n_dy)
    dthetas = dtheta0 + jnp.linspace(-r_theta, r_theta, n_dtheta)

    def mhd_at(dx, dy, dtheta):
        moved = transform_points(new_pts, dx, dy, dtheta)
        return modified_hausdorff_distance(ref_pts, moved, ref_mask, new_mask)

    grid = jax.vmap(jax.vmap(jax.vmap(mhd_at, (None, None, 0)), (None, 0, None)), (0, None, None))
    return grid(dxs, dys, dthetas), dxs, dys, dthetas


def match_scan_to_reference(ref_pts, ref_mask, new_pts, new_mask,
                             dx0=0.0, dy0=0.0, dtheta0=0.0,
                             r_xy=0.3, r_theta=0.2, n_xy=15, n_theta=15):
    """
    Real MHD scan-to-reference matching (Nazate-Burgos et al.'s own real
    method for this problem): grid-search the rigid transform around an
    initial guess (e.g. from odometry) that minimizes the modified Hausdorff
    distance between the transformed new scan and the reference. MHD has no
    useful closed-form gradient (min over a discrete point set), so this is
    a real, deliberate grid search, not a differentiable relaxation.
    """
    costs, dxs, dys, dthetas = _mhd_grid(ref_pts, ref_mask, new_pts, new_mask,
                                          dx0, dy0, dtheta0, r_xy, r_theta, n_xy, n_xy, n_theta)
    i, j, k = jnp.unravel_index(jnp.argmin(costs), costs.shape)
    return float(dxs[i]), float(dys[j]), float(dthetas[k]), float(costs[i, j, k])
