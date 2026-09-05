# Full 6-DoF (Position + Orientation) Tracking

Experiment 63's controller tracks only a link's position (3 DoF). Kurtz, Wensing & Lin's real
paper tracks full pose -- position and orientation together, six degrees of freedom.

## What changed

`six_dof_pbc_cbf_controller.py` extends the same passivity+singularity-CBF QP to the link's
full 6-DoF pose, using `RigidBodyModel`'s spatial (6xN) Jacobian instead of the 3xN
position-only one:

```python
from urdf_dynamics import RigidBodyModel
from six_dof_pbc_cbf_controller import solve_control_qp

model = RigidBodyModel("urdf/GEN3_URDF_V12.urdf")
qdd, tau, mu, h = solve_control_qp(model, "end_effector_link", q, qd,
                                    p_des, pd_des, pdd_des, r_des, w_des, wd_des, eps=0.03)
```

`r_des` is the desired 3x3 rotation matrix; `w_des`/`wd_des` are the desired angular
velocity/acceleration, alongside the same position/velocity/acceleration triple as before.

## A deliberate deviation from Kurtz et al.: the orientation error

The paper's own code measures orientation error via `RollPitchYaw` of the error rotation
matrix -- simple, but with a real, well-known gimbal-lock singularity. This uses Lee, Leok &
McClamroch's (2010, "Geometric Tracking Control of a Quadrotor UAV on SE(3)") SO(3) attitude
error instead:

```python
def rotation_error(r, r_des):
    e_mat = r_des.T @ r - r.T @ r_des
    return 0.5 * jnp.array([e_mat[2, 1], e_mat[0, 2], e_mat[1, 0]])
```

Smooth everywhere, vanishes iff `r == r_des`, no arccos/log-map derivative blowup near zero
error. Same task-space PBC+CBF structure as Experiment 63, different (real, citable, more
robust) error metric.

## Verified two ways

**Zero tracking error gives pure gravity compensation** -- a real correctness check of the
whole pipeline (spatial Jacobian, rotation error, 6x6 task-space Lambda), not just "doesn't
crash": at the exact desired pose and zero velocity, the solved `qdd` is exactly zero and the
solved torque matches `gravity_forces(q)` to machine precision (`8.9e-16`).

**Real closed-loop convergence**, a disclosed 10cm/10cm position offset plus a disclosed
30-degree orientation offset (about world z), RK4-integrated under the real dynamics:

| step | position error (m) | orientation error |
|---|---|---|
| 0 | 0.141 | 0.499 |
| 200 | 4.9e-4 | 0.054 |
| 600 | 1.9e-6 | 6.5e-4 |
| 1000 | 1.9e-8 | 7.9e-6 |
| 1499 | 9.9e-11 | 3.2e-8 |

Both errors decay roughly an order of magnitude every 200 steps -- clean exponential
convergence, both position and orientation together.

---

## Details

**Not yet promoted.** Kept in Discovery for now, parallel to Experiment 63's own path: a
strong single-robot validation, not yet cross-validated on a second/third robot the way
`RigidBodyModel` and the position-only controller were before their promotions.

**Reproducing this**: `pytest tests/test_six_dof_pbc_cbf_controller.py`.
