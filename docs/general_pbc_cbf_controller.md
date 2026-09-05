# Generalizing the Passivity-CBF Controller to Any Robot

Experiment 62 generalized the dynamics engine to any URDF. Experiment 61's controller still
called Gen3-only functions (`end_effector_position`, `end_effector_jacobian` from
`gen3_dynamics.py`) directly, so it stayed tied to one robot even after the dynamics
underneath it became generic.

## What changed

`general_pbc_cbf_controller.py` is the same passivity+singularity-CBF QP as Experiment 61 --
same math, same OSQP-infeasibility fallback -- but every call into `gen3_dynamics.py` is
replaced by a call into a `RigidBodyModel` instance and a target link name:

```python
from urdf_dynamics import RigidBodyModel
from general_pbc_cbf_controller import solve_control_qp

model = RigidBodyModel("urdf/GEN3-6DOF.urdf")
qdd, tau, mu, h = solve_control_qp(model, "bracelet_with_vision_link", q, qd, p_des, pd_des, pdd_des, eps=0.03)
```

Any robot the parser can load, any named link as the tracked task-space point.

## Cross-check against Experiment 61

At the exact joint state that triggered Experiment 61's real OSQP-infeasibility bug (see
[docs/gen3_dynamics_and_cbf_controller.md](gen3_dynamics_and_cbf_controller.md)), the general
controller reproduces the hardcoded one to machine precision (`qdd`, `tau`, `mu`, `h` all
within `1e-9`) -- including correctly taking the same infeasibility-fallback branch.

## Validated on two more independent robots

Driving each robot's own task-space link toward its own singular pose (`mu=0`, a true
singularity in each case, same as Experiment 62 found):

| robot | link | min(mu), no CBF | min(mu), CBF eps=0.03 | final err (CBF) |
|---|---|---|---|---|
| Gen3 6-DoF | `bracelet_with_vision_link` | 0.00006 | 0.02995 | 0.27633 m |
| Franka Panda | `panda_hand` | 0.00296 | 0.02997 | 0.05034 m |

Both: the CBF holds manipulability almost exactly at its declared floor (0.17% and 0.1% below
eps respectively -- both tighter than the 7-DoF Gen3's own 100Hz result). The nonzero residual
tracking error is expected, not a bug: the commanded target is the singular pose itself
(`mu=0`), unreachable once the CBF enforces `mu>=eps` -- the controller correctly gets as close
as the constraint allows and stops there.

---

## Details

**Not yet promoted**, but now validated on three independent robots (Gen3 7-DoF, Gen3 6-DoF,
Franka Panda -- a different manufacturer), matching `RigidBodyModel`'s own validation bar
before its promotion.

**Reproducing this**: `pytest tests/test_general_pbc_cbf_controller.py`;
`python scripts/rigid_body_dynamics/general_controller_multirobot_validation.py` for the
closed-loop numbers above.
