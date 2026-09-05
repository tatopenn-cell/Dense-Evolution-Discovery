# Enforcing Real Joint Limits

The URDF parser read `<link>`/`<joint>` geometry and inertia but never touched each joint's
own `<limit>` tag -- nothing stopped the controller from commanding a joint past its real
physical bound.

## What changed

`RigidBodyModel` now parses each joint's real `lower`/`upper`/`velocity` limit (`±inf` where
the URDF declares none, e.g. a `continuous` joint's position, or a joint with no `<limit>`
element at all):

```python
model = RigidBodyModel("panda.urdf")
model.q_min, model.q_max   # real per-joint position bounds
model.qd_max                # real per-joint velocity bound (symmetric, from the URDF)
```

`solve_control_qp` enforces them as an additional constraint, following Kurtz, Wensing & Lin's
own "joint" CBF constraint type: a relative-degree-1 exponential CBF for the velocity limit, and
a relative-degree-2 (nested class-K) CBF for the position limit, both real class-K gains = 1,
matching their own published code. Both reduce to a simple box on the joint acceleration `qdd`.

## A real box, not a clamp

At a joint sitting right at its real bound with velocity driving further past it (Franka
Panda's `panda_joint4`, real range `[-3.1416, 0.0]`, `q=-0.001`, `qd=5.0`):

```
unconstrained nominal command:  qdd = -205.8
real CBF box for this joint:    qdd in [-7.175, -4.999]
solved qdd:                     -7.175  (the box's own edge)
```

The nominal task-space PD command already wants to decelerate this joint, just far more
aggressively than needed; the real limit box tames it to the gentler range that keeps the
joint's own CBF condition satisfied, rather than either ignoring the limit or overreacting.

---

## Details

**Only added when a real limit exists.** A robot whose URDF declares no limits anywhere (e.g.
Kinova's `GEN3_URDF_V12.urdf`, which has no `<limit>` tags at all) gets the exact same QP as
before this feature -- the joint-limit box rows are only added when at least one real, finite
bound is present, so the Experiment 61 cross-check (machine-precision match against the
hand-transcribed controller) still holds. An earlier attempt added the box rows unconditionally
(inert `+/-1e20` bounds when a robot has no real limits) -- harmless mathematically, but it
measurably perturbed OSQP's internal scaling by ~1e-4, breaking that cross-check for no
physical reason.

**A real, separate instability found along the way, not this feature's bug**: comparing a
9-DoF Panda run with real limits enforced against the same run with all limits manually
removed showed the "removed" run oscillating and diverging (`qd` norm swinging 5-15,
`tau` reaching into the thousands) while the real-limits run stayed smooth throughout (`qd`
norm ~2). Real joint velocity limits happen to stabilize an already-marginal
redundancy-resolution scenario here -- a legitimate, if incidental, finding, not evidence of a
bug in the limit-enforcement code itself (confirmed separately with a direct, single-QP-call
check of the box math above, independent of any closed-loop dynamics).

**Reproducing this**: `pytest tests/test_general_pbc_cbf_controller.py`.
