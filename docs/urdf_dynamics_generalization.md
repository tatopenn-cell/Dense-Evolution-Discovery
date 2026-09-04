# Generalizing the Dynamics Engine: One Real Parser, Three Real Robots

Experiment 61 built a real rigid-body dynamics engine for a Kinova Gen3, but every number in it
-- masses, centers of mass, inertia tensors, joint origins -- was typed in by hand from that
one robot's URDF. That was the reason it stayed in Discovery: every other Dense-Armor module
works for any joint array of any length, and this one worked for exactly one specific robot.

## Step 1. A real URDF parser, not a robot-specific table

`urdf_dynamics.py` reads a real URDF file's `<link>` and `<joint>` elements with Python's
standard `xml.etree.ElementTree` -- no new heavy dependency. It builds a real parent/child
adjacency from the `<joint><parent>`/`<child>` tags, so it supports any kinematic **tree**
(a gripper with two independent fingers, say), not only a single serial chain like Experiment
61's hardcoded arm.

```python
from urdf_dynamics import RigidBodyModel

model = RigidBodyModel("urdf/GEN3_URDF_V12.urdf")
model.n            # 7 -- read from the file, not hardcoded
model.mass_matrix(q)
```

## Step 2. Any joint axis, any joint type

Experiment 61's forward kinematics assumed every joint rotates about its own local z-axis,
true for the Kinova Gen3 but not guaranteed in general. The general version uses Rodrigues'
rotation formula for an arbitrary axis:

```python
def axis_angle_matrix(axis, angle):
    axis = axis / jnp.linalg.norm(axis)
    k = jnp.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    return jnp.eye(3) + jnp.sin(angle) * k + (1.0 - jnp.cos(angle)) * (k @ k)
```

Prismatic joints (a linear slide, not a rotation) are handled too: the joint's contribution to
a link's position is `axis_world * q[i]` instead of a rotation, and its Jacobian column is the
joint axis directly rather than `axis x (p_link - p_joint)`.

## Step 3. Same physics, verified the same way

`mass_matrix`, `bias_forces`, `gravity_forces` are built exactly as in Experiment 61 --
`jax.grad`/`jax.jvp` on the kinetic and potential energy, not a hand-derived Christoffel
expression -- just generalized to walk the parsed tree instead of a hardcoded 7-link list.

## Step 4. Validated on three real, independent robots

First, a direct cross-check against Experiment 61's own hand-transcribed numbers, on the same
real Gen3 7-DoF URDF: mass matrix and gravity forces match to machine precision (`1e-16`). (A
small, expected, and understood 1.4 J difference showed up in total *energy* at first -- traced
to the general parser correctly including the fixed base link's own static gravitational
potential energy, which Experiment 61's hardcoded arrays never listed at all. A constant,
q-independent offset, confirmed by checking it was identical -- to 1e-10 -- across ten random
configurations, and equal to `mass_base * g * z_base` exactly.)

Then, two more real robots the code had never seen:

| robot | source | DoF | joint types |
|---|---|---|---|
| Kinova Gen3 6-DoF | github.com/vincekurtz/kinova_drake | 6 | all revolute, different chain (`bicep_link`/`forearm_link`) |
| Franka Emika Panda | bulletphysics/bullet3 (pybullet's real data) | 9 | 7 revolute + 2 prismatic |

Both: mass matrix symmetric and positive-definite at 20 random configurations, and free-dynamics
energy conservation with the correct 4th-order RK4 convergence.

| robot | dt=1e-2 | dt=1e-3 | dt=1e-4 |
|---|---|---|---|
| Gen3 7-DoF | 9.3e-2 | 4.4e-6 | 4.1e-10 |
| Gen3 6-DoF | 2.9e-1 | 5.4e-6 | 3.8e-10 |
| Panda | 5.9e-7 | 6.0e-11 | 5.5e-15 |

(relative energy drift; the Panda's published inertia tensors are simpler placeholder values,
which is why its baseline drift at the largest step is already much smaller -- not a difference
in correctness, a difference in how stiff the underlying dynamics are.)

---

## Details

**Mimic joints not modeled.** The Panda's real URDF ties its two finger joints together with a
`<mimic>` tag (closing one closes the other); this parser doesn't read that tag, so the energy
check above treats both fingers as independent, unconstrained DOF. That's still a valid (if
slightly different, more permissive) mechanical system, and the energy-conservation property
holds regardless -- but it means this isn't yet a byte-for-byte simulation of the real
constrained gripper.

**Inertia values are the file's, not verified against the manufacturer.** The Panda URDF used
here (pybullet's `franka_panda/panda.urdf`) ships simplified placeholder inertia tensors
(uniform `0.1` diagonal for most links) rather than Franka's true measured values -- a known
property of that specific public file, not something altered here. The masses are real
published figures; the inertia tensors are a common simplification, disclosed rather than
presented as precise.

**Reproducing this**: `python scripts/rigid_body_dynamics/urdf_generality_check.py` runs the
SPD and energy-conservation checks on both new robots; `pytest tests/test_urdf_dynamics.py`
runs the full regression suite (DOF count, SPD, energy conservation, and the Experiment 61
cross-check) across all three.
