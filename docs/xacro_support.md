# Loading Robots from Xacro Macros

Real robot descriptions are rarely shipped as a single flat URDF. Manufacturers publish them as
`.xacro` macro files: parametrized building blocks (`xacro:macro`), math expressions
(`${-pi/2}`), conditionals (`xacro:unless`) and includes (`xacro:include`) that get expanded into
a plain URDF before anything can parse them. `RigidBodyModel` only accepted the expanded form.

## What changed

Point `RigidBodyModel` at a `.xacro` file directly:

```python
model = RigidBodyModel("panda_arm_hand.urdf.xacro")
model.n   # 9 -- 7 arm joints + 2 gripper fingers, same as the plain URDF
```

The real `xacro` package (the same expander the ROS ecosystem itself uses, `pip install xacro`,
no ROS install required) does the expansion; nothing about macros, math or conditionals is
reimplemented here. A `.xacro` extension routes through `xacro.process_file(path).toxml()` first;
any other extension parses as a plain URDF exactly as before.

## A real inconsistency found along the way

Expanding the Franka Panda's own published macros (`panda_arm.xacro` + `hand.xacro`) first
produced a 7-joint model and a `KeyError: 'panda_hand'` -- the hand macro attaches itself with
`connected_to="panda_link8"`, but the arm macro's own `panda_link8`/`panda_joint8` block was
commented out in the source. The separately checked-in, pre-expanded `panda_arm_hand.urdf` in the
same upstream repo does include that link, meaning it was generated from an earlier, uncommented
version of the same macro. Restoring that block (same real values: mass 0.005, inertia 0.00003,
origin `0 0 0.107`) reconnects the tree and the hand resolves correctly.

---

## Details

**Source files**: `scripts/rigid_body_dynamics/urdf/xacro_panda/` (`panda_arm.xacro`,
`hand.xacro`, `panda_arm_hand.urdf.xacro`), fetched from `clvrai/furniture`.

**Not the same Panda as `panda.urdf`**: this repo already carries a different, plain Franka
Panda URDF (official `franka_description` inertial numbers) used elsewhere in this test suite.
The `clvrai/furniture` xacro source used here has its own, different published inertial
parameters (e.g. `link0` mass 4.0 here vs. 2.9 there) -- both are real, both are published, they
are simply two independent parameter sets for the same physical robot, not a bug in the
expansion. The regression guard here instead checks that expanding the same xacro source twice
gives identical results (`xacro.process_file()` has no hidden global state across calls).

**Reproducing this**: `pytest tests/test_xacro_support.py`.
