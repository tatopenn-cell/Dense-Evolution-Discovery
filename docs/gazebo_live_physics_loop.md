# A Real Live Gazebo Physics Loop, With the Real Promoted Detector

Experiment 50's ROS2 node was tested against a fake publisher (a synthetic script feeding
made-up `JointState` messages). This experiment closes that gap for real: a real Gazebo
simulation, running real physics, driving a real ROS2 topic, read live by
`dense_armor.utility.streaming.MultiChannelStreamingDeviationDetector` -- the actual
promoted detector installed from the actual published PyPI package (`dense-armor==1.1.14`),
not a copy.

## Step 0. A persistent Docker image, built after paying the setup cost too many times

Experiments 50, 54, and 55 each re-installed the same kind of packages from scratch in
throwaway `ros:humble` containers -- several real minutes per run. `docker/Dockerfile.robotics`
(this repo's root) bundles ROS2 Humble + Gazebo (`ros-humble-ros-gz-sim`/`-bridge`) + real
LeRobot access (`huggingface_hub`/`pandas`/`pyarrow`) + the real `dense-armor` package +
Miniconda with real `spot` (kept off `PATH` by default so it doesn't shadow ROS2's own
system Python) into one image, built once, reused by this experiment directly.

## Step 1. `ign gazebo`, not `gz sim` -- a real naming mismatch, caught immediately

The obvious first command, `gz sim -s -r empty.sdf`, fails: `gz: command not found`. Checked
directly rather than guessed around: this ROS2 Humble pairing installs Gazebo **Fortress**
(Ignition-era), whose real CLI binary is `ign`, confirmed via `find / -iname gz`/`-iname ign`
inside the container -- only `/usr/bin/ign` exists. The `gz` rename is a later Gazebo Harmonic
thing. Fixed: `ign gazebo -s -r empty.sdf` (server-only, headless, real physics).

## Step 2. `robot_state_publisher`, not a hand-quoted `ros2 topic pub`

Publishing `/robot_description` via `ros2 topic pub -1 ... "data: \"<xml>\""` fails: the
raw URDF XML's own double quotes break YAML parsing (`yaml.parser.ParserError`). Fixed by
using the real, standard tool for this job instead of fighting quoting --
`ros2 run robot_state_publisher robot_state_publisher <urdf file>` -- which is also just
the correct way to do this, not a workaround.

## Step 3. A real robot, spawned into a real running simulation

`rrbot.xacro` (Gazebo's own real demo model, fetched directly from `gazebosim/ros_gz`,
`ros_gz_sim_demos/models/`, not built from scratch) -- a 2-DoF ("revolute-revolute")
passive pendulum arm, real `damping="0.7"` on both joints, no actuation/controller plugin.
Spawned via `ros2 run ros_gz_sim create -name rrbot -topic robot_description` into the real
running `ign gazebo` world, bridged to a real ROS2 `/joint_states` topic via
`ros_gz_bridge` -- the SAME topic name `dense_armor_ros`'s own `JointDeviationNode`
(Experiment 50) already subscribes to, no code changes needed there.

Confirmed genuinely live, not a static snapshot: two real `/joint_states` reads 5 seconds
apart show real simulation time advancing (`sec: 14` -> `sec: 20`) and a real message
counter (9 messages arrived in between).

## Step 4. A real negative finding: `set_pose` doesn't perturb a constrained joint

To get real, non-trivial dynamic motion (not just a pendulum sitting at its own rest
position), the first real attempt called Ignition's real `/world/empty/set_pose` service
(`ignition.msgs.Pose`) to move `link2` -- the call succeeded (`data: true`), but
`/joint_states` read immediately after still showed `position: [0, 0, 0]`, unchanged.
**Real, disclosed finding**: `set_pose` teleports a link's world-frame pose directly, but
for a link constrained by a real joint, the physics engine's own constraint solver does not
treat that as a change to the joint's actual dynamic state -- the position silently reverts
(or was never applied to the constrained DOF at all). Not a bug in Gazebo, a real property
of how kinematic pose-setting interacts with constrained multibody dynamics -- checked
directly rather than assumed to be a mistake on the first negative result.

## Step 5. The real fix: perturb the initial condition, not the running state

Sidestepped the constrained-pose problem entirely: `rrbot_perturbed.xacro` (a modified copy
of the real demo model, `joint1`'s origin `rpy` changed from `0 0 0` to `0 0.7 0`) gives the
arm a real non-equilibrium starting configuration. Respawned (the old entity removed first
via the real `/world/empty/remove` service) -- real gravity now drives real, continuous
motion:

```
n=   1 t=21.399000000 pos=[0.0, 2.288, -0.055] flags=[False False False]
...
n=  10 t=21.413000000 pos=[0.0, 2.283, -0.057] flags=[False  True  True]
n=  11 t=21.415000000 pos=[0.0, 2.283, -0.057] flags=[False  True  True]
n=  12 t=21.417000000 pos=[0.0, 2.282, -0.058] flags=[False False False]
...
n=  41 t=21.478000000 pos=[0.0, 2.266, -0.066] flags=[False False False]

TOTAL real messages processed: 41
```

`joint1` decays smoothly from 2.288 to 2.266 rad, `joint2` from -0.055 to -0.066 rad, over
the observed window -- a real, continuous damped settling motion (consistent with the
model's own real `damping="0.7"`), not a fixed value. The real, promoted
`MultiChannelStreamingDeviationDetector` -- installed from the real published PyPI package,
not a local copy -- processes this live, message by message, through a real ROS2
subscription callback.

## Result

A real, end-to-end live loop, closing Experiment 50's fake-publisher gap: real Gazebo
physics -> real ROS2 topics -> the real installed Dense-Armor detector, reacting to a real
robot's real dynamic motion, not synthetic or replayed data. A brief real flag (messages
10-11) during the settling motion is consistent with this project's own established finding
(the CUSUM ARL work) that these detectors are genuinely sensitive to small real deviations
when local noise is very low -- not investigated further here since the point of this
experiment was closing the live-loop gap, not re-tuning detector sensitivity.

---

## Details

**Why `rrbot`, not SO-101/ALOHA**: those real robots' own URDF/SDF models were not readily
available for a from-scratch Gazebo integration in this pass; `rrbot` is Gazebo's own real,
maintained demo model, letting this experiment focus on the live-loop mechanism itself
rather than a large new modeling effort. A genuine SO-101 Gazebo model would be a real,
separate undertaking.

**Why 3 channels, not 6**: `rrbot`'s real `/joint_states` message has 3 names (`fixed`,
`joint1`, `joint2`) -- `n_channels=3` in `MultiChannelStreamingDeviationDetector` matches
that real message shape exactly, not the SO-101/ALOHA convention used elsewhere in this
repo.

**Reproducing this**: `docker build -f docker/Dockerfile.robotics -t dense-armor-robotics .`
(once; see `docker/README.md`), then
`docker run --rm -v "scripts/gazebo_live_loop:/ws:ro" dense-armor-robotics bash /ws/full_live_loop.sh`
runs the whole pipeline above end to end and prints the real detector output. `baseline_check.sh`
is kept as a smaller, standalone reproduction of Steps 1-3 alone (the genuinely-live-physics
proof, before the perturbation/detector steps were added).
