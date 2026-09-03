# A Minimal ROS2 Node for Multi-Joint Deviation Detection

Third standard building block toward real robotics adoption -- the single most "standard"
ecosystem integration point, since practically every real robotics project checked in
Experiments 46-47 (Gazebo, TurtleBot, IsaacLab) runs on ROS/ROS2.

## Honest disclosure, up front: this was never run against a live ROS2 system

This environment has no ROS2 installation -- checked directly before writing anything:
no `rclpy` module, no `ros2` CLI, no Docker available either to run an official ROS2
image. Installing ROS2 natively on Windows is heavy (Chocolatey + Visual Studio Build
Tools, several GB) and wasn't authorized for this experiment. What follows is real
engineering grounded in verified, current sources -- not a claim that it has been
integration-tested live.

## What was actually verified, and how

**The rclpy API pattern.** Fetched directly from `ros2/examples` (humble branch, the
current, real repository, not from memory) before writing anything:

```python
# ros2/examples, rclpy/topics/minimal_publisher/.../publisher_member_function.py
class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        ...
```

The `Node.__init__`/`create_subscription`/`create_publisher`/callback pattern used in
`ros2_deviation_node.py` matches this exactly.

**The message types.** Fetched directly from `ros2/common_interfaces` (humble branch)
before choosing them:

```
# sensor_msgs/msg/JointState.msg
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
```

`JointState.position` is the standard field for a robot arm's joint positions -- the
subscriber reads it directly, matching the same signal type Experiments 43/46/47/49 have
been analyzing all along (LeRobot's `action`/`observation.state`).

For output, checked `std_msgs`' own message list directly: every `*MultiArray` type
(`Float64MultiArray`, `UInt8MultiArray`, etc.) carries the same note -- **"deprecated as of
Foxy, recommended to create your own semantically meaningful message"**. Used
`UInt8MultiArray` anyway, disclosed rather than silently treated as current best practice:
a custom `.msg` needs a full package with message generation (a much heavier addition than
this single-file node), and `*MultiArray` remains what real ROS2 packages commonly publish
despite the recommendation. A future iteration could add a proper
`dense_armor_msgs/DeviationFlags` message.

## What was actually run, and what wasn't

`import rclpy` fails outright in this environment, so `joint_deviation_node.py` itself
cannot even be imported here, let alone run. Split the callback logic into
`joint_deviation_logic.py` -- zero rclpy dependency, fully testable on its own -- so at
least the part that CAN be verified here, was:

```python
def process_joint_positions(detector, position):
    if len(position) != detector.n_channels:
        return None
    flags = detector.update(position)
    return [1 if f else 0 for f in flags]
```

```
test_wrong_length_returns_none: PASSED
test_correct_length_returns_list_of_0_1: PASSED
test_matches_direct_detector_call_on_real_lerobot_data: n=303 frames, all match
test_matches_direct_detector_call_on_real_lerobot_data: PASSED
```

Real, executable tests (not mocked, not skipped) -- including feeding all 303 real frames
of LeRobot episode 0 through `process_joint_positions` and checking every single output
matches calling `MultiChannelStreamingDeviationDetector` directly. `joint_deviation_node.py`
itself only compiles syntactically (`python -m py_compile`, confirmed) -- the thin rclpy
wiring layer on top, genuinely untested here.

## Update: a real installable ament_python package, ROS2 parameters, importing from the real library

The first version was a loose script -- not something a real ROS2 user could `ros2 run`.
Rebuilt as a proper `ament_python` package (`package.xml`/`setup.py`/`setup.cfg`/
`resource/`), matching `ros2/examples`' own `minimal_publisher` package layout, fetched
directly before writing anything. Two concrete gaps closed:

- **ROS2 parameters, not hardcoded constructor args.** `declare_parameter`/`get_parameter`'s
  real signatures were fetched directly from `ros2/rclpy`'s `node.py` and `parameter.py`
  source (humble branch) before using them -- `n_joints`/`radius`/`ref_mult`/`n_sigmas` are
  now configurable via a launch file/YAML, the idiomatic ROS2 way, instead of requiring a
  code change to retarget a different robot.
- **Imports the real, promoted `MultiChannelStreamingDeviationDetector` from
  `dense_armor.utility.streaming`** (Dense-Armor's own library, promoted after Experiments
  48-49), not a local copy -- `dense-armor` is declared as a real pip dependency in
  `setup.py`'s `install_requires`. Re-verified this actually works: reinstalled Dense-Armor
  in editable mode from its local repo (reversible, restored to the prior PyPI install
  afterward) and re-ran the real tests against the promoted library code directly -- same
  result, all real LeRobot frames match.

Still not colcon-built or run against a live ROS2 system -- that remains the one gap that
genuinely cannot be closed without installing ROS2 (or Docker), which needs the maintainer's
explicit go-ahead given the weight of that install.

## Honest scope

This is real engineering against verified, current sources, not a live-tested ROS2
integration. Before trusting this in an actual robot pipeline: `colcon build` it against a
real ROS2 installation (Foxy/Humble/Jazzy), confirm it actually builds, confirm the topic
names/QoS settings suit the target robot, and consider replacing `UInt8MultiArray` with a
proper custom message once the package is real enough to warrant one.

## Reproducing this

`scripts/dense_armor_streaming/ros2_package/` -- a full `ament_python` package:
`package.xml`, `setup.py`, `setup.cfg`, `resource/dense_armor_ros`,
`dense_armor_ros/joint_deviation_logic.py` (tested, rclpy-free),
`dense_armor_ros/joint_deviation_node.py` (untested rclpy wiring, ROS2 parameters),
`test/test_joint_deviation_logic.py` (the real tests -- reuses already-cached LeRobot data,
no new download).
