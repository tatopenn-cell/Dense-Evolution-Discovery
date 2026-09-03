# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/ros2_deviation_node.py
========================================================
A minimal ROS2 node wrapping Experiment 49's
MultiChannelStreamingDeviationDetector: subscribes to
`sensor_msgs/msg/JointState` (the standard ROS2 message for a robot
arm's joint positions -- name/position/velocity/effort arrays, verified
directly against the real message definition in ros2/common_interfaces
before writing this, not assumed), runs the already-validated streaming
detector on the `position` array, and publishes one deviation flag per
joint on a separate topic.

HONEST DISCLOSURE, stated once and meant literally: this environment
has no ROS2 installation (checked directly -- no `rclpy`, no `ros2`
CLI, no Docker available either to run an official ROS2 image). This
node has NEVER been run against a live ROS2 system. What IS verified:
- The rclpy API used here (Node, create_subscription, create_publisher,
  the Node/callback structure) matches the real, current
  ros2/examples repository (humble branch, fetched directly before
  writing this file, not from memory).
- The message fields used (JointState.position, UInt8MultiArray.data)
  match the real message definitions in ros2/common_interfaces
  (humble branch, fetched directly).
- The DETECTION LOGIC (MultiChannelStreamingDeviationDetector) is
  already verified bit-exact against real robot data in Experiment 49
  -- this file only wires it to rclpy's subscribe/publish pattern.
- The callback logic itself lives in ros2_deviation_logic.py, split
  out deliberately because `import rclpy` fails outright in this
  environment (no ROS2 installed) -- that module has zero rclpy
  dependency and is directly, really tested (not mocked) in
  test_ros2_deviation_logic.py. This file is the thin, untested-here
  wiring layer on top of it.

OUTPUT MESSAGE TYPE: std_msgs/msg/UInt8MultiArray. Checked directly:
ros2/common_interfaces marks every std_msgs *MultiArray type
(Float64MultiArray, UInt8MultiArray, etc.) as "deprecated as of Foxy,
recommended to create your own semantically meaningful message" --
disclosed here rather than silently used as if it were the current
best practice. Used anyway because a custom .msg requires a full
package with message generation (a much heavier addition than this
single-file node), and *MultiArray remains what real ROS2 packages
commonly publish despite the recommendation. A future iteration could
replace this with a custom `dense_armor_msgs/DeviationFlags` message.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8MultiArray

from multichannel import MultiChannelStreamingDeviationDetector
from ros2_deviation_logic import process_joint_positions


class JointDeviationNode(Node):
    """Subscribes to `joint_states` (sensor_msgs/JointState), publishes
    `joint_deviation_flags` (std_msgs/UInt8MultiArray, one 0/1 per
    joint, same order as the incoming JointState.name/.position)."""

    def __init__(self, n_joints: int, radius: int = 10, ref_mult: int = 3, n_sigmas: float = 3.0):
        super().__init__('dense_armor_joint_deviation')
        self._detector = MultiChannelStreamingDeviationDetector(
            n_channels=n_joints, radius=radius, ref_mult=ref_mult, n_sigmas=n_sigmas,
        )
        self.subscription = self.create_subscription(
            JointState, 'joint_states', self._on_joint_state, 10)
        self.publisher_ = self.create_publisher(
            UInt8MultiArray, 'joint_deviation_flags', 10)

    def _on_joint_state(self, msg: JointState) -> None:
        data = process_joint_positions(self._detector, msg.position)
        if data is None:
            self.get_logger().warn(
                f"expected {self._detector.n_channels} joints, got {len(msg.position)} -- skipping this message")
            return
        out = UInt8MultiArray()
        out.data = data
        self.publisher_.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointDeviationNode(n_joints=6)  # SO-101-style 6-joint arm; adjust for your robot
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
