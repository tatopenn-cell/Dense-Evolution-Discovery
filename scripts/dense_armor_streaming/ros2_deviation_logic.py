# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/ros2_deviation_logic.py
=========================================================
The pure, rclpy-free logic behind ros2_deviation_node.py -- split out
deliberately so it can be tested in an environment without ROS2
installed (this one: no rclpy, no ros2 CLI, no Docker available
either). ros2_deviation_node.py imports THIS module and wires it to
rclpy's Node/subscription/publisher machinery; nothing here imports
rclpy or any ROS2 message type, so it is fully testable on its own.
"""
from typing import List, Sequence

from multichannel import MultiChannelStreamingDeviationDetector


def process_joint_positions(
    detector: MultiChannelStreamingDeviationDetector, position: Sequence[float],
) -> List[int]:
    """Feeds one JointState.position-shaped sequence through the
    detector, returns a list of 0/1 ints (one per joint) -- exactly
    what ros2_deviation_node.py packs into a UInt8MultiArray.msg's
    `.data` field. Returns None if the length doesn't match the
    detector's own n_channels (the node's own "skip this message"
    case), so the caller can decide what to log/do."""
    if len(position) != detector.n_channels:
        return None
    flags = detector.update(position)
    return [1 if f else 0 for f in flags]
