# -*- coding: utf-8 -*-
"""
dense_armor_ros/joint_deviation_logic.py
============================================
The pure, rclpy-free logic behind joint_deviation_node.py -- split out
deliberately so it's testable in an environment without ROS2 installed
(this one has none: no rclpy, no ros2 CLI, no Docker either).
joint_deviation_node.py imports THIS module and wires it to rclpy's
Node/subscription/publisher machinery; nothing here imports rclpy or
any ROS2 message type.

Imports the REAL, promoted MultiChannelStreamingDeviationDetector from
dense_armor.utility.streaming (Dense-Armor >=1.1.13, unreleased at
promotion time -- see that PR) -- not a local copy. `pip install
dense-armor` is a real dependency of this package (declared in
setup.py's install_requires), not a rosdep key.
"""
from typing import List, Optional, Sequence

from dense_armor.utility.streaming import MultiChannelStreamingDeviationDetector


def process_joint_positions(
    detector: MultiChannelStreamingDeviationDetector, position: Sequence[float],
) -> Optional[List[int]]:
    """Feeds one JointState.position-shaped sequence through the
    detector, returns a list of 0/1 ints (one per joint) -- exactly
    what joint_deviation_node.py packs into a UInt8MultiArray.msg's
    `.data` field. Returns None if the length doesn't match the
    detector's own n_channels (the node's own "skip this message"
    case), so the caller can decide what to log/do."""
    if len(position) != detector.n_channels:
        return None
    flags = detector.update(position)
    return [1 if f else 0 for f in flags]
