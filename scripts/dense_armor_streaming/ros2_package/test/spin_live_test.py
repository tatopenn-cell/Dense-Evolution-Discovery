# -*- coding: utf-8 -*-
"""
Live rclpy.spin() integration test for dense_armor_ros.JointDeviationNode.

Runs three real rclpy nodes in one process, all wired through the real
ROS2 pub/sub machinery (not mocked): a fake JointState publisher, the
real JointDeviationNode under test, and a flag collector subscriber.
Uses a real SingleThreadedExecutor (the same machinery rclpy.spin()
wraps) so the whole callback graph -- publish -> node's subscription
callback -> node's own publish -> collector's subscription callback --
actually runs through ROS2's real event loop.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8MultiArray

from dense_armor_ros.joint_deviation_node import JointDeviationNode

N_JOINTS = 6
BASELINE_MSGS = 60
DEVIATION_MSGS = 30
DEVIATED_JOINT = 2
DEVIATION_OFFSET = 5.0


class FakePublisher(Node):
    def __init__(self):
        super().__init__('fake_joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.count = 0
        self.timer = self.create_timer(0.02, self._tick)

    def _tick(self):
        msg = JointState()
        msg.name = [f'joint_{i}' for i in range(N_JOINTS)]
        base = [0.5 + 0.001 * ((self.count + i) % 5) for i in range(N_JOINTS)]
        if self.count >= BASELINE_MSGS:
            base[DEVIATED_JOINT] += DEVIATION_OFFSET
        msg.position = base
        self.publisher_.publish(msg)
        self.count += 1


class FlagCollector(Node):
    def __init__(self):
        super().__init__('flag_collector')
        self.received = []
        self.create_subscription(UInt8MultiArray, 'joint_deviation_flags', self._on_flags, 10)

    def _on_flags(self, msg: UInt8MultiArray):
        self.received.append(list(msg.data))


def main():
    rclpy.init()
    publisher = FakePublisher()
    detector_node = JointDeviationNode()
    collector = FlagCollector()

    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(detector_node)
    executor.add_node(collector)

    total_msgs = BASELINE_MSGS + DEVIATION_MSGS
    deadline = time.time() + 10.0
    while publisher.count < total_msgs or len(collector.received) < total_msgs:
        executor.spin_once(timeout_sec=0.1)
        if time.time() > deadline:
            print(f"TIMEOUT: published={publisher.count} received={len(collector.received)}")
            break

    print(f"published={publisher.count} flag_messages_received={len(collector.received)}")

    baseline_flags = collector.received[:BASELINE_MSGS]
    deviation_flags = collector.received[BASELINE_MSGS:]

    baseline_any_true = any(any(f) for f in baseline_flags)
    deviation_joint_flagged = any(f[DEVIATED_JOINT] for f in deviation_flags if len(f) > DEVIATED_JOINT)
    deviation_other_joints_flagged = any(
        any(v for i, v in enumerate(f) if i != DEVIATED_JOINT) for f in deviation_flags
    )

    print(f"baseline_any_true={baseline_any_true}")
    print(f"deviation_joint_{DEVIATED_JOINT}_flagged={deviation_joint_flagged}")
    print(f"deviation_other_joints_flagged={deviation_other_joints_flagged}")

    ok = (not baseline_any_true) and deviation_joint_flagged and (not deviation_other_joints_flagged)
    print("LIVE SPIN TEST: " + ("PASS" if ok else "FAIL"))

    for n in (publisher, detector_node, collector):
        n.destroy_node()
    rclpy.shutdown()

    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
