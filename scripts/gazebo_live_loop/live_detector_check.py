import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from dense_armor.utility.streaming import MultiChannelStreamingDeviationDetector

det = MultiChannelStreamingDeviationDetector(n_channels=3, radius=10, ref_mult=2, n_sigmas=3.0)
count = [0]


class LiveCheck(Node):
    def __init__(self):
        super().__init__('live_detector_check')
        self.create_subscription(JointState, '/joint_states', self.cb, 10)

    def cb(self, msg):
        flags = det.update(list(msg.position))
        count[0] += 1
        print(f"n={count[0]:4d} t={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
              f"pos={[round(p, 3) for p in msg.position]} flags={flags}")


rclpy.init()
node = LiveCheck()
try:
    rclpy.spin_once(node, timeout_sec=0.5)
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.5)
finally:
    node.destroy_node()
    rclpy.shutdown()

print(f"\nTOTAL real messages processed: {count[0]}")
