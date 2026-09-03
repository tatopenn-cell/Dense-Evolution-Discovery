import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8MultiArray
from dense_armor.utility.streaming import MultiChannelStreamingDeviationDetector

print("TUTTI GLI IMPORT DI BASE FUNZIONANO")

det = MultiChannelStreamingDeviationDetector(n_channels=6, radius=5, ref_mult=2, n_sigmas=3.0)
flags = det.update([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print(f"detector funzionante, flags={flags}")
