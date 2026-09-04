# -*- coding: utf-8 -*-
"""
scripts/gazebo_live_loop/live_safety_loop.py
====================================================================
Chains every real, promoted Dense-Armor safety primitive into one live
loop, driven by real Ignition Fortress physics (not replay, not
synthetic data), against a real actuated single-joint pendulum (model
"pendulum", joint "joint1"):

    sensor (/joint_states) -> streaming detector -> LLM decides a
    corrective target -> rate_limiter -> cbf_filter (sub-stepped, per
    real tick) -> motor (real joint velocity command back into Gazebo)

Everything here was checked directly against the real running system
before being relied on, not assumed:
- ignition-gazebo-joint-controller-system (confirmed present on disk),
  commanded via /model/pendulum/joint/joint1/cmd_vel (ignition.msgs.
  Double), confirmed empirically: publishing 1.5 drove the joint to a
  real measured velocity of 1.4999999999875004; the ROS2 bridge for the
  same topic confirmed too (-1.0 -> -1.0000000000426079).
- /joint_states carries exactly one real channel, "joint1".
- The pendulum's own inertia values were originally wrong (guessed,
  not computed) and caused a real physics-solver explosion -- fixed by
  computing the real box inertia tensor for both links directly.

REAL BUG FOUND AND FIXED IN THE FIRST LIVE RUN (kept here, not hidden):
JointStatePublisher had no declared <update_rate>, so it free-ran at
the physics step rate -- confirmed directly via `ros2 topic hz
/joint_states`: about 960 Hz, not a realistic sensor rate. This
session's Python callback (JAX-jitted dense_armor calls, real first-
call compile overhead) could not keep up, so `rclpy.spin_once` only
ever dequeued one message per real Python iteration while the
simulation kept running underneath -- the callback saw a wildly
under-sampled, aliased slice of the real physics, with huge real
simulated-time gaps between the states it actually reacted to. That
starved cbf_safety_filter of the fine time resolution its continuous-
time forward-invariance guarantee needs (the exact discrete-overshoot
failure mode Experiment 54 already found and documented -- reproduced
here live, in a closed loop, not just replayed on a recorded array).
Real joint position went past the declared 2.5 rad CBF boundary and
hit the real 3.14 rad hard mechanical limit.

TWO real fixes applied, not one -- lowering the publish rate alone is
not sufficient (Experiment 54 found even a single un-substepped CBF
evaluation per real sample can let h go negative, e.g. min h=-0.4811
with n_substeps=1 on real SO-101/ALOHA data at 30-50Hz):
1. JointStatePublisher's <update_rate> set to a real, declared 20 Hz
   (a realistic robot joint-state publish rate, not physics-step rate).
2. Each control tick now calls the real, promoted `cbf_filtered_
   trajectory` (n_substeps=20, this project's own established default)
   over [current_position, position_implied_by_this_tick's_desired_
   velocity], instead of a single raw `cbf_safety_filter` evaluation --
   the same sub-stepped integration Experiment 54 validated, reused
   directly rather than reimplemented.

Design choices, unchanged from the first run:
1. The LLM step only fires when the streaming detector flags a real
   deviation, not every tick.
2. "LLM decides" is realized as Claude (this session), via a real file
   handshake -- same substitution as Experiment 55, same reason (free,
   genuine reasoning about the specific flagged event).
3. rate_limiter bounds how fast the chosen target is approached; the
   sub-stepped cbf_filter then ensures the real resulting motion never
   crosses the declared safety boundary.
"""
import json
import pathlib
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from dense_armor.utility.streaming import MultiChannelStreamingDeviationDetector
from dense_armor.utility.rate_limiter import rate_limited_follower
from dense_armor.utility.cbf_filter import cbf_filtered_trajectory

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_DECISION_DIR = _THIS_DIR / "live_loop_handshake"
_FLAG_PATH = _DECISION_DIR / "flagged_state.json"
_DECISION_PATH = _DECISION_DIR / "decision.json"

N_CHANNELS = 1
JOINT_INDEX = 0
MAX_VEL = 1.5
MAX_ACCEL = 3.0
SAFE_JOINT_LIMIT = 2.5    # rad, real hard limit is +-3.14
CBF_ALPHA = 2.0
CBF_SUBSTEPS = 20          # matches Experiment 54's own established default
DECISION_TIMEOUT_S = 300.0


def _wait_for_decision(flagged_state: dict) -> float:
    _DECISION_DIR.mkdir(parents=True, exist_ok=True)
    if _DECISION_PATH.exists():
        _DECISION_PATH.unlink()
    with open(_FLAG_PATH, "w", encoding="utf-8") as f:
        json.dump(flagged_state, f, indent=2)
    print(f"[live loop] Real deviation flagged, waiting for a real decision in {_DECISION_PATH} ...", flush=True)
    start = time.time()
    while not _DECISION_PATH.exists():
        if time.time() - start > DECISION_TIMEOUT_S:
            raise TimeoutError(f"No decision written within {DECISION_TIMEOUT_S}s -- aborting, not guessing.")
        time.sleep(0.5)
    with open(_DECISION_PATH, encoding="utf-8") as f:
        decision = json.load(f)
    _DECISION_PATH.unlink()
    print(f"[live loop] Real decision received: {decision}", flush=True)
    return float(decision["target_velocity"])


class LiveSafetyLoop(Node):
    def __init__(self, baseline_target: float):
        super().__init__("live_safety_loop")
        self.det = MultiChannelStreamingDeviationDetector(
            n_channels=N_CHANNELS, radius=10, ref_mult=2, n_sigmas=3.0,
        )
        self.cmd_buffer = []
        self.current_target = baseline_target
        self.count = 0
        self.last_stamp = None
        self.create_subscription(JointState, "/joint_states", self.cb, 10)
        self.cmd_pub = self.create_publisher(Float64, "/model/pendulum/joint/joint1/cmd_vel", 10)

    def cb(self, msg: JointState):
        self.count += 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        real_dt = (stamp - self.last_stamp) if self.last_stamp is not None else 0.05
        real_dt = max(real_dt, 1e-3)
        self.last_stamp = stamp

        flags = self.det.update(list(msg.position))
        target = self.current_target

        if any(flags):
            flagged_state = dict(
                t=self.count, sec=stamp, real_dt=real_dt,
                positions=list(msg.position), velocities=list(msg.velocity),
                flags=list(bool(f) for f in flags), current_target=self.current_target,
            )
            target = _wait_for_decision(flagged_state)
            self.current_target = target

        self.cmd_buffer.append(target)
        limited = rate_limited_follower(
            np.array(self.cmd_buffer), max_vel=MAX_VEL, max_accel=MAX_ACCEL, dt=1.0,
        )
        u_des = float(limited[-1])

        current_pos = float(msg.position[JOINT_INDEX])
        implied_target_pos = current_pos + u_des * real_dt
        safe_traj = cbf_filtered_trajectory(
            np.array([current_pos, implied_target_pos]),
            obstacle=SAFE_JOINT_LIMIT, safe_dist=0.3, alpha_gain=CBF_ALPHA, n_substeps=CBF_SUBSTEPS,
        )
        u_safe = float((safe_traj[1] - safe_traj[0]) / real_dt)

        self.cmd_pub.publish(Float64(data=u_safe))
        print(f"n={self.count:4d} sec={stamp:.2f} dt={real_dt:.3f} pos={current_pos:.3f} "
              f"target={target:.3f} u_des={u_des:.3f} u_safe={u_safe:.3f} flags={flags}", flush=True)


def main():
    import sys
    baseline = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    n_ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rclpy.init()
    node = LiveSafetyLoop(baseline_target=baseline)
    try:
        for _ in range(n_ticks):
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
