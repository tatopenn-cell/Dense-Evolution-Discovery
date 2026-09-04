#!/bin/bash
# Full, reproducible live safety loop: sensor -> streaming -> LLM decides ->
# rate_limiter -> cbf_filter -> motor, chained end to end against real
# Ignition Fortress physics (see docs/live_safety_loop.md for the real bugs
# found and fixed along the way -- bad inertia values causing a physics
# explosion, JointStatePublisher's free-running publish rate, the CBF's
# discrete-overshoot vulnerability reproduced live and fixed with sub-
# stepping). Run with:
#   docker run --rm -v "<this folder>:/ws:ro" dense-armor-robotics bash /ws/live_safety_loop.sh [baseline_target] [n_ticks]
set -e
source /opt/ros/humble/setup.bash
BASELINE=${1:-0.8}
N_TICKS=${2:-400}

echo "[$(date +%T)] starting headless Ignition (real physics, real 50Hz joint-state publish rate)"
ign gazebo -s -r /ws/actuated_pendulum.sdf > /tmp/gz.log 2>&1 &
sleep 5

echo "[$(date +%T)] bridging the real velocity command topic (ROS2 -> Ignition)"
ros2 run ros_gz_bridge parameter_bridge \
  /model/pendulum/joint/joint1/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double > /tmp/bridge_cmd.log 2>&1 &
sleep 2

echo "[$(date +%T)] bridging real joint_states (Ignition -> ROS2)"
ros2 run ros_gz_bridge parameter_bridge \
  '/world/empty/model/pendulum/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model' \
  --ros-args -r /world/empty/model/pendulum/joint_state:=/joint_states > /tmp/bridge_state.log 2>&1 &
sleep 4

mkdir -p /ws/live_loop_handshake
echo "[$(date +%T)] === running the real live safety loop (baseline target ${BASELINE} rad/s, ${N_TICKS} ticks) ==="
python3 -u /ws/live_safety_loop.py "$BASELINE" "$N_TICKS"

echo "[$(date +%T)] done"
