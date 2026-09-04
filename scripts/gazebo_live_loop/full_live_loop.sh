#!/bin/bash
# Full, reproducible live Gazebo -> real detector loop, consolidated from an
# interactive debugging session (see docs/gazebo_live_physics_loop.md for the
# real bugs found and fixed along the way -- ign gazebo not gz sim, YAML
# quoting, set_pose not perturbing a constrained joint's state).
# Run with: docker run --rm -v "<this folder>:/ws:ro" dense-armor-robotics bash /ws/full_live_loop.sh
set -e
source /opt/ros/humble/setup.bash

echo "[$(date +%T)] starting headless Gazebo (ign gazebo, real physics)"
ign gazebo -s -r empty.sdf > /tmp/gz.log 2>&1 &
sleep 6

echo "[$(date +%T)] spawning rrbot with a real perturbed initial joint1 angle (rpy=0.7 rad on joint1's origin)"
xacro /ws/rrbot_perturbed.xacro > /tmp/rrbot.urdf
ros2 run robot_state_publisher robot_state_publisher /tmp/rrbot.urdf > /tmp/rsp.log 2>&1 &
sleep 3
timeout 20 ros2 run ros_gz_sim create -name rrbot -topic robot_description
sleep 3

echo "[$(date +%T)] bridging real joint_states (Gazebo -> ROS2)"
ros2 run ros_gz_bridge parameter_bridge \
  /world/empty/model/rrbot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model \
  --ros-args -r /world/empty/model/rrbot/joint_state:=/joint_states > /tmp/bridge.log 2>&1 &
sleep 8

echo "[$(date +%T)] === running the real dense_armor StreamingDeviationDetector live ==="
python3 /ws/live_detector_check.py

echo "[$(date +%T)] done"
