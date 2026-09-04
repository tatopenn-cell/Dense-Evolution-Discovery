#!/bin/bash
set -e
echo "[$(date +%T)] apt update"
apt-get update -qq
echo "[$(date +%T)] apt install minimal packages (no rviz2/rqt)"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  ros-humble-ros-gz-sim ros-humble-ros-gz-bridge ros-humble-xacro ros-humble-robot-state-publisher >/dev/null 2>&1
echo "[$(date +%T)] apt install done"
source /opt/ros/humble/setup.bash

echo "[$(date +%T)] starting headless Gazebo (ign gazebo, server only)"
ign gazebo -s -r empty.sdf &
GZ_PID=$!
sleep 6
echo "[$(date +%T)] ign gazebo started, pid=$GZ_PID"

xacro /ws/rrbot.xacro > /tmp/rrbot.urdf
echo "[$(date +%T)] xacro processed -- starting robot_state_publisher to publish /robot_description"
ros2 run robot_state_publisher robot_state_publisher /tmp/rrbot.urdf &
RSP_PID=$!
sleep 3

echo "[$(date +%T)] spawning rrbot in the real simulation"
timeout 20 ros2 run ros_gz_sim create -name rrbot -topic robot_description
sleep 3

echo "[$(date +%T)] starting the real ros_gz bridge for joint_states"
ros2 run ros_gz_bridge parameter_bridge \
  /world/empty/model/rrbot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model \
  --ros-args -r /world/empty/model/rrbot/joint_state:=/joint_states &
BRIDGE_PID=$!
sleep 5

echo "[$(date +%T)] === REAL joint_states message #1 from live Gazebo physics ==="
timeout 10 ros2 topic echo /joint_states --once
sleep 2
echo "[$(date +%T)] === REAL joint_states message #2 (real physics evolving) ==="
timeout 10 ros2 topic echo /joint_states --once

kill $GZ_PID $RSP_PID $BRIDGE_PID 2>/dev/null || true
echo "[$(date +%T)] done"
