#!/bin/bash
set -e
apt-get update -qq
apt-get install -y -qq python3-pip >/dev/null 2>&1
pip3 install /dense-armor-wheel/dense_armor-1.1.13-py3-none-any.whl 2>&1 | tail -10
echo "--- TEST IMPORT REALE ---"
python3 /ws/src/dense_armor_ros/test/live_container_check.py
echo "--- COLCON BUILD ---"
cd /ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install 2>&1
echo "--- TEST IMPORT NODO VIA WORKSPACE ---"
source install/setup.bash
python3 -c "from dense_armor_ros.joint_deviation_node import JointDeviationNode; print('NODO IMPORTABILE E COSTRUITO CORRETTAMENTE')"
echo "--- LIVE rclpy.spin() TEST (fake publisher -> node -> flag collector) ---"
python3 /ws/src/dense_armor_ros/test/spin_live_test.py
