# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/test_ros2_deviation_logic.py
==============================================================
Real, executable tests (not mocked, not skipped) for
ros2_deviation_logic.py -- the only part of the ROS2 integration this
environment can actually run, since rclpy itself isn't installed here.
Run with: python -m pytest test_ros2_deviation_logic.py -v
(or just `python test_ros2_deviation_logic.py` for a plain run).
"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from multichannel import MultiChannelStreamingDeviationDetector
from ros2_deviation_logic import process_joint_positions


def test_wrong_length_returns_none():
    det = MultiChannelStreamingDeviationDetector(n_channels=6, radius=5, ref_mult=2, n_sigmas=3.0)
    assert process_joint_positions(det, [0.0] * 5) is None
    assert process_joint_positions(det, [0.0] * 7) is None


def test_correct_length_returns_list_of_0_1():
    det = MultiChannelStreamingDeviationDetector(n_channels=3, radius=5, ref_mult=2, n_sigmas=3.0)
    out = process_joint_positions(det, [0.0, 0.0, 0.0])
    assert out is not None
    assert len(out) == 3
    assert all(v in (0, 1) for v in out)


def test_matches_direct_detector_call_on_real_lerobot_data():
    """The real correctness check: feeding real 6-joint LeRobot data
    through process_joint_positions must match calling the detector
    directly -- same real data Experiment 49 was validated on."""
    from huggingface_hub import hf_hub_download
    data_root = pathlib.Path(__file__).resolve().parent.parent / "robot_sensor_validation" / "lerobot_data"
    parquet_path = hf_hub_download(
        repo_id="lerobot/svla_so101_pickplace", repo_type="dataset",
        filename="data/chunk-000/file-000.parquet", local_dir=str(data_root),
    )
    df = pd.read_parquet(parquet_path)
    sub = df[df.episode_index == 0].sort_values("frame_index")
    action = np.stack(sub["action"].values)  # (n, 6) real joint positions

    det_a = MultiChannelStreamingDeviationDetector(n_channels=6, radius=5, ref_mult=2, n_sigmas=3.0)
    det_b = MultiChannelStreamingDeviationDetector(n_channels=6, radius=5, ref_mult=2, n_sigmas=3.0)

    via_logic = [process_joint_positions(det_a, action[i].tolist()) for i in range(len(action))]
    via_direct = [[1 if f else 0 for f in det_b.update(action[i])] for i in range(len(action))]

    assert via_logic == via_direct, "process_joint_positions must match calling the detector directly"
    print(f"test_matches_direct_detector_call_on_real_lerobot_data: n={len(action)} frames, all match")


if __name__ == "__main__":
    test_wrong_length_returns_none()
    print("test_wrong_length_returns_none: PASSED")
    test_correct_length_returns_list_of_0_1()
    print("test_correct_length_returns_list_of_0_1: PASSED")
    test_matches_direct_detector_call_on_real_lerobot_data()
    print("test_matches_direct_detector_call_on_real_lerobot_data: PASSED")
