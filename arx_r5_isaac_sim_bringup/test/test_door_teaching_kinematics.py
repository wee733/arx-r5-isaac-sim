# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Check the imported ARX zero pose and bounded inverse kinematics."""
from pathlib import Path
import numpy as np
import pytest

URDF = Path(__file__).resolve().parents[3] / 'arx-r5-moveit' / (
    'isaac_ros_manipulation_arx_r5a_robot_description/urdf/r5a_cumotion.urdf')


def test_zero_pose_matches_usd_import():
    from arx_r5_isaac_sim_bringup.door_teaching_kinematics import ArmKinematics
    arm = ArmKinematics(URDF)
    pose = arm.fk(np.zeros(6))
    assert pose[:3, 3] == pytest.approx([0.1002, 0.0000004344, 0.1635], abs=1e-5)


def test_unreachable_goal_rejected():
    from arx_r5_isaac_sim_bringup.door_teaching_kinematics import ArmKinematics
    arm = ArmKinematics(URDF)
    target = np.eye(4)
    target[0, 3] = 10
    with pytest.raises(ValueError, match='IK'):
        arm.ik(target, np.array([0, 1, 1.5, 0, 0, 0]), attempts=2)
