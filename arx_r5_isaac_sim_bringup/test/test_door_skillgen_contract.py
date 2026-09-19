# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Pure contract tests for the ARX door SkillGen environment boundary."""

from pathlib import Path

import numpy as np
import pytest

from arx_r5_isaac_sim_bringup.door_skillgen.contract import (
    DoorSkillGenContract,
    DoorSuccessThresholds,
    evaluate_door_success,
    first_batch_placements,
    monotonic_signal,
)
from arx_r5_isaac_sim_bringup.door_skillgen.seed import load_seed
from arx_r5_isaac_sim_bringup.door_teaching_kinematics import ArmKinematics


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / 'generated/door_teaching/upright02/candidate_006'
URDF = (
    ROOT.parent
    / 'arx-r5-moveit'
    / 'isaac_ros_manipulation_arx_r5a_robot_description'
    / 'urdf/r5a_cumotion.urdf'
)


@pytest.fixture(scope='module')
def seed():
    """Return the reviewed demonstration used by the first batch."""
    return load_seed(CANDIDATE)


@pytest.fixture(scope='module')
def contract(seed):
    """Build the pure joint/pose conversion contract."""
    return DoorSkillGenContract.from_metadata(
        ArmKinematics(URDF),
        seed.metadata,
    )


def test_action_pose_round_trip_preserves_seed_branch(contract, seed):
    """Seeded IK must recover the demonstrated joint branch."""
    action = np.asarray(seed.action[1505], dtype=float)
    pose = contract.action_to_target_pose(action)
    recovered = contract.target_pose_to_action(pose, action)
    np.testing.assert_allclose(recovered[:6], action[:6], atol=5e-4)
    assert recovered[6] == pytest.approx(action[6])


def test_first_batch_has_fixed_order_radius_height_and_facing(seed):
    """The approved five positions are deterministic and face the handle."""
    placements = first_batch_placements(seed.metadata)
    assert [item.angle_deg for item in placements] == [0.0, -3.0, 3.0, -5.0, 5.0]
    handle = np.asarray(seed.metadata['scene_placement']['handle_world_xyz'])
    for placement in placements:
        base = np.asarray(placement.position)
        assert np.linalg.norm(base[:2] - handle[:2]) == pytest.approx(0.59)
        assert base[2] == pytest.approx(0.63)
        yaw = 2 * np.arctan2(
            placement.quaternion_xyzw[2],
            placement.quaternion_xyzw[3],
        )
        forward = np.asarray((np.cos(yaw), np.sin(yaw)))
        assert np.dot(handle[:2] - base[:2], forward) == pytest.approx(0.59)


def test_success_requires_open_door_contacts_clearance_and_no_arm_contact():
    """Every physical guard contributes a named failure reason."""
    thresholds = DoorSuccessThresholds()
    valid = evaluate_door_success(
        door_angle_rad=np.deg2rad(30.1),
        finger_forces=np.asarray(((0, -4, 0), (0, 4, 0))),
        panel_normal=np.asarray((0, 1, 0)),
        camera_clearance_m=0.02,
        arm_contact_forces=np.zeros((6, 3)),
        thresholds=thresholds,
    )
    assert valid.success
    blocked = evaluate_door_success(
        door_angle_rad=np.deg2rad(28.9),
        finger_forces=np.asarray(((0, -2, 0), (0, 4, 0))),
        panel_normal=np.asarray((0, 1, 0)),
        camera_clearance_m=0.005,
        arm_contact_forces=np.asarray(((50, 0, 0),) * 6),
        thresholds=thresholds,
    )
    assert not blocked.success
    assert set(blocked.reasons) == {
        'door_angle_below_threshold',
        'finger_contact_below_threshold',
        'finger_normal_force_below_threshold',
        'camera_clearance_below_threshold',
        'arm_contact_above_threshold',
    }


def test_success_uses_pilot_clearance_and_normal_force():
    """Tangential contact cannot replace the reviewed opposed squeeze."""
    metrics = dict(
        door_angle_rad=np.deg2rad(30),
        panel_normal=np.asarray((0, 1, 0)),
        camera_clearance_m=0.012,
        arm_contact_forces=np.zeros((6, 3)),
    )
    assert evaluate_door_success(
        finger_forces=np.asarray(((0, -4, 0), (0, 4, 0))), **metrics,
    ).success
    rejected = evaluate_door_success(
        finger_forces=np.asarray(((8, -0.1, 0), (8, 0.1, 0))), **metrics,
    )
    assert 'finger_normal_force_below_threshold' in rejected.reasons


def test_subtask_signals_are_binary_and_monotonic():
    """Mimic segmentation signals transition from false to true once."""
    signal = monotonic_signal(12, 7)
    np.testing.assert_array_equal(signal[:7], 0)
    np.testing.assert_array_equal(signal[7:], 1)
    with pytest.raises(ValueError):
        monotonic_signal(3, 4)
