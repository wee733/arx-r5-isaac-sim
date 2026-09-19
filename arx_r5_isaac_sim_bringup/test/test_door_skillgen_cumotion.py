# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native cuMotion adapter used by SkillGen."""

import numpy as np
import torch

from arx_r5_isaac_sim_bringup.door_skillgen.cumotion_planner import (
    DoorCuMotionPlanner,
)


class LinearKinematics:
    """Map the first three joints to a link6 translation."""

    def fk(self, joints):
        """Return one deterministic base-frame pose."""
        result = np.eye(4)
        result[:3, 3] = np.asarray(joints)[:3]
        return result

    def ik(self, target, seed):
        """Retain the seed branch while solving the translational fixture."""
        result = np.asarray(seed).copy()
        result[:3] = target[:3, 3]
        return result


class FakeRobotData:
    """Minimal articulation state used by the adapter."""

    joint_pos = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.028, 0.028]
    ])


class FakeRobot:
    """Minimal robot object accepted by MotionPlannerBase."""

    data = FakeRobotData()


class FakeEnv:
    """Expose the explicit planner boundary implemented by the door env."""

    device = 'cpu'

    def __init__(self):
        self.scene = {'robot': FakeRobot()}

    def get_robot_base_pose(self, env_id):
        """Return base-to-world with a non-zero origin."""
        assert env_id == 0
        result = torch.eye(4)
        result[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        return result

    def get_arm_joint_positions(self, env_id):
        """Return the six controlled joints in contract order."""
        return self.scene['robot'].data.joint_pos[env_id, :6]

    def export_cumotion_cuboids(self, env_id):
        """Return one complete collision primitive."""
        assert env_id == 0
        return [{
            'name': 'door_frame',
            'center': [0.0, 0.0, 1.0],
            'size': [1.0, 0.1, 2.0],
            'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0],
        }]


class FakeRunner:
    """Record requests and return an explicit native-worker result."""

    def __init__(self, success=True):
        self.requests = []
        self.success = success

    def plan(self, request):
        """Return two joint waypoints or one retained failure."""
        self.requests.append(request)
        if not self.success:
            return {
                'success': False,
                'status': 'INVALID_START_STATE',
                'reason': 'fixture failure',
                'positions': [],
            }
        return {
            'success': True,
            'status': 'SUCCESS',
            'positions': [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.2, 0.4, 0.6, 0.4, 0.5, 0.6],
            ],
            'timestamps': [0.0, 1.0 / 30.0],
        }


def target_world():
    """Return a target one metre along base X from the base origin."""
    result = torch.eye(4)
    result[:3, 3] = torch.tensor([2.0, 2.0, 3.0])
    return result


def test_planner_frames_request_and_iterates_fk_waypoints():
    """Wrong world/base conversion or waypoint FK must fail this test."""
    env = FakeEnv()
    runner = FakeRunner()
    planner = DoorCuMotionPlanner(
        env,
        env.scene['robot'],
        runner=runner,
        kinematics=LinearKinematics(),
    )

    assert planner.update_world_and_plan_motion(target_world())

    request = runner.requests[0]
    np.testing.assert_allclose(request['start_positions'], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(request['target_joints'], [1., 0., 0., .4, .5, .6])
    assert 'target_position' not in request
    np.testing.assert_allclose(request['target_pose_base']['position'], [1., 0., 0.])
    assert request['cuboids'][0]['name'] == 'door_frame'
    poses = planner.get_planned_poses()
    assert len(poses) == 2
    np.testing.assert_allclose(poses[1][:3, 3], [1.2, 2.4, 3.6])
    assert planner.has_next_waypoint()
    np.testing.assert_allclose(planner.get_next_waypoint_ee_pose()[:3, 3], [1.1, 2.2, 3.3])


def test_planner_retains_native_failure_and_clears_old_plan():
    """A failed replan must not execute waypoints left by an older plan."""
    env = FakeEnv()
    runner = FakeRunner()
    planner = DoorCuMotionPlanner(
        env,
        env.scene['robot'],
        runner=runner,
        kinematics=LinearKinematics(),
    )
    assert planner.update_world_and_plan_motion(target_world())
    runner.success = False

    assert not planner.update_world_and_plan_motion(target_world())

    assert not planner.has_next_waypoint()
    assert planner.get_planned_poses() == []
    assert planner.last_result['status'] == 'INVALID_START_STATE'
    assert planner.last_result['reason'] == 'fixture failure'


def test_planner_standoff_is_applied_along_tool_axis_without_mutating_target():
    """A moving-panel transfer ends outside the contact entrance."""
    env = FakeEnv()
    runner = FakeRunner()
    planner = DoorCuMotionPlanner(env, env.scene['robot'], runner=runner,
                                 kinematics=LinearKinematics())
    planner.approach_standoff_m = 0.04
    target = target_world()
    assert planner.update_world_and_plan_motion(target)
    np.testing.assert_allclose(runner.requests[0]['target_joints'][:3], [.96, 0, 0])
    np.testing.assert_allclose(target[:3, 3], [2, 2, 3])


def test_planner_can_use_a_free_space_intermediate_pose():
    """The transfer pose can have a different orientation from the skill."""
    env, runner = FakeEnv(), FakeRunner()
    planner = DoorCuMotionPlanner(env, env.scene['robot'], runner=runner,
                                 kinematics=LinearKinematics())
    intermediate = target_world().numpy().copy()
    intermediate[2, 3] += 0.02
    intermediate[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    planner.transition_goal_override = intermediate
    assert planner.update_world_and_plan_motion(target_world())
    np.testing.assert_allclose(runner.requests[0]['target_joints'][:3], [1., 0., .02], atol=1e-7)
    np.testing.assert_allclose(
        runner.requests[0]['target_pose_base']['quaternion_xyzw'],
        [0., 0., 2**-0.5, 2**-0.5], atol=1e-7,
    )
