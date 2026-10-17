# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Absolute ARX joint actions for the official AutoData embodiment interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter

from arx_door.kinematics import ArmKinematics
from arx_door.skillgen.compat import as_torch


PLAN_Q_KEY = '_arx_plan_q'
PLAN_GRIPPER_KEY = '_arx_plan_gripper'
TERMINAL_KEY = '_arx_terminal'
ARM_NAMES = [f'joint{index}' for index in range(1, 7)]
POSITION_TOLERANCE_M = 0.0005
ROTATION_TOLERANCE_RAD = 0.005


def numpy_value(value: Any) -> np.ndarray:
    """Convert a CPU/CUDA tensor to host numerical data explicitly."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def checked_pose(value: Any) -> np.ndarray:
    """Require a finite rigid homogeneous transform, in metres."""
    pose = numpy_value(value)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('pose must be a finite 4x4 matrix')
    if not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-7, rtol=0.):
        raise ValueError('pose must have a homogeneous bottom row')
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.) or not np.isclose(
        np.linalg.det(rotation), 1., atol=1e-5, rtol=0.,
    ):
        raise ValueError('pose rotation must be proper and orthonormal')
    return pose


def validate_joints(value: Any, kinematics: ArmKinematics) -> np.ndarray:
    """Require exactly six finite ARX joint targets within the URDF limits [rad]."""
    joints = numpy_value(value)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError('arm command must contain six finite joints')
    if np.any(joints < kinematics.lower) or np.any(joints > kinematics.upper):
        raise ValueError('arm command is outside URDF joint limits')
    return joints


def validate_fk(joints: np.ndarray, target_base: np.ndarray, kinematics: ArmKinematics) -> None:
    """Reject a plan/pose pair that disagrees by more than 0.5 mm or 0.005 rad."""
    actual = kinematics.fk(joints)
    position_error = np.linalg.norm(actual[:3, 3] - target_base[:3, 3])
    angle_error = np.linalg.norm(Rotation.from_matrix(target_base[:3, :3].T @ actual[:3, :3]).as_rotvec())
    if position_error > POSITION_TOLERANCE_M or angle_error > ROTATION_TOLERANCE_RAD:
        raise ValueError(f'FK mismatch: {position_error:.6f} m, {angle_error:.6f} rad')


def validate_plan_result(
    result: dict, start: np.ndarray, target_base: np.ndarray, kinematics: ArmKinematics,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the native worker's numerical trajectory before execution.

    This pure contract can run without CUDA or the motion-planner package. It
    validates declared collision checking; it does not establish that a solver
    actually performed those checks, which requires runtime acceptance.
    """
    if result.get('joint_names') != ARM_NAMES:
        raise ValueError('native planner joint names must be joint1..6 in order')
    if result.get('collision_checking') is not True:
        raise ValueError('native planner must report collision checking')
    if result.get('frame_id') != 'base_link' or result.get('tool_frame') != 'link6':
        raise ValueError('native planner must use base_link and link6 frames')
    positions = numpy_value(result.get('positions'))
    timestamps = numpy_value(result.get('timestamps'))
    if positions.ndim != 2 or positions.shape[1] != 6 or len(positions) < 1:
        raise ValueError('native positions must have nonempty shape (N, 6)')
    if timestamps.shape != (len(positions),) or not np.isfinite(timestamps).all():
        raise ValueError('native timestamps must contain one finite value per position')
    if timestamps[0] < 0. or (len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.)):
        raise ValueError('native timestamps must be nonnegative and strictly increasing')
    for joints in positions:
        validate_joints(joints, kinematics)
    if not np.allclose(positions[0], start, atol=1e-4, rtol=0.):
        raise ValueError('native plan does not start at the measured arm state')
    validate_fk(positions[-1], target_base, kinematics)
    return positions, timestamps


def require_zero_noise(value: Any) -> None:
    """The first migration supports zero noise in both transit and skill phases."""
    if value is None:
        return
    array = numpy_value(value)
    if not np.isfinite(array).all() or np.any(array != 0.):
        raise ValueError('ARX AutoData action noise must be zero')


def selected_env_ids(env: Any, env_ids: Sequence[int] | None) -> list[int]:
    """Validate scene indices without permitting negative cross-env indexing."""
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().tolist()
    selected = list(range(env.num_envs)) if env_ids is None else list(env_ids)
    if any(not isinstance(index, (int, np.integer)) or not 0 <= index < env.num_envs for index in selected):
        raise ValueError('env_id is outside the live environment batch')
    return selected


def state_to_env_pose(states: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
    """Convert pinned Lab 3 xyz/xyzw link states to env-relative transforms.

    Lab 91a0906 BaseArticulationData.root_link_pose_w/body_link_pose_w
    explicitly document XYZW. Legacy Lab 2 state ordering is not supported by
    this new fixed-runtime entry point.
    """
    if states.ndim != 2 or states.shape[-1] < 7 or not torch.isfinite(states[:, :7]).all():
        raise ValueError('link state must contain finite position and xyzw quaternion')
    # Use the true link frame rather than COM data; preserve the live device.
    quaternion = states[:, 3:7]
    if torch.any(torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1.) > 1e-3):
        raise ValueError('link quaternion must have unit norm')
    poses = torch.eye(4, dtype=states.dtype, device=states.device).repeat(len(states), 1, 1)
    if len(states) == 0:
        return poses
    rotations = Rotation.from_quat(numpy_value(quaternion)).as_matrix()
    poses[:, :3, :3] = torch.as_tensor(rotations, dtype=states.dtype, device=states.device)
    poses[:, :3, 3] = states[:, :3] - origins.to(device=states.device, dtype=states.dtype)
    return poses


def body_pose_env(env: Any, asset_name: str, body_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    """Read a live articulation link in its own environment's local frame."""
    selected = selected_env_ids(env, env_ids)
    asset = env.scene[asset_name]
    body_ids, names = asset.find_bodies([body_name], preserve_order=True)
    if list(names) != [body_name] or len(body_ids) != 1:
        raise ValueError(f'missing body {body_name!r} in {asset_name!r}')
    states = as_torch(asset.data.body_link_state_w)[selected, int(body_ids[0])]
    return state_to_env_pose(states, as_torch(env.scene.env_origins)[selected])


class ArxAdapter(EmbodimentAdapter):
    """Six absolute arm targets [rad] and one active gripper target [m].

    Planned joints arrive only in reserved passthrough metadata. They do not
    become extra action dimensions; skill segments use current-state seeded IK.
    """

    robot_asset_name = 'robot'

    def __init__(self, urdf_path: str | Path) -> None:
        self.env = None
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.kinematics = ArmKinematics(self.urdf_path)
        joint = ET.parse(self.urdf_path).getroot().find("joint[@name='joint7']")
        if joint is None or joint.get('type') != 'prismatic' or joint.find('limit') is None:
            raise ValueError('URDF joint7 must be the bounded prismatic gripper coordinate')
        limit = joint.find('limit')
        self.gripper_limits = float(limit.get('lower')), float(limit.get('upper'))
        if not np.isfinite(self.gripper_limits).all() or self.gripper_limits[0] > self.gripper_limits[1]:
            raise ValueError('joint7 limits must be finite and ordered')

    def get_eef_names(self) -> tuple[str, ...]:
        return ('link6',)

    def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        assert self.env is not None, 'Call bind_env(env) before reading state.'
        return {'link6': body_pose_env(self.env, self.robot_asset_name, 'link6', env_ids)}

    def get_joint_positions(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Unwrap Lab 3 ProxyArray before indexing the live joint batch."""
        assert self.env is not None, 'Call bind_env(env) before reading state.'
        selected = selected_env_ids(self.env, env_ids)
        return as_torch(self.env.scene[self.robot_asset_name].data.joint_pos)[selected]

    def get_base_poses(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Read env-relative base_link poses, never articulation COM poses."""
        assert self.env is not None, 'Call bind_env(env) before reading state.'
        selected = selected_env_ids(self.env, env_ids)
        states = as_torch(self.env.scene[self.robot_asset_name].data.root_link_state_w)[selected]
        return state_to_env_pose(states, as_torch(self.env.scene.env_origins)[selected])

    def get_named_joint_positions(self, names: Sequence[str], env_id: int) -> np.ndarray:
        """Read measured joints in the requested explicit order."""
        assert self.env is not None, 'Call bind_env(env) before reading state.'
        selected_env_ids(self.env, [env_id])
        robot = self.env.scene[self.robot_asset_name]
        ids, actual_names = robot.find_joints(list(names), preserve_order=True)
        if list(actual_names) != list(names) or len(ids) != len(names):
            raise ValueError(f'ARX joint order mismatch: expected {list(names)}, got {actual_names}')
        values = numpy_value(as_torch(robot.data.joint_pos)[env_id, ids])
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError('measured joint positions must be finite')
        return values

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.env is not None, 'Call bind_env(env) before converting actions.'
        commands = numpy_value(action)
        if commands.ndim != 2 or commands.shape[1] != 7:
            raise ValueError('ARX action batch must have shape (N, 7)')
        if len(commands) == self.env.num_envs:
            bases = numpy_value(self.get_base_poses())
        elif self.env.num_envs == 1:
            bases = np.repeat(numpy_value(self.get_base_poses()), len(commands), axis=0)
        else:
            raise ValueError('action batch cannot be mapped to environment bases')
        for command in commands:
            validate_joints(command[:6], self.kinematics)
            self._gripper(command[6:])
        poses = np.stack([base @ self.kinematics.fk(command[:6]) for base, command in zip(bases, commands)])
        return {'link6': torch.as_tensor(poses, dtype=action.dtype, device=action.device)}

    def _gripper(self, value: Any) -> float:
        array = numpy_value(value)
        if array.size != 1 or not np.isfinite(array).all():
            raise ValueError('gripper passthrough must contain one finite position [m]')
        grip = float(array.reshape(-1)[0])
        if not self.gripper_limits[0] <= grip <= self.gripper_limits[1]:
            raise ValueError('joint7 gripper command is outside URDF limits [m]')
        return grip

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        passthrough_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        assert self.env is not None, 'Call bind_env(env) before converting actions.'
        selected_env_ids(self.env, [env_id])
        if set(target_eef_pose_dict) != {'link6'}:
            raise ValueError('ARX requires exactly the link6 end effector')
        for noise in (action_noise_dict or {}).values():
            require_zero_noise(noise)
        target = checked_pose(target_eef_pose_dict['link6'])
        base = numpy_value(self.get_base_poses([env_id])[0])
        target_base = np.linalg.inv(base) @ target
        has_plan_joints = PLAN_Q_KEY in passthrough_action_dict
        has_plan_gripper = PLAN_GRIPPER_KEY in passthrough_action_dict
        if has_plan_joints != has_plan_gripper:
            raise ValueError('planned joint/gripper metadata must be present together')
        grip = self._gripper(passthrough_action_dict['link6'])
        if has_plan_joints:
            planned_gripper = numpy_value(passthrough_action_dict[PLAN_GRIPPER_KEY])
            if planned_gripper.shape != (2,) or not np.isfinite(planned_gripper).all():
                raise ValueError('planned gripper metadata must contain joint7/joint8 [m]')
            for position in planned_gripper:
                self._gripper(position)
            if abs(grip - planned_gripper[0]) > 1e-8:
                raise ValueError('command does not match the planned gripper collision model')
            measured_gripper = self.get_named_joint_positions(['joint7', 'joint8'], env_id)
            # Match the worker's 0.1 mm auxiliary mimic-state tolerance. A
            # larger live drift invalidates the frozen collision model; never
            # continue by silently switching to source gripper values or IK.
            if np.any(np.abs(measured_gripper - planned_gripper) > 1e-4 + 1e-8):
                raise ValueError('stale planned gripper model: measured fingers drifted by more than 0.1 mm')
            arm = validate_joints(passthrough_action_dict[PLAN_Q_KEY], self.kinematics)
        else:
            seed = self.get_named_joint_positions(ARM_NAMES, env_id)
            arm = validate_joints(self.kinematics.ik(target_base, seed), self.kinematics)
        validate_fk(arm, target_base, self.kinematics)
        terminal = numpy_value(passthrough_action_dict.get(TERMINAL_KEY, False))
        if terminal.size != 1 or not np.isfinite(terminal).all() or not np.isin(terminal, [0., 1.]).all():
            raise ValueError('terminal metadata must contain one boolean')
        if not hasattr(self.env, '_arx_terminal'):
            self.env._arx_terminal = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.env._arx_terminal[env_id] = bool(terminal.reshape(-1)[0])
        return torch.as_tensor(np.r_[arm, grip], dtype=torch.float32, device=self.env.device)

    def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.ndim < 2 or actions.shape[-1] != 7 or not torch.isfinite(actions).all():
            raise ValueError('ARX actions must end in seven finite coordinates')
        return {'link6': actions[..., 6:7]}
