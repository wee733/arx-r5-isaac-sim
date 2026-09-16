# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the native Isaac Lab ARX door SkillGen environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import image, joint_pos, joint_vel, last_action, time_out
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


ARM_JOINT_NAMES = tuple(f'joint{index}' for index in range(1, 7))
GRIPPER_JOINT_NAMES = ('joint7', 'joint8')
PANEL_BOX_CENTER = (0.3795, 0.0228, 0.915)
PANEL_BOX_HALF_SIZE = (0.3625, 0.019, 0.395)


class MirroredGripperAction(ActionTerm):
    """Preserve a 7-D external action while driving joint7 and joint8."""

    cfg: MirroredGripperActionCfg

    def __init__(self, cfg: MirroredGripperActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(
            list(cfg.joint_names),
            preserve_order=True,
        )
        if tuple(names) != tuple(cfg.joint_names):
            raise ValueError(
                f'expected gripper joints {cfg.joint_names}, resolved {names}'
            )
        self._raw_actions = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f'gripper action must have shape {self._raw_actions.shape}'
            )
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, 0.0, 0.044)

    def apply_actions(self) -> None:
        targets = self._processed_actions.expand(-1, 2)
        self._asset.set_joint_position_target(
            targets,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class MirroredGripperActionCfg(ActionTermCfg):
    """One action value mapped to both physical finger joints."""

    class_type: type[ActionTerm] = MirroredGripperAction
    joint_names: tuple[str, str] = GRIPPER_JOINT_NAMES


def robot_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'),
) -> torch.Tensor:
    """Return six arm joints plus the active joint7 coordinate."""
    robot = env.scene[asset_cfg.name]
    ids, _ = robot.find_joints([*ARM_JOINT_NAMES, 'joint7'], preserve_order=True)
    return robot.data.joint_pos[:, ids]


def robot_velocity(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg('robot'),
) -> torch.Tensor:
    """Return velocity in the same seven-dimensional order as state."""
    robot = env.scene[asset_cfg.name]
    ids, _ = robot.find_joints([*ARM_JOINT_NAMES, 'joint7'], preserve_order=True)
    return robot.data.joint_vel[:, ids]


def door_joint_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg('door'),
) -> torch.Tensor:
    """Return hinge, handle, and latch coordinates for diagnostics."""
    door = env.scene[asset_cfg.name]
    ids, _ = door.find_joints(
        ['hinge_joint', 'handle_joint', 'latch_joint'],
        preserve_order=True,
    )
    return door.data.joint_pos[:, ids]


def finger_contacts(env) -> torch.Tensor:
    """Return (env, finger, object, xyz) forces; object order is knob/panel."""
    return torch.stack([
        env.scene.sensors[name].data.force_matrix_w[:, 0, [1, 0]]
        for name in ('left_panel_contact', 'right_panel_contact')
    ], dim=1)


def arm_contact_forces(env) -> torch.Tensor:
    """Return actual non-finger contact forces for trajectory auditing."""
    return env.scene.sensors['arm_contacts'].data.net_forces_w


def _body_pose(asset, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids, names = asset.find_bodies([body_name], preserve_order=True)
    if names != [body_name]:
        raise ValueError(f'body {body_name!r} is missing from {asset.cfg.prim_path}')
    state = asset.data.body_link_state_w[:, ids[0]]
    return state[:, :3], state[:, 3:7]


def camera_panel_clearance(env) -> torch.Tensor:
    """Signed virtual wrist-viewpoint clearance to the panel OBB."""
    panel_pos, panel_quat = _body_pose(env.scene['door'], 'door_panel')
    camera_pos = env.scene.sensors['wrist_camera'].data.pos_w
    center_local = torch.tensor(
        PANEL_BOX_CENTER,
        dtype=panel_pos.dtype,
        device=panel_pos.device,
    ).expand_as(panel_pos)
    center_world = panel_pos + math_utils.quat_apply(panel_quat, center_local)
    point_local = math_utils.quat_apply_inverse(
        panel_quat,
        camera_pos - center_world,
    )
    half = torch.tensor(
        PANEL_BOX_HALF_SIZE,
        dtype=panel_pos.dtype,
        device=panel_pos.device,
    )
    delta = torch.abs(point_local) - half
    outside = torch.linalg.vector_norm(torch.clamp_min(delta, 0.0), dim=-1)
    inside = torch.minimum(torch.max(delta, dim=-1).values, torch.zeros_like(outside))
    return (outside + inside).unsqueeze(-1)


def door_gap_reached(env, angle_deg: float = 10.0) -> torch.Tensor:
    """Signal completion of the knob/gap interaction macro skill."""
    return door_joint_state(env)[:, 0:1] >= torch.deg2rad(
        torch.tensor(angle_deg, device=env.device)
    )


def door_opened(env, angle_deg: float = 29.0) -> torch.Tensor:
    """Signal that the requested physical opening was achieved."""
    return door_joint_state(env)[:, 0:1] >= torch.deg2rad(
        torch.tensor(angle_deg, device=env.device)
    )


def door_success(
    env,
    *,
    door_angle_deg: float = 29.0,
    finger_force_n: float = 3.0,
    camera_clearance_m: float = 0.01,
    arm_contact_force_n: float = 40.0,
) -> torch.Tensor:
    """Apply the reviewed physical acceptance gates to every environment."""
    opened = door_opened(env, door_angle_deg).squeeze(-1)
    left = env.scene.sensors['left_panel_contact'].data.force_matrix_w[:, 0, 0]
    right = env.scene.sensors['right_panel_contact'].data.force_matrix_w[:, 0, 0]
    fingers = torch.stack((left, right), dim=1)
    contact = torch.linalg.vector_norm(fingers, dim=-1).amin(dim=1) >= finger_force_n

    _, panel_quat = _body_pose(env.scene['door'], 'door_panel')
    local_normal = torch.tensor(
        (0.0, 1.0, 0.0),
        dtype=panel_quat.dtype,
        device=panel_quat.device,
    ).expand(panel_quat.shape[0], 3)
    panel_normal = math_utils.quat_apply(panel_quat, local_normal)
    normal_force = torch.sum(fingers * panel_normal[:, None, :], dim=-1)
    opposed = (normal_force[:, 0] * normal_force[:, 1] < 0) & (
        normal_force.abs().amin(dim=1) >= finger_force_n
    )

    clearance = camera_panel_clearance(env).squeeze(-1) >= camera_clearance_m
    arm_forces = env.scene.sensors['arm_contacts'].data.net_forces_w
    arm_clear = torch.linalg.vector_norm(arm_forces, dim=-1).amax(dim=1) <= arm_contact_force_n
    return opened & contact & opposed & clearance & arm_clear


__all__ = [
    'MirroredGripperActionCfg',
    'camera_panel_clearance',
    'door_gap_reached',
    'door_joint_state',
    'door_opened',
    'door_success',
    'joint_pos',
    'joint_vel',
    'last_action',
    'image',
    'robot_state',
    'robot_velocity',
    'time_out',
]
