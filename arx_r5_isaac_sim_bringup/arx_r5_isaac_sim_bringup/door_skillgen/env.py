# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Native Isaac Lab Mimic environment for the ARX door task."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from arx_r5_isaac_sim_bringup.door_teaching_kinematics import ArmKinematics

from .collision import DynamicDoorCuboids


ARM_NAMES = [f'joint{index}' for index in range(1, 7)]
DOOR_BODIES = ('door_panel', 'door_handle', 'latch_link')


class DoorSkillGenEnv(ManagerBasedRLMimicEnv):
    """Expose ARX absolute joint control through Isaac Lab Mimic's pose API."""

    def __init__(self, cfg, **kwargs) -> None:
        super().__init__(cfg=cfg, **kwargs)
        if not cfg.urdf_path or not Path(cfg.urdf_path).is_file():
            raise FileNotFoundError(
                'ARX_DOOR_SKILLGEN_URDF must name the R5A cuMotion URDF'
            )
        self.kinematics = ArmKinematics(cfg.urdf_path)
        self._dynamic_cuboids: DynamicDoorCuboids | None = None
        self._start_signal_latch = {
            name: torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            for name in ('knob_interaction', 'edge_interaction')
        }
        self._term_signal_latch = {
            name: torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            for name in ('knob_interaction', 'edge_interaction')
        }

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        for values in (*self._start_signal_latch.values(), *self._term_signal_latch.values()):
            values[env_ids] = False
        self._dynamic_cuboids = None

    def _body_pose_tensor(
        self,
        asset_name: str,
        body_name: str,
        env_ids: Sequence[int] | slice,
    ) -> torch.Tensor:
        asset = self.scene[asset_name]
        body_ids, names = asset.find_bodies([body_name], preserve_order=True)
        if names != [body_name]:
            raise ValueError(f'body {body_name!r} is missing from {asset_name}')
        state = asset.data.body_link_state_w[env_ids, body_ids[0]]
        return PoseUtils.make_pose(
            state[..., :3],
            PoseUtils.matrix_from_quat(state[..., 3:7]),
        )

    def get_robot_eef_pose(
        self,
        eef_name: str,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return actual world-frame link6 poses."""
        if eef_name != 'link6':
            raise KeyError(f'unknown ARX end effector {eef_name!r}')
        selected = slice(None) if env_ids is None else env_ids
        return self._body_pose_tensor('robot', 'link6', selected)

    def get_robot_base_pose(self, env_id: int = 0) -> np.ndarray:
        """Return world-from-base_link for the cuMotion adapter."""
        robot = self.scene['robot']
        state = robot.data.root_link_state_w[int(env_id)]
        matrix = PoseUtils.make_pose(
            state[:3].unsqueeze(0),
            PoseUtils.matrix_from_quat(state[3:7].unsqueeze(0)),
        )[0]
        return matrix.detach().cpu().numpy()

    def get_arm_joint_positions(self, env_id: int = 0) -> np.ndarray:
        """Return six current joints in the native ARX order."""
        robot = self.scene['robot']
        ids, _ = robot.find_joints(ARM_NAMES, preserve_order=True)
        return robot.data.joint_pos[int(env_id), ids].detach().cpu().numpy()

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Use seeded IK and retain the single active gripper coordinate."""
        if set(target_eef_pose_dict) != {'link6'}:
            raise ValueError('ARX door generation requires exactly link6')
        target = target_eef_pose_dict['link6'].detach().cpu().numpy().copy()
        if target.shape != (4, 4):
            raise ValueError('target link6 pose must have shape (4, 4)')
        noise = 0.0
        if action_noise_dict is not None:
            noise = float(action_noise_dict.get('link6', 0.0))
        if noise:
            target[:3, 3] += np.random.normal(0.0, noise, size=3)
            delta = Rotation.from_rotvec(np.random.normal(0.0, noise, size=3))
            target[:3, :3] = delta.as_matrix() @ target[:3, :3]
        target_base = np.linalg.inv(self.get_robot_base_pose(env_id)) @ target
        seed = self.get_arm_joint_positions(env_id)
        arm = self.kinematics.ik(target_base, seed)
        gripper = gripper_action_dict['link6']
        grip = float(gripper.detach().cpu().reshape(-1)[0])
        command = torch.as_tensor(
            np.concatenate((arm, [grip])),
            dtype=torch.float32,
            device=self.device,
        )
        return command.unsqueeze(0)

    def action_to_target_eef_pose(
        self,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map batched 7-D absolute joint actions to world link6 poses."""
        commands = action.detach().cpu().numpy()
        if commands.ndim != 2 or commands.shape[1] != 7:
            raise ValueError('ARX action batch must have shape (N, 7)')
        if commands.shape[0] == self.num_envs:
            bases = [self.get_robot_base_pose(index) for index in range(self.num_envs)]
        elif self.num_envs == 1:
            bases = [self.get_robot_base_pose(0)] * commands.shape[0]
        else:
            raise ValueError('action batch cannot be mapped to environment bases')
        poses = np.stack(
            [base @ self.kinematics.fk(command[:6]) for base, command in zip(bases, commands)]
        )
        return {
            'link6': torch.as_tensor(
                poses,
                dtype=torch.float32,
                device=self.device,
            )
        }

    def actions_to_gripper_actions(
        self,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract the final coordinate without changing batch dimensions."""
        if actions.shape[-1] != 7:
            raise ValueError('ARX actions must end in seven coordinates')
        return {'link6': actions[..., -1:]}

    def get_object_poses(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return actual articulated reference body poses in world frame."""
        selected = slice(None) if env_ids is None else env_ids
        knob = self._body_pose_tensor('door', 'door_handle', selected)
        # The approved seed uses grasp_target, 67 mm in front of the body
        # origin. Both source and live frames must refer to that same point.
        offset = torch.tensor((0.0, -0.067, 0.0), device=self.device)
        knob[:, :3, 3] += torch.matmul(knob[:, :3, :3], offset)
        return {
            'knob': knob,
            'panel': self._body_pose_tensor('door', 'door_panel', selected),
        }

    def _update_signals(self) -> None:
        door = self.scene['door']
        hinge_ids, _ = door.find_joints(['hinge_joint'], preserve_order=True)
        hinge = door.data.joint_pos[:, hinge_ids[0]:hinge_ids[0] + 1]
        link6 = self.get_robot_eef_pose('link6')[:, :3, 3]
        knob = self.get_object_poses()['knob'][:, :3, 3]
        gripper_ids, _ = self.scene['robot'].find_joints(['joint7'], preserve_order=True)
        gripper = self.scene['robot'].data.joint_pos[:, gripper_ids[0]:gripper_ids[0] + 1]
        self._start_signal_latch['knob_interaction'] |= (
            torch.linalg.vector_norm(link6 - knob, dim=-1, keepdim=True) < 0.22
        )
        self._term_signal_latch['knob_interaction'] |= hinge >= torch.deg2rad(
            torch.tensor(10.0, device=self.device)
        )
        self._start_signal_latch['edge_interaction'] |= (
            self._term_signal_latch['knob_interaction'] & (gripper >= 0.04)
        )
        self._term_signal_latch['edge_interaction'] |= hinge >= torch.deg2rad(
            torch.tensor(29.0, device=self.device)
        )

    def get_subtask_start_signals(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return monotonic physical start signals for future annotations."""
        self._update_signals()
        selected = slice(None) if env_ids is None else env_ids
        return {key: value[selected] for key, value in self._start_signal_latch.items()}

    def get_subtask_term_signals(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return monotonic door-angle completion signals."""
        self._update_signals()
        selected = slice(None) if env_ids is None else env_ids
        return {key: value[selected] for key, value in self._term_signal_latch.items()}

    def get_expected_attached_object(self, *unused) -> None:
        """The articulated door must never become a rigid planner attachment."""
        return None

    def _door_body_world_numpy(self, env_id: int) -> dict[str, np.ndarray]:
        return {
            name: self._body_pose_tensor('door', name, [env_id])[0]
            .detach()
            .cpu()
            .numpy()
            for name in DOOR_BODIES
        }

    def export_cumotion_cuboids(self, env_id: int = 0) -> list[dict]:
        """Refresh closed-scene cuboids from current articulated body poses."""
        if not self.cfg.cuboids_path:
            raise ValueError('ARX_DOOR_SKILLGEN_CUBOIDS is not configured')
        base = self.get_robot_base_pose(env_id)
        body = self._door_body_world_numpy(env_id)
        if self._dynamic_cuboids is None:
            self._dynamic_cuboids = DynamicDoorCuboids.from_file(
                self.cfg.cuboids_path,
                base,
                body,
            )
        return self._dynamic_cuboids.update(base, body)
