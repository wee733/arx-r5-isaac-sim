# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Runtime-independent contracts shared by the ARX door SkillGen tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from arx_r5_isaac_sim_bringup.door_workcell import BasePose, sample_base_pose


FIRST_BATCH_ANGLES_DEG = (0.0, -3.0, 3.0, -5.0, 5.0)


class InverseKinematics(Protocol):
    """Kinematic operations needed at the SkillGen action boundary."""

    def fk(self, joints: np.ndarray) -> np.ndarray:
        """Return base-to-end-effector pose for six arm joints."""

    def ik(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        attempts: int = 12,
    ) -> np.ndarray:
        """Return six arm joints for a base-frame target pose."""


@dataclass(frozen=True)
class DoorSuccessThresholds:
    """Physical acceptance thresholds inherited from the reviewed pilot."""

    door_angle_deg: float = 29.0
    finger_force_n: float = 3.0
    camera_clearance_m: float = 0.01
    arm_contact_force_n: float = 40.0


@dataclass(frozen=True)
class DoorSuccess:
    """Structured acceptance decision retained for generated trials."""

    success: bool
    reasons: tuple[str, ...]


class DoorSkillGenContract:
    """Map absolute ARX joint actions to world-frame SkillGen poses."""

    def __init__(
        self,
        kinematics: InverseKinematics,
        base_world: np.ndarray,
    ) -> None:
        matrix = np.asarray(base_world, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('base_world must be a finite 4x4 matrix')
        if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-9):
            raise ValueError('base_world must be a homogeneous transform')
        self.kinematics = kinematics
        self.base_world = matrix.copy()

    @classmethod
    def from_metadata(
        cls,
        kinematics: InverseKinematics,
        metadata: Mapping[str, Any],
    ) -> 'DoorSkillGenContract':
        """Build the contract from an approved attempt manifest."""
        placement = metadata['scene_placement']['placement']
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            placement['quaternion_xyzw']
        ).as_matrix()
        matrix[:3, 3] = placement['position']
        return cls(kinematics, matrix)

    def action_to_target_pose(self, action: Sequence[float]) -> np.ndarray:
        """Convert one seven-dimensional joint target to a world pose."""
        command = _action(action)
        return self.base_world @ self.kinematics.fk(command[:6])

    def target_pose_to_action(
        self,
        target_world: np.ndarray,
        seed_action: Sequence[float],
    ) -> np.ndarray:
        """Solve a world pose while preserving the seed's gripper command."""
        target = np.asarray(target_world, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError('target_world must be a finite 4x4 matrix')
        seed = _action(seed_action)
        target_base = np.linalg.inv(self.base_world) @ target
        arm = self.kinematics.ik(target_base, seed[:6])
        return np.concatenate((np.asarray(arm, dtype=np.float64), seed[6:]))


def _action(value: Sequence[float]) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('action must contain seven finite joint targets')
    return action


def first_batch_placements(
    metadata: Mapping[str, Any],
    angles_deg: Sequence[float] = FIRST_BATCH_ANGLES_DEG,
) -> tuple[BasePose, ...]:
    """Return the explicitly ordered first-batch robot base placements."""
    scene = metadata['scene_placement']
    config = scene['config']
    return tuple(
        sample_base_pose(
            scene['handle_world_xyz'],
            radius_m=float(config['radius_m']),
            base_height_m=float(config['base_height_m']),
            half_angle_deg=float(config['half_angle_deg']),
            front_normal_xy=tuple(config['front_normal_xy']),
            yaw_mode=str(config['yaw_mode']),
            seed=int(scene.get('seed', 0)),
            episode_index=index,
            angle_deg=float(angle),
        )
        for index, angle in enumerate(angles_deg)
    )


def monotonic_signal(frame_count: int, transition_index: int) -> np.ndarray:
    """Create an Isaac Lab Mimic-compatible binary transition signal."""
    if frame_count <= 0:
        raise ValueError('frame_count must be positive')
    if not 0 <= transition_index < frame_count:
        raise ValueError('transition_index must reference an existing frame')
    signal = np.zeros((frame_count, 1), dtype=np.uint8)
    signal[transition_index:] = 1
    return signal


def evaluate_door_success(
    *,
    door_angle_rad: float,
    finger_forces: np.ndarray,
    panel_normal: np.ndarray,
    camera_clearance_m: float,
    arm_contact_forces: np.ndarray,
    thresholds: DoorSuccessThresholds | None = None,
) -> DoorSuccess:
    """Evaluate the physical guards used by the reviewed door pilot."""
    limits = thresholds or DoorSuccessThresholds()
    fingers = np.asarray(finger_forces, dtype=np.float64)
    normal = np.asarray(panel_normal, dtype=np.float64)
    arm = np.asarray(arm_contact_forces, dtype=np.float64)
    if fingers.shape != (2, 3):
        raise ValueError('finger_forces must have shape (2, 3)')
    if normal.shape != (3,) or np.linalg.norm(normal) <= 0:
        raise ValueError('panel_normal must be a nonzero three-vector')
    if arm.ndim != 2 or arm.shape[1:] != (3,):
        raise ValueError('arm_contact_forces must have shape (N, 3)')
    values = np.concatenate(
        (
            np.asarray((door_angle_rad, camera_clearance_m)),
            fingers.reshape(-1),
            normal,
            arm.reshape(-1),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError('success metrics must be finite')

    reasons: list[str] = []
    if np.rad2deg(door_angle_rad) < limits.door_angle_deg:
        reasons.append('door_angle_below_threshold')
    if float(np.min(np.linalg.norm(fingers, axis=1))) < limits.finger_force_n:
        reasons.append('finger_contact_below_threshold')
    unit_normal = normal / np.linalg.norm(normal)
    components = fingers @ unit_normal
    if np.min(np.abs(components)) < limits.finger_force_n:
        reasons.append('finger_normal_force_below_threshold')
    if components[0] * components[1] >= 0:
        reasons.append('finger_contact_not_opposed')
    if camera_clearance_m < limits.camera_clearance_m:
        reasons.append('camera_clearance_below_threshold')
    if arm.size and float(np.max(np.linalg.norm(arm, axis=1))) > limits.arm_contact_force_n:
        reasons.append('arm_contact_above_threshold')
    return DoorSuccess(not reasons, tuple(reasons))
