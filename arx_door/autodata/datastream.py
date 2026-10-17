# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Articulated door state and isolated collision worlds for AutoData."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from autodata_core.pool import DataGenInfoPool
from autodata_interfaces.datastream.datastream import Datastream
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor

from arx_door.autodata.adapter import ArxAdapter, ARM_NAMES, body_pose_env, numpy_value, selected_env_ids, state_to_env_pose
from arx_door.skillgen.collision import DynamicDoorCuboids
from arx_door.skillgen.compat import as_torch


DOOR_BODIES = ('door_panel', 'door_handle', 'latch_link')


class DoorDatastream(Datastream):
    """Use AutoData's source pool and task interface with articulated objects.

    Collision references are captured once, before generation randomizes the
    copied closed scene. Each environment owns a separate reference/updater.
    Resetting an episode must not redefine the authored cuboids' reference.
    """

    def __init__(
        self,
        env: Any,
        task_descriptor: TaskDescriptor,
        embodiment_adapter: ArxAdapter,
        source_pool: DataGenInfoPool | None = None,
        source_dataset_path: str | None = None,
        select_demo_keys: list[str] | None = None,
        asyncio_lock: asyncio.Lock | None = None,
        uses_start_signals: bool = False,
        *,
        cuboids_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            env=env,
            task_descriptor=task_descriptor,
            embodiment_adapter=embodiment_adapter,
            source_pool=source_pool,
            source_dataset_path=source_dataset_path,
            select_demo_keys=select_demo_keys,
            asyncio_lock=asyncio_lock,
            uses_start_signals=uses_start_signals,
        )
        self.cuboids_path: Path | None = None
        self._dynamic_cuboids: dict[int, DynamicDoorCuboids] = {}
        if cuboids_path is not None:
            self.configure_collision_world(cuboids_path)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Read rigid props plus real knob and panel articulation links [m]."""
        selected = selected_env_ids(self.env, env_ids)
        origins = as_torch(self.env.scene.env_origins)[selected]
        poses = {}
        # The pinned Lab 3 exposes ProxyArray; unwrap its .torch view before
        # indexing. The upstream generic reader only unwraps bare Warp arrays.
        for name, obj in self.env.scene.rigid_objects.items():
            position = as_torch(obj.data.root_pos_w)[selected]
            quaternion = as_torch(obj.data.root_quat_w)[selected]
            poses[name] = state_to_env_pose(torch.cat([position, quaternion], dim=-1), origins)
        knob = body_pose_env(self.env, 'door', 'door_handle', env_ids)
        offset = torch.tensor([0., -0.067, 0.], dtype=knob.dtype, device=knob.device)
        # Keep the original seed's grasp_target, 67 mm from the handle origin.
        knob[:, :3, 3] += knob[:, :3, :3] @ offset
        poses['knob'] = knob
        poses['panel'] = body_pose_env(self.env, 'door', 'door_panel', env_ids)
        return poses

    def get_robot_root_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return base_link (not COM) in the same env-relative frame as EEFs."""
        return self.embodiment_adapter.get_base_poses(env_ids)

    def get_arm_joint_positions(self, env_id: int = 0):
        """Read joint1..6 in native planner order [rad]."""
        return self.embodiment_adapter.get_named_joint_positions(ARM_NAMES, env_id)

    def get_gripper_joint_positions(self, env_id: int = 0):
        """Read actual active and passive fingers, joint7/joint8 [m]."""
        return self.embodiment_adapter.get_named_joint_positions(['joint7', 'joint8'], env_id)

    def get_door_body_poses(self, env_id: int = 0) -> dict:
        """Return numerical environment-local door link matrices."""
        return {
            name: numpy_value(body_pose_env(self.env, 'door', name, [env_id])[0])
            for name in DOOR_BODIES
        }

    def configure_collision_world(self, cuboids_path: str | Path) -> None:
        """Bind exported base-frame geometry to each copied closed scene.

        Call during environment setup, before randomization or replay. A later
        planner may repeat the same path, but cannot silently replace geometry.
        """
        path = Path(cuboids_path).expanduser().resolve()
        if self.cuboids_path is not None:
            if self.cuboids_path != path:
                raise ValueError('collision geometry cannot change after references were captured')
            return
        references = {}
        for env_id in selected_env_ids(self.env, None):
            references[env_id] = DynamicDoorCuboids.from_file(
                path,
                numpy_value(self.get_robot_root_pose([env_id])[0]),
                self.get_door_body_poses(env_id),
            )
        self._dynamic_cuboids = references
        self.cuboids_path = path

    def export_cumotion_cuboids(self, env_id: int = 0) -> list[dict]:
        """Refresh only the selected environment's base-frame collision world."""
        selected_env_ids(self.env, [env_id])
        if self.cuboids_path is None:
            raise ValueError('cuboids_path must be configured before generation')
        return self._dynamic_cuboids[env_id].update(
            numpy_value(self.get_robot_root_pose([env_id])[0]), self.get_door_body_poses(env_id),
        )
