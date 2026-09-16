# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Update exported cuMotion cuboids from live articulated door body poses."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


BODY_PREFIXES = {
    '/World/Door/door_panel/': 'door_panel',
    '/World/Door/door_handle/': 'door_handle',
    '/World/Door/latch_link/': 'latch_link',
}


def pose_matrix(
    position: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Build a finite homogeneous matrix from a ROS-order pose."""
    position_array = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position_array.shape != (3,) or quaternion.shape != (4,):
        raise ValueError('position/quaternion must have shapes (3,) and (4,)')
    if not np.isfinite(position_array).all() or not np.isfinite(quaternion).all():
        raise ValueError('pose must be finite')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = position_array
    return result


def _cuboid_pose(cuboid: Mapping[str, object]) -> np.ndarray:
    return pose_matrix(cuboid['center'], cuboid['quaternion_xyzw'])


@dataclass
class DynamicDoorCuboids:
    """Retain static geometry and move door-linked cuboids with their bodies."""

    cuboids: list[dict]
    initial_base_world: np.ndarray
    initial_body_world: dict[str, np.ndarray]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        base_world: np.ndarray,
        body_world: Mapping[str, np.ndarray],
    ) -> 'DynamicDoorCuboids':
        """Load one validated, closed-scene exporter result."""
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        cuboids = payload.get('cuboids')
        if not isinstance(cuboids, list) or not cuboids:
            raise ValueError('cuMotion collision file contains no cuboids')
        for index, cuboid in enumerate(cuboids):
            if not isinstance(cuboid, dict):
                raise ValueError(f'cuboid {index} must be an object')
            _cuboid_pose(cuboid)
            size = np.asarray(cuboid.get('size'), dtype=np.float64)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError(f'cuboid {index} has invalid size')
        return cls(
            deepcopy(cuboids),
            np.asarray(base_world, dtype=np.float64).copy(),
            {
                key: np.asarray(value, dtype=np.float64).copy()
                for key, value in body_world.items()
            },
        )

    def update(
        self,
        base_world: np.ndarray,
        body_world: Mapping[str, np.ndarray],
    ) -> list[dict]:
        """Return current base-frame cuboids without mutating source geometry."""
        base_now = np.asarray(base_world, dtype=np.float64)
        base_now_inverse = np.linalg.inv(base_now)
        output: list[dict] = []
        for source in self.cuboids:
            name = str(source['name'])
            body_name = next(
                (body for prefix, body in BODY_PREFIXES.items() if name.startswith(prefix)),
                None,
            )
            cuboid_base_zero = _cuboid_pose(source)
            cuboid_world_zero = self.initial_base_world @ cuboid_base_zero
            if body_name is None:
                cuboid_world_now = cuboid_world_zero
            else:
                if body_name not in self.initial_body_world or body_name not in body_world:
                    raise ValueError(f'missing live pose for {body_name}')
                cuboid_world_now = (
                    np.asarray(body_world[body_name], dtype=np.float64)
                    @ np.linalg.inv(self.initial_body_world[body_name])
                    @ cuboid_world_zero
                )
            cuboid_base_now = base_now_inverse @ cuboid_world_now
            updated = deepcopy(source)
            updated['center'] = cuboid_base_now[:3, 3].tolist()
            updated['quaternion_xyzw'] = Rotation.from_matrix(
                cuboid_base_now[:3, :3]
            ).as_quat().tolist()
            output.append(updated)
        return output


__all__ = ['DynamicDoorCuboids', 'pose_matrix']
