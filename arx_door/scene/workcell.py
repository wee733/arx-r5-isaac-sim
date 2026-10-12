# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Reproducible planar base placement around a closed door's handle."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class BasePose:
    """World pose with ROS quaternion ordering (x, y, z, w)."""

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    angle_deg: float


def sample_base_pose(
    handle_xyz, *, radius_m=0.59, base_height_m=0.63,
    half_angle_deg=30.0, front_normal_xy=(0.0, -1.0),
    yaw_mode='face_handle', seed=0, episode_index=0, angle_deg=None,
):
    """Sample uniform arc angle, independently reproducible per episode.

    Positive angle is counterclockwise viewed from +Z. The radius is horizontal,
    not a 3D distance to the handle or a constant perpendicular door clearance.
    """
    values = (*handle_xyz, radius_m, base_height_m, half_angle_deg, *front_normal_xy)
    if len(handle_xyz) != 3 or len(front_normal_xy) != 2:
        raise ValueError('handle_xyz and front_normal_xy must have lengths 3 and 2')
    if not all(math.isfinite(value) for value in values):
        raise ValueError('geometry values must be finite')
    if radius_m <= 0 or base_height_m < 0 or not 0 <= half_angle_deg < 90:
        raise ValueError('invalid radius, base height or angular range')
    if yaw_mode not in ('face_handle', 'fixed'):
        raise ValueError('yaw_mode must be face_handle or fixed')
    if not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError('episode_index must be a nonnegative integer')
    nx, ny = front_normal_xy
    norm = math.hypot(nx, ny)
    if norm == 0:
        raise ValueError('front normal must be nonzero')
    nx, ny = nx / norm, ny / norm
    if angle_deg is None:
        angle_deg = random.Random(f'{seed}:{episode_index}').uniform(
            -half_angle_deg, half_angle_deg)
    if not math.isfinite(angle_deg) or abs(angle_deg) > half_angle_deg:
        raise ValueError('angle is outside the configured arc')
    angle = math.radians(angle_deg)
    dx = radius_m * (nx * math.cos(angle) - ny * math.sin(angle))
    dy = radius_m * (nx * math.sin(angle) + ny * math.cos(angle))
    yaw = math.atan2(-dy, -dx) if yaw_mode == 'face_handle' else math.atan2(-ny, -nx)
    return BasePose(
        (handle_xyz[0] + dx, handle_xyz[1] + dy, base_height_m),
        (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
        angle_deg,
    )
