# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""A body-fixed "ego" camera rigidly attached to the ARX base.

On B2+ARX the second view is a camera on the dog body, not a world-fixed
overview camera; training on the overview would rely on a view that does not
exist after the arm moves onto the dog. The fixed-base workcell has a static
base, so the ego camera is static in the world and its pose is the robot
placement composed with a mount in the base frame (+X toward the door, +Z up).
The default mount is a simulation choice to be recalibrated against the real
B2 body camera before sim-to-real use.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_EGO_MOUNT = {
    'position_base_m': [0.10, 0.0, 0.12],
    'yaw_deg': 0.0,
    'pitch_up_deg': 20.0,
    'horizontal_fov_deg': 87.0,
    'width': 640,
    'height': 480,
    'prim_path': '/World/SkillGenEgoCamera',
}
HORIZONTAL_APERTURE_MM = 20.955


def ego_mount(value: dict | None) -> dict:
    """Fill defaults and validate an ego camera mount specification."""
    mount = {**DEFAULT_EGO_MOUNT, **(value or {})}
    position = np.asarray(mount['position_base_m'], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('ego camera position_base_m must contain three finite numbers')
    if not 20.0 <= float(mount['horizontal_fov_deg']) < 170.0:
        raise ValueError('ego camera horizontal_fov_deg must be within [20, 170)')
    if int(mount['width']) <= 0 or int(mount['height']) <= 0:
        raise ValueError('ego camera resolution must be positive')
    return mount


def ego_world_pose(placement: dict, mount: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return world position and xyzw quaternion in Isaac's "world" camera convention."""
    base = Rotation.from_quat(placement['quaternion_xyzw'])
    local = Rotation.from_euler('ZY', [mount['yaw_deg'], -mount['pitch_up_deg']], degrees=True)
    position = np.asarray(placement['position'], dtype=float) + base.apply(mount['position_base_m'])
    return position, (base * local).as_quat()


def focal_length_mm(horizontal_fov_deg: float) -> float:
    """Pinhole focal length that gives the requested horizontal field of view."""
    return HORIZONTAL_APERTURE_MM / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def ego_camera_cfg(placement: dict, mount: dict, *, quaternion_from_xyzw):
    """Build the Isaac Lab camera configuration; imports Isaac Lab lazily."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    position, quaternion = ego_world_pose(placement, mount)
    return CameraCfg(
        prim_path=mount['prim_path'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_mm(mount['horizontal_fov_deg']),
            horizontal_aperture=HORIZONTAL_APERTURE_MM, clipping_range=(0.02, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(position.tolist()),
                                   rot=tuple(np.asarray(quaternion_from_xyzw(quaternion), dtype=float).tolist()),
                                   convention='world'),
        width=int(mount['width']), height=int(mount['height']), data_types=['rgb'],
        update_period=0.0, update_latest_camera_pose=True,
    )
