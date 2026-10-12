# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Geometric checks for a virtual viewpoint, independent of camera housing."""
import numpy as np


def point_box_clearance(point_world, world_from_box, half_size):
    """Return signed point-to-OBB clearance in metres; negative is inside.

    The transform must be rigid; dimensions belong in half_size. This checks
    the viewpoint only and does not certify a physical camera housing.
    """
    point = np.asarray(point_world, dtype=float)
    transform = np.asarray(world_from_box, dtype=float)
    half = np.asarray(half_size, dtype=float)
    if (point.shape != (3,) or transform.shape != (4, 4) or half.shape != (3,)
            or not np.isfinite(point).all() or not np.isfinite(transform).all()
            or not np.isfinite(half).all() or np.any(half <= 0)):
        raise ValueError('point, rigid box transform and positive half sizes must be finite')
    local = transform[:3, :3].T @ (point-transform[:3, 3])
    delta = np.abs(local)-half
    return float(np.linalg.norm(np.maximum(delta, 0)) + min(float(np.max(delta)), 0))
