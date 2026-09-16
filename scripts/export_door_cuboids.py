#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Export enabled USD workcell colliders as conservative base_link cuboids.

Run with Isaac Sim's Python and omni.usd.libs on PYTHONPATH. The stage is opened
with LoadNone, so sensor payloads are not loaded. The sibling scene JSON supplies
the world pose of base_link; the USD file is never modified.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def matrix_from_pose(position, quaternion_xyzw):
    """Return a column-vector homogeneous transform from an xyzw pose."""
    x, y, z, w = quaternion_xyzw
    rotation = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(Gf.Matrix3d(rotation)).T
    matrix[:3, 3] = position
    return matrix


def quaternion_from_matrix(rotation):
    """Convert a column-vector rotation matrix to an xyzw quaternion."""
    quaternion = Gf.Matrix3d(*rotation.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    return [*map(float, quaternion.GetImaginary()), float(quaternion.GetReal())]


def collider_bounds(prim):
    """Bound only this collision shape, never its visual siblings."""
    if prim.IsA(UsdGeom.Cube):
        half_size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2.0
        return np.full(3, -half_size), np.full(3, half_size)
    if prim.IsA(UsdGeom.Mesh):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f'missing or invalid collision mesh points: {prim.GetPath()}')
        return points.min(axis=0), points.max(axis=0)
    raise ValueError(f'unsupported enabled collider {prim.GetTypeName()}: {prim.GetPath()}')


def export(scene):
    """Include every enabled Cube/Mesh collider under /World exactly once."""
    manifest = json.loads(scene.with_suffix('.json').read_text())
    placement = manifest['placement']
    world_from_base = matrix_from_pose(placement['position'], placement['quaternion_xyzw'])
    base_from_world = np.linalg.inv(world_from_base)
    stage = Usd.Stage.Open(str(scene), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f'cannot open USD stage {scene}')
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise ValueError('scene must be authored in metres')
    root = stage.GetPrimAtPath('/World')
    if not root:
        raise ValueError('scene has no /World prim')
    cache = UsdGeom.XformCache()
    cuboids = []
    excluded = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision.GetCollisionEnabledAttr().Get():
            continue
        minimum, maximum = collider_bounds(prim)
        world_from_local = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        path = str(prim.GetPath())
        if path == '/World/BaseSupport':
            # This support is vertical and shares the fixed robot installation
            # plane. Remove exactly its top 6 cm from planning geometry only.
            vertical = world_from_local[:3, 2]
            scale = np.linalg.norm(vertical)
            if scale <= 0 or not np.allclose(vertical / scale, [0, 0, 1], atol=1e-7):
                raise ValueError('BaseSupport must have its local +Z aligned with world +Z')
            maximum[2] -= 0.06 / scale
            if maximum[2] <= minimum[2]:
                raise ValueError('BaseSupport is shorter than the requested 0.06 m trim')
            excluded.append({
                'prim': path, 'top_trim_m': 0.06,
                'reason': 'fixed mounting contact; avoid inflated base collision sphere overlap',
                'physics_geometry_modified': False,
            })
        base_from_local = base_from_world @ world_from_local
        linear = base_from_local[:3, :3]
        u, _, vh = np.linalg.svd(linear)
        orientation = u @ vh
        if np.linalg.det(orientation) < 0:
            u[:, -1] *= -1
            orientation = u @ vh
        local_corners = np.array(list(itertools.product(*zip(minimum, maximum))))
        base_corners = local_corners @ linear.T + base_from_local[:3, 3]
        oriented = base_corners @ orientation
        lower, upper = oriented.min(axis=0), oriented.max(axis=0)
        size = upper - lower
        if np.any(size <= 0):
            raise ValueError(f'zero-volume collider cannot be represented by a cuboid: {path}')
        center = orientation @ ((lower + upper) / 2.0)
        cuboids.append({
            'name': path,
            'center': center.tolist(),
            'quaternion_xyzw': quaternion_from_matrix(orientation),
            'size': size.tolist(),
        })
    if not cuboids:
        raise ValueError('no enabled colliders exported')
    return {
        'cuboids': cuboids,
        'metadata': {
            'scene': str(scene), 'frame_id': 'base_link',
            'world_from_base': world_from_base.tolist(),
            'collision_count': len(cuboids),
            'geometry': 'conservative OBBs of enabled collision Cube/Mesh local bounds',
            'payload_policy': 'LoadNone',
            'physics_snapshot': 'authored USD poses; re-export updated stage after door motion',
            'exclusions': excluded,
        },
    }


def main():
    """Write cuboids and provenance for a cuMotion request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.scene.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Exported {len(result["cuboids"])} enabled collision shapes to {args.output}')


if __name__ == '__main__':
    main()
