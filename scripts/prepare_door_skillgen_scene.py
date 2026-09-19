#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Prepare one resettable SkillGen scene from the approved simulator snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from arx_r5_isaac_sim_bringup.door_skillgen.contract import first_batch_placements
from arx_r5_isaac_sim_bringup.door_skillgen.seed import load_seed
from arx_r5_isaac_sim_bringup.usd_assets import rebase_local_assets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'generated/door_teaching/upright02/live_scene.usd'
DEFAULT_SEED = ROOT / 'generated/door_teaching/upright02/candidate_006'


def _set_pose(prim: Usd.Prim, position, quaternion_xyzw) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(opSuffix='skillgenPlacement').Set(Gf.Vec3d(*position))
    x, y, z, w = quaternion_xyzw
    xform.AddOrientOp(opSuffix='skillgenPlacement').Set(
        Gf.Quatf(w, Gf.Vec3f(x, y, z))
    )


def _canonicalize_xform(prim: Usd.Prim) -> None:
    """Rewrite one local transform in Isaac Lab's canonical TRS order."""
    xform = UsdGeom.Xformable(prim)
    local = xform.GetLocalTransformation()
    transform = Gf.Transform(local)
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat()
    scale = transform.GetScale()
    xform.ClearXformOpOrder()
    # Isaac Lab's Fabric view reads scale into a Vec3dArray. Authored float3
    # scales in simulator snapshots cannot be assigned to that USD array.
    prim.RemoveProperty('xformOp:scale')
    # The pinned camera view writes Quatd when synchronizing world poses to USD.
    prim.RemoveProperty('xformOp:orient')
    xform.AddTranslateOp().Set(translation)
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(
            float(quaternion.GetReal()),
            Gf.Vec3d(*map(float, quaternion.GetImaginary())),
        )
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(float, scale))
    )


def _restore_zero_door_bodies(stage: Usd.Stage) -> None:
    """Reconstruct closed-body transforms from the authored joint frames."""
    for name in ('hinge_joint', 'handle_joint', 'latch_joint'):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f'/World/Door/{name}'))
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        frames = []
        for pos, rot in (
            (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        ):
            frame = Gf.Matrix4d().SetRotate(Gf.Quatd(rot))
            frame.SetTranslateOnly(Gf.Vec3d(pos))
            frames.append(frame)
        cache = UsdGeom.XformCache()
        world = frames[1].GetInverse() * frames[0] * cache.GetLocalToWorldTransform(parent)
        local = world * cache.GetLocalToWorldTransform(child.GetParent()).GetInverse()
        transform = Gf.Transform(local)
        rotation = transform.GetRotation().GetQuat()
        _set_pose(child, transform.GetTranslation(), (
            *rotation.GetImaginary(), rotation.GetReal(),
        ))


def _set_joint_position(
    stage: Usd.Stage,
    path: str,
    value: float,
    *,
    angular: bool,
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f'missing joint prim: {path}')
    kind = 'angular' if angular else 'linear'
    authored_value = math.degrees(value) if angular else value
    state = prim.GetAttribute(f'state:{kind}:physics:position')
    if state:
        state.Set(float(authored_value))
    velocity = prim.GetAttribute(f'state:{kind}:physics:velocity')
    if velocity:
        velocity.Set(0.0)
    drive = UsdPhysics.DriveAPI.Get(prim, kind)
    if drive:
        drive.CreateTargetPositionAttr(float(authored_value))
        drive.CreateTargetVelocityAttr(0.0)


def prepare_scene(
    source: Path,
    seed_dir: Path,
    output: Path,
    angle_deg: float,
) -> dict:
    """Copy the proven stage and publish a closed, placed trial scene."""
    seed = load_seed(seed_dir)
    placements = first_batch_placements(seed.metadata, (angle_deg,))
    placement = placements[0]
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.stem}.tmp{output.suffix}')
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None or not layer.Export(str(temporary)):
        raise RuntimeError(f'cannot copy source stage {source}')
    copied_layer = Sdf.Layer.FindOrOpen(str(temporary))
    rebase_local_assets(copied_layer, source, output)
    copied_layer.Save()
    stage = Usd.Stage.Open(str(temporary), Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f'cannot open copied stage {temporary}')
    stage.SetEditTarget(stage.GetRootLayer())
    robot = stage.GetPrimAtPath('/R5a')
    if not robot:
        raise ValueError('approved scene has no /R5a articulation')
    _set_pose(robot, placement.position, placement.quaternion_xyzw)
    support = stage.GetPrimAtPath('/World/BaseSupport')
    if support:
        x, y, z = placement.position
        support.GetAttribute('xformOp:translate').Set(Gf.Vec3d(x, y, z / 2.0))
    _restore_zero_door_bodies(stage)

    initial = np.asarray(seed.state[0], dtype=float)
    for index, value in enumerate(initial[:6], start=1):
        _set_joint_position(
            stage,
            f'/R5a/joints/joint{index}',
            float(value),
            angular=True,
        )
    for name in ('joint7', 'joint8'):
        _set_joint_position(
            stage,
            f'/R5a/joints/{name}',
            float(initial[6]),
            angular=False,
        )
    for name, value, angular in (
        ('hinge_joint', 0.0, True),
        ('handle_joint', 0.0, True),
        ('latch_joint', 0.0, False),
    ):
        _set_joint_position(
            stage,
            f'/World/Door/{name}',
            value,
            angular=angular,
        )

    for stale in (
        '/Render',
        '/Orchestrator',
        '/OmniverseKit_Persp',
        '/OmniverseKit_Front',
        '/OmniverseKit_Top',
        '/OmniverseKit_Right',
    ):
        if stage.GetPrimAtPath(stale):
            stage.RemovePrim(stale)
    for camera_path in (
        '/R5a/link6/TeachingWristCamera',
        '/World/OverviewCamera',
    ):
        camera = stage.GetPrimAtPath(camera_path)
        if not camera:
            raise ValueError(f'approved scene has no camera {camera_path}')
        _canonicalize_xform(camera)

    # A camera parented to a GPU articulation inherits stale USD parent poses
    # in the pinned renderer bridge. Keep its original mount as a reference,
    # and render an independent camera under static World, tracked from PhysX.
    Sdf.CopySpec(stage.GetRootLayer(), '/R5a/link6/TeachingWristCamera',
                 stage.GetRootLayer(), '/World/SkillGenWristCamera')

    manifest = {
        'schema_version': 1,
        'purpose': 'isaac_lab_skillgen_physics_replay',
        'source_scene': str(source),
        'source_seed': str(seed.path),
        'source_seed_quality': seed.metadata['quality_status'],
        'placement': asdict(placement),
        'handle_world_xyz': seed.metadata['scene_placement']['handle_world_xyz'],
        'config': seed.metadata['scene_placement']['config'],
        'initial_state': initial.tolist(),
        'camera_contract': seed.metadata['camera'],
    }
    custom = dict(stage.GetRootLayer().customLayerData)
    custom['doorSkillGenManifest'] = json.dumps(manifest, sort_keys=True)
    stage.GetRootLayer().customLayerData = custom
    stage.GetRootLayer().Save()
    os.replace(temporary, output)
    output.with_suffix('.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return manifest


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--seed', type=Path, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--angle-deg', type=float, required=True)
    args = parser.parse_args()
    manifest = prepare_scene(args.source, args.seed, args.output, args.angle_deg)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
