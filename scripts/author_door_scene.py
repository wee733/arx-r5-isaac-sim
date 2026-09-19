#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Author an ARX door workcell and an episode placement manifest, offline."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import yaml

from arx_r5_isaac_sim_bringup.door_workcell import sample_base_pose


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'arx_r5_isaac_sim_bringup'


def set_pose(prim, xyz, xyzw=(0.0, 0.0, 0.0, 1.0)):
    """Replace a placement transform, explicitly eliminating the old base tilt."""
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(opSuffix='doorPlacement').Set(Gf.Vec3d(*xyz))
    xform.AddOrientOp(opSuffix='doorPlacement').Set(
        Gf.Quatf(xyzw[3], Gf.Vec3f(*xyzw[:3])))


def box(stage, path, size, center, color):
    """Create a static collision box in metres."""
    from pxr import Gf, UsdGeom, UsdPhysics
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def author(args):
    """Keep the robot's internal mesh references and compose the local door asset."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
    config = yaml.safe_load(args.config.read_text())
    if config.pop('schema_version') != 1:
        raise ValueError('unsupported door scene schema')
    handle_path = config.pop('handle_reference_prim')
    if args.yaw_mode:
        config['yaw_mode'] = args.yaw_mode
    output = args.output.resolve()
    source = PACKAGE / 'assets/scenes/arx_sim.usd'
    if output == source.resolve():
        raise ValueError('output must not replace the source robot scene')
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None or not layer.Export(str(output)):
        raise RuntimeError('cannot copy source robot scene')
    # Strip the old workcell before opening; avoids its remote ground reference.
    layer = Sdf.Layer.FindOrOpen(str(output))
    from arx_r5_isaac_sim_bringup.usd_assets import rebase_local_assets
    rebase_local_assets(layer, source, output)
    edits = Sdf.BatchNamespaceEdit()
    for path in ('/World/defaultGroundPlane', '/World/Workspace', '/World/Sensors', '/Render'):
        if layer.GetPrimAtPath(path):
            edits.Add(path, Sdf.Path.emptyPath)
    if not layer.Apply(edits):
        raise RuntimeError('cannot remove the old tabletop')
    stage = Usd.Stage.Open(layer, Usd.Stage.LoadNone)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    door = UsdGeom.Xform.Define(stage, '/World/Door')
    asset = PACKAGE / 'assets/door/door.usda'
    door.GetPrim().GetReferences().AddReference(os.path.relpath(asset, output.parent))
    reference = stage.GetPrimAtPath(handle_path)
    if not reference:
        raise ValueError(f'handle reference is missing: {handle_path}')
    handle = tuple(UsdGeom.XformCache().GetLocalToWorldTransform(reference).ExtractTranslation())
    pose = sample_base_pose(
        handle, **config, seed=args.seed, episode_index=args.episode_index,
        angle_deg=args.angle_deg)
    set_pose(stage.GetPrimAtPath('/R5a'), pose.position, pose.quaternion_xyzw)
    box(stage, '/World/Ground', (5.0, 5.0, 0.10), (0, 0, -0.05), (0.23, 0.25, 0.27))
    # A neutral mounting pedestal, not a model of the user's unmeasured carrier.
    x, y, z = pose.position
    box(stage, '/World/BaseSupport', (0.20, 0.20, z), (x, y, z / 2), (0.15, 0.18, 0.22))
    light = UsdLux.DomeLight.Define(stage, '/World/EnvironmentLight')
    light.CreateIntensityAttr(700)
    # Guides have no collision and can be hidden via the guide purpose toggle.
    points = [sample_base_pose(handle, **config, angle_deg=-config['half_angle_deg']
              + 2 * config['half_angle_deg'] * i / 60).position for i in range(61)]
    arc = UsdGeom.BasisCurves.Define(stage, '/World/PlacementArc')
    arc.CreateTypeAttr('linear')
    arc.CreateCurveVertexCountsAttr([len(points)])
    arc.CreatePointsAttr([Gf.Vec3f(px, py, 0.005) for px, py, _ in points])
    arc.CreateWidthsAttr([0.006])
    arc.SetWidthsInterpolation('constant')
    arc.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.8, 0.4)])
    arc.CreatePurposeAttr(UsdGeom.Tokens.guide)
    camera = UsdGeom.Camera.Define(stage, '/World/OverviewCamera')
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(2.2, -3.1, 2.1), Gf.Vec3d(0, -0.35, 0.95), Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(30)
    manifest = {
        'schema_version': 1, 'seed': args.seed, 'episode_index': args.episode_index,
        'world_frame': 'world', 'base_frame': 'base_link',
        'handle_reference_prim': handle_path, 'handle_world_xyz': handle,
        'placement': asdict(pose), 'config': config,
        'assumptions': ['handle height 0.928 m from existing asset; measurement pending',
                        'base_link origin is the installation plane',
                        'base stationary during each episode; randomized between episodes'],
    }
    stage.GetRootLayer().customLayerData = {'doorPlacementManifest': json.dumps(manifest)}
    stage.GetRootLayer().Save()
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'scene': str(output), **manifest}, indent=2))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PACKAGE / 'config/door_scene.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/door/scene.usd')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--angle-deg', type=float, help='Explicit angle within the configured arc')
    parser.add_argument('--yaw-mode', choices=['face_handle', 'fixed'])
    author(parser.parse_args())


if __name__ == '__main__':
    main()
