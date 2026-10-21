#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Place the robot and randomize visuals in a door scene from the approved simulator snapshot.

The result feeds scripts/door/autodata/prepare.py (asset bundle) for generation or evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from arx_door.contract import first_batch_placements  # noqa: E402
from arx_door.demos.seed import load_seed  # noqa: E402
from arx_door.scene.usd_assets import rebase_local_assets  # noqa: E402


DEFAULT_SOURCE = ROOT / 'generated/door_teaching/upright02/live_scene.usd'
DEFAULT_SEED = ROOT / 'generated/door_teaching/upright02/candidate_006'

# Moderate matte/satin colours leave the chrome knob and door edge readable.
# These are RGB shader inputs, not image-space recolouring or new geometry.
DOOR_PALETTE = (
    ('muted_blue', (0.30, 0.48, 0.67)),
    ('sage_green', (0.37, 0.55, 0.41)),
    ('warm_sand', (0.68, 0.54, 0.36)),
    ('dusty_red', (0.61, 0.32, 0.29)),
    ('cool_grey', (0.43, 0.49, 0.56)),
    ('ivory', (0.76, 0.73, 0.63)),
)
BACKGROUND_PALETTE = (
    (0.32, 0.35, 0.39), (0.48, 0.43, 0.35),
    (0.36, 0.43, 0.45), (0.49, 0.48, 0.43),
)
DOOR_COLOUR_SHADERS = (
    '/World/Door/door_panel/Visuals/Looks/Portable_White_glossy_laminate_002/PreviewSurface',
    '/World/Door/door_panel/Visuals/Looks/Portable_Warm_tan_fibreboard_back_002/PreviewSurface',
)
WALL_ROOTS = ('/World/Wall', '/World/Walls', '/World/BackWall', '/World/Background')


def appearance_for_seed(visual_seed: int) -> dict:
    """Describe a reproducible appearance draw; zero makes no appearance edits."""
    if not isinstance(visual_seed, int) or not 0 <= visual_seed < 2**32:
        raise ValueError('visual seed must be an unsigned 32-bit integer')
    appearance = {'schema': 'arx-door-appearance-v1', 'visual_seed': visual_seed,
                  'enabled': visual_seed != 0}
    if visual_seed == 0:
        return appearance
    rng = random.Random(visual_seed)
    palette_name, colour = DOOR_PALETTE[(visual_seed - 1) % len(DOOR_PALETTE)]
    appearance.update({
        'palette': palette_name,
        'door_rgb': [round(value * rng.uniform(0.94, 1.06), 6) for value in colour],
        'background_rgb': list(rng.choice(BACKGROUND_PALETTE)),
        'wall_rgb': [round(rng.uniform(0.52, 0.72), 6) for _ in range(3)],
        'light_intensity_multiplier': round(rng.uniform(0.80, 1.20), 6),
        'light_temperature_kelvin': rng.randint(4300, 7300),
    })
    return appearance


def _recolour_geometry(stage: Usd.Stage, paths, colour, material_name: str) -> list[str]:
    """Bind a dedicated visual material, leaving material:binding:physics intact."""
    targets = [stage.GetPrimAtPath(path) for path in paths]
    targets = [prim for prim in targets if prim and prim.IsA(UsdGeom.Gprim)]
    if not targets:
        return []
    material = UsdShade.Material.Define(stage, f'/World/SkillGenAppearance/{material_name}')
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('PreviewSurface'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for prim in targets:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return [str(prim.GetPath()) for prim in targets]


def apply_appearance(stage: Usd.Stage, visual_seed: int) -> dict:
    """Change only existing illumination and non-handle surface appearance."""
    appearance = appearance_for_seed(visual_seed)
    if not appearance['enabled']:
        return appearance
    changed_shaders = []
    for path in DOOR_COLOUR_SHADERS:
        shader = UsdShade.Shader(stage.GetPrimAtPath(path))
        if not shader:
            continue
        diffuse = shader.GetInput('diffuseColor')
        if not diffuse:
            raise ValueError(f'door colour shader has no diffuseColor: {path}')
        # Keep normal maps, roughness, opacity and handle materials unchanged.
        # Disconnect the old albedo texture so its white/tan colour cannot hide the draw.
        diffuse.DisconnectSource()
        diffuse.Set(Gf.Vec3f(*appearance['door_rgb']))
        changed_shaders.append(path)
    if not changed_shaders:
        raise ValueError('appearance draw requires the door panel colour shaders')
    appearance['door_shader_targets'] = changed_shaders
    appearance['background_targets'] = _recolour_geometry(
        stage, ('/World/Ground',), appearance['background_rgb'], 'Background')
    wall_paths = []
    for path in WALL_ROOTS:
        root = stage.GetPrimAtPath(path)
        if root:
            wall_paths.extend(
                str(prim.GetPath()) for prim in Usd.PrimRange(root)
                if prim.IsA(UsdGeom.Gprim) and 'Colliders' not in prim.GetPath().pathString)
    appearance['wall_targets'] = _recolour_geometry(
        stage, wall_paths, appearance['wall_rgb'], 'Wall')
    lights = []
    for path in ('/World/EnvironmentLight', '/World/DistantLight'):
        light = UsdLux.LightAPI(stage.GetPrimAtPath(path))
        if not light:
            continue
        original = light.GetIntensityAttr().Get()
        if original is None or not math.isfinite(original) or original <= 0:
            raise ValueError(f'appearance draw requires a positive source light: {path}')
        intensity = float(original) * appearance['light_intensity_multiplier']
        light.CreateIntensityAttr(intensity)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(float(appearance['light_temperature_kelvin']))
        lights.append({'path': path, 'source_intensity': float(original),
                       'intensity': intensity,
                       'temperature_kelvin': appearance['light_temperature_kelvin']})
    if not lights:
        raise ValueError('appearance draw requires existing scene illumination')
    appearance['lights'] = lights
    return appearance


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
    *,
    visual_seed: int = 0,
    radius_m: float | None = None,
    allow_agent_reviewed_seeds: bool = False,
) -> dict:
    """Copy the proven stage and publish a closed, placed trial scene."""
    if radius_m is not None and (not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('radius must be finite and positive')
    seed = load_seed(seed_dir, allow_agent_reviewed=allow_agent_reviewed_seeds)
    # Placement is a trial dimension, not a new source recording. Preserve the
    # reviewed seed's original base/configuration and all identity evidence.
    trial_config = dict(seed.metadata['scene_placement']['config'])
    if radius_m is not None:
        trial_config['radius_m'] = float(radius_m)
    placement_metadata = {**seed.metadata, 'scene_placement': {
        **seed.metadata['scene_placement'], 'config': trial_config,
    }}
    placements = first_batch_placements(placement_metadata, (angle_deg,))
    placement = placements[0]
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise ValueError(f'existing scene is never overwritten: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.stem}.tmp{output.suffix}')
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None or not layer.Export(str(temporary)):
        raise RuntimeError(f'cannot copy source stage {source}')
    copied_layer = Sdf.Layer.FindOrOpen(str(temporary))
    asset_resolution = rebase_local_assets(copied_layer, source, output)
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

    appearance = apply_appearance(stage, visual_seed)

    manifest = {
        'schema_version': 1,
        'purpose': 'isaac_lab_skillgen_physics_replay',
        'source_scene': str(source),
        'source_seed': str(seed.path),
        'source_seed_quality': seed.metadata['quality_status'],
        'asset_resolution': asset_resolution,
        'behavior_family': seed.metadata.get('expert_parameters', {}).get(
            'behavior_family', 'edge_regrasp'),
        'placement': asdict(placement),
        'handle_world_xyz': seed.metadata['scene_placement']['handle_world_xyz'],
        'config': trial_config,
        'initial_state': initial.tolist(),
        'camera_contract': seed.metadata['camera'],
        'visual_seed': visual_seed,
        'appearance': appearance,
    }
    if seed.metadata['quality_status'] == 'agent_reviewed_seed':
        manifest.update(source_seed_agent_review=seed.metadata['agent_review'],
                        source_human_review_status='pending')
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
    parser.add_argument('--allow-agent-reviewed-seeds', action='store_true',
                        help='accept an agent-reviewed source only after rechecking its evidence')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--angle-deg', type=float, required=True)
    parser.add_argument('--radius-m', type=float,
                        help='positive horizontal handle-to-base radius; default preserves the source')
    parser.add_argument('--visual-seed', type=int, default=0,
                        help='zero preserves original appearance; nonzero changes only visuals')
    args = parser.parse_args()
    manifest = prepare_scene(args.source, args.seed, args.output, args.angle_deg,
                             visual_seed=args.visual_seed,
                             radius_m=args.radius_m,
                             allow_agent_reviewed_seeds=args.allow_agent_reviewed_seeds)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
