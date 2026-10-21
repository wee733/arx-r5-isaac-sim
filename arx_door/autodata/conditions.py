# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Reproducible scene conditions for real AutoData trajectory generation.

Each condition is a portable asset bundle. A fresh Arena environment and cuRobo
world are built from it, so geometry, collision planning, replay and images use
the same placement. Cloning an environment does not create another condition.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def sample_conditions(count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError('count must be positive')
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        rows.append(dict(condition_id=f'condition_{index:04d}', seed=seed + index,
            door_translation_m=[float(rng.uniform(-.008, .008)),
                                float(rng.uniform(-.008, .008)), float(rng.uniform(-.02, .02))],
            door_yaw_deg=float(rng.uniform(-3., 3.)),
            panel_color=rng.uniform(.12, .75, 3).tolist(),
            panel_roughness=float(rng.uniform(.25, .8)),
            texture_family=['grain', 'paint', 'brushed'][index % 3],
            texture_seed=int(rng.integers(0, 2**31)),
            dome_intensity=float(rng.uniform(200., 450.)),
            key_intensity=float(rng.uniform(650., 1500.)),
            light_temperature_k=float(rng.uniform(3500., 7500.)),
            light_angles_deg=[float(rng.uniform(-65., -30.)), float(rng.uniform(-30., 30.)),
                              float(rng.uniform(-60., 60.))]))
    return rows


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose['quaternion_xyzw']).as_matrix()
    result[:3, 3] = pose['position']
    return result


def _texture(path, condition):
    from PIL import Image
    rng = np.random.default_rng(condition['texture_seed'])
    y, x = np.mgrid[:512, :512] / 512
    family = condition['texture_family']
    noise = rng.normal(0., .025, x.shape)
    if family == 'grain':
        pattern = .12 * np.sin(100*x + 5*np.sin(9*y)) + noise
    elif family == 'brushed':
        pattern = rng.normal(0., .075, (512, 1)) + noise * .3
    else:
        pattern = .05 * np.sin(20*x) * np.sin(24*y) + noise
    rgb = np.clip(np.asarray(condition['panel_color'])[None, None] * (1 + pattern[..., None]), 0, 1)
    Image.fromarray(np.uint8(rgb * 255)).save(path)


def prepare_condition(base_manifest: Path, output: Path, condition: dict) -> Path:
    """Keep colliders/material physics untouched; randomize visuals and door root pose."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    if output.exists():
        raise FileExistsError(output)
    manifest = copy.deepcopy(json.loads(base_manifest.read_text()))
    shutil.copytree(base_manifest.parent, output)
    source_door = copy.deepcopy(manifest['assets']['door']['initial_pose'])
    door = manifest['assets']['door']['initial_pose']
    door['position'] = (np.array(door['position']) + condition['door_translation_m']).tolist()
    door['quaternion_xyzw'] = (Rotation.from_euler('z', condition['door_yaw_deg'], degrees=True)
                              * Rotation.from_quat(door['quaternion_xyzw'])).as_quat().tolist()
    manifest['trial_parameters'] = copy.deepcopy(condition)
    manifest['trial_parameters']['scope'] = 'door_pose_and_rendering; closed_hinge; unchanged_physics'
    # The source actions are in the robot frame. Save the expected rigid transform
    # independently of generated actions so data audits can check retargeting.
    base = pose_matrix(manifest['assets']['robot']['initial_pose'])
    delta = np.linalg.inv(base) @ pose_matrix(door) @ np.linalg.inv(pose_matrix(source_door)) @ base
    manifest['trial_parameters']['expected_skill_transform_robot'] = delta.tolist()
    _texture(output / 'panel_texture.png', condition)
    stage = Usd.Stage.Open(str(output / manifest['assets']['door']['usd_path']))
    material = UsdShade.Material.Define(stage, '/Asset/RandomizedPanelMaterial')
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(condition['panel_roughness'])
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    reader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    texture = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('panel_texture.png'))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    bound = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and '/door_panel/Visuals/' in str(prim.GetPath()) and any(
                word in prim.GetName() for word in ('White_glossy_laminate', 'Warm_tan_fibreboard')):
            mesh = UsdGeom.Mesh(prim)
            # Planar UVs on the door face; physics meshes and transforms are untouched.
            points = np.asarray(mesh.GetPointsAttr().Get())
            extents = np.ptp(points, axis=0)
            axes = np.argsort(extents)[-2:]
            uv = (points[:, axes] - points[:, axes].min(axis=0)) / extents[axes]
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*map(float, p)) for p in uv])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            bound.append(str(prim.GetPath()))
    if len(bound) < 2:
        raise ValueError('expected both panel visual faces')
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output / manifest['assets']['workcell']['usd_path']))
    dome = UsdLux.DomeLight(stage.GetPrimAtPath('/Asset/EnvironmentLight'))
    key = UsdLux.DistantLight(stage.GetPrimAtPath('/Asset/DistantLight'))
    dome.GetIntensityAttr().Set(condition['dome_intensity'])
    key.GetIntensityAttr().Set(condition['key_intensity'])
    key.GetEnableColorTemperatureAttr().Set(True)
    key.GetColorTemperatureAttr().Set(condition['light_temperature_k'])
    xf = UsdGeom.Xformable(key.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*condition['light_angles_deg']))
    stage.GetRootLayer().Save()
    manifest['randomized_visual_prims'] = bound
    target = output / base_manifest.name
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    return target
