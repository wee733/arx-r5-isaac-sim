# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Prepare portable, relocatable USD copies without starting Isaac Sim."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import tempfile

from arx_door.scene.usd_assets import copy_flattened_scene
from arx_door.skillgen.physics_profile import source_cpu_scene_settings

SCHEMA = 'arx-autodata-assets-v1'


def load_asset_manifest(manifest: str | Path | dict) -> dict:
    """Resolve package-relative USD paths against the mounted manifest directory.

    An in-memory manifest returned by ``prepare_assets`` carries ``asset_root``.
    JSON deliberately omits that host path, so the whole package can be mounted
    at any container path. ``source_scene`` is provenance, never a runtime asset.
    """
    if isinstance(manifest, dict):
        result = deepcopy(manifest)
        root_value = result.get('asset_root')
        if root_value is None:
            raise ValueError('in-memory asset manifest requires asset_root')
        root = Path(root_value).resolve()
    else:
        path = Path(manifest).resolve()
        result = json.loads(path.read_text())
        root = path.parent
    if result.get('schema') != SCHEMA:
        raise ValueError(f'unsupported asset manifest schema: {result.get("schema")}')
    for entry in result.get('assets', {}).values():
        value = Path(entry['usd_path'])
        if (value.is_absolute() and not result.get('asset_paths_resolved')) or '..' in value.parts:
            raise ValueError('asset USD paths must be relative to the mounted manifest')
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError('asset USD path escapes the mounted manifest')
        if not path.is_file():
            raise FileNotFoundError(path)
        entry['usd_path'] = str(path)
    result['asset_root'] = str(root)
    result['asset_paths_resolved'] = True
    return result


def _pose(stage, path: str) -> dict:
    from pxr import Gf, UsdGeom
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    transform = Gf.Transform(matrix)
    quat = transform.GetRotation().GetQuat()
    scale = [float(value) for value in transform.GetScale()]
    if not all(math.isfinite(value) and value > 0 for value in scale):
        raise ValueError(f'asset root has unsupported scale at {path}')
    return {'position': [float(value) for value in transform.GetTranslation()],
            'quaternion_xyzw': [*map(float, quat.GetImaginary()), float(quat.GetReal())],
            'scale': scale}


def _camera(stage) -> dict:
    from pxr import UsdGeom
    mount_path = '/R5a/link6/TeachingWristCamera'
    prim = stage.GetPrimAtPath(mount_path)
    if not prim or prim.GetTypeName() != 'Camera':
        raise ValueError(f'approved link6 camera mount is missing: {mount_path}')
    # USD Gf uses row vectors; adapters/recorders use column-vector matrices.
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    mount = [[float(matrix[col][row]) for col in range(4)] for row in range(4)]
    if not all(math.isfinite(value) for row in mount for value in row):
        raise ValueError('approved wrist mount contains non-finite values')
    sensor = stage.GetPrimAtPath('/World/SkillGenWristCamera') or prim
    attrs = {}
    for name in ('focalLength', 'horizontalAperture', 'verticalAperture',
                 'horizontalApertureOffset', 'verticalApertureOffset',
                 'clippingRange', 'focusDistance'):
        value = sensor.GetAttribute(name).Get()
        attrs[name] = list(map(float, value)) if name == 'clippingRange' else float(value)
    return {'mount_source_prim': mount_path, 'mount_link': 'link6',
            'mount_convention': 'opengl', 'mount_transform': mount,
            'prim_path': '{ENV_REGEX_NS}/WristCamera',
            'width': 640, 'height': 480, 'attributes': attrs,
            'update_period_s': 0.0, 'use_tiled_camera': False}


def _materialize(stage):
    """Flatten instance contents, not only references to private prototypes."""
    from pxr import Usd
    copied = Usd.Stage.Open(stage.Flatten())
    while True:
        instances = [prim for prim in copied.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    return copied.Flatten()


def _extract(flattened, source_path: str, destination: Path, *, remove=()) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    source = Usd.Stage.Open(flattened)
    root = Sdf.Path(source_path)
    target = Sdf.Path('/Asset')
    layer = Sdf.Layer.CreateAnonymous('arena_asset.usda')
    if not Sdf.CopySpec(flattened, root, layer, target):
        raise ValueError(f'could not copy USD subtree {source_path}')
    stage = Usd.Stage.Open(layer)
    prim = stage.GetPrimAtPath(target)
    prim.SetTypeName('Xform')
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    initial_pose = _pose(source, source_path)
    # Arena/Lab owns the root pose. Leaving its source transform here would
    # apply robot placement twice; root scale is passed through the spawner.
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(opSuffix='arena').Set(Gf.Matrix4d(1.0))
    for path in remove:
        stage.RemovePrim(str(Sdf.Path(path).ReplacePrefix(root, target)))
    mappings = [(root, target)]
    remapped, owners_cleared = 0, 0

    def relocate(path):
        nonlocal remapped
        absolute = path if path.IsAbsolutePath() else path.MakeAbsolutePath(root)
        # CopySpec already remaps paths internal to the copied subtree.
        if absolute.HasPrefix(target):
            return absolute
        for old, new in mappings:
            if absolute.HasPrefix(old):
                return absolute.ReplacePrefix(old, new)
        original = source.GetPrimAtPath(absolute.GetPrimPath())
        if not original:
            raise ValueError(f'USD target is missing: {absolute}')
        # External materials/shader nodes are copied into this same asset,
        # keeping binding and shader connections private to each environment.
        dependency = original.GetPath()
        shared = target.AppendPath('_shared/dependency_' + str(len(mappings)))
        stage.DefinePrim('/Asset/_shared', 'Scope')
        Sdf.CopySpec(flattened, dependency, layer, shared)
        mappings.append((dependency, shared))
        remapped += 1
        return absolute.ReplacePrefix(dependency, shared)

    visited = set()
    while True:
        candidates = [p for p in stage.Traverse() if p.GetPath() not in visited]
        if not candidates:
            break
        for current in candidates:
            visited.add(current.GetPath())
            for rel in current.GetRelationships():
                if rel.GetName() == 'physics:simulationOwner':
                    # The source PhysicsScene is intentionally not duplicated;
                    # the generated Lab scene owns all copied rigid bodies.
                    if rel.GetTargets():
                        owners_cleared += 1
                        rel.SetTargets([])
                else:
                    rel.SetTargets([relocate(path) for path in rel.GetTargets()])
            for attr in current.GetAttributes():
                if attr.GetConnections():
                    attr.SetConnections([relocate(path) for path in attr.GetConnections()])
    articulation_roots = [str(p.GetPath()) for p in stage.Traverse()
                          if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not layer.Export(str(destination)):
        raise RuntimeError(f'asset export failed: {destination}')
    return {'usd_path': destination.name, 'source_prim': source_path,
            'prim_path': '/Asset', 'initial_pose': initial_pose,
            'articulation_roots': articulation_roots,
            'copied_external_subtrees': remapped, 'simulation_owners_cleared': owners_cleared}


def _pack_resources(layer, source: Path, destination: Path, copied: dict) -> None:
    from pxr import UsdUtils
    def relocate(value):
        if not value or '://' in value:
            return value
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else source.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            # SDK module names resolve inside Kit; an absolute .mdl path still
            # must exist or it would leak a host-only dependency.
            if Path(value).name == value and value.lower().endswith('.mdl'):
                return value
            raise FileNotFoundError(f'missing local USD resource: {value}')
        if candidate not in copied:
            authored = Path('resources') / f'{len(copied):03d}_{candidate.name}'
            (destination.parent / authored).parent.mkdir(exist_ok=True)
            shutil.copy2(candidate, destination.parent / authored)
            copied[candidate] = str(authored)
        return copied[candidate]
    UsdUtils.ModifyAssetPaths(layer, relocate)
    layer.Export(str(destination))


def _validate_asset(path: Path) -> dict:
    from pxr import Sdf, Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    if any(layer.realPath and Path(layer.realPath).resolve() != path.resolve() for layer in stage.GetUsedLayers()):
        raise ValueError(f'asset still depends on an external USD layer: {path}')
    relationships = prim_count = rigid_body_count = collision_count = 0
    for prim in stage.Traverse():
        for prop in [*prim.GetRelationships(), *prim.GetAttributes()]:
            targets = prop.GetTargets() if hasattr(prop, 'GetTargets') else prop.GetConnections()
            for target in targets:
                if not target.HasPrefix(Sdf.Path('/Asset')) or not stage.GetObjectAtPath(target):
                    raise ValueError(f'asset has unresolved or external target: {prop.GetPath()} -> {target}')
                relationships += 1
        prim_count += 1
        rigid_body_count += prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collision_count += prim.HasAPI(UsdPhysics.CollisionAPI)
    return {'prim_count': prim_count,
            'checked_targets': relationships,
            'rigid_body_count': rigid_body_count,
            'collision_count': collision_count,
            'evidence_level': 'static_usd_only'}



def _swept_spacing(stage, camera: dict) -> tuple[float, dict]:
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    names = ['base_link', *(f'link{i}' for i in range(1, 7))]
    points = []
    for name in names:
        prim = stage.GetPrimAtPath('/R5a/' + name)
        if not prim:
            raise ValueError(f'cannot derive robot sweep without body {name}')
        points.append(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
    # Distances between serial joint origins are invariant under joint rotation.
    # Sum them rather than using the current compact robot's displayed bbox.
    chain_reach = sum((b - a).GetLength() for a, b in zip(points, points[1:]))
    mount_radius = math.sqrt(sum(camera['mount_transform'][i][3] ** 2 for i in range(3)))
    reach = float(chain_reach + mount_radius + 0.1)  # finger/collider surface allowance
    root = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/R5a')).ExtractTranslation()
    door_range = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy']).ComputeWorldBound(
        stage.GetPrimAtPath('/World/Door')).ComputeAlignedRange()
    door_extent = door_range.GetSize()
    # A full door-width radius about its hinge bounds the panel's planar sweep.
    door_radius = max(float(door_extent[0]), float(door_extent[1]))
    extent = max(max(abs(float(root[i])) for i in range(2)) + reach, door_radius)
    spacing = max(3.0, 2 * extent + 0.25)
    return spacing, {'arm_chain_reach_m': float(chain_reach),
                     'arm_swept_radius_m': reach, 'wrist_mount_radius_m': mount_radius,
                     'finger_surface_allowance_m': 0.1, 'clearance_m': 0.25,
                     'door_swept_radius_m': door_radius, 'evidence_level': 'conservative_static'}


def prepare_assets(scene_path: str | Path, output_dir: str | Path) -> dict:
    """Export robot, articulated door, workcell and shared ground copies.

    Source USD, meshes and textures are read only. Physics/mimic/material
    relationships are remapped into each asset's default prim; local texture
    resources are included. No simulator or upstream generation script runs.
    """
    from pxr import Sdf, Usd
    source, output = Path(scene_path).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'asset output must be empty: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.arx-assets-', dir=output.parent) as temporary:
        staging = Path(temporary)
        flattened_path = staging / 'source.usd'
        copy_flattened_scene(source, flattened_path)
        source_stage = Usd.Stage.Open(str(flattened_path))
        for root in ('/R5a', '/World/Door', '/World'):
            if not source_stage.GetPrimAtPath(root):
                raise ValueError(f'required source subtree is missing: {root}')
        settings = source_cpu_scene_settings(source_stage)
        camera = _camera(source_stage)
        flattened = _materialize(source_stage)
        package = staging / 'package'
        package.mkdir()
        assets = {}
        for name, root in (('robot', '/R5a'), ('door', '/World/Door'), ('workcell', '/World')):
            remove = ('/World/Door', '/World/Ground', '/World/OverviewCamera',
                      '/World/SkillGenWristCamera', '/World/PlacementArc') if name == 'workcell' else ()
            assets[name] = _extract(flattened, root, package / f'{name}.usd', remove=remove)
        if source_stage.GetPrimAtPath('/World/Ground'):
            assets['ground'] = _extract(flattened, '/World/Ground', package / 'ground.usd')
            # Geometry roots can be cubes; preserve their authored type when
            # using the default prim as a global shared collider.
            ground = Usd.Stage.Open(str(package / 'ground.usd'))
            ground.GetDefaultPrim().SetTypeName(source_stage.GetPrimAtPath('/World/Ground').GetTypeName())
            ground.GetRootLayer().Save()
        resources = {}
        for entry in assets.values():
            path = package / entry['usd_path']
            _pack_resources(Sdf.Layer.FindOrOpen(str(path)), flattened_path, path, resources)
            entry['static_validation'] = _validate_asset(path)
        env_spacing, spacing_basis = _swept_spacing(source_stage, camera)
        manifest = {'schema': SCHEMA, 'source_scene': str(source),
                    'source_settings': settings, 'camera': camera, 'assets': assets,
                    'env_spacing_m': env_spacing,
                    'spacing_basis': spacing_basis,
                    'resource_count': len(resources), 'evidence_level': 'static_usd_only'}
        (package / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            output.rmdir()  # previously verified empty; never replace existing data
        package.rename(output)
    return {**manifest, 'asset_root': str(output)}


__all__ = ['prepare_assets', 'load_asset_manifest']
