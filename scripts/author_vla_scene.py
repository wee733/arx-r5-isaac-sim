#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Author the VLA collection workcell from the AprilTag workcell USD.

The source scene keeps a table, an AprilTag cube and a drop tag. This script
derives the collection scene from it: table and tags removed, the floor raised
to z = 0, the robot moved onto its 0.35 m column, and a low platform added for
the block to rest on.

Every number comes from ``config/vla_task.yaml`` and ``config/vla_scene.yaml``,
so the generated USD matches its own contract by construction. That is what the
hand-edited AprilTag scene lost: its ``/R5a`` pose silently drifted away from
``arx_sim_usd_scene.yaml`` and the simulator now refuses to start on it.

Run through ``scripts/run_author_vla_scene.sh``, which puts the Isaac Sim USD
libraries on the path.
"""

import argparse
from pathlib import Path
import sys
from typing import Sequence

from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.collision_scene import render_collision_scene
from arx_r5_isaac_sim_bringup.vla.task_config import (
    assert_valid_against_scene,
    load_task_config,
    TaskConfig,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'arx_r5_isaac_sim_bringup'
SOURCE_USD = PACKAGE_ROOT / 'assets' / 'scenes' / 'arx_sim.usd'
DESTINATION_USD = PACKAGE_ROOT / 'assets' / 'scenes' / 'arx_vla_scene.usd'
TASK_CONFIG = PACKAGE_ROOT / 'config' / 'vla_task.yaml'
SCENE_CONFIG = PACKAGE_ROOT / 'config' / 'vla_scene.yaml'
COLLISION_SCENE = PACKAGE_ROOT / 'config' / 'vla_ground.scene'

WORKSPACE_PRIM_PATH = '/World/Workspace'
GROUND_PLANE_PRIM_PATH = '/World/defaultGroundPlane'
SOURCE_BLOCK_PRIM_PATH = '/World/Workspace/TaggedCube'
SOURCE_BLOCK_BODY_NAME = 'Body'

# Everything the collection workcell does not have.
REMOVED_PRIM_PATHS = (
    '/World/Workspace/TableTop',
    '/World/Workspace/TableLeg_0',
    '/World/Workspace/TableLeg_1',
    '/World/Workspace/TableLeg_2',
    '/World/Workspace/TableLeg_3',
    '/World/Workspace/DropTag_1',
    '/World/Workspace/TaggedCube/AprilTag_0',
)


def _set_transform(prim, translation, rotation=None, scale=None) -> None:
    """Overwrite a prim's translate/orient/scale ops in place.

    The source scene mixes single- and double-precision ops, so each value is
    rebuilt at the precision the existing op already declares.
    """
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise RuntimeError(f'prim is not transformable: {prim.GetPath()}')
    operations = {
        operation.GetOpName(): operation
        for operation in xformable.GetOrderedXformOps()
    }

    def require(name: str):
        operation = operations.get(name)
        if operation is None:
            raise RuntimeError(
                f'{prim.GetPath()} has no {name} op; the source scene must '
                'author translate/orient/scale in that order'
            )
        return operation

    def vector(operation, values):
        vector_type = (
            Gf.Vec3f
            if operation.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
            else Gf.Vec3d
        )
        return vector_type(*(float(value) for value in values))

    translate_op = require('xformOp:translate')
    translate_op.Set(vector(translate_op, translation))
    if rotation is not None:
        orient_op = require('xformOp:orient')
        # USD stores quaternions as (real, imaginary); the contract is xyzw.
        single = orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
        quaternion_type = Gf.Quatf if single else Gf.Quatd
        imaginary_type = Gf.Vec3f if single else Gf.Vec3d
        orient_op.Set(quaternion_type(
            float(rotation[3]),
            imaginary_type(
                float(rotation[0]),
                float(rotation[1]),
                float(rotation[2]),
            ),
        ))
    if scale is not None:
        scale_op = require('xformOp:scale')
        scale_op.Set(vector(scale_op, scale))


def _define_box(stage, prim_path: str, size, center) -> None:
    """Create a unit Cube scaled into a static collider at a world pose.

    Only ``UsdPhysics.CollisionAPI`` is authored here. The runtime's
    ``_configure_authored_workspace_contacts`` applies the PhysX contact and
    rest offsets to every collider under ``/World/Workspace`` on load.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.GetSizeAttr().Set(1.0)
    prim = cube.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(
        float(value) for value in center
    )))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    )
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(
        float(value) for value in size
    )))
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)


def _define_place_marker(stage, task: TaskConfig) -> None:
    """Create the visual-only decal marking the placement target.

    A thin unlit cube rather than a textured quad: it needs no UVs, no material
    asset and no light to read as a solid colour in both camera streams. It
    carries no collision API, so cuMotion and PhysX both ignore it.
    """
    from pxr import Gf, Sdf, UsdGeom

    marker = task.place_marker
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(marker.prim_path))
    cube.GetSizeAttr().Set(1.0)
    prim = cube.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # Placed at the nominal target; the runtime moves it on every reset.
    surface_z = task.workcell.platform.top_z
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(
        float(task.block.nominal_center[0]),
        float(task.block.nominal_center[1]),
        float(surface_z + marker.height_above_surface),
    ))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    )
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(
        float(marker.size[0]),
        float(marker.size[1]),
        float(marker.height_above_surface),
    ))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*(
        float(channel) for channel in marker.color
    ))])


def author_stage(stage, task: TaskConfig, scene) -> None:
    """Turn an opened AprilTag workcell stage into the collection workcell."""
    from pxr import Sdf, UsdGeom

    for prim_path in REMOVED_PRIM_PATHS:
        if not stage.GetPrimAtPath(prim_path).IsValid():
            raise RuntimeError(
                f'source scene is missing an expected prim: {prim_path}'
            )
        if not stage.RemovePrim(Sdf.Path(prim_path)):
            raise RuntimeError(f'failed to remove prim: {prim_path}')

    robot_prim = stage.GetPrimAtPath(scene.robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f'robot prim not found: {scene.robot_prim_path}')
    _set_transform(
        robot_prim,
        scene.expected_world_to_base_translation,
        scene.expected_world_to_base_rotation,
    )

    ground_prim = stage.GetPrimAtPath(GROUND_PLANE_PRIM_PATH)
    if not ground_prim.IsValid():
        raise RuntimeError(
            f'ground plane prim not found: {GROUND_PLANE_PRIM_PATH}'
        )
    _set_transform(ground_prim, (0.0, 0.0, task.workcell.ground_z))

    _define_box(
        stage,
        task.workcell.mount.prim_path,
        task.workcell.mount.size,
        task.workcell.mount.center,
    )
    _define_box(
        stage,
        task.workcell.platform.prim_path,
        task.workcell.platform.size,
        task.workcell.platform.center,
    )
    _define_place_marker(stage, task)

    block_prim = stage.GetPrimAtPath(SOURCE_BLOCK_PRIM_PATH)
    if not block_prim.IsValid():
        raise RuntimeError(
            f'source block prim not found: {SOURCE_BLOCK_PRIM_PATH}'
        )
    # Put the block Xform on the block centre and zero the body offset, so the
    # rigid body the runtime adds and the grasp distance check both key off the
    # geometric centre instead of an authored corner offset.
    _set_transform(block_prim, task.block.nominal_center)
    body_prim = stage.GetPrimAtPath(
        f'{SOURCE_BLOCK_PRIM_PATH}/{SOURCE_BLOCK_BODY_NAME}'
    )
    if not body_prim.IsValid():
        raise RuntimeError('source block has no Body geometry')
    _set_transform(body_prim, (0.0, 0.0, 0.0), scale=task.block.size)

    if not UsdGeom.Xformable(block_prim):
        raise RuntimeError('block prim is not transformable')
    if not stage.GetPrimAtPath(WORKSPACE_PRIM_PATH).IsValid():
        raise RuntimeError(f'workspace prim disappeared: {WORKSPACE_PRIM_PATH}')

    destination = Sdf.Path(task.block.prim_path)
    if str(destination) != SOURCE_BLOCK_PRIM_PATH:
        if not _rename_prim(stage, SOURCE_BLOCK_PRIM_PATH, str(destination)):
            raise RuntimeError(
                f'failed to rename {SOURCE_BLOCK_PRIM_PATH} to {destination}'
            )


def _rename_prim(stage, source_path: str, destination_path: str) -> bool:
    """Move a prim within the root layer, preserving its whole subtree."""
    from pxr import Sdf

    layer = stage.GetRootLayer()
    with Sdf.ChangeBlock():
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.Path(source_path), Sdf.Path(destination_path))
        return layer.Apply(edit)


def author_scene(
    source_usd: Path,
    destination_usd: Path,
    task: TaskConfig,
    scene,
) -> None:
    """Copy the source workcell and rewrite it into the collection scene."""
    from pxr import Sdf, Usd

    if source_usd.resolve() == destination_usd.resolve():
        raise ValueError('destination must differ from the source USD')
    source_layer = Sdf.Layer.FindOrOpen(str(source_usd))
    if source_layer is None:
        raise FileNotFoundError(f'failed to open USD layer: {source_usd}')
    destination_usd.parent.mkdir(parents=True, exist_ok=True)
    if not source_layer.Export(str(destination_usd)):
        raise RuntimeError(f'failed to copy USD layer to {destination_usd}')

    # The ZED and RealSense assets are remote payloads. Opening without them
    # keeps the script usable offline; the payload arcs are preserved verbatim
    # in the exported layer and resolve normally inside Isaac Sim.
    stage = Usd.Stage.Open(str(destination_usd), Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f'failed to open copied stage: {destination_usd}')
    author_stage(stage, task, scene)
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f'failed to save authored stage: {destination_usd}')


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Author the VLA collection workcell USD and collision scene.',
    )
    parser.add_argument('--source-usd', default=str(SOURCE_USD))
    parser.add_argument('--destination-usd', default=str(DESTINATION_USD))
    parser.add_argument('--task-config', default=str(TASK_CONFIG))
    parser.add_argument('--scene-config', default=str(SCENE_CONFIG))
    parser.add_argument('--collision-scene', default=str(COLLISION_SCENE))
    parser.add_argument(
        '--collision-scene-only',
        action='store_true',
        help='Regenerate the cuMotion .scene file without touching the USD.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the collection USD and its cuMotion collision scene."""
    args = _build_argument_parser().parse_args(argv)
    task = load_task_config(args.task_config)
    scene = load_usd_scene_config(args.scene_config)
    assert_valid_against_scene(task, scene)

    collision_scene_path = Path(args.collision_scene).expanduser()
    collision_scene_path.write_text(
        render_collision_scene(task, scene),
        encoding='utf-8',
    )
    print(f'[author-vla-scene] collision scene: {collision_scene_path}')

    if args.collision_scene_only:
        return 0

    destination = Path(args.destination_usd).expanduser()
    author_scene(
        Path(args.source_usd).expanduser(),
        destination,
        task,
        scene,
    )
    print(f'[author-vla-scene] scene: {destination}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
