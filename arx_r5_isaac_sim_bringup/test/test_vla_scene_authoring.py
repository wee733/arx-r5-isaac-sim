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

"""
Verify the generated VLA workcell USD matches its own contract.

The AprilTag scene drifted away from ``arx_sim_usd_scene.yaml`` because it was
edited by hand in the Isaac Sim viewport; the simulator now refuses to open it.
These tests exist so the generated scene cannot repeat that.

The USD Python bindings live inside the Isaac Sim wheel and are unavailable to
a plain interpreter, so every test here skips when ``pxr`` cannot be imported.
"""

import math
from pathlib import Path

from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.task_config import load_task_config

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCENE_USD = PACKAGE_ROOT / 'assets' / 'scenes' / 'arx_vla_scene.usd'
TASK_CONFIG = PACKAGE_ROOT / 'config' / 'vla_task.yaml'
SCENE_CONFIG = PACKAGE_ROOT / 'config' / 'vla_scene.yaml'

REMOVED_PRIM_PATHS = (
    '/World/Workspace/TableTop',
    '/World/Workspace/TableLeg_0',
    '/World/Workspace/TableLeg_1',
    '/World/Workspace/TableLeg_2',
    '/World/Workspace/TableLeg_3',
    '/World/Workspace/DropTag_1',
    '/World/Workspace/TaggedCube',
)

try:
    import pxr
except ImportError:
    pxr = None


pytestmark = pytest.mark.skipif(
    pxr is None,
    reason='the USD Python bindings ship with Isaac Sim',
)


@pytest.fixture(name='stage', scope='module')
def stage_fixture():
    """Open the generated workcell without resolving remote sensor payloads."""
    from pxr import Usd

    if not SCENE_USD.is_file():
        pytest.skip(
            f'{SCENE_USD.name} has not been generated; '
            'run scripts/run_author_vla_scene.sh'
        )
    stage = Usd.Stage.Open(str(SCENE_USD), Usd.Stage.LoadNone)
    assert stage is not None, f'failed to open {SCENE_USD}'
    return stage


@pytest.fixture(name='task', scope='module')
def task_fixture():
    """Return the task configuration the scene was generated from."""
    return load_task_config(TASK_CONFIG)


@pytest.fixture(name='scene', scope='module')
def scene_fixture():
    """Return the USD scene contract the simulator validates against."""
    return load_usd_scene_config(SCENE_CONFIG)


def _world_pose(stage, prim_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f'missing prim: {prim_path}'
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    return tuple(float(component) for component in translation)


def _local_scale(stage, prim_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f'missing prim: {prim_path}'
    for operation in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if operation.GetOpName() == 'xformOp:scale':
            return tuple(float(component) for component in operation.Get())
    raise AssertionError(f'{prim_path} has no scale op')


def test_table_and_tags_are_gone(stage):
    """The collection workcell has no table and no AprilTags."""
    for prim_path in REMOVED_PRIM_PATHS:
        assert not stage.GetPrimAtPath(prim_path).IsValid(), prim_path


def test_workspace_contains_only_the_new_geometry(stage):
    """Only the column, the platform, the block and its target remain."""
    workspace = stage.GetPrimAtPath('/World/Workspace')
    assert workspace.IsValid()
    names = sorted(child.GetName() for child in workspace.GetChildren())
    assert names == ['Block', 'Mount', 'PlaceMarker', 'Platform']


def test_place_marker_is_a_visual_only_decal(stage, task):
    """The marker must not enter the collision world or the physics scene."""
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(task.place_marker.prim_path)
    assert prim.IsValid()
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)


def test_place_marker_geometry_matches_the_task(stage, task):
    """The decal must be the configured footprint, flat on the platform."""
    from pxr import UsdGeom

    marker = task.place_marker
    scale = _local_scale(stage, marker.prim_path)
    assert scale[0] == pytest.approx(marker.size[0])
    assert scale[1] == pytest.approx(marker.size[1])
    assert scale[2] == pytest.approx(marker.height_above_surface)

    position = _world_pose(stage, marker.prim_path)
    expected_z = (
        task.workcell.platform.top_z + marker.height_above_surface
    )
    assert position[2] == pytest.approx(expected_z, abs=1e-6)

    colour = UsdGeom.Cube(
        stage.GetPrimAtPath(marker.prim_path)
    ).GetDisplayColorAttr().Get()
    assert colour is not None and len(colour) == 1
    for index in range(3):
        assert colour[0][index] == pytest.approx(marker.color[index], abs=1e-6)


def test_robot_matches_the_scene_contract(stage, scene):
    """The authored /R5a pose must equal the contract the simulator asserts."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(scene.robot_prim_path)
    assert prim.IsValid()
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = tuple(
        float(component) for component in matrix.ExtractTranslation()
    )
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    rotation = (
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quaternion.GetReal()),
    )
    magnitude = math.sqrt(sum(value * value for value in rotation))
    rotation = tuple(value / magnitude for value in rotation)

    for index in range(3):
        assert translation[index] == pytest.approx(
            scene.expected_world_to_base_translation[index],
            abs=1e-6,
        )
    dot = abs(sum(
        rotation[index] * scene.expected_world_to_base_rotation[index]
        for index in range(4)
    ))
    assert math.degrees(2.0 * math.acos(min(1.0, dot))) < 0.01


def test_ground_plane_is_at_the_configured_height(stage, task):
    """The floor moved up from the old table-leg depth to the workcell floor."""
    position = _world_pose(stage, '/World/defaultGroundPlane')
    assert position[2] == pytest.approx(task.workcell.ground_z, abs=1e-6)


def test_block_geometry_matches_the_task(stage, task):
    """The block must be the configured size at the configured pose."""
    position = _world_pose(stage, task.block.prim_path)
    for index in range(3):
        assert position[index] == pytest.approx(
            task.block.nominal_center[index],
            abs=1e-6,
        )
    scale = _local_scale(stage, task.block.body_prim_path)
    for index in range(3):
        assert scale[index] == pytest.approx(task.block.size[index], abs=1e-6)


def test_block_body_is_centred_on_its_parent(stage, task):
    """The body offset must be zero so the rigid body sits on the centre."""
    parent = _world_pose(stage, task.block.prim_path)
    body = _world_pose(stage, task.block.body_prim_path)
    for index in range(3):
        assert body[index] == pytest.approx(parent[index], abs=1e-9)


def test_mount_and_platform_match_the_task(stage, task):
    """The generated boxes must match the configuration they came from."""
    for spec in (task.workcell.mount, task.workcell.platform):
        position = _world_pose(stage, spec.prim_path)
        scale = _local_scale(stage, spec.prim_path)
        for index in range(3):
            assert position[index] == pytest.approx(
                spec.center[index],
                abs=1e-6,
            )
            assert scale[index] == pytest.approx(spec.size[index], abs=1e-6)


def test_platform_supports_the_block(stage, task):
    """The block must rest exactly on the platform's top face."""
    platform_top = task.workcell.platform.top_z
    block_bottom = task.block.nominal_center[2] - task.block.size[2] / 2.0
    assert block_bottom == pytest.approx(platform_top, abs=1e-9)
    assert _world_pose(stage, task.block.prim_path)[2] == pytest.approx(
        task.block.nominal_center[2],
        abs=1e-6,
    )


def test_new_boxes_are_colliders(stage, task):
    """The .scene file feeds cuMotion, but PhysX needs real colliders too."""
    from pxr import UsdPhysics

    for spec in (task.workcell.mount, task.workcell.platform):
        prim = stage.GetPrimAtPath(spec.prim_path)
        assert prim.HasAPI(UsdPhysics.CollisionAPI), spec.prim_path


def test_cameras_and_robot_links_survived(stage, scene):
    """
    Authoring must not disturb the articulation or the sensor mounts.

    Only the mount prims are checked. The ZED X and RealSense assets arrive as
    remote payloads, so their interiors do not exist on a ``LoadNone`` stage;
    the simulator asserts those extrinsics itself once they resolve.
    """
    for prim_path in (
        '/R5a/base_link/ZED_X',
        '/R5a/link6/rsd455',
    ):
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.IsValid(), prim_path
        assert prim.HasAuthoredPayloads(), (
            f'{prim_path} lost its sensor payload during authoring'
        )
    for prim_path in (
        scene.base_prim_path,
        '/R5a/link6',
        '/R5a/link6/grasp_frame',
        '/R5a/joints/joint8',
    ):
        assert stage.GetPrimAtPath(prim_path).IsValid(), prim_path


def test_sensor_rigid_body_paths_stay_under_their_mounts(scene):
    """The contract's sensor paths must live inside the payload mounts."""
    for prim_path in scene.sensor_rigid_body_paths:
        assert prim_path.startswith(
            ('/R5a/base_link/ZED_X', '/R5a/link6/rsd455')
        ), prim_path


def test_source_scene_is_untouched():
    """The AprilTag workcell must remain available as its own scene."""
    from pxr import Usd

    source = PACKAGE_ROOT / 'assets' / 'scenes' / 'arx_sim.usd'
    if not source.is_file():
        pytest.skip('the AprilTag workcell USD is not present')
    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadNone)
    assert stage.GetPrimAtPath('/World/Workspace/TableTop').IsValid()
    assert stage.GetPrimAtPath('/World/Workspace/TaggedCube').IsValid()
