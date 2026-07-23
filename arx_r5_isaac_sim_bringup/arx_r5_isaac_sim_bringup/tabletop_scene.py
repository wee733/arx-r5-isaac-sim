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

"""Create and animate the ARX R5A tabletop AprilTag demo scene."""

from dataclasses import dataclass
from math import sqrt

from .demo_config import DemoConfig
from .pose_math import (
    multiply_quaternions,
    normalize_quaternion,
    rotate_vector,
)


CAMERA_PRIM_PATH = '/World/Sensors/Camera_1'
CUBE_ROOT_PATH = '/World/Workspace/TaggedCube'
LINK6_NAME = 'link6'


def _tag_quad(stage, path: str, size: float, texture_path, translation) -> None:
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    half_size = size / 2.0
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-half_size, -half_size, 0.0),
        Gf.Vec3f(half_size, -half_size, 0.0),
        Gf.Vec3f(half_size, half_size, 0.0),
        Gf.Vec3f(-half_size, half_size, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([
        Gf.Vec3f(-half_size, -half_size, 0.0),
        Gf.Vec3f(half_size, half_size, 0.0),
    ])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    texture_coordinates = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'st',
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    texture_coordinates.Set(Vt.Vec2fArray([
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ]))
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*translation))

    material_path = f'{path}/Material'
    omni.kit.commands.execute(
        'CreateMdlMaterialPrim',
        mtl_url='OmniPBR.mdl',
        mtl_name='OmniPBR',
        mtl_path=material_path,
    )
    material_prim = stage.GetPrimAtPath(material_path)
    shader = UsdShade.Shader(
        omni.usd.get_shader_from_material(material_prim, get_prim=True)
    )
    shader.CreateInput('diffuse_texture', Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path))
    )
    shader.CreateInput('project_uvw', Sdf.ValueTypeNames.Bool).Set(False)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
        UsdShade.Material(material_prim),
        UsdShade.Tokens.strongerThanDescendants,
    )


def _find_named_prim(stage, root_path: str, name: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    matches = [prim for prim in Usd.PrimRange(root) if prim.GetName() == name]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(
            f'expected one {name!r} prim below {root_path}, found {paths}'
        )
    return matches[0]


class VisualAttachmentController:
    """Mirror grasp/release events by moving the rendered cube with link6."""

    def __init__(self, stage, robot, robot_path: str, config: DemoConfig) -> None:
        from pxr import UsdGeom

        self._stage = stage
        self._robot = robot
        self._config = config.attachment
        self._link6_prim = _find_named_prim(stage, robot_path, LINK6_NAME)
        cube_root = UsdGeom.Xformable(stage.GetPrimAtPath(CUBE_ROOT_PATH))
        operations = {
            operation.GetOpName(): operation
            for operation in cube_root.GetOrderedXformOps()
        }
        self._translate = operations['xformOp:translate']
        self._orient = operations['xformOp:orient']
        self._attached = False
        self._relative_position = None
        self._relative_rotation = None

    @property
    def attached(self) -> bool:
        """Return whether the visual cube currently follows the gripper."""
        return self._attached

    def _grasp_pose(self):
        from pxr import Gf, UsdGeom

        cache = UsdGeom.XformCache()
        link6_transform = cache.GetLocalToWorldTransform(self._link6_prim)
        position = link6_transform.Transform(
            Gf.Vec3d(*self._config.grasp_frame_offset)
        )
        orientation = link6_transform.ExtractRotationQuat()
        return position, orientation

    def update(self) -> None:
        """Update attachment state from the simulated gripper aperture."""
        from pxr import Gf

        positions = self._robot.get_joint_positions()
        if positions is None:
            return
        joint_names = tuple(self._robot.dof_names)
        if 'joint7' not in joint_names:
            return
        aperture = float(positions[joint_names.index('joint7')])
        grasp_position, grasp_orientation = self._grasp_pose()
        grasp_rotation = normalize_quaternion((
            *tuple(grasp_orientation.GetImaginary()),
            grasp_orientation.GetReal(),
        ))
        cube_position = self._translate.Get()
        distance = sqrt(sum(
            (float(cube_position[index]) - float(grasp_position[index])) ** 2
            for index in range(3)
        ))

        if (
            not self._attached
            and aperture <= self._config.close_threshold
            and distance <= self._config.maximum_distance
        ):
            self._attached = True
            inverse_grasp_rotation = (
                -grasp_rotation[0],
                -grasp_rotation[1],
                -grasp_rotation[2],
                grasp_rotation[3],
            )
            cube_rotation_value = self._orient.Get()
            cube_rotation = normalize_quaternion((
                *tuple(cube_rotation_value.GetImaginary()),
                cube_rotation_value.GetReal(),
            ))
            self._relative_position = rotate_vector(
                inverse_grasp_rotation,
                tuple(
                    float(cube_position[index])
                    - float(grasp_position[index])
                    for index in range(3)
                ),
            )
            self._relative_rotation = multiply_quaternions(
                inverse_grasp_rotation,
                cube_rotation,
            )
            print(
                '[arx-r5-sim] tagged cube attached at grasp_frame',
                flush=True,
            )

        if not self._attached:
            return

        cube_offset = rotate_vector(
            grasp_rotation,
            self._relative_position,
        )
        cube_position = tuple(
            float(grasp_position[index]) + cube_offset[index]
            for index in range(3)
        )
        cube_rotation = multiply_quaternions(
            grasp_rotation,
            self._relative_rotation,
        )
        self._translate.Set(Gf.Vec3d(*cube_position))
        self._orient.Set(Gf.Quatd(
            cube_rotation[3],
            cube_rotation[0],
            cube_rotation[1],
            cube_rotation[2],
        ))
        if aperture >= self._config.open_threshold:
            self._attached = False
            self._relative_position = None
            self._relative_rotation = None
            print(
                '[arx-r5-sim] tagged cube released from grasp_frame',
                flush=True,
            )


@dataclass(frozen=True)
class TabletopScene:
    """Runtime handles for the generated tabletop scene."""

    camera_path: str
    attachment: VisualAttachmentController


def create_tabletop_scene(
    world,
    stage,
    robot,
    robot_path: str,
    config: DemoConfig,
) -> TabletopScene:
    """Create the table, tagged cube, destination tag, and fixed RGB-D camera."""
    import numpy as np
    from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, '/World/Workspace')
    UsdGeom.Xform.Define(stage, '/World/Sensors')

    table = config.table
    world.scene.add(FixedCuboid(
        prim_path='/World/Workspace/TableTop',
        name='table_top',
        position=np.asarray(table.top_center),
        scale=np.asarray(table.top_size),
        size=1.0,
        color=np.asarray([0.46, 0.36, 0.26]),
    ))
    leg_height = table.leg_size[2]
    leg_center_z = (
        table.top_center[2]
        - table.top_size[2] / 2.0
        - leg_height / 2.0
    )
    x_offset = table.top_size[0] / 2.0 - table.leg_inset
    y_offset = table.top_size[1] / 2.0 - table.leg_inset
    leg_index = 0
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            world.scene.add(FixedCuboid(
                prim_path=f'/World/Workspace/TableLeg_{leg_index}',
                name=f'table_leg_{leg_index}',
                position=np.asarray([
                    table.top_center[0] + x_sign * x_offset,
                    table.top_center[1] + y_sign * y_offset,
                    leg_center_z,
                ]),
                scale=np.asarray(table.leg_size),
                size=1.0,
                color=np.asarray([0.32, 0.25, 0.18]),
            ))
            leg_index += 1

    source = config.source_object
    cube_root = UsdGeom.Xform.Define(stage, CUBE_ROOT_PATH)
    cube_xform = UsdGeom.Xformable(cube_root)
    cube_xform.AddTranslateOp().Set(Gf.Vec3d(*source.center))
    cube_xform.AddOrientOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    VisualCuboid(
        prim_path=f'{CUBE_ROOT_PATH}/Body',
        name='tagged_cube_body',
        translation=np.zeros(3),
        scale=np.asarray(source.size),
        size=1.0,
        color=np.asarray([0.88, 0.04, 0.03]),
    )
    _tag_quad(
        stage,
        f'{CUBE_ROOT_PATH}/AprilTag_{source.tag_id}',
        source.tag_quad_size,
        source.tag_texture,
        (0.0, 0.0, source.size[2] / 2.0 + 0.0002),
    )

    drop = config.drop_target
    _tag_quad(
        stage,
        f'/World/Workspace/DropTag_{drop.tag_id}',
        drop.tag_quad_size,
        drop.tag_texture,
        drop.center,
    )

    camera_config = config.camera
    camera = UsdGeom.Camera.Define(stage, CAMERA_PRIM_PATH)
    camera.CreateFocalLengthAttr(camera_config.focal_length)
    camera.CreateHorizontalApertureAttr(camera_config.horizontal_aperture)
    camera.CreateVerticalApertureAttr(
        camera_config.horizontal_aperture
        * camera_config.height
        / camera_config.width
    )
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 20.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*camera_config.position))
    camera_rotation = camera_config.usd_rotation
    camera_xform.AddOrientOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Quatd(
        camera_rotation[3],
        camera_rotation[0],
        camera_rotation[1],
        camera_rotation[2],
    ))

    attachment = VisualAttachmentController(stage, robot, robot_path, config)
    print(
        '[arx-r5-sim] tabletop demo: '
        f'tag {source.tag_id} cube -> tag {drop.tag_id} target',
        flush=True,
    )
    print(
        '[arx-r5-sim] camera topics: '
        f'{camera_config.color_image_topic}, {camera_config.color_info_topic}',
        flush=True,
    )
    return TabletopScene(
        camera_path=CAMERA_PRIM_PATH,
        attachment=attachment,
    )
