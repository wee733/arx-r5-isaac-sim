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

"""Build and run the fixed-base ARX R5A scene in Isaac Sim 5.1."""

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Optional, Sequence

from arx_r5_isaac_sim_bringup.contracts import (
    ALL_JOINTS,
    CLOCK_TOPIC,
    DEFAULT_JOINT_POSITIONS,
    DESCRIPTION_PACKAGE,
    GRIPPER_COMMAND_JOINT,
    GRIPPER_MAX_VELOCITY,
    GRIPPER_MIMIC_JOINT,
    JOINT_COMMANDS_TOPIC,
    JOINT_POSITION_LIMITS,
    JOINT_STATES_TOPIC,
    positions_from_sequence,
    resolve_description_mesh_uris,
)
from arx_r5_isaac_sim_bringup.demo_config import (
    DemoConfig,
    load_demo_config,
)
from arx_r5_isaac_sim_bringup.tabletop_scene import create_tabletop_scene
from arx_r5_isaac_sim_bringup.usd_scene import (
    AuthoredLayoutConfig,
    CameraPublisherConfig,
    load_usd_scene_config,
    UsdSceneConfig,
)


BRINGUP_PACKAGE = 'arx_r5_isaac_sim_bringup'
TABLETOP_DEMO_CONFIG = 'tabletop_apriltag_demo.yaml'
AUTHORED_USD_DEMO_CONFIG = 'authored_usd_apriltag_demo.yaml'
USD_SCENE_CONFIG = 'arx_sim_usd_scene.yaml'
USD_SCENE_ASSET = 'arx_sim.usd'
AUTHORED_WORKSPACE_PRIM_PATH = '/World/Workspace'
AUTHORED_GRIPPER_MIMIC_JOINT_PATH = '/R5a/joints/joint8'
AUTHORED_GRIPPER_COLLISION_PRIMS = (
    (
        '/colliders/link7/link7/node_STL_BINARY_',
        '/R5a/link7/collisions/link7/node_STL_BINARY_',
    ),
    (
        '/colliders/link8/link8/node_STL_BINARY_',
        '/R5a/link8/collisions/link8/node_STL_BINARY_',
    ),
)
AUTHORED_GRIPPER_BODY_PRIMS = (
    '/R5a/link7',
    '/R5a/link8',
)


@dataclass(frozen=True)
class RosTransform:
    """One transform published from the Isaac Sim ROS 2 bridge."""

    parent_frame: str
    child_frame: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


class BilateralFingerContactTracker:
    """Track simultaneous object contact with both gripper finger bodies."""

    def __init__(
        self,
        object_body_path: str,
        finger_body_paths: Sequence[str],
    ) -> None:
        """Initialize the two-finger contact contract."""
        self._object_body_path = str(object_body_path)
        self._finger_body_paths = frozenset(
            str(path) for path in finger_body_paths
        )
        if not self._object_body_path:
            raise ValueError('object body path must be non-empty')
        if len(self._finger_body_paths) != 2 or '' in self._finger_body_paths:
            raise ValueError(
                'exactly two non-empty finger body paths are required'
            )
        if self._object_body_path in self._finger_body_paths:
            raise ValueError('object and finger body paths must differ')
        self._active_contacts = set()

    def update(
        self,
        actor0: str,
        collider0: str,
        actor1: str,
        collider1: str,
        active: bool,
    ) -> None:
        """Add or remove one PhysX actor/collider contact pair."""
        endpoint0 = (str(actor0), str(collider0))
        endpoint1 = (str(actor1), str(collider1))
        actors = {endpoint0[0], endpoint1[0]}
        if self._object_body_path not in actors:
            return
        if not (actors & self._finger_body_paths):
            return
        contact = tuple(sorted((endpoint0, endpoint1)))
        if active:
            self._active_contacts.add(contact)
        else:
            self._active_contacts.discard(contact)

    def clear(self) -> None:
        """Forget contact state after a release or simulation reset."""
        self._active_contacts.clear()

    @property
    def has_bilateral_contact(self) -> bool:
        """Return whether the object currently touches both finger bodies."""
        touching_fingers = set()
        for endpoint0, endpoint1 in self._active_contacts:
            actors = {endpoint0[0], endpoint1[0]}
            if self._object_body_path in actors:
                touching_fingers.update(actors & self._finger_body_paths)
        return touching_fingers == self._finger_body_paths


def _parse_position_vector(value: str) -> dict[str, float]:
    values = [float(item.strip()) for item in value.split(',') if item.strip()]
    positions = dict(positions_from_sequence(values))
    invalid = {
        name: position
        for name, position in positions.items()
        if not math.isfinite(position)
        or not (
            JOINT_POSITION_LIMITS[name][0]
            <= position
            <= JOINT_POSITION_LIMITS[name][1]
        )
    }
    if invalid:
        raise ValueError(f'initial joint positions exceed URDF limits: {invalid}')
    if positions['joint8'] != positions['joint7']:
        raise ValueError('joint8 must equal its source mimic joint7 at startup')
    return positions


def _description_share_from_ament() -> Optional[Path]:
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None

    try:
        return Path(get_package_share_directory(DESCRIPTION_PACKAGE))
    except Exception:  # ament raises different lookup errors across ROS releases
        return None


def _source_tree_candidates() -> list[Path]:
    module_path = Path(__file__).resolve()
    candidates = []
    for parent in module_path.parents:
        candidates.extend(
            [
                parent / 'arx-r5-moveit' / DESCRIPTION_PACKAGE,
                parent / 'isaac_ros_manipulation' / 'arx-r5-moveit' /
                DESCRIPTION_PACKAGE,
            ]
        )
    return candidates


def find_description_share(explicit_path: Optional[str]) -> Path:
    """Find the installed or source-tree ARX description package."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get('ARX_R5_DESCRIPTION_SHARE'):
        candidates.append(Path(os.environ['ARX_R5_DESCRIPTION_SHARE']))

    ament_share = _description_share_from_ament()
    if ament_share is not None:
        candidates.append(ament_share)
    candidates.extend(_source_tree_candidates())

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / 'urdf' / 'r5a_cumotion.urdf').is_file():
            return candidate

    raise FileNotFoundError(
        'ARX R5A description package was not found. Source the '
        'arx-r5-moveit install space, set ARX_R5_DESCRIPTION_SHARE, or pass '
        '--description-share.'
    )


def find_demo_config(
    explicit_path: Optional[str],
    default_filename: str = TABLETOP_DEMO_CONFIG,
) -> Path:
    """Find the installed or source-tree tabletop demo configuration."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get('ARX_R5_SIM_DEMO_CONFIG'):
        candidates.append(Path(os.environ['ARX_R5_SIM_DEMO_CONFIG']))

    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory(BRINGUP_PACKAGE))
            / 'config'
            / default_filename
        )
    except Exception:
        pass

    candidates.append(
        Path(__file__).resolve().parents[1]
        / 'config'
        / default_filename
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'demo config {default_filename!r} was not found; pass --demo-config or set '
        'ARX_R5_SIM_DEMO_CONFIG'
    )


def find_usd_scene_config(explicit_path: Optional[str]) -> Path:
    """Find the authored workcell's sensor and frame contract."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get('ARX_R5_SIM_USD_CONFIG'):
        candidates.append(Path(os.environ['ARX_R5_SIM_USD_CONFIG']))
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory(BRINGUP_PACKAGE))
            / 'config'
            / USD_SCENE_CONFIG
        )
    except Exception:
        pass
    candidates.append(
        Path(__file__).resolve().parents[1] / 'config' / USD_SCENE_CONFIG
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        'authored USD scene config was not found; pass --usd-scene-config or '
        'set ARX_R5_SIM_USD_CONFIG'
    )


def find_usd_scene(explicit_path: Optional[str]) -> Path:
    """Find the committed authored workcell or an explicitly selected USD."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get('ARX_R5_SIM_USD'):
        candidates.append(Path(os.environ['ARX_R5_SIM_USD']))
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory(BRINGUP_PACKAGE))
            / 'assets'
            / 'scenes'
            / USD_SCENE_ASSET
        )
    except Exception:
        pass
    candidates.append(
        Path(__file__).resolve().parents[1]
        / 'assets'
        / 'scenes'
        / USD_SCENE_ASSET
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        'authored ARX USD scene was not found; pass --usd or set ARX_R5_SIM_USD'
    )


def materialize_isaac_urdf(
    source_urdf: Path,
    description_share: Path,
    destination: Path,
) -> Path:
    """Create an importer-safe URDF with resolved package mesh paths."""
    source_text = source_urdf.read_text(encoding='utf-8')
    resolved_text = resolve_description_mesh_uris(
        source_text,
        description_share,
    )
    unresolved_prefix = f'package://{DESCRIPTION_PACKAGE}/'
    if unresolved_prefix in resolved_text:
        raise ValueError(f'failed to resolve mesh URI prefix {unresolved_prefix}')
    destination.write_text(resolved_text, encoding='utf-8')
    return destination


def _build_argument_parser() -> argparse.ArgumentParser:
    default_positions = ','.join(
        str(DEFAULT_JOINT_POSITIONS[name]) for name in ALL_JOINTS
    )
    parser = argparse.ArgumentParser(
        description='Run the ARX R5A cuMotion execution scene in Isaac Sim.',
    )
    parser.add_argument(
        '--description-share',
        help='Path to the ARX robot-description package share/source directory.',
    )
    parser.add_argument(
        '--urdf',
        help=(
            'Plain URDF to import; defaults to '
            '<description-share>/urdf/r5a_cumotion.urdf.'
        ),
    )
    parser.add_argument(
        '--usd',
        nargs='?',
        const='',
        help=(
            'Open the authored dual-camera workcell instead of importing the '
            'URDF. With no path, use the committed assets/scenes/arx_sim.usd.'
        ),
    )
    parser.add_argument(
        '--usd-scene-config',
        help='Sensor/prim contract for the authored USD workcell.',
    )
    parser.add_argument(
        '--camera-mode',
        choices=('none', 'zedx', 'd455', 'both'),
        default='both',
        help='Camera streams to publish when --usd is selected.',
    )
    parser.add_argument(
        '--authored-layout',
        default='as-authored',
        help=(
            'Named non-persistent layout from arx_sim_usd_scene.yaml. '
            'Camera demo wrappers use reachable; as-authored preserves every '
            'source transform.'
        ),
    )
    parser.add_argument(
        '--reset-usd-joints',
        action='store_true',
        help='Reset an authored USD articulation to --initial-positions.',
    )
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--duration',
        type=float,
        default=0.0,
        help='Simulation seconds to run; zero runs until the app closes.',
    )
    parser.add_argument('--physics-dt', type=float, default=0.01)
    parser.add_argument('--rendering-dt', type=float, default=1.0 / 60.0)
    parser.add_argument(
        '--convex-decomposition',
        action='store_true',
        help='Use convex decomposition instead of one convex hull per collider.',
    )
    parser.add_argument('--drive-stiffness', type=float, default=625.0)
    parser.add_argument('--drive-damping', type=float, default=50.0)
    parser.add_argument(
        '--gripper-drive-max-force',
        type=float,
        default=8.0,
        help=(
            'Maximum force in newtons for both finger linear drives. '
            'TopicBasedSystem sends joint7 targets to joint7 and joint8.'
        ),
    )
    parser.add_argument(
        '--grasp-contact-steps',
        type=int,
        default=3,
        help=(
            'Consecutive physics steps with bilateral finger contact before '
            'creating the authored-workcell FixedJoint.'
        ),
    )
    parser.add_argument(
        '--initial-positions',
        default=default_positions,
        help='Comma-separated joint1..joint8 positions.',
    )
    parser.add_argument(
        '--ros-domain-id',
        type=int,
        help='Set ROS_DOMAIN_ID before the ROS 2 bridge is loaded.',
    )
    parser.add_argument(
        '--no-ros',
        action='store_true',
        help='Build the scene without the ROS 2 Action Graph.',
    )
    parser.add_argument(
        '--save-usd',
        help='Export the generated scene to this .usd/.usda/.usdc path.',
    )
    parser.add_argument(
        '--exit-after-build',
        action='store_true',
        help='Exit after import, graph creation, initialization, and optional export.',
    )
    parser.add_argument(
        '--scene',
        choices=('empty', 'tabletop'),
        default='empty',
        help='Optional workcell scene to add around the fixed-base robot.',
    )
    parser.add_argument(
        '--demo-config',
        help='Tabletop demo YAML; defaults to the package configuration.',
    )
    return parser


def _find_articulation_root(stage, robot_path: str) -> str:
    from pxr import Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f'imported robot prim does not exist: {robot_path}')

    articulation_roots = [
        prim
        for prim in Usd.PrimRange(robot_prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(articulation_roots) != 1:
        paths = [str(prim.GetPath()) for prim in articulation_roots]
        raise RuntimeError(
            f'expected exactly one articulation root below {robot_path}, '
            f'found {paths}'
        )
    return str(articulation_roots[0].GetPath())


def _validate_grasp_frame_import(stage, robot_path: str) -> None:
    from pxr import Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_path)
    grasp_frames = [
        prim for prim in Usd.PrimRange(robot_prim)
        if prim.GetName() == 'grasp_frame'
    ]
    if len(grasp_frames) > 1:
        paths = [str(prim.GetPath()) for prim in grasp_frames]
        raise RuntimeError(f'expected at most one grasp_frame prim, found {paths}')
    if grasp_frames and grasp_frames[0].HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            'grasp_frame was imported as a rigid body; fixed URDF links must '
            'be merged to preserve robot dynamics'
        )
    if not grasp_frames:
        print(
            '[arx-r5-sim] grasp_frame merged into link6 without a rigid body',
            flush=True,
        )


def _configure_joint_drives(
    stage,
    robot_path: str,
    stiffness: float,
    damping: float,
    gripper_max_force: float,
    target_positions: Optional[dict[str, float]],
) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_path)
    joint_prims = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(robot_prim)
        if prim.IsA(UsdPhysics.RevoluteJoint)
        or prim.IsA(UsdPhysics.PrismaticJoint)
    }
    missing = sorted(set(ALL_JOINTS) - set(joint_prims))
    if missing:
        raise RuntimeError(f'cannot configure missing joint drives: {missing}')

    preserved_gripper_target = None
    if target_positions is None:
        command_prim = joint_prims[GRIPPER_COMMAND_JOINT]
        command_drive = UsdPhysics.DriveAPI.Get(command_prim, 'linear')
        if not command_drive:
            raise RuntimeError('joint7 has no authored linear drive target')
        preserved_gripper_target = (
            command_drive.GetTargetPositionAttr().Get()
        )
        if preserved_gripper_target is None:
            raise RuntimeError('joint7 authored drive target is not set')

    for joint_name in ALL_JOINTS:
        joint_prim = joint_prims[joint_name]
        if (
            joint_name == GRIPPER_MIMIC_JOINT and
            joint_prim.HasAPI(PhysxSchema.PhysxMimicJointAPI)
        ):
            raise RuntimeError(
                'joint8 cannot combine a PhysX mimic constraint with its '
                'explicit TopicBasedSystem drive'
            )
        drive_name = (
            'linear' if joint_prim.IsA(UsdPhysics.PrismaticJoint) else 'angular'
        )
        drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_name)
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        if joint_name in (GRIPPER_COMMAND_JOINT, GRIPPER_MIMIC_JOINT):
            if drive_name != 'linear':
                raise RuntimeError(
                    f'{joint_name} must use a linear PhysX drive'
                )
            lower, upper = JOINT_POSITION_LIMITS[joint_name]
            prismatic_joint = UsdPhysics.PrismaticJoint(joint_prim)
            prismatic_joint.CreateLowerLimitAttr(lower)
            prismatic_joint.CreateUpperLimitAttr(upper)
            PhysxSchema.PhysxJointAPI.Apply(
                joint_prim
            ).CreateMaxJointVelocityAttr(GRIPPER_MAX_VELOCITY)
            drive.CreateTypeAttr(UsdPhysics.Tokens.acceleration)
            drive.CreateMaxForceAttr(gripper_max_force)
        if target_positions is None:
            if joint_name not in (
                GRIPPER_COMMAND_JOINT,
                GRIPPER_MIMIC_JOINT,
            ):
                continue
            target = preserved_gripper_target
        else:
            target = target_positions[joint_name]
        if drive_name == 'angular':
            target = math.degrees(target)
        drive.CreateTargetPositionAttr(target)

    print(
        '[arx-r5-sim] gripper drives joint7/joint8 max force: '
        f'{gripper_max_force:.3f} N each (joint8 mirrors joint7 commands)',
        flush=True,
    )


def _gf_matrix_pose(matrix):
    """Return a normalized ROS xyz/xyzw pose from a USD Gf matrix."""
    translation_value = matrix.ExtractTranslation()
    quaternion_value = matrix.ExtractRotationQuat()
    rotation = (
        float(quaternion_value.GetImaginary()[0]),
        float(quaternion_value.GetImaginary()[1]),
        float(quaternion_value.GetImaginary()[2]),
        float(quaternion_value.GetReal()),
    )
    magnitude = math.sqrt(sum(component * component for component in rotation))
    if magnitude <= 1e-12:
        raise ValueError('USD transform produced a zero quaternion')
    return (
        tuple(float(component) for component in translation_value),
        tuple(component / magnitude for component in rotation),
    )


def _unit_vector(values: Sequence[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(float(value) ** 2 for value in values))
    if magnitude <= 1e-12:
        raise ValueError('cannot normalize a zero vector')
    return tuple(float(value) / magnitude for value in values)


def _direction_error_degrees(
    actual: Sequence[float],
    expected: Sequence[float],
) -> float:
    actual_unit = _unit_vector(actual)
    expected_unit = _unit_vector(expected)
    dot_product = sum(
        actual_unit[index] * expected_unit[index]
        for index in range(len(actual_unit))
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_product))))


def _quaternion_error_degrees(
    actual: Sequence[float],
    expected: Sequence[float],
) -> float:
    actual_unit = _unit_vector(actual)
    expected_unit = _unit_vector(expected)
    dot_product = abs(sum(
        actual_unit[index] * expected_unit[index]
        for index in range(4)
    ))
    return math.degrees(2.0 * math.acos(min(1.0, dot_product)))


def _camera_optical_transform(
    stage,
    camera: CameraPublisherConfig,
    expected_translation: Optional[Sequence[float]] = None,
) -> RosTransform:
    """Extract parent-to-camera ROS optical extrinsics from authored USD."""
    from pxr import Gf, UsdGeom

    camera_prim = stage.GetPrimAtPath(camera.camera_prim_path)
    parent_prim = stage.GetPrimAtPath(camera.parent_prim_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            f'camera profile {camera.name!r} has no USD Camera at '
            f'{camera.camera_prim_path}'
        )
    if not parent_prim.IsValid() or not parent_prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(
            f'camera profile {camera.name!r} has no Xform parent at '
            f'{camera.parent_prim_path}'
        )

    cache = UsdGeom.XformCache()
    camera_to_parent, _ = cache.ComputeRelativeTransform(
        camera_prim,
        parent_prim,
    )
    # USD Camera coordinates are +X right, +Y up, -Z forward. ROS optical
    # coordinates are +X right, +Y down, +Z forward. Gf uses row-vector
    # matrices, so the local Rx(pi) correction premultiplies the relative pose.
    usd_camera_to_ros_optical = Gf.Matrix4d(
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    optical_to_parent = usd_camera_to_ros_optical * camera_to_parent
    translation, rotation = _gf_matrix_pose(optical_to_parent)
    expected_translation = tuple(
        camera.expected_parent_to_optical_translation
        if expected_translation is None
        else expected_translation
    )
    translation_error = math.dist(translation, expected_translation)
    if translation_error > 1e-5:
        raise RuntimeError(
            f'camera profile {camera.name!r} moved {translation_error:.6f} m '
            'from its configured parent-to-optical translation; update the '
            'USD scene and arx_sim_usd_scene.yaml together'
        )
    optical_forward = tuple(float(component) for component in (
        optical_to_parent.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    ))
    direction_error = _direction_error_degrees(
        optical_forward,
        camera.expected_optical_forward,
    )
    if direction_error > 0.01:
        raise RuntimeError(
            f'camera profile {camera.name!r} optical axis differs by '
            f'{direction_error:.6f} degrees from its configured mount '
            'contract; update the USD scene and arx_sim_usd_scene.yaml '
            'together'
        )
    return RosTransform(
        parent_frame=camera.parent_frame,
        child_frame=camera.optical_frame,
        translation=translation,
        rotation=rotation,
    )


def _normalize_camera_projection(stage, camera: CameraPublisherConfig) -> None:
    """Keep rendered pixels and ROS CameraInfo on one pinhole model."""
    from pxr import UsdGeom

    camera_prim = stage.GetPrimAtPath(camera.camera_prim_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            f'camera prim does not exist: {camera.camera_prim_path}'
        )
    usd_camera = UsdGeom.Camera(camera_prim)
    horizontal_aperture = float(usd_camera.GetHorizontalApertureAttr().Get())
    if not math.isfinite(horizontal_aperture) or horizontal_aperture <= 0.0:
        raise RuntimeError(
            f'invalid horizontal aperture at {camera.camera_prim_path}'
        )
    vertical_aperture = horizontal_aperture * camera.height / camera.width
    if not usd_camera.GetVerticalApertureAttr().Set(vertical_aperture):
        raise RuntimeError(
            f'failed to normalize vertical aperture at {camera.camera_prim_path}'
        )
    print(
        f'[arx-r5-sim] {camera.name} aperture: '
        f'{horizontal_aperture:.6f} x {vertical_aperture:.6f} mm '
        f'for {camera.width}x{camera.height}',
        flush=True,
    )


def _world_to_base_transform(stage, config: UsdSceneConfig) -> RosTransform:
    """Extract the authored world-to-base transform without assuming identity."""
    from pxr import UsdGeom

    base_prim = stage.GetPrimAtPath(config.base_prim_path)
    if not base_prim.IsValid() or not base_prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f'base prim does not exist: {config.base_prim_path}')
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(base_prim)
    translation, rotation = _gf_matrix_pose(matrix)
    rotation_error = _quaternion_error_degrees(
        rotation,
        config.expected_world_to_base_rotation,
    )
    if rotation_error > 0.01:
        raise RuntimeError(
            f'authored world-to-base rotation differs by '
            f'{rotation_error:.6f} degrees from the scene contract; keep the '
            'robot root upright and apply camera pitch on the camera mount'
        )
    return RosTransform(
        parent_frame=config.world_frame,
        child_frame=config.base_frame,
        translation=translation,
        rotation=rotation,
    )


def _remove_sensor_body_apis(stage, prim_paths: Sequence[str]) -> None:
    """Make sensor assets inherit their articulation-link parent transform."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f'expected a sensor RigidBodyAPI at authored prim {prim_path}'
            )
        if UsdGeom.Xformable(prim).GetResetXformStack():
            raise RuntimeError(
                f'sensor mount must inherit its parent transform: {prim_path}'
            )
        for api_schema in (
            UsdPhysics.RigidBodyAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            UsdPhysics.MassAPI,
        ):
            if prim.HasAPI(api_schema) and not prim.RemoveAPI(api_schema):
                raise RuntimeError(
                    f'failed to remove {api_schema.__name__} from {prim_path}'
                )
        # The source schema's attributes remain composed even after its API
        # token is deleted. Mark the stale body explicitly disabled so Isaac
        # Sim's tensor discovery does not treat this Xform as an articulation
        # link while PhysX's hierarchy validator sees no nested body schema.
        rigid_body_enabled = prim.GetAttribute('physics:rigidBodyEnabled')
        if rigid_body_enabled.IsValid():
            rigid_body_enabled.Set(False)
        nested_bodies = [
            str(candidate.GetPath())
            for candidate in Usd.PrimRange(prim)
            if candidate.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if nested_bodies:
            raise RuntimeError(
                f'sensor subtree still contains rigid bodies: {nested_bodies}'
            )
        print(
            f'[arx-r5-sim] mounted sensor as link payload: {prim_path}',
            flush=True,
        )


def _apply_authored_layout(stage, layout: AuthoredLayoutConfig) -> None:
    """Apply a named layout only to the anonymous runtime session layer."""
    from pxr import Gf, UsdGeom

    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise RuntimeError(
            'authored layout overrides require the runtime session layer as '
            'the USD edit target'
        )

    for prim_path, translation in layout.prim_translations.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(
                f'layout {layout.name!r} references invalid Xform {prim_path}'
            )
        translate_ops = [
            operation
            for operation in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if len(translate_ops) != 1:
            raise RuntimeError(
                f'layout {layout.name!r} expected one translate op at '
                f'{prim_path}, found {len(translate_ops)}'
            )
        operation = translate_ops[0]
        value = (
            Gf.Vec3f(*translation)
            if operation.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
            else Gf.Vec3d(*translation)
        )
        if not operation.Set(value):
            raise RuntimeError(
                f'failed to apply layout {layout.name!r} at {prim_path}'
            )
        print(
            f'[arx-r5-sim] layout {layout.name}: {prim_path} -> '
            f'{tuple(translation)}',
            flush=True,
        )


def _configure_authored_source_object(
    stage,
    config: UsdSceneConfig,
) -> None:
    """Add non-persistent rigid-body physics to the authored red block."""
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    object_prim = stage.GetPrimAtPath(config.source_object_prim_path)
    collision_prim = stage.GetPrimAtPath(config.source_collision_prim_path)
    if not object_prim.IsValid() or not object_prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(
            f'source object prim does not exist: {config.source_object_prim_path}'
        )
    if not collision_prim.IsValid() or not collision_prim.IsA(UsdGeom.Gprim):
        raise RuntimeError(
            'source collision geometry does not exist: '
            f'{config.source_collision_prim_path}'
        )

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(object_prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    # The 50 x 50 x 150 mm block is intentionally slender and otherwise tips
    # from tiny solver impulses before perception has stabilized. Keep the
    # staged object collidable but kinematic until a verified gripper close;
    # the attachment controller enables dynamics immediately before creating
    # the physical grasp joint.
    rigid_body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(object_prim).CreateMassAttr(
        config.source_object_mass_kg
    )
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(object_prim)
    # PhysX rejects CCD on a kinematic body. Enable it only when the grasp
    # controller switches the object to dynamic immediately before attaching.
    physx_body.CreateEnableCCDAttr(False)
    contact_report = PhysxSchema.PhysxContactReportAPI.Apply(object_prim)
    contact_report.CreateThresholdAttr(0.0)

    collision = UsdPhysics.CollisionAPI.Apply(collision_prim)
    collision.CreateCollisionEnabledAttr(True)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    physx_collision.CreateContactOffsetAttr(0.002)
    physx_collision.CreateRestOffsetAttr(0.0)

    material_path = Sdf.Path('/World/Runtime/TaggedObjectPhysicsMaterial')
    UsdGeom.Scope.Define(stage, material_path.GetParentPath())
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(1.2)
    material_api.CreateDynamicFrictionAttr(1.0)
    material_api.CreateRestitutionAttr(0.0)
    binding = collision_prim.CreateRelationship(
        'material:binding:physics',
        custom=False,
    )
    binding.SetTargets([material_path])
    print(
        '[arx-r5-sim] enabled collision and rigid-body physics for '
        f'{config.source_object_prim_path}',
        flush=True,
    )


def _configure_authored_gripper_collisions(stage) -> None:
    """Preserve the finger gaps with multi-hull collision geometry."""
    from pxr import UsdPhysics

    approximation = UsdPhysics.Tokens.convexDecomposition
    for source_path, _ in AUTHORED_GRIPPER_COLLISION_PRIMS:
        collision_prim = stage.GetPrimAtPath(source_path)
        if not collision_prim.IsValid():
            raise RuntimeError(
                f'authored gripper collision prim does not exist: {source_path}'
            )
        mesh_collision = UsdPhysics.MeshCollisionAPI(collision_prim)
        if not mesh_collision:
            raise RuntimeError(
                'authored gripper collision prim has no MeshCollisionAPI: '
                f'{source_path}'
            )
        approximation_attribute = mesh_collision.GetApproximationAttr()
        if (
            not approximation_attribute.IsValid() or
            not approximation_attribute.Set(approximation)
        ):
            raise RuntimeError(
                'failed to set convex decomposition on authored gripper '
                f'collision: {source_path}'
            )


def _disable_authored_gripper_mimic(stage) -> None:
    """Replace the compliant authored mimic with an explicit second drive."""
    from pxr import PhysxSchema, Sdf

    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise RuntimeError(
            'authored gripper mimic overrides require the runtime session '
            'layer as the USD edit target'
        )

    joint_prim = stage.GetPrimAtPath(AUTHORED_GRIPPER_MIMIC_JOINT_PATH)
    if not joint_prim.IsValid():
        raise RuntimeError(
            'authored gripper mimic joint does not exist: '
            f'{AUTHORED_GRIPPER_MIMIC_JOINT_PATH}'
        )
    mimic_schemas = [
        schema
        for schema in joint_prim.GetAppliedSchemas()
        if schema.startswith('PhysxMimicJointAPI:')
    ]
    if len(mimic_schemas) != 1:
        raise RuntimeError(
            'expected exactly one authored PhysxMimicJointAPI on '
            f'{AUTHORED_GRIPPER_MIMIC_JOINT_PATH}, found {mimic_schemas}'
        )
    instance_name = mimic_schemas[0].split(':', 1)[1]
    if not joint_prim.RemoveAPI(
        PhysxSchema.PhysxMimicJointAPI,
        instance_name,
    ):
        raise RuntimeError(
            'failed to remove authored PhysxMimicJointAPI from '
            f'{AUTHORED_GRIPPER_MIMIC_JOINT_PATH}'
        )
    if joint_prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
        raise RuntimeError(
            'authored PhysxMimicJointAPI remained after session override: '
            f'{AUTHORED_GRIPPER_MIMIC_JOINT_PATH}'
        )
    property_prefix = f'physxMimicJoint:{instance_name}:'
    for attribute in joint_prim.GetAttributes():
        if attribute.GetName().startswith(property_prefix):
            attribute.Set(Sdf.ValueBlock())
    for relationship in joint_prim.GetRelationships():
        if relationship.GetName().startswith(property_prefix):
            relationship.SetTargets([])
    print(
        '[arx-r5-sim] disabled compliant joint8 PhysX mimic; '
        'using explicit dual finger drives',
        flush=True,
    )


def _validate_authored_gripper_collision_instances(stage) -> None:
    """Verify source overrides propagated through the robot instances."""
    from pxr import UsdPhysics

    expected = UsdPhysics.Tokens.convexDecomposition
    for _, instance_path in AUTHORED_GRIPPER_COLLISION_PRIMS:
        collision_prim = stage.GetPrimAtPath(instance_path)
        if not collision_prim.IsValid() or not collision_prim.IsInstanceProxy():
            raise RuntimeError(
                'authored gripper collision instance proxy does not exist: '
                f'{instance_path}'
            )
        mesh_collision = UsdPhysics.MeshCollisionAPI(collision_prim)
        actual = (
            mesh_collision.GetApproximationAttr().Get()
            if mesh_collision
            else None
        )
        if actual != expected:
            raise RuntimeError(
                'authored gripper collision override did not reach instance '
                f'proxy {instance_path}: expected {expected}, got {actual}'
            )
        print(
            '[arx-r5-sim] gripper collision instance uses '
            f'{actual}: {instance_path}',
            flush=True,
        )


def _configure_authored_workspace_contacts(stage) -> None:
    """Replace oversized authored contact envelopes on environment shapes."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    workspace_prim = stage.GetPrimAtPath(AUTHORED_WORKSPACE_PRIM_PATH)
    if not workspace_prim.IsValid():
        raise RuntimeError(
            'authored workspace prim does not exist: '
            f'{AUTHORED_WORKSPACE_PRIM_PATH}'
        )
    collision_prims = [
        prim
        for prim in Usd.PrimRange(workspace_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not collision_prims:
        raise RuntimeError('authored workspace contains no collision shapes')
    for collision_prim in collision_prims:
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
        physx_collision.CreateContactOffsetAttr(0.002)
        physx_collision.CreateRestOffsetAttr(0.0)
    print(
        '[arx-r5-sim] normalized contact offsets for '
        f'{len(collision_prims)} authored workspace collision shapes',
        flush=True,
    )


def _repair_authored_tag_textures(
    stage,
    config: UsdSceneConfig,
    demo_config: DemoConfig,
) -> None:
    """Point copied USD material inputs at package-owned AprilTag textures."""
    from pxr import Sdf

    texture_contracts = (
        (config.source_tag_texture_prim, demo_config.source_object.tag_texture),
        (config.target_tag_texture_prim, demo_config.drop_target.tag_texture),
    )
    for shader_path, texture_path in texture_contracts:
        shader_prim = stage.GetPrimAtPath(shader_path)
        texture_attribute = shader_prim.GetAttribute('inputs:diffuse_texture')
        if not shader_prim.IsValid() or not texture_attribute.IsValid():
            raise RuntimeError(
                f'AprilTag texture shader input does not exist: {shader_path}'
            )
        texture_attribute.Set(Sdf.AssetPath(str(texture_path)))


def _attach_prepared_authored_stage(
    simulation_app,
    context,
    usd_scene_path: Path,
    config: UsdSceneConfig,
    demo_config: DemoConfig,
    layout: AuthoredLayoutConfig,
):
    """Prepare session overrides before Hydra and PhysX parse the stage."""
    from pxr import Sdf, Usd

    root_layer = Sdf.Layer.FindOrOpen(str(usd_scene_path))
    if root_layer is None:
        raise RuntimeError(f'failed to open USD layer: {usd_scene_path}')
    session_layer = Sdf.Layer.CreateAnonymous('arx_runtime_session.usda')
    stage = Usd.Stage.Open(root_layer, session_layer, Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f'failed to compose USD stage: {usd_scene_path}')
    stage.SetEditTarget(session_layer)

    _apply_authored_layout(stage, layout)
    _disable_authored_gripper_mimic(stage)
    _configure_authored_gripper_collisions(stage)
    _configure_authored_workspace_contacts(stage)
    _repair_authored_tag_textures(stage, config, demo_config)
    _remove_sensor_body_apis(stage, config.sensor_rigid_body_paths)
    _configure_authored_source_object(stage, config)

    attached, error = simulation_app.run_coroutine(
        context.attach_stage_async(stage)
    )
    if not attached:
        raise RuntimeError(
            f'failed to attach prepared USD stage {usd_scene_path}: {error}'
        )
    attached_stage = context.get_stage()
    if attached_stage is None:
        raise RuntimeError(f'USD stage was not attached: {usd_scene_path}')
    attached_stage.SetEditTarget(attached_stage.GetSessionLayer())
    _validate_authored_gripper_collision_instances(attached_stage)
    return attached_stage


class AuthoredObjectAttachmentController:
    """Create a fixed joint only after a verified bilateral finger contact."""

    _JOINT_PATH = '/World/Runtime/TaggedObjectGraspJoint'

    def __init__(
        self,
        stage,
        robot,
        scene_config: UsdSceneConfig,
        demo_config: DemoConfig,
        required_contact_steps: int = 3,
    ) -> None:
        """Subscribe to contact reports for the authored graspable object."""
        self._stage = stage
        self._robot = robot
        self._object_path = scene_config.source_object_prim_path
        self._body_path = scene_config.grasp_body_prim_path
        self._grasp_frame_path = scene_config.grasp_frame_prim_path
        self._config = demo_config.attachment
        self._attached = False
        if required_contact_steps < 1:
            raise ValueError('required contact steps must be at least one')
        self._required_contact_steps = int(required_contact_steps)
        self._bilateral_contact_steps = 0
        self._contacts = BilateralFingerContactTracker(
            self._object_path,
            AUTHORED_GRIPPER_BODY_PRIMS,
        )
        for prim_path in (
            self._object_path,
            self._body_path,
            self._grasp_frame_path,
        ):
            if not stage.GetPrimAtPath(prim_path).IsValid():
                raise RuntimeError(
                    f'physical grasp contract prim does not exist: {prim_path}'
                )
        for prim_path in AUTHORED_GRIPPER_BODY_PRIMS:
            if not stage.GetPrimAtPath(prim_path).IsValid():
                raise RuntimeError(
                    f'gripper finger body does not exist: {prim_path}'
                )

        from omni.physx import get_physx_simulation_interface

        self._contact_report_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report,
            )
        )

    @property
    def attached(self) -> bool:
        """Return whether the block is constrained to link6."""
        return self._attached

    def _distance_to_grasp_frame(self) -> float:
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        object_position = cache.GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(self._object_path)
        ).ExtractTranslation()
        grasp_position = cache.GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(self._grasp_frame_path)
        ).ExtractTranslation()
        return math.sqrt(sum(
            (float(object_position[index]) - float(grasp_position[index])) ** 2
            for index in range(3)
        ))

    def _on_contact_report(self, contact_headers, _contact_data) -> None:
        """Track PhysX FOUND/PERSIST/LOST events for each finger body."""
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        for header in contact_headers:
            if header.type == ContactEventType.CONTACT_LOST:
                active = False
            elif header.type in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            ):
                active = True
            else:
                continue
            self._contacts.update(
                str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                active,
            )

    def _create_joint(self) -> None:
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

        object_prim = self._stage.GetPrimAtPath(self._object_path)
        body_prim = self._stage.GetPrimAtPath(self._body_path)
        cache = UsdGeom.XformCache()
        body_world = cache.GetLocalToWorldTransform(
            body_prim
        ).RemoveScaleShear()
        object_world = cache.GetLocalToWorldTransform(
            object_prim
        ).RemoveScaleShear()
        # Gf matrices use row vectors: local * body_world == anchor_world.
        # Compute both local frames from one world anchor so PhysX sees an
        # already assembled joint and never snaps the object into place.
        anchor_world = object_world
        local0 = (
            anchor_world * body_world.GetInverse()
        ).RemoveScaleShear()
        local1 = (
            anchor_world * object_world.GetInverse()
        ).RemoveScaleShear()

        frame0_position, frame0_rotation = _gf_matrix_pose(
            local0 * body_world
        )
        frame1_position, frame1_rotation = _gf_matrix_pose(
            local1 * object_world
        )
        frame_translation_error = math.dist(
            frame0_position,
            frame1_position,
        )
        frame_rotation_error = _quaternion_error_degrees(
            frame0_rotation,
            frame1_rotation,
        )
        if frame_translation_error > 1e-6 or frame_rotation_error > 1e-4:
            raise RuntimeError(
                'fixed-joint frames do not share one world pose: '
                f'translation error={frame_translation_error:.9f} m, '
                f'rotation error={frame_rotation_error:.9f} deg'
            )

        rigid_body = UsdPhysics.RigidBodyAPI(object_prim)
        if not rigid_body:
            raise RuntimeError(
                f'grasp object has no rigid-body API: {self._object_path}'
            )
        rigid_body.CreateKinematicEnabledAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            object_prim
        ).CreateEnableCCDAttr(True)
        self._stage.RemovePrim(self._JOINT_PATH)
        joint = UsdPhysics.FixedJoint.Define(
            self._stage,
            Sdf.Path(self._JOINT_PATH),
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self._object_path)])
        for index, local_frame in enumerate((local0, local1)):
            translation = local_frame.ExtractTranslation()
            rotation = local_frame.ExtractRotationQuat()
            imaginary = rotation.GetImaginary()
            getattr(joint, f'CreateLocalPos{index}Attr')(Gf.Vec3f(*(
                float(component) for component in translation
            )))
            getattr(joint, f'CreateLocalRot{index}Attr')(Gf.Quatf(
                float(rotation.GetReal()),
                Gf.Vec3f(*(float(component) for component in imaginary)),
            ))
        joint.CreateExcludeFromArticulationAttr(True)
        self._attached = True
        print(
            '[arx-r5-sim] red block physically attached to link6',
            flush=True,
        )

    def _remove_joint(self) -> None:
        self._stage.RemovePrim(self._JOINT_PATH)
        self._attached = False
        self._bilateral_contact_steps = 0
        self._contacts.clear()
        print('[arx-r5-sim] red block physically released', flush=True)

    def update(self) -> None:
        """Mirror close/open gripper transitions with a physical constraint."""
        positions = self._robot.get_joint_positions()
        if positions is None:
            return
        joint_names = tuple(self._robot.dof_names)
        if not all(
            joint_name in joint_names
            for joint_name in (GRIPPER_COMMAND_JOINT, GRIPPER_MIMIC_JOINT)
        ):
            return
        finger_positions = {
            joint_name: float(positions[joint_names.index(joint_name)])
            for joint_name in (GRIPPER_COMMAND_JOINT, GRIPPER_MIMIC_JOINT)
        }
        aperture = sum(finger_positions.values()) / len(finger_positions)
        if self._attached:
            if aperture >= self._config.open_threshold:
                self._remove_joint()
            return
        if aperture > self._config.close_threshold:
            self._bilateral_contact_steps = 0
            return
        grasp_distance = self._distance_to_grasp_frame()
        if (
            not self._contacts.has_bilateral_contact or
            grasp_distance > self._config.maximum_distance
        ):
            self._bilateral_contact_steps = 0
            return
        self._bilateral_contact_steps += 1
        if self._bilateral_contact_steps < self._required_contact_steps:
            return
        print(
            '[arx-r5-sim] bilateral PhysX contact confirmed: '
            f'steps={self._bilateral_contact_steps}, '
            f'joint7={finger_positions[GRIPPER_COMMAND_JOINT]:.6f} m, '
            f'joint8={finger_positions[GRIPPER_MIMIC_JOINT]:.6f} m, '
            f'mean aperture={aperture:.6f} m, '
            f'grasp-frame distance={grasp_distance:.6f} m',
            flush=True,
        )
        self._create_joint()


def _create_ros_action_graph(
    robot_path: str,
    cameras: Sequence[tuple[str, str, object]] = (),
    static_transforms: Sequence[RosTransform] = (),
) -> None:
    import omni.graph.core as og
    import usdrt.Sdf

    target = [usdrt.Sdf.Path(robot_path)]
    create_nodes = [
        ('OnPhysicsStep', 'isaacsim.core.nodes.OnPhysicsStep'),
        ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
        ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
        ('PublishClock', 'isaacsim.ros2.bridge.ROS2PublishClock'),
        ('PublishJointState', 'isaacsim.ros2.bridge.ROS2PublishJointState'),
        ('SubscribeJointState', 'isaacsim.ros2.bridge.ROS2SubscribeJointState'),
        ('ArticulationController', 'isaacsim.core.nodes.IsaacArticulationController'),
    ]
    connections = [
        ('OnPhysicsStep.outputs:step', 'PublishClock.inputs:execIn'),
        ('OnPhysicsStep.outputs:step', 'PublishJointState.inputs:execIn'),
        ('OnPhysicsStep.outputs:step', 'SubscribeJointState.inputs:execIn'),
        ('OnPhysicsStep.outputs:step', 'ArticulationController.inputs:execIn'),
        ('Context.outputs:context', 'PublishClock.inputs:context'),
        ('Context.outputs:context', 'PublishJointState.inputs:context'),
        ('Context.outputs:context', 'SubscribeJointState.inputs:context'),
        ('ReadSimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
        ('ReadSimTime.outputs:simulationTime', 'PublishJointState.inputs:timeStamp'),
        ('SubscribeJointState.outputs:jointNames', 'ArticulationController.inputs:jointNames'),
        (
            'SubscribeJointState.outputs:positionCommand',
            'ArticulationController.inputs:positionCommand',
        ),
        (
            'SubscribeJointState.outputs:velocityCommand',
            'ArticulationController.inputs:velocityCommand',
        ),
        (
            'SubscribeJointState.outputs:effortCommand',
            'ArticulationController.inputs:effortCommand',
        ),
    ]
    values = [
        ('PublishClock.inputs:topicName', CLOCK_TOPIC),
        ('PublishJointState.inputs:topicName', JOINT_STATES_TOPIC),
        ('PublishJointState.inputs:targetPrim', target),
        ('SubscribeJointState.inputs:topicName', JOINT_COMMANDS_TOPIC),
        ('ArticulationController.inputs:targetPrim', target),
    ]

    for camera_name, camera_path, camera_config in cameras:
        node_suffix = ''.join(
            character if character.isalnum() else '_'
            for character in camera_name
        )
        render_product = f'CreateCameraRenderProduct_{node_suffix}'
        color_image = f'PublishColorImage_{node_suffix}'
        color_info = f'PublishColorInfo_{node_suffix}'
        depth_image = f'PublishDepthImage_{node_suffix}'
        depth_info = f'PublishDepthInfo_{node_suffix}'
        create_nodes.extend([
            (render_product, 'isaacsim.core.nodes.IsaacCreateRenderProduct'),
            (color_image, 'isaacsim.ros2.bridge.ROS2CameraHelper'),
            (color_info, 'isaacsim.ros2.bridge.ROS2CameraInfoHelper'),
            (depth_image, 'isaacsim.ros2.bridge.ROS2CameraHelper'),
            (depth_info, 'isaacsim.ros2.bridge.ROS2CameraInfoHelper'),
        ])
        connections.extend([
            (
                'OnPhysicsStep.outputs:step',
                f'{render_product}.inputs:execIn',
            ),
            (
                f'{render_product}.outputs:execOut',
                f'{color_image}.inputs:execIn',
            ),
            (
                f'{render_product}.outputs:execOut',
                f'{color_info}.inputs:execIn',
            ),
            (
                f'{render_product}.outputs:execOut',
                f'{depth_image}.inputs:execIn',
            ),
            (
                f'{render_product}.outputs:execOut',
                f'{depth_info}.inputs:execIn',
            ),
        ])
        for publisher in (
            color_image,
            color_info,
            depth_image,
            depth_info,
        ):
            connections.extend([
                ('Context.outputs:context', f'{publisher}.inputs:context'),
                (
                    f'{render_product}.outputs:renderProductPath',
                    f'{publisher}.inputs:renderProductPath',
                ),
            ])
        values.extend([
            (
                f'{render_product}.inputs:cameraPrim',
                [usdrt.Sdf.Path(camera_path)],
            ),
            (f'{render_product}.inputs:width', camera_config.width),
            (f'{render_product}.inputs:height', camera_config.height),
            (f'{color_image}.inputs:type', 'rgb'),
            (
                f'{color_image}.inputs:topicName',
                camera_config.color_image_topic,
            ),
            (f'{color_image}.inputs:frameId', camera_config.optical_frame),
            (
                f'{color_info}.inputs:topicName',
                camera_config.color_info_topic,
            ),
            (f'{color_info}.inputs:frameId', camera_config.optical_frame),
            (f'{depth_image}.inputs:type', 'depth'),
            (
                f'{depth_image}.inputs:topicName',
                camera_config.depth_image_topic,
            ),
            (f'{depth_image}.inputs:frameId', camera_config.optical_frame),
            (
                f'{depth_info}.inputs:topicName',
                camera_config.depth_info_topic,
            ),
            (f'{depth_info}.inputs:frameId', camera_config.optical_frame),
        ])
        for publisher in (
            color_image,
            color_info,
            depth_image,
            depth_info,
        ):
            values.extend([
                (
                    f'{publisher}.inputs:frameSkipCount',
                    camera_config.frame_skip_count,
                ),
                (f'{publisher}.inputs:queueSize', 1),
                (f'{publisher}.inputs:useSystemTime', False),
            ])

    for transform_index, transform in enumerate(static_transforms):
        publisher = f'PublishStaticTransform_{transform_index}'
        create_nodes.append((
            publisher,
            'isaacsim.ros2.bridge.ROS2PublishRawTransformTree',
        ))
        connections.extend([
            ('OnPhysicsStep.outputs:step', f'{publisher}.inputs:execIn'),
            ('Context.outputs:context', f'{publisher}.inputs:context'),
            ('ReadSimTime.outputs:simulationTime', f'{publisher}.inputs:timeStamp'),
        ])
        values.extend([
            (f'{publisher}.inputs:parentFrameId', transform.parent_frame),
            (f'{publisher}.inputs:childFrameId', transform.child_frame),
            (f'{publisher}.inputs:translation', transform.translation),
            (f'{publisher}.inputs:rotation', transform.rotation),
            (f'{publisher}.inputs:staticPublisher', True),
            (f'{publisher}.inputs:topicName', 'tf_static'),
        ])

    og.Controller.edit(
        {
            'graph_path': '/World/ROS2Graph',
            'evaluator_name': 'execution',
            'pipeline_stage': (
                og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND
            ),
        },
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.CONNECT: connections,
            og.Controller.Keys.SET_VALUES: values,
        },
    )


def _set_initial_positions(robot, requested_positions: dict[str, float]) -> None:
    from isaacsim.core.utils.types import ArticulationAction
    import numpy as np

    dof_names = tuple(robot.dof_names)
    missing = sorted(set(ALL_JOINTS) - set(dof_names))
    if missing:
        raise RuntimeError(
            f'Isaac Sim articulation is missing expected joints: {missing}; '
            f'found {dof_names}'
        )

    positions = robot.get_joint_positions()
    if positions is None:
        positions = np.zeros(len(dof_names), dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
    for joint_name, value in requested_positions.items():
        positions[dof_names.index(joint_name)] = value

    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros_like(positions))
    commanded_indices = np.asarray(
        [dof_names.index(name) for name in ALL_JOINTS],
        dtype=np.int32,
    )
    robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=positions[commanded_indices],
            joint_indices=commanded_indices,
        )
    )


def _export_stage(stage, destination: str) -> None:
    from isaacsim.core.utils.stage import save_stage

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_stage(
        str(destination_path),
        save_and_reload_in_place=False,
    ):
        raise RuntimeError(f'failed to export USD stage to {destination_path}')
    print(f'[arx-r5-sim] exported scene: {destination_path}', flush=True)


def _run_simulation(args: argparse.Namespace) -> None:
    if args.physics_dt <= 0.0 or args.rendering_dt <= 0.0:
        raise ValueError('physics and rendering time steps must be positive')
    if args.duration < 0.0:
        raise ValueError('duration must be non-negative')
    if args.drive_stiffness <= 0.0 or args.drive_damping < 0.0:
        raise ValueError('drive stiffness must be positive and damping non-negative')
    if (
        not math.isfinite(args.gripper_drive_max_force) or
        args.gripper_drive_max_force <= 0.0
    ):
        raise ValueError('gripper drive max force must be finite and positive')
    if args.grasp_contact_steps < 1:
        raise ValueError('grasp contact steps must be at least one')

    authored_usd = args.usd is not None
    if authored_usd and args.scene != 'empty':
        raise ValueError('--usd cannot be combined with --scene tabletop')

    demo_config: Optional[DemoConfig] = None
    if authored_usd or args.scene == 'tabletop':
        default_demo_config = (
            AUTHORED_USD_DEMO_CONFIG
            if authored_usd
            else TABLETOP_DEMO_CONFIG
        )
        demo_config = load_demo_config(
            find_demo_config(args.demo_config, default_demo_config)
        )
    usd_scene_config = None
    usd_scene_path = None
    authored_layout = None
    if authored_usd:
        usd_scene_config = load_usd_scene_config(
            find_usd_scene_config(args.usd_scene_config)
        )
        usd_scene_path = find_usd_scene(args.usd)
        try:
            authored_layout = usd_scene_config.layouts[args.authored_layout]
        except KeyError as error:
            available = ', '.join(sorted(usd_scene_config.layouts))
            raise ValueError(
                f'unknown authored layout {args.authored_layout!r}; '
                f'available: {available}'
            ) from error
        if (
            args.save_usd and
            Path(args.save_usd).expanduser().resolve() == usd_scene_path
        ):
            raise ValueError(
                '--save-usd must not overwrite the selected authored USD; '
                'choose a different destination path'
            )

    if args.ros_domain_id is not None:
        os.environ['ROS_DOMAIN_ID'] = str(args.ros_domain_id)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            'headless': args.headless,
            'width': 1280,
            'height': 720,
        }
    )

    world = None
    failure = None
    try:
        from isaacsim.asset.importer.urdf import _urdf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.viewports import set_camera_view
        import omni.kit.commands
        import omni.usd
        from pxr import PhysxSchema, Sdf, UsdLux

        enable_extension('isaacsim.asset.importer.urdf')
        if not args.no_ros:
            enable_extension('isaacsim.ros2.bridge')
        simulation_app.update()

        requested_positions = _parse_position_vector(args.initial_positions)
        camera_streams = []
        static_transforms = []

        if authored_usd:
            assert usd_scene_config is not None
            assert usd_scene_path is not None
            assert authored_layout is not None
            print(f'[arx-r5-sim] opening authored USD: {usd_scene_path}', flush=True)
            context = omni.usd.get_context()
            assert demo_config is not None
            stage = _attach_prepared_authored_stage(
                simulation_app,
                context,
                usd_scene_path,
                usd_scene_config,
                demo_config,
                authored_layout,
            )
            world = World(
                stage_units_in_meters=1.0,
                physics_dt=args.physics_dt,
                rendering_dt=args.rendering_dt,
            )
            robot_path = usd_scene_config.robot_prim_path
            if args.camera_mode == 'both':
                selected_cameras = tuple(usd_scene_config.cameras.values())
            elif args.camera_mode == 'none':
                selected_cameras = ()
            else:
                selected_cameras = (
                    usd_scene_config.cameras[args.camera_mode],
                )
            camera_streams = [
                (camera.name, camera.camera_prim_path, camera)
                for camera in selected_cameras
            ]
            for camera in selected_cameras:
                _normalize_camera_projection(stage, camera)
            expected_camera_translations = (
                authored_layout.camera_parent_to_optical_translations
            )
            static_transforms = [
                _world_to_base_transform(stage, usd_scene_config),
                *(
                    _camera_optical_transform(
                        stage,
                        camera,
                        expected_camera_translations.get(camera.name),
                    )
                    for camera in selected_cameras
                ),
            ]
        else:
            description_share = find_description_share(args.description_share)
            source_urdf = (
                Path(args.urdf).expanduser().resolve()
                if args.urdf
                else description_share / 'urdf' / 'r5a_cumotion.urdf'
            )
            if not source_urdf.is_file():
                raise FileNotFoundError(f'URDF does not exist: {source_urdf}')
            print(f'[arx-r5-sim] description: {description_share}', flush=True)
            print(f'[arx-r5-sim] importing: {source_urdf}', flush=True)
            world = World(
                stage_units_in_meters=1.0,
                physics_dt=args.physics_dt,
                rendering_dt=args.rendering_dt,
            )
            ground_height = 0.0
            if demo_config is not None:
                ground_height = (
                    demo_config.table.top_center[2]
                    - demo_config.table.top_size[2] / 2.0
                    - demo_config.table.leg_size[2]
                )
            world.scene.add_default_ground_plane(z_position=ground_height)
            stage = omni.usd.get_context().get_stage()

            distant_light = UsdLux.DistantLight.Define(
                stage,
                Sdf.Path('/World/DistantLight'),
            )
            distant_light.CreateIntensityAttr(1000.0)

            with tempfile.TemporaryDirectory(
                prefix='arx_r5a_isaac_sim_'
            ) as temp_dir:
                resolved_urdf = materialize_isaac_urdf(
                    source_urdf,
                    description_share,
                    Path(temp_dir) / 'r5a_isaac_sim.urdf',
                )
                status, import_config = omni.kit.commands.execute(
                    'URDFCreateImportConfig'
                )
                if not status:
                    raise RuntimeError(
                        'failed to create the Isaac Sim URDF import config'
                    )
                import_config.merge_fixed_joints = True
                import_config.convex_decomp = args.convex_decomposition
                import_config.fix_base = True
                import_config.make_default_prim = False
                import_config.create_physics_scene = False
                import_config.import_inertia_tensor = True
                # TopicBasedSystem explicitly publishes joint7's target for
                # both fingers. A PhysX mimic and a drive on joint8 are an
                # invalid combination, so import both fingers as driven DOFs.
                import_config.parse_mimic = False
                import_config.self_collision = False
                import_config.density = 0.0
                import_config.distance_scale = 1.0
                import_config.collision_from_visuals = False
                import_config.default_drive_type = (
                    _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
                )
                import_config.default_drive_strength = args.drive_stiffness
                import_config.default_position_drive_damping = args.drive_damping

                status, robot_path = omni.kit.commands.execute(
                    'URDFParseAndImportFile',
                    urdf_path=str(resolved_urdf),
                    import_config=import_config,
                    get_articulation_root=False,
                )
                if not status or not robot_path:
                    raise RuntimeError(f'failed to import URDF: {resolved_urdf}')
                robot_path = str(robot_path)

        articulation_path = _find_articulation_root(stage, robot_path)
        _validate_grasp_frame_import(stage, robot_path)
        _configure_joint_drives(
            stage,
            robot_path,
            stiffness=args.drive_stiffness,
            damping=args.drive_damping,
            gripper_max_force=args.gripper_drive_max_force,
            target_positions=(
                requested_positions
                if not authored_usd or args.reset_usd_joints
                else None
            ),
        )
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(
            stage.GetPrimAtPath(articulation_path)
        )
        articulation_api.CreateSolverPositionIterationCountAttr(64)
        articulation_api.CreateSolverVelocityIterationCountAttr(16)

        robot = world.scene.add(
            SingleArticulation(prim_path=articulation_path, name='arx_r5a')
        )

        tabletop_scene = None
        authored_attachment = None
        if demo_config is not None and not authored_usd:
            tabletop_scene = create_tabletop_scene(
                world,
                stage,
                robot,
                robot_path,
                demo_config,
            )
            camera_streams = [(
                'camera_1',
                tabletop_scene.camera_path,
                demo_config.camera,
            )]

        set_camera_view(
            eye=[1.1, 1.1, 0.8],
            target=[0.0, 0.0, 0.3],
            camera_prim_path='/OmniverseKit_Persp',
        )
        world.reset()
        if not authored_usd or args.reset_usd_joints:
            _set_initial_positions(robot, requested_positions)
        if authored_usd:
            assert usd_scene_config is not None
            assert demo_config is not None
            authored_attachment = AuthoredObjectAttachmentController(
                stage,
                robot,
                usd_scene_config,
                demo_config,
                required_contact_steps=args.grasp_contact_steps,
            )
        if not args.no_ros:
            _create_ros_action_graph(
                articulation_path,
                cameras=camera_streams,
                static_transforms=static_transforms,
            )
        simulation_app.update()

        print(f'[arx-r5-sim] robot prim: {robot_path}', flush=True)
        print(f'[arx-r5-sim] articulation root: {articulation_path}', flush=True)
        print(f'[arx-r5-sim] joints: {tuple(robot.dof_names)}', flush=True)
        if not args.no_ros:
            print(
                '[arx-r5-sim] ROS topics: '
                f'/{JOINT_STATES_TOPIC} -> ros2_control, '
                f'ros2_control -> /{JOINT_COMMANDS_TOPIC}',
                flush=True,
            )
            for camera_name, _, camera_config in camera_streams:
                print(
                    f'[arx-r5-sim] {camera_name} RGB: '
                    f'{camera_config.color_image_topic} '
                    f'[{camera_config.optical_frame}]',
                    flush=True,
                )

        if args.save_usd:
            _export_stage(stage, args.save_usd)
        if args.exit_after_build:
            return

        world.play()
        step_count = 0
        max_steps = (
            int(args.duration / args.physics_dt) if args.duration > 0.0 else None
        )
        while simulation_app.is_running():
            world.step(render=True)
            if tabletop_scene is not None:
                tabletop_scene.attachment.update()
            if authored_attachment is not None:
                authored_attachment.update()
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break
    except BaseException:
        failure = sys.exc_info()
        traceback.print_exc()
    finally:
        if world is not None:
            try:
                world.stop()
            except Exception:
                traceback.print_exc()
        simulation_app.close()
    if failure is not None:
        _, error, error_traceback = failure
        raise error.with_traceback(error_traceback)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run the Isaac Sim application."""
    args = _build_argument_parser().parse_args(argv)
    _run_simulation(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
