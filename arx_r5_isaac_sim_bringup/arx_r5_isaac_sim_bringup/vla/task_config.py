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
Load, validate and apply the VLA data-collection task configuration.

This module is the geometric single source of truth for the collection
workcell. The USD authoring script, the cuMotion collision scene and the
episode driver all derive their numbers from here, so the rendered scene, the
collision world and the planned goals cannot drift apart.

It imports neither Isaac Sim nor ROS, so it stays unit-testable and can move to
a real-robot overlay unchanged.
"""

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, isfinite, pi, radians, sin, sqrt
from pathlib import Path
import random
from typing import Optional, Sequence, Tuple

from arx_r5_isaac_sim_bringup.pose_math import (
    conjugate_quaternion,
    cross,
    multiply_quaternions,
    normalize_vector,
    quaternion_from_axes,
    quaternion_from_yaw,
    rotate_vector,
)

import yaml


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Range = Tuple[float, float]

# link6 +X points at the fingers and joint7/joint8 separate along +/-Y, so a
# top grasp is fully defined by the block's yaw about world Z.
WORLD_UP: Vector3 = (0.0, 0.0, 1.0)
WORLD_DOWN: Vector3 = (0.0, 0.0, -1.0)

# Rejection-sampling budget for the place pose. With the packaged zone the
# separation constraint rejects well under half of all draws, so exhausting
# this many attempts means the geometry changed, not that the draw was unlucky.
_MAX_SEPARATION_ATTEMPTS = 64


def _vector3(values, field_name: str) -> Vector3:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f'{field_name} must contain three numeric values')
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{field_name} must contain three numeric values'
        ) from error
    if not all(isfinite(value) for value in vector):
        raise ValueError(f'{field_name} must contain only finite values')
    return vector


def _positive(value, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f'{field_name} must be greater than zero')
    return number


def _non_negative(value, field_name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f'{field_name} must not be negative')
    return number


def _range(values, field_name: str) -> Range:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f'{field_name} must contain two numeric values')
    try:
        low, high = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{field_name} must contain two numeric values'
        ) from error
    if not (isfinite(low) and isfinite(high)):
        raise ValueError(f'{field_name} must contain only finite values')
    if low > high:
        raise ValueError(f'{field_name} lower bound must not exceed the upper')
    return (low, high)


def _positive_size(values, field_name: str) -> Vector3:
    size = _vector3(values, field_name)
    if any(dimension <= 0.0 for dimension in size):
        raise ValueError(f'{field_name} dimensions must be greater than zero')
    return size


@dataclass(frozen=True)
class BoxSpec:
    """An axis-aligned box in the world frame."""

    size: Vector3
    center: Vector3

    @property
    def top_z(self) -> float:
        """Return the world Z of the box's upper face."""
        return self.center[2] + self.size[2] / 2.0


@dataclass(frozen=True)
class MountSpec:
    """
    The column the robot is bolted to.

    ``collision_*`` is deliberately shorter than the rendered prim: a cuMotion
    box that reaches base_link would make every start state collide.
    """

    prim_path: str
    size: Vector3
    center: Vector3
    collision_size: Vector3
    collision_center: Vector3

    @property
    def top_z(self) -> float:
        """Return the world Z of the rendered column's upper face."""
        return self.center[2] + self.size[2] / 2.0

    @property
    def collision_box(self) -> BoxSpec:
        """Return the shortened box handed to cuMotion."""
        return BoxSpec(size=self.collision_size, center=self.collision_center)


@dataclass(frozen=True)
class PlatformSpec:
    """The low platform objects rest on."""

    prim_path: str
    size: Vector3
    center: Vector3

    @property
    def top_z(self) -> float:
        """Return the world Z of the platform's working surface."""
        return self.center[2] + self.size[2] / 2.0

    @property
    def box(self) -> BoxSpec:
        """Return this platform as a plain box."""
        return BoxSpec(size=self.size, center=self.center)


@dataclass(frozen=True)
class WorkcellSpec:
    """Static geometry shared by the USD scene and the collision world."""

    ground_z: float
    mount: MountSpec
    platform: PlatformSpec
    ground: BoxSpec


@dataclass(frozen=True)
class BlockSpec:
    """The manipulated block."""

    prim_path: str
    body_prim_path: str
    size: Vector3
    nominal_center: Vector3
    mass_kg: float

    @property
    def height(self) -> float:
        """Return the block's Z extent."""
        return self.size[2]

    def center_z_on(self, surface_z: float) -> float:
        """Return the block centre height when resting on a surface."""
        return surface_z + self.height / 2.0


@dataclass(frozen=True)
class PlaceMarkerSpec:
    """Visual-only decal marking where the block must be placed."""

    prim_path: str
    size: Tuple[float, float]
    height_above_surface: float
    color: Vector3


@dataclass(frozen=True)
class ZoneSpec:
    """A rectangular sampling region for one object pose."""

    x: Range
    y: Range
    yaw_deg: Range

    def corners(self) -> Tuple[Tuple[float, float], ...]:
        """Return the four extreme (x, y) combinations of this zone."""
        return tuple(
            (x_value, y_value)
            for x_value in self.x
            for y_value in self.y
        )

    def sample(self, rng: random.Random) -> Tuple[float, float, float]:
        """Return a random (x, y, yaw_radians) inside this zone."""
        return (
            rng.uniform(*self.x),
            rng.uniform(*self.y),
            radians(rng.uniform(*self.yaw_deg)),
        )


class GraspStyle:
    """How the gripper approaches the block."""

    TOP = 'top'
    SIDE = 'side'
    ALL = (TOP, SIDE)


@dataclass(frozen=True)
class GraspSpec:
    """Grasp geometry and gripper commands."""

    style: str
    depth_below_top: float
    side_grasp_height: float
    side_approach_distance: float
    grasp_frame_offset: float
    approach_height: float
    lift_height: float
    gripper_open: float
    gripper_close: float
    attach_max_distance: float

    @property
    def is_side(self) -> bool:
        """Return whether the gripper approaches horizontally."""
        return self.style == GraspStyle.SIDE


@dataclass(frozen=True)
class ReachEnvelope:
    """Conservative bounds on where cuMotion is known to find top grasps."""

    frame: str
    max_x: float
    max_radius: float
    min_radius: float

    def violations(self, point: Vector3, label: str) -> Tuple[str, ...]:
        """Return human-readable reasons this base-frame point is rejected."""
        radius = sqrt(sum(component * component for component in point))
        problems = []
        if point[0] > self.max_x:
            problems.append(
                f'{label}: {self.frame} x={point[0]:.4f} exceeds '
                f'max_x={self.max_x:.4f}'
            )
        if radius > self.max_radius:
            problems.append(
                f'{label}: {self.frame} radius={radius:.4f} exceeds '
                f'max_radius={self.max_radius:.4f}'
            )
        if radius < self.min_radius:
            problems.append(
                f'{label}: {self.frame} radius={radius:.4f} is below '
                f'min_radius={self.min_radius:.4f}'
            )
        return tuple(problems)


@dataclass(frozen=True)
class CameraVisibility:
    """Frustum bounds asserted against the eye-to-hand camera."""

    camera: str
    max_horizontal_deg: float
    max_vertical_deg: float
    min_depth: float

    def violations(
        self,
        offsets: Tuple[float, float, float],
        label: str,
    ) -> Tuple[str, ...]:
        """Return reasons an optical-frame offset falls outside the frustum."""
        horizontal_deg, vertical_deg, depth = offsets
        problems = []
        if depth < self.min_depth:
            problems.append(
                f'{label}: {self.camera} depth={depth:.4f} is closer than '
                f'min_depth={self.min_depth:.4f}'
            )
        if abs(horizontal_deg) > self.max_horizontal_deg:
            problems.append(
                f'{label}: {self.camera} horizontal offset '
                f'{horizontal_deg:.2f} deg exceeds '
                f'{self.max_horizontal_deg:.2f} deg'
            )
        if abs(vertical_deg) > self.max_vertical_deg:
            problems.append(
                f'{label}: {self.camera} vertical offset '
                f'{vertical_deg:.2f} deg exceeds '
                f'{self.max_vertical_deg:.2f} deg'
            )
        return tuple(problems)


@dataclass(frozen=True)
class EpisodeSpec:
    """Episode lifecycle knobs."""

    home_positions_file: str
    grasp_contact_steps: int
    reset_settle_sec: float


@dataclass(frozen=True)
class PlacementValidationSpec:
    """
    Tolerance and settling policy for the final placement check.

    A release acknowledgement only says that the simulator removed the
    temporary grasp constraint.  It does not say where the dynamic block
    came to rest.  These values define the separate, ground-truth check the
    episode driver performs after the retreat waypoint. ``timeout_sec`` is
    measured in simulation time; the driver separately enforces a wall-clock
    watchdog so a paused or missing ``/clock`` cannot block shutdown.
    """

    release_clearance: float
    xy_tolerance: float
    z_tolerance: float
    max_tilt_deg: float
    yaw_tolerance_deg: float
    yaw_symmetry_order: int
    stability_tolerance: float
    stability_orientation_tolerance_deg: float
    stable_duration_sec: float
    timeout_sec: float

    def errors(
        self,
        actual: Vector3,
        target: Vector3,
        actual_rotation: Optional[Quaternion] = None,
        target_yaw: float = 0.0,
    ) -> Tuple[float, ...]:
        """
        Return translation errors and, when supplied, orientation errors.

        The first two values are horizontal and vertical centre error in
        metres.  If ``actual_rotation`` is supplied, tilt and symmetry-aware
        yaw error in degrees follow.  Keeping the translation-only form makes
        this helper useful to callers that merely want distance diagnostics;
        final placement acceptance always supplies the object orientation.
        """
        xy_error = sqrt(
            (float(actual[0]) - float(target[0])) ** 2
            + (float(actual[1]) - float(target[1])) ** 2
        )
        z_error = abs(float(actual[2]) - float(target[2]))
        if actual_rotation is None:
            return xy_error, z_error
        tilt_error, yaw_error = self.orientation_errors(
            actual_rotation,
            target_yaw,
        )
        return xy_error, z_error, tilt_error, yaw_error

    def orientation_errors(
        self,
        actual_rotation: Quaternion,
        target_yaw: float,
    ) -> Tuple[float, float]:
        """
        Return upright tilt and symmetry-aware target-yaw errors.

        The block's local Z axis must remain aligned with world Z.  Its square
        horizontal cross-section makes quarter turns physically equivalent,
        so yaw error is wrapped into one half of that 90-degree symmetry
        interval instead of comparing raw Euler angles.
        """
        local_z = rotate_vector(actual_rotation, WORLD_UP)
        vertical_component = max(-1.0, min(1.0, float(local_z[2])))
        tilt_error = degrees(acos(vertical_component))

        local_x = rotate_vector(actual_rotation, (1.0, 0.0, 0.0))
        actual_yaw = atan2(float(local_x[1]), float(local_x[0]))
        symmetry_period = 2.0 * pi / float(self.yaw_symmetry_order)
        yaw_delta = float(actual_yaw) - float(target_yaw)
        wrapped_yaw_delta = (
            (yaw_delta + symmetry_period / 2.0) % symmetry_period
            - symmetry_period / 2.0
        )
        return tilt_error, abs(degrees(wrapped_yaw_delta))

    def accepts(
        self,
        actual: Vector3,
        target: Vector3,
        actual_rotation: Optional[Quaternion] = None,
        target_yaw: float = 0.0,
    ) -> bool:
        """Return whether a stable block pose is on the sampled target."""
        xy_error, z_error = self.errors(actual, target)
        if actual_rotation is None:
            # Translation-only checks remain available for geometry callers;
            # the episode driver's final acceptance always supplies rotation.
            return (
                xy_error <= self.xy_tolerance
                and z_error <= self.z_tolerance
            )
        tilt_error, yaw_error = self.orientation_errors(
            actual_rotation,
            target_yaw,
        )
        return (
            xy_error <= self.xy_tolerance
            and z_error <= self.z_tolerance
            and tilt_error <= self.max_tilt_deg
            and yaw_error <= self.yaw_tolerance_deg
        )


@dataclass(frozen=True)
class StableTranslationWindow:
    """A stable-pose window anchored to its first translation sample."""

    anchor: Vector3
    first_stamp_ns: int
    latest_stamp_ns: int
    sample_count: int

    @property
    def duration_sec(self) -> float:
        """Return simulated time covered by the stable window."""
        return max(0, self.latest_stamp_ns - self.first_stamp_ns) / 1e9


@dataclass(frozen=True)
class StablePoseWindow:
    """A stable translation/orientation window anchored to its first pose."""

    anchor_translation: Vector3
    anchor_rotation: Quaternion
    first_stamp_ns: int
    latest_stamp_ns: int
    sample_count: int

    @property
    def duration_sec(self) -> float:
        """Return simulated time covered by the stable pose window."""
        return max(0, self.latest_stamp_ns - self.first_stamp_ns) / 1e9


def quaternion_distance_degrees(
    left: Quaternion,
    right: Quaternion,
) -> float:
    """Return shortest angular distance, treating q and -q as identical."""
    left_norm = sqrt(sum(float(value) ** 2 for value in left))
    right_norm = sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError('pose stability quaternion must be non-zero')
    dot = abs(sum(
        float(left[index]) * float(right[index])
        for index in range(4)
    ) / (left_norm * right_norm))
    return degrees(2.0 * acos(max(-1.0, min(1.0, dot))))


def update_stable_pose_window(
    previous: Optional[StablePoseWindow],
    translation: Vector3,
    rotation: Quaternion,
    stamp_ns: int,
    translation_tolerance: float,
    orientation_tolerance_deg: float,
) -> StablePoseWindow:
    """Advance only while both centre and orientation remain stable."""
    translation = tuple(float(value) for value in translation)
    rotation = tuple(float(value) for value in rotation)
    stamp_ns = int(stamp_ns)
    if previous is None:
        return StablePoseWindow(
            translation,
            rotation,
            stamp_ns,
            stamp_ns,
            1,
        )
    if stamp_ns <= previous.latest_stamp_ns:
        return previous
    movement = sqrt(sum(
        (
            float(translation[index]) -
            float(previous.anchor_translation[index])
        ) ** 2
        for index in range(3)
    ))
    rotation_error = quaternion_distance_degrees(
        previous.anchor_rotation,
        rotation,
    )
    if (
        movement <= float(translation_tolerance)
        and rotation_error <= float(orientation_tolerance_deg)
    ):
        return StablePoseWindow(
            previous.anchor_translation,
            previous.anchor_rotation,
            previous.first_stamp_ns,
            stamp_ns,
            previous.sample_count + 1,
        )
    return StablePoseWindow(
        translation,
        rotation,
        stamp_ns,
        stamp_ns,
        1,
    )


def update_stable_translation_window(
    previous: Optional[StableTranslationWindow],
    current: Vector3,
    stamp_ns: int,
    tolerance: float,
) -> StableTranslationWindow:
    """
    Advance a duration-based stability window or anchor a new one.

    Movement is measured from the *first* sample in the current window, not
    merely from the immediately preceding sample.  Therefore a block that
    drifts 1 mm on every sample cannot be certified stable under a 2 mm gate.
    Duplicate or out-of-order timestamps do not advance the window.
    """
    current = tuple(float(value) for value in current)
    stamp_ns = int(stamp_ns)
    if previous is None:
        return StableTranslationWindow(current, stamp_ns, stamp_ns, 1)
    if stamp_ns <= previous.latest_stamp_ns:
        return previous
    movement = sqrt(sum(
        (float(current[index]) - float(previous.anchor[index])) ** 2
        for index in range(3)
    ))
    if movement <= float(tolerance):
        return StableTranslationWindow(
            previous.anchor,
            previous.first_stamp_ns,
            stamp_ns,
            previous.sample_count + 1,
        )
    return StableTranslationWindow(current, stamp_ns, stamp_ns, 1)


def next_stable_sample_count(
    previous: Optional[Vector3],
    current: Vector3,
    previous_count: int,
    tolerance: float,
) -> int:
    """
    Advance the legacy adjacent-sample counter.

    New placement verification uses :func:`update_stable_pose_window`. This
    compatibility helper remains for downstream users of the task-config
    module, but adjacent-sample counts must not be used to certify placement.
    """
    if previous is None:
        return 1
    movement = sqrt(sum(
        (float(current[index]) - float(previous[index])) ** 2
        for index in range(3)
    ))
    if movement <= float(tolerance):
        return max(1, int(previous_count)) + 1
    return 1


@dataclass(frozen=True)
class TaskConfig:
    """Complete VLA collection task definition."""

    instruction: str
    workcell: WorkcellSpec
    block: BlockSpec
    place_marker: PlaceMarkerSpec
    sampling_zone: ZoneSpec
    min_separation: float
    grasp: GraspSpec
    reach_envelope: ReachEnvelope
    camera_visibility: CameraVisibility
    episode: EpisodeSpec
    placement: PlacementValidationSpec

    def block_center_world(
        self,
        x_value: float,
        y_value: float,
    ) -> Vector3:
        """Return the block centre for a platform-resting (x, y) placement."""
        return (
            x_value,
            y_value,
            self.block.center_z_on(self.workcell.platform.top_z),
        )

    def grasp_pose_world(
        self,
        block_center: Vector3,
        yaw_radians: float,
        retreat: float = 0.0,
        lift: float = 0.0,
    ) -> Tuple[Vector3, Quaternion]:
        """
        Return the world link6 pose for a grasp of a placed block.

        ``retreat`` backs the pose away from the block along the approach
        direction: straight up for a top grasp, horizontally along the tool
        axis for a side grasp. ``lift`` always raises vertically, because
        lifting a grasped block is vertical regardless of how it was grasped.
        Every waypoint in an episode is one of these two offsets applied to the
        grasp pose itself.
        """
        if self.grasp.is_side:
            translation, rotation = self._side_grasp_pose(
                block_center,
                yaw_radians,
                retreat,
            )
        else:
            translation, rotation = self._top_grasp_pose(
                block_center,
                yaw_radians,
                retreat,
            )
        if lift:
            translation = (
                translation[0],
                translation[1],
                translation[2] + lift,
            )
        return translation, rotation

    def _top_grasp_pose(
        self,
        block_center: Vector3,
        yaw_radians: float,
        retreat: float,
    ) -> Tuple[Vector3, Quaternion]:
        grasp_point_z = (
            block_center[2] + self.block.height / 2.0 -
            self.grasp.depth_below_top
        )
        # A top grasp retreats straight up, so retreat and lift coincide.
        translation = (
            block_center[0],
            block_center[1],
            grasp_point_z + self.grasp.grasp_frame_offset + retreat,
        )
        return translation, top_grasp_rotation(yaw_radians)

    def _side_grasp_pose(
        self,
        block_center: Vector3,
        yaw_radians: float,
        retreat: float,
    ) -> Tuple[Vector3, Quaternion]:
        """
        Return a horizontal grasp pose approaching along the block's X axis.

        The gripper closes on the two block faces normal to the block's own Y
        axis, so the grasp survives the block's yaw randomization. The wrist
        sits back along the approach direction by grasp_frame_offset plus the
        retreat distance.
        """
        rotation = side_grasp_rotation(yaw_radians)
        # link6 +X is the tool axis, pointing from the wrist at the block.
        approach = rotate_vector(rotation, (1.0, 0.0, 0.0))
        grasp_z = (
            self.workcell.platform.top_z + self.grasp.side_grasp_height
        )
        distance = self.grasp.grasp_frame_offset + retreat
        return (
            (
                block_center[0] - approach[0] * distance,
                block_center[1] - approach[1] * distance,
                grasp_z - approach[2] * distance,
            ),
            rotation,
        )

    @property
    def retreat_distance(self) -> float:
        """Return how far the approach and retreat waypoints back off."""
        if self.grasp.is_side:
            return self.grasp.side_approach_distance
        return self.grasp.approach_height

    def equivalent_grasp_rotations(
        self,
        rotation: Quaternion,
    ) -> Tuple[Quaternion, ...]:
        """
        Return physically equivalent wrist rotations for this block.

        The VLA block has a square horizontal cross-section.  During a top
        grasp, rotating the parallel jaws by a quarter turn selects the other
        identical pair of faces, so all four quarter-turn poses describe the
        same task.  Supplying that symmetry as a cuMotion goalset prevents a
        random block yaw from needlessly saturating wrist joint6.

        Side grasps and non-square blocks have no such four-way symmetry and
        therefore keep only the requested orientation.
        """
        normalized = multiply_quaternions(
            tuple(float(value) for value in rotation),
            (0.0, 0.0, 0.0, 1.0),
        )
        if (
            self.grasp.is_side or
            abs(self.block.size[0] - self.block.size[1]) > 1e-9
        ):
            return (normalized,)

        # With the quaternion convention used here, a negative local-X turn
        # is a positive yaw around the downward-pointing tool axis.
        local_x_turns = (0.0, -pi / 2.0, pi, pi / 2.0)
        return tuple(
            multiply_quaternions(
                normalized,
                (sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)),
            )
            for angle in local_x_turns
        )


def top_grasp_rotation(yaw_radians: float) -> Quaternion:
    """
    Return the world link6 orientation for a top grasp at a given yaw.

    link6 +X points straight down so grasp_frame lands on the block axis, and
    link6 +Y (the joint7/joint8 separation axis) is aligned with the block's
    own X axis so the fingers close on a matched pair of faces.
    """
    block_x_axis = rotate_vector(
        quaternion_from_yaw(yaw_radians),
        (1.0, 0.0, 0.0),
    )
    link6_x = WORLD_DOWN
    link6_y = normalize_vector(block_x_axis)
    link6_z = normalize_vector(cross(link6_x, link6_y))
    return quaternion_from_axes(link6_x, link6_y, link6_z)


def side_grasp_rotation(yaw_radians: float) -> Quaternion:
    """
    Return the world link6 orientation for a side grasp at a given yaw.

    link6 +X (the tool axis) points horizontally along the block's own +X axis,
    so the wrist approaches one vertical face head-on. link6 +Y, the finger
    separation axis, is horizontal and normal to that approach, so the fingers
    close on the block's two remaining vertical faces. Keeping both axes
    horizontal puts link6 +Z straight up, which keeps the wrist upright.
    """
    block_yaw = quaternion_from_yaw(yaw_radians)
    approach = normalize_vector(rotate_vector(block_yaw, (1.0, 0.0, 0.0)))
    link6_x = approach
    link6_y = normalize_vector(cross(WORLD_UP, link6_x))
    link6_z = normalize_vector(cross(link6_x, link6_y))
    return quaternion_from_axes(link6_x, link6_y, link6_z)


def world_to_base(
    point_world: Vector3,
    base_translation: Sequence[float],
    base_rotation: Sequence[float],
) -> Vector3:
    """Express a world point in the robot base frame."""
    delta = tuple(
        float(point_world[index]) - float(base_translation[index])
        for index in range(3)
    )
    return rotate_vector(conjugate_quaternion(base_rotation), delta)


def pose_world_to_base(
    translation: Vector3,
    rotation: Quaternion,
    base_translation: Sequence[float],
    base_rotation: Sequence[float],
) -> Tuple[Vector3, Quaternion]:
    """
    Express a world pose in the robot base frame.

    cuMotion goals are stamped in base_link while the task geometry is authored
    in world, so every waypoint passes through here on its way to the planner.
    """
    return (
        world_to_base(translation, base_translation, base_rotation),
        multiply_quaternions(conjugate_quaternion(base_rotation), rotation),
    )


def optical_frame_offsets(
    point_base: Vector3,
    camera_translation: Sequence[float],
    camera_forward: Sequence[float],
) -> Tuple[float, float, float]:
    """
    Return (horizontal_deg, vertical_deg, depth) of a base-frame point.

    The camera's ROS optical frame is reconstructed from its forward axis using
    the usual convention: +Z forward, +X right, +Y down, with the image kept
    upright relative to base_link +Z.
    """
    optical_z = normalize_vector(camera_forward)
    optical_x = normalize_vector(cross(optical_z, WORLD_UP))
    optical_y = normalize_vector(cross(optical_z, optical_x))
    relative = tuple(
        float(point_base[index]) - float(camera_translation[index])
        for index in range(3)
    )

    def project(axis: Vector3) -> float:
        return sum(relative[index] * axis[index] for index in range(3))

    depth = project(optical_z)
    if depth <= 1e-9:
        return (180.0, 180.0, depth)
    return (
        degrees(atan2(project(optical_x), depth)),
        degrees(atan2(project(optical_y), depth)),
        depth,
    )


def load_task_config(path: str | Path) -> TaskConfig:
    """Load and validate the VLA task configuration from YAML."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'VLA task config not found: {config_path}')
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or raw.get('schema_version') != 1:
        raise ValueError('VLA task config must use schema_version 1')

    instruction = str(raw.get('instruction', '')).strip()
    if not instruction:
        raise ValueError('instruction must be a non-empty string')

    workcell_raw = raw.get('workcell')
    if not isinstance(workcell_raw, dict):
        raise ValueError('workcell must be a mapping')

    mount_raw = workcell_raw.get('mount')
    if not isinstance(mount_raw, dict):
        raise ValueError('workcell.mount must be a mapping')
    mount = MountSpec(
        prim_path=_prim_path(mount_raw.get('prim_path'), 'workcell.mount'),
        size=_positive_size(mount_raw.get('size'), 'workcell.mount.size'),
        center=_vector3(mount_raw.get('center'), 'workcell.mount.center'),
        collision_size=_positive_size(
            mount_raw.get('collision_size'),
            'workcell.mount.collision_size',
        ),
        collision_center=_vector3(
            mount_raw.get('collision_center'),
            'workcell.mount.collision_center',
        ),
    )

    platform_raw = workcell_raw.get('platform')
    if not isinstance(platform_raw, dict):
        raise ValueError('workcell.platform must be a mapping')
    platform = PlatformSpec(
        prim_path=_prim_path(
            platform_raw.get('prim_path'),
            'workcell.platform',
        ),
        size=_positive_size(
            platform_raw.get('size'),
            'workcell.platform.size',
        ),
        center=_vector3(
            platform_raw.get('center'),
            'workcell.platform.center',
        ),
    )

    ground_raw = workcell_raw.get('ground')
    if not isinstance(ground_raw, dict):
        raise ValueError('workcell.ground must be a mapping')
    ground = BoxSpec(
        size=_positive_size(ground_raw.get('size'), 'workcell.ground.size'),
        center=_vector3(ground_raw.get('center'), 'workcell.ground.center'),
    )

    workcell = WorkcellSpec(
        ground_z=float(workcell_raw.get('ground_z', 0.0)),
        mount=mount,
        platform=platform,
        ground=ground,
    )

    block_raw = raw.get('block')
    if not isinstance(block_raw, dict):
        raise ValueError('block must be a mapping')
    block = BlockSpec(
        prim_path=_prim_path(block_raw.get('prim_path'), 'block'),
        body_prim_path=_prim_path(
            block_raw.get('body_prim_path'),
            'block.body',
        ),
        size=_positive_size(block_raw.get('size'), 'block.size'),
        nominal_center=_vector3(
            block_raw.get('nominal_center'),
            'block.nominal_center',
        ),
        mass_kg=_positive(block_raw.get('mass_kg'), 'block.mass_kg'),
    )

    marker_raw = raw.get('place_marker')
    if not isinstance(marker_raw, dict):
        raise ValueError('place_marker must be a mapping')
    marker_size = marker_raw.get('size')
    if not isinstance(marker_size, (list, tuple)) or len(marker_size) != 2:
        raise ValueError('place_marker.size must contain two numeric values')
    marker_size = tuple(float(value) for value in marker_size)
    if any(value <= 0.0 for value in marker_size):
        raise ValueError('place_marker.size values must be greater than zero')
    place_marker = PlaceMarkerSpec(
        prim_path=_prim_path(marker_raw.get('prim_path'), 'place_marker'),
        size=marker_size,
        height_above_surface=_positive(
            marker_raw.get('height_above_surface'),
            'place_marker.height_above_surface',
        ),
        color=_vector3(marker_raw.get('color'), 'place_marker.color'),
    )
    if any(not 0.0 <= channel <= 1.0 for channel in place_marker.color):
        raise ValueError('place_marker.color channels must be within [0, 1]')

    sampling_zone = _zone(raw.get('sampling_zone'), 'sampling_zone')
    min_separation = _positive(raw.get('min_separation'), 'min_separation')
    # A separation the zone cannot satisfy would make every episode's
    # rejection sampler run to its retry limit and fall back.
    zone_diagonal = sqrt(
        (sampling_zone.x[1] - sampling_zone.x[0]) ** 2 +
        (sampling_zone.y[1] - sampling_zone.y[0]) ** 2
    )
    if min_separation >= zone_diagonal:
        raise ValueError(
            f'min_separation {min_separation:.4f} m does not fit inside the '
            f'sampling zone (diagonal {zone_diagonal:.4f} m)'
        )

    grasp_raw = raw.get('grasp')
    if not isinstance(grasp_raw, dict):
        raise ValueError('grasp must be a mapping')
    style = str(grasp_raw.get('style', GraspStyle.TOP)).strip().lower()
    if style not in GraspStyle.ALL:
        raise ValueError(
            f'grasp.style must be one of {GraspStyle.ALL}, got {style!r}'
        )
    grasp = GraspSpec(
        style=style,
        depth_below_top=_positive(
            grasp_raw.get('depth_below_top'),
            'grasp.depth_below_top',
        ),
        side_grasp_height=_positive(
            grasp_raw.get('side_grasp_height'),
            'grasp.side_grasp_height',
        ),
        side_approach_distance=_positive(
            grasp_raw.get('side_approach_distance'),
            'grasp.side_approach_distance',
        ),
        grasp_frame_offset=_positive(
            grasp_raw.get('grasp_frame_offset'),
            'grasp.grasp_frame_offset',
        ),
        approach_height=_positive(
            grasp_raw.get('approach_height'),
            'grasp.approach_height',
        ),
        lift_height=_positive(
            grasp_raw.get('lift_height'),
            'grasp.lift_height',
        ),
        gripper_open=_positive(
            grasp_raw.get('gripper_open'),
            'grasp.gripper_open',
        ),
        gripper_close=_non_negative(
            grasp_raw.get('gripper_close'),
            'grasp.gripper_close',
        ),
        attach_max_distance=_positive(
            grasp_raw.get('attach_max_distance'),
            'grasp.attach_max_distance',
        ),
    )
    if grasp.gripper_close >= grasp.gripper_open:
        raise ValueError(
            'grasp.gripper_close must be below grasp.gripper_open'
        )
    if grasp.depth_below_top >= block.height:
        raise ValueError('grasp.depth_below_top must be inside the block')
    if grasp.side_grasp_height >= block.height:
        raise ValueError(
            'grasp.side_grasp_height must be below the block top face'
        )

    envelope_raw = raw.get('reach_envelope')
    if not isinstance(envelope_raw, dict):
        raise ValueError('reach_envelope must be a mapping')
    reach_envelope = ReachEnvelope(
        frame=str(envelope_raw.get('frame', 'base_link')).strip(),
        max_x=_positive(envelope_raw.get('max_x'), 'reach_envelope.max_x'),
        max_radius=_positive(
            envelope_raw.get('max_radius'),
            'reach_envelope.max_radius',
        ),
        min_radius=_positive(
            envelope_raw.get('min_radius'),
            'reach_envelope.min_radius',
        ),
    )
    if reach_envelope.min_radius >= reach_envelope.max_radius:
        raise ValueError(
            'reach_envelope.min_radius must be below max_radius'
        )

    visibility_raw = raw.get('camera_visibility')
    if not isinstance(visibility_raw, dict):
        raise ValueError('camera_visibility must be a mapping')
    camera_visibility = CameraVisibility(
        camera=str(visibility_raw.get('camera', '')).strip(),
        max_horizontal_deg=_positive(
            visibility_raw.get('max_horizontal_deg'),
            'camera_visibility.max_horizontal_deg',
        ),
        max_vertical_deg=_positive(
            visibility_raw.get('max_vertical_deg'),
            'camera_visibility.max_vertical_deg',
        ),
        min_depth=_positive(
            visibility_raw.get('min_depth'),
            'camera_visibility.min_depth',
        ),
    )
    if not camera_visibility.camera:
        raise ValueError('camera_visibility.camera must be non-empty')

    episode_raw = raw.get('episode')
    if not isinstance(episode_raw, dict):
        raise ValueError('episode must be a mapping')
    grasp_contact_steps = int(episode_raw.get('grasp_contact_steps', 0))
    if grasp_contact_steps < 1:
        raise ValueError('episode.grasp_contact_steps must be at least one')
    episode = EpisodeSpec(
        home_positions_file=str(
            episode_raw.get('home_positions_file', '')
        ).strip(),
        grasp_contact_steps=grasp_contact_steps,
        reset_settle_sec=_non_negative(
            episode_raw.get('reset_settle_sec', 0.0),
            'episode.reset_settle_sec',
        ),
    )
    if not episode.home_positions_file:
        raise ValueError('episode.home_positions_file must be non-empty')

    placement_raw = raw.get('placement')
    if not isinstance(placement_raw, dict):
        raise ValueError('placement must be a mapping')
    # ``stable_samples`` was the original adjacent-frame gate.  Validate a
    # legacy value if present so stale configs fail loudly, but use elapsed
    # simulator time for all new placement verification.
    if 'stable_samples' in placement_raw:
        stable_samples = int(placement_raw['stable_samples'])
        if stable_samples < 2:
            raise ValueError('placement.stable_samples must be at least two')
    yaw_symmetry_order = int(placement_raw.get('yaw_symmetry_order', 0))
    if yaw_symmetry_order < 1:
        raise ValueError('placement.yaw_symmetry_order must be at least one')
    if float(yaw_symmetry_order) != float(
        placement_raw.get('yaw_symmetry_order', 0)
    ):
        raise ValueError('placement.yaw_symmetry_order must be an integer')
    max_tilt_deg = _positive(
        placement_raw.get('max_tilt_deg'),
        'placement.max_tilt_deg',
    )
    if max_tilt_deg > 180.0:
        raise ValueError('placement.max_tilt_deg must not exceed 180 degrees')
    yaw_tolerance_deg = _positive(
        placement_raw.get('yaw_tolerance_deg'),
        'placement.yaw_tolerance_deg',
    )
    max_distinct_yaw_error = 180.0 / float(yaw_symmetry_order)
    if yaw_tolerance_deg > max_distinct_yaw_error:
        raise ValueError(
            'placement.yaw_tolerance_deg must not exceed half of the '
            'symmetry period'
        )
    stable_duration_sec = _positive(
        placement_raw.get('stable_duration_sec'),
        'placement.stable_duration_sec',
    )
    timeout_sec = _positive(
        placement_raw.get('timeout_sec'),
        'placement.timeout_sec',
    )
    if stable_duration_sec > timeout_sec:
        raise ValueError(
            'placement.stable_duration_sec must not exceed '
            'placement.timeout_sec'
        )
    placement = PlacementValidationSpec(
        release_clearance=_positive(
            placement_raw.get('release_clearance'),
            'placement.release_clearance',
        ),
        xy_tolerance=_positive(
            placement_raw.get('xy_tolerance'),
            'placement.xy_tolerance',
        ),
        z_tolerance=_positive(
            placement_raw.get('z_tolerance'),
            'placement.z_tolerance',
        ),
        max_tilt_deg=max_tilt_deg,
        yaw_tolerance_deg=yaw_tolerance_deg,
        yaw_symmetry_order=yaw_symmetry_order,
        stability_tolerance=_positive(
            placement_raw.get('stability_tolerance'),
            'placement.stability_tolerance',
        ),
        stability_orientation_tolerance_deg=_positive(
            placement_raw.get('stability_orientation_tolerance_deg'),
            'placement.stability_orientation_tolerance_deg',
        ),
        stable_duration_sec=stable_duration_sec,
        timeout_sec=timeout_sec,
    )

    return TaskConfig(
        instruction=instruction,
        workcell=workcell,
        block=block,
        place_marker=place_marker,
        sampling_zone=sampling_zone,
        min_separation=min_separation,
        grasp=grasp,
        reach_envelope=reach_envelope,
        camera_visibility=camera_visibility,
        episode=episode,
        placement=placement,
    )


def _prim_path(value, field_name: str) -> str:
    path = str(value or '').strip()
    if not path.startswith('/') or path == '/':
        raise ValueError(
            f'{field_name} prim_path must be an absolute USD path'
        )
    return path


def _zone(raw, field_name: str) -> ZoneSpec:
    if not isinstance(raw, dict):
        raise ValueError(f'{field_name} must be a mapping')
    return ZoneSpec(
        x=_range(raw.get('x'), f'{field_name}.x'),
        y=_range(raw.get('y'), f'{field_name}.y'),
        yaw_deg=_range(raw.get('yaw_deg'), f'{field_name}.yaw_deg'),
    )


def validate_against_scene(task: TaskConfig, scene) -> Tuple[str, ...]:
    """
    Return every reach and visibility problem in the randomization zones.

    Both zone extremes are checked, for the block centre and for the derived
    link6 approach and grasp goals, so an unreachable or invisible corner is
    rejected at startup instead of failing mid-episode.
    """
    base_translation = scene.expected_world_to_base_translation
    base_rotation = scene.expected_world_to_base_rotation
    camera = scene.cameras.get(task.camera_visibility.camera)
    if camera is None:
        available = ', '.join(sorted(scene.cameras))
        raise ValueError(
            f'camera_visibility.camera {task.camera_visibility.camera!r} is '
            f'not in the USD scene contract; available: {available}'
        )

    problems = []
    # One shared region now serves both the block and the target, so a single
    # corner sweep covers every pose either of them can take.
    zones = (('sampling_zone', task.sampling_zone),)
    for zone_name, zone in zones:
        for x_value, y_value in zone.corners():
            block_center = task.block_center_world(x_value, y_value)
            label = f'{zone_name}({x_value:.3f}, {y_value:.3f})'

            block_base = world_to_base(
                block_center,
                base_translation,
                base_rotation,
            )
            problems.extend(task.camera_visibility.violations(
                optical_frame_offsets(
                    block_base,
                    camera.expected_parent_to_optical_translation,
                    camera.expected_optical_forward,
                ),
                f'{label} block centre',
            ))

            for offset_name, retreat, lift in (
                ('grasp', 0.0, 0.0),
                ('approach', task.retreat_distance, 0.0),
                ('lift', 0.0, task.grasp.lift_height),
            ):
                # Yaw does move the link6 origin for a side grasp, so sample
                # the extremes of the yaw range as well as the zone corners.
                yaw_samples = (
                    (
                        radians(zone.yaw_deg[0]),
                        0.0,
                        radians(zone.yaw_deg[1]),
                    )
                    if task.grasp.is_side
                    else (0.0,)
                )
                for yaw in yaw_samples:
                    link6_world, _ = task.grasp_pose_world(
                        block_center,
                        yaw,
                        retreat=retreat,
                        lift=lift,
                    )
                    problems.extend(task.reach_envelope.violations(
                        world_to_base(
                            link6_world,
                            base_translation,
                            base_rotation,
                        ),
                        f'{label} link6 {offset_name}',
                    ))
    return tuple(problems)


def assert_valid_against_scene(task: TaskConfig, scene) -> None:
    """Raise if a randomized pose leaves reach or camera view."""
    problems = validate_against_scene(task, scene)
    if problems:
        details = '\n  '.join(problems)
        raise ValueError(
            'VLA task configuration is geometrically inconsistent with the '
            f'USD scene contract:\n  {details}'
        )


def sample_scene_layout(
    task: TaskConfig,
    seed: int,
) -> Tuple[Tuple[Vector3, float], Tuple[Vector3, float]]:
    """
    Return deterministic ((block_center, yaw), (place_center, yaw)).

    Both poses come from the same region, so the carry direction covers every
    azimuth instead of always pointing one way. The place pose is drawn by
    rejection sampling until it clears ``min_separation`` from the block, which
    keeps the marker from overlapping the block's start footprint.

    The same seed always produces the same layout, which is what lets the ROS
    episode driver and the Isaac Sim process agree on where the block is
    without exchanging poses.
    """
    rng = random.Random(seed)
    source_x, source_y, source_yaw = task.sampling_zone.sample(rng)
    minimum = task.min_separation

    place_x, place_y, place_yaw = task.sampling_zone.sample(rng)
    for _ in range(_MAX_SEPARATION_ATTEMPTS):
        if sqrt(
            (place_x - source_x) ** 2 + (place_y - source_y) ** 2
        ) >= minimum:
            break
        place_x, place_y, place_yaw = task.sampling_zone.sample(rng)
    else:
        # Config validation proves the zone can hold the separation, so this
        # is an unlucky draw rather than an impossible one. Push the target
        # away along the sampled direction and clamp it back into the zone:
        # a deterministic layout matters more than a perfectly uniform one.
        place_x, place_y = _push_apart(
            task.sampling_zone,
            (source_x, source_y),
            (place_x, place_y),
            minimum,
            rng,
        )

    return (
        (task.block_center_world(source_x, source_y), source_yaw),
        (task.block_center_world(place_x, place_y), place_yaw),
    )


def _push_apart(
    zone: ZoneSpec,
    source: Tuple[float, float],
    place: Tuple[float, float],
    minimum: float,
    rng: random.Random,
) -> Tuple[float, float]:
    """Move ``place`` to ``minimum`` from ``source``, staying inside ``zone``."""
    delta_x = place[0] - source[0]
    delta_y = place[1] - source[1]
    distance = sqrt(delta_x * delta_x + delta_y * delta_y)
    if distance < 1e-9:
        angle = rng.uniform(-pi, pi)
        delta_x, delta_y = cos(angle), sin(angle)
        distance = 1.0
    scale = minimum / distance
    return (
        min(max(source[0] + delta_x * scale, zone.x[0]), zone.x[1]),
        min(max(source[1] + delta_y * scale, zone.y[0]), zone.y[1]),
    )


def find_task_config(
    explicit_path: Optional[str],
    package_share: Optional[Path] = None,
    default_filename: str = 'vla_task.yaml',
) -> Path:
    """Find the installed or source-tree VLA task configuration."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if package_share is not None:
        candidates.append(Path(package_share) / 'config' / default_filename)
    candidates.append(
        Path(__file__).resolve().parents[1] / 'config' / default_filename
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f'VLA task config {default_filename!r} was not found; pass an '
        'explicit path'
    )
