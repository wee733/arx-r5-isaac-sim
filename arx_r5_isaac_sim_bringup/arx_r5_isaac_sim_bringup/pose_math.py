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

"""Dependency-free rigid-transform helpers for the demo goal adapter."""

from math import cos, isfinite, sin, sqrt
from typing import Sequence, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


def normalize_quaternion(quaternion: Sequence[float]) -> Quaternion:
    """Return a normalized finite xyzw quaternion."""
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(isfinite(value) for value in values):
        raise ValueError('quaternion must contain four finite values')
    magnitude = sqrt(sum(value * value for value in values))
    if magnitude <= 1e-12:
        raise ValueError('quaternion must have non-zero magnitude')
    return tuple(value / magnitude for value in values)


def _multiply_raw(left: Quaternion, right: Quaternion) -> Quaternion:
    left_x, left_y, left_z, left_w = left
    right_x, right_y, right_z, right_w = right
    return (
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
    )


def multiply_quaternions(
    left: Quaternion,
    right: Quaternion,
) -> Quaternion:
    """Return the normalized xyzw quaternion product ``left * right``."""
    return normalize_quaternion(_multiply_raw(
        normalize_quaternion(left),
        normalize_quaternion(right),
    ))


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector by a normalized xyzw quaternion."""
    quaternion = normalize_quaternion(quaternion)
    vector_quaternion = (vector[0], vector[1], vector[2], 0.0)
    conjugate = (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])
    rotated = _multiply_raw(
        _multiply_raw(quaternion, vector_quaternion),
        conjugate,
    )
    return rotated[:3]


def compose_pose(
    parent_translation: Vector3,
    parent_rotation: Quaternion,
    child_translation: Vector3,
    child_rotation: Quaternion,
) -> tuple[Vector3, Quaternion]:
    """Compose parent-to-frame and frame-to-child rigid transforms."""
    parent_rotation = normalize_quaternion(parent_rotation)
    child_rotation = normalize_quaternion(child_rotation)
    rotated_translation = rotate_vector(parent_rotation, child_translation)
    translation = tuple(
        parent_translation[index] + rotated_translation[index]
        for index in range(3)
    )
    rotation = normalize_quaternion(
        _multiply_raw(parent_rotation, child_rotation)
    )
    return translation, rotation


def conjugate_quaternion(quaternion: Quaternion) -> Quaternion:
    """Return the inverse rotation of a normalized xyzw quaternion."""
    normalized = normalize_quaternion(quaternion)
    return (-normalized[0], -normalized[1], -normalized[2], normalized[3])


def invert_pose(
    translation: Vector3,
    rotation: Quaternion,
) -> tuple[Vector3, Quaternion]:
    """Return the inverse of a parent-to-child rigid transform."""
    inverse_rotation = conjugate_quaternion(rotation)
    rotated = rotate_vector(inverse_rotation, translation)
    return tuple(-component for component in rotated), inverse_rotation


def transform_point(
    translation: Vector3,
    rotation: Quaternion,
    point: Vector3,
) -> Vector3:
    """Map a point through a parent-to-child rigid transform."""
    rotated = rotate_vector(rotation, point)
    return tuple(translation[index] + rotated[index] for index in range(3))


def cross(left: Vector3, right: Vector3) -> Vector3:
    """Return the cross product of two three-vectors."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def normalize_vector(vector: Sequence[float]) -> Vector3:
    """Return a unit three-vector."""
    values = tuple(float(value) for value in vector)
    if len(values) != 3 or not all(isfinite(value) for value in values):
        raise ValueError('vector must contain three finite values')
    magnitude = sqrt(sum(value * value for value in values))
    if magnitude <= 1e-12:
        raise ValueError('vector must have non-zero magnitude')
    return tuple(value / magnitude for value in values)


def quaternion_from_axes(
    x_axis: Vector3,
    y_axis: Vector3,
    z_axis: Vector3,
) -> Quaternion:
    """
    Return the xyzw quaternion whose columns are the given unit axes.

    The three axes must form a right-handed orthonormal basis expressed in the
    parent frame; the result rotates parent coordinates into that basis.
    """
    matrix = (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = sqrt(1.0 + trace) * 2.0
        quaternion = (
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        )
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[2][1] - matrix[1][2]) / scale,
        )
    elif matrix[1][1] > matrix[2][2]:
        scale = sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        quaternion = (
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
        )
    else:
        scale = sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        quaternion = (
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    return normalize_quaternion(quaternion)


def quaternion_from_yaw(yaw_radians: float) -> Quaternion:
    """Return the xyzw quaternion for a rotation about the parent Z axis."""
    return (0.0, 0.0, sin(yaw_radians / 2.0), cos(yaw_radians / 2.0))
