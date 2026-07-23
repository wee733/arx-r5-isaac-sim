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

from math import isfinite, sqrt
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
