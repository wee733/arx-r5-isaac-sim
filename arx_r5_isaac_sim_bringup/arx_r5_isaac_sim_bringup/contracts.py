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

"""Shared interface contract for ARX R5A, ROS 2 control, and Isaac Sim."""

from pathlib import Path
from typing import Mapping, Sequence


DESCRIPTION_PACKAGE = 'isaac_ros_manipulation_arx_r5a_robot_description'
ARM_JOINTS = (
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
)
GRIPPER_COMMAND_JOINT = 'joint7'
GRIPPER_MIMIC_JOINT = 'joint8'
GRIPPER_MAX_VELOCITY = 0.02
INDEPENDENT_JOINTS = ARM_JOINTS + (GRIPPER_COMMAND_JOINT,)
ALL_JOINTS = ARM_JOINTS + (GRIPPER_COMMAND_JOINT, GRIPPER_MIMIC_JOINT)

BASE_FRAME = 'base_link'
TOOL_FRAME = 'link6'
PLANNING_GROUP = 'manipulator'

JOINT_COMMANDS_TOPIC = 'isaac_joint_commands'
JOINT_STATES_TOPIC = 'isaac_joint_states'
CLOCK_TOPIC = 'clock'

DEFAULT_JOINT_POSITIONS = {
    'joint1': 0.0,
    'joint2': 1.0,
    'joint3': 1.5,
    'joint4': 0.0,
    'joint5': 0.0,
    'joint6': 0.0,
    'joint7': 0.044,
    'joint8': 0.044,
}
JOINT_POSITION_LIMITS = {
    'joint1': (-3.14, 2.6),
    'joint2': (-0.01, 3.0),
    'joint3': (0.0, 3.14),
    'joint4': (-1.3, 1.3),
    'joint5': (-1.57, 1.57),
    'joint6': (-2.1, 2.1),
    'joint7': (0.0, 0.044),
    'joint8': (0.0, 0.044),
}


def resolve_description_mesh_uris(
    urdf_text: str,
    package_share: Path,
    package_name: str = DESCRIPTION_PACKAGE,
) -> str:
    """Replace this description package's URIs with absolute mesh paths."""
    package_share = package_share.expanduser().resolve()
    package_prefix = f'package://{package_name}/'
    return urdf_text.replace(package_prefix, f'{package_share.as_posix()}/')


def positions_from_sequence(values: Sequence[float]) -> Mapping[str, float]:
    """Map an ordered position vector to the canonical eight-joint contract."""
    if len(values) != len(ALL_JOINTS):
        raise ValueError(
            f'expected {len(ALL_JOINTS)} joint positions, got {len(values)}'
        )
    return dict(zip(ALL_JOINTS, (float(value) for value in values)))


def validate_joint_set(joint_names: Sequence[str]) -> None:
    """Raise a useful error when a joint sequence differs from the contract."""
    actual = tuple(joint_names)
    if actual != ALL_JOINTS:
        raise ValueError(f'expected joints {ALL_JOINTS}, got {actual}')
