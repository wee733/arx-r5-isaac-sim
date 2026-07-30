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
Render the cuMotion static collision scene from the task configuration.

Isaac ROS 4.5 parses a MoveIt ``.scene`` file and stamps every collision object
with the ``world`` frame even though the numbers below are expressed in
base_link; ``planning_scene_frame_adapter`` rewrites that header at runtime.
Generating the file from :mod:`task_config` keeps the collision world, the
rendered USD and the planned goals derived from one set of numbers.
"""

from typing import Sequence, Tuple

from arx_r5_isaac_sim_bringup.pose_math import conjugate_quaternion
from arx_r5_isaac_sim_bringup.vla.task_config import (
    BoxSpec,
    TaskConfig,
    Vector3,
    world_to_base,
)


SCENE_HEADER = '(noname)+'
SCENE_FOOTER = '.'


def _format_floats(values: Sequence[float]) -> str:
    # `+ 0.0` collapses negative zero, which the quaternion inverse produces
    # for the unused X and Z components and which reads as a typo in a scene
    # file people compare by eye.
    return ' '.join(f'{float(value) + 0.0:.9g}' for value in values)


def _collision_object(
    name: str,
    translation: Vector3,
    rotation: Sequence[float],
    size: Vector3,
) -> Tuple[str, ...]:
    """Return the MoveIt .scene lines for one axis-aligned box."""
    return (
        f'* {name}',
        _format_floats(translation),
        _format_floats(rotation),
        '1',
        'box',
        _format_floats(size),
        '0 0 0',
        '0 0 0 1',
        '0 0 0 0',
        '0',
    )


def render_collision_scene(task: TaskConfig, scene) -> str:
    """
    Return the cuMotion .scene text for the static VLA workcell.

    The boxes stay axis-aligned in the world, so in base_link they all carry
    the inverse of the robot's own forward pitch.
    """
    base_translation = scene.expected_world_to_base_translation
    base_rotation = scene.expected_world_to_base_rotation
    box_rotation = conjugate_quaternion(base_rotation)

    boxes = (
        ('Ground', task.workcell.ground),
        ('Mount', task.workcell.mount.collision_box),
        ('Platform', task.workcell.platform.box),
    )
    lines = [SCENE_HEADER]
    for name, box in boxes:
        if not isinstance(box, BoxSpec):
            raise TypeError(f'{name} must be a BoxSpec')
        lines.extend(_collision_object(
            name,
            world_to_base(box.center, base_translation, base_rotation),
            box_rotation,
            box.size,
        ))
    lines.append(SCENE_FOOTER)
    return '\n'.join(lines) + '\n'
