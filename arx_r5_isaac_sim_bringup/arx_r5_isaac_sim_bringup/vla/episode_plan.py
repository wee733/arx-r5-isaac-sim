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
Waypoint planning for a scripted pick-and-place episode.

Kept free of ROS so the whole trajectory shape stays unit-testable without a
sourced environment, and so a real-robot driver can reuse it unchanged.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from arx_r5_isaac_sim_bringup.contracts import ARM_JOINTS
from arx_r5_isaac_sim_bringup.vla.episode_events import EpisodePhase
from arx_r5_isaac_sim_bringup.vla.task_config import (
    Quaternion,
    sample_scene_layout,
    TaskConfig,
    Vector3,
)

import yaml


Waypoint = Tuple[str, Vector3, Quaternion]


def phase_rotation_candidates(
    task: TaskConfig,
    phase: str,
    nominal_rotation: Sequence[float],
    source_rotation: Optional[Sequence[float]],
    place_rotation: Optional[Sequence[float]],
) -> Tuple[Quaternion, ...]:
    """
    Return the physically equivalent wrist goals for one phase.

    Once a grasp orientation has been selected, APPROACH through LIFT must
    retain it so the wrist does not turn around the block during acquisition.
    TRANSFER is different: the square block may be carried at any equivalent
    quarter-turn orientation. Keeping the selected source orientation first
    avoids an unnecessary turn when it remains reachable, while the remaining
    candidates let cuMotion choose another IK branch at the place position.

    PLACE chooses among orientations equivalent to the requested target yaw.
    RETREAT then keeps the orientation that PLACE actually selected. Side
    grasps and non-square blocks naturally produce one candidate through
    :meth:`TaskConfig.equivalent_grasp_rotations`.
    """
    nominal = tuple(float(value) for value in nominal_rotation)
    source = (
        None if source_rotation is None
        else tuple(float(value) for value in source_rotation)
    )
    placed = (
        None if place_rotation is None
        else tuple(float(value) for value in place_rotation)
    )

    if phase in (
        EpisodePhase.APPROACH,
        EpisodePhase.GRASP,
        EpisodePhase.LIFT,
    ):
        if source is not None:
            return (source,)
    elif phase == EpisodePhase.TRANSFER:
        if source is not None:
            return task.equivalent_grasp_rotations(source)
    elif placed is not None:
        return (placed,)
    return task.equivalent_grasp_rotations(nominal)


def load_home_positions(path: str | Path) -> List[float]:
    """
    Read the between-episode arm configuration from YAML.

    Returns the six arm joints in the canonical ``ARM_JOINTS`` order; the
    gripper is commanded separately through its own action.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(
            f'home positions file is not a mapping: {config_path}'
        )
    positions = raw.get('initial_positions')
    if not isinstance(positions, dict):
        raise ValueError(f'{config_path} has no initial_positions mapping')
    missing = sorted(set(ARM_JOINTS) - set(positions))
    if missing:
        raise ValueError(f'{config_path} is missing arm joints: {missing}')
    return [float(positions[joint_name]) for joint_name in ARM_JOINTS]


def build_episode_waypoints(task: TaskConfig, seed: int) -> List[Waypoint]:
    """
    Return the ordered (phase, translation, rotation) world waypoints.

    The seed alone fixes the entire episode: it selects the block and place
    poses, and every waypoint is a retreat or lift offset from one of the two
    grasp poses, so the sequence is reproducible from the recorded seed alone.

    ``retreat`` backs away along the approach direction (up for a top grasp,
    horizontally for a side grasp); ``lift`` always raises vertically.
    """
    (block_center, block_yaw), (place_center, place_yaw) = sample_scene_layout(
        task,
        seed,
    )
    retreat = task.retreat_distance
    lift = task.grasp.lift_height
    # A top grasp should retreat to the same proven carry pose used by LIFT
    # and TRANSFER.  Adding approach_height on top of release_clearance would
    # put the raised-palm geometry above that IK envelope.  Side grasps still
    # need their horizontal tool-axis retreat at the release height.
    if task.grasp.is_side:
        retreat_offset = retreat
        retreat_lift = task.placement.release_clearance
    else:
        retreat_offset = 0.0
        retreat_lift = lift
    plan = (
        # (phase, centre, yaw, retreat, lift)
        (EpisodePhase.APPROACH, block_center, block_yaw, retreat, 0.0),
        (EpisodePhase.GRASP, block_center, block_yaw, 0.0, 0.0),
        (EpisodePhase.LIFT, block_center, block_yaw, 0.0, lift),
        (EpisodePhase.TRANSFER, place_center, place_yaw, 0.0, lift),
        # The official cuMotion cuboid attachment is conservatively wrapped
        # by collision spheres whose envelope extends beyond the real block,
        # and the ARX XRDF adds another attached-object buffer. Releasing at
        # the final resting centre would therefore plan that inflated model
        # into the support platform. Stop above it, detach, and let PhysX
        # settle the block onto the unchanged ground-truth target.
        (
            EpisodePhase.PLACE,
            place_center,
            place_yaw,
            0.0,
            task.placement.release_clearance,
        ),
        (
            EpisodePhase.RETREAT,
            place_center,
            place_yaw,
            retreat_offset,
            retreat_lift,
        ),
    )
    return [
        (phase,) + task.grasp_pose_world(
            center,
            yaw,
            retreat=retreat_distance,
            lift=lift_distance,
        )
        for phase, center, yaw, retreat_distance, lift_distance in plan
    ]
