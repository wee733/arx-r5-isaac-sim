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

"""Test the side-grasp style and the visual placement marker."""

import math
from pathlib import Path

from arx_r5_isaac_sim_bringup.pose_math import rotate_vector
from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.episode_events import EpisodePhase
from arx_r5_isaac_sim_bringup.vla.episode_plan import (
    build_episode_waypoints,
    phase_rotation_candidates,
)
from arx_r5_isaac_sim_bringup.vla.task_config import (
    GraspStyle,
    load_task_config,
    sample_scene_layout,
    side_grasp_rotation,
    validate_against_scene,
    world_to_base,
)

import pytest

import yaml


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / 'config'
TASK_CONFIG = CONFIG_DIRECTORY / 'vla_task.yaml'
SCENE_CONFIG = CONFIG_DIRECTORY / 'vla_scene.yaml'


@pytest.fixture(name='scene')
def scene_fixture():
    """Return the packaged VLA USD scene contract."""
    return load_usd_scene_config(SCENE_CONFIG)


def _task_with(tmp_path: Path, mutate):
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding='utf-8'))
    mutate(raw)
    destination = tmp_path / 'vla_task.yaml'
    destination.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_task_config(destination)


@pytest.fixture(name='side_task')
def side_task_fixture(tmp_path):
    """Return the packaged task switched to side grasps."""
    def use_side(raw):
        raw['grasp']['style'] = GraspStyle.SIDE

    return _task_with(tmp_path, use_side)


# ----------------------------------------------------------- grasp styles --


def test_default_style_is_top():
    """The validated style stays the default."""
    assert load_task_config(TASK_CONFIG).grasp.style == GraspStyle.TOP
    assert not load_task_config(TASK_CONFIG).grasp.is_side


def test_unknown_style_is_rejected(tmp_path):
    """A typo must fail at startup, not silently fall back to top grasps."""
    def bad_style(raw):
        raw['grasp']['style'] = 'diagonal'

    with pytest.raises(ValueError, match='grasp.style must be one of'):
        _task_with(tmp_path, bad_style)


def test_side_grasp_tool_axis_is_horizontal(side_task):
    """The wrist must approach the block level, not from above."""
    for yaw_deg in (-45.0, 0.0, 30.0, 45.0):
        rotation = side_grasp_rotation(math.radians(yaw_deg))
        tool_axis = rotate_vector(rotation, (1.0, 0.0, 0.0))
        assert tool_axis[2] == pytest.approx(0.0, abs=1e-9)


def test_side_grasp_approaches_along_the_block_x_axis(side_task):
    """The approach must follow the block's yaw so faces stay square on."""
    for yaw_deg in (-45.0, 0.0, 30.0):
        yaw = math.radians(yaw_deg)
        tool_axis = rotate_vector(side_grasp_rotation(yaw), (1.0, 0.0, 0.0))
        expected = (math.cos(yaw), math.sin(yaw), 0.0)
        for index in range(3):
            assert tool_axis[index] == pytest.approx(expected[index], abs=1e-9)


def test_side_grasp_fingers_close_horizontally(side_task):
    """The finger axis must be horizontal and normal to the approach."""
    for yaw_deg in (-45.0, 0.0, 45.0):
        rotation = side_grasp_rotation(math.radians(yaw_deg))
        tool_axis = rotate_vector(rotation, (1.0, 0.0, 0.0))
        finger_axis = rotate_vector(rotation, (0.0, 1.0, 0.0))
        assert finger_axis[2] == pytest.approx(0.0, abs=1e-9)
        dot = sum(tool_axis[i] * finger_axis[i] for i in range(3))
        assert dot == pytest.approx(0.0, abs=1e-9)


def test_side_grasp_keeps_the_wrist_upright(side_task):
    """link6 +Z must point up so the gripper is not rolled over."""
    for yaw_deg in (-45.0, 0.0, 45.0):
        rotation = side_grasp_rotation(math.radians(yaw_deg))
        up_axis = rotate_vector(rotation, (0.0, 0.0, 1.0))
        assert up_axis[2] == pytest.approx(1.0, abs=1e-9)


def test_side_grasp_sits_below_the_block_top(side_task):
    """Grasping below mid-height keeps the hold under the centre of mass."""
    surface = side_task.workcell.platform.top_z
    grasp_z = surface + side_task.grasp.side_grasp_height
    block_top = surface + side_task.block.height
    assert surface < grasp_z < block_top
    assert side_task.grasp.side_grasp_height < side_task.block.height / 2.0


def test_side_grasp_wrist_is_offset_along_the_approach(side_task):
    """The wrist must sit back from the block by the grasp frame offset."""
    center = side_task.block.nominal_center
    translation, rotation = side_task.grasp_pose_world(center, 0.0)
    approach = rotate_vector(rotation, (1.0, 0.0, 0.0))
    # grasp_frame is grasp_frame_offset along +X from link6, and must land on
    # the block's own axis.
    grasp_frame = tuple(
        translation[index] + approach[index] * side_task.grasp.grasp_frame_offset
        for index in range(3)
    )
    assert grasp_frame[0] == pytest.approx(center[0], abs=1e-9)
    assert grasp_frame[1] == pytest.approx(center[1], abs=1e-9)
    expected_z = (
        side_task.workcell.platform.top_z + side_task.grasp.side_grasp_height
    )
    assert grasp_frame[2] == pytest.approx(expected_z, abs=1e-9)


def test_side_grasp_retreat_is_horizontal(side_task):
    """The retreat must back straight out, not lift the block sideways."""
    center = side_task.block.nominal_center
    grasp, _ = side_task.grasp_pose_world(center, 0.0)
    retreated, _ = side_task.grasp_pose_world(
        center,
        0.0,
        retreat=side_task.retreat_distance,
    )
    assert retreated[2] == pytest.approx(grasp[2], abs=1e-9)
    horizontal = math.dist(retreated[:2], grasp[:2])
    assert horizontal == pytest.approx(side_task.retreat_distance, abs=1e-9)


def test_side_grasp_retreat_distance_follows_the_style(side_task):
    """A side grasp backs off by its own configured distance."""
    assert side_task.retreat_distance == pytest.approx(
        side_task.grasp.side_approach_distance
    )


def test_side_grasp_zones_are_reachable(side_task, scene):
    """Side grasps put the wrist lower and further out than top grasps."""
    assert validate_against_scene(side_task, scene) == ()


def test_side_grasp_waypoints_have_the_same_phases(side_task):
    """Switching style must not change the episode structure."""
    phases = [phase for phase, _, _ in build_episode_waypoints(side_task, 3)]
    assert phases == [
        EpisodePhase.APPROACH,
        EpisodePhase.GRASP,
        EpisodePhase.LIFT,
        EpisodePhase.TRANSFER,
        EpisodePhase.PLACE,
        EpisodePhase.RETREAT,
    ]


def test_side_grasp_lift_clears_the_platform(side_task):
    """The block must come off the surface before it is transferred."""
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(side_task, 8)
    }
    assert waypoints[EpisodePhase.LIFT][2] - waypoints[EpisodePhase.GRASP][2] \
        == pytest.approx(side_task.grasp.lift_height)
    assert waypoints[EpisodePhase.TRANSFER][2] == pytest.approx(
        waypoints[EpisodePhase.LIFT][2]
    )


def test_side_grasp_retreat_remains_horizontal(side_task):
    """Top-grasp IK protection must not change the side retreat contract."""
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(side_task, 8)
    }
    place = waypoints[EpisodePhase.PLACE]
    retreat = waypoints[EpisodePhase.RETREAT]

    assert retreat[2] == pytest.approx(place[2])
    horizontal = math.hypot(retreat[0] - place[0], retreat[1] - place[1])
    assert horizontal == pytest.approx(
        side_task.grasp.side_approach_distance,
        abs=1e-9,
    )


def test_side_grasp_transfer_does_not_invent_yaw_symmetry(side_task):
    """Only the square top grasp may switch quarter-turn IK branches."""
    waypoints = {
        phase: rotation
        for phase, _, rotation in build_episode_waypoints(side_task, 3)
    }
    source_rotation = waypoints[EpisodePhase.GRASP]

    assert phase_rotation_candidates(
        side_task,
        EpisodePhase.TRANSFER,
        waypoints[EpisodePhase.TRANSFER],
        source_rotation,
        None,
    ) == (source_rotation,)


def test_side_grasp_waypoints_stay_in_the_reach_envelope(side_task, scene):
    """Every sampled side-grasp waypoint must stay inside the envelope."""
    for seed in range(25):
        for _, translation, _ in build_episode_waypoints(side_task, seed):
            assert side_task.reach_envelope.violations(
                world_to_base(
                    translation,
                    scene.expected_world_to_base_translation,
                    scene.expected_world_to_base_rotation,
                ),
                'waypoint',
            ) == ()


def test_side_grasp_height_must_be_inside_the_block(tmp_path):
    """A grasp above the block top would close on nothing."""
    def raise_grasp(raw):
        raw['grasp']['style'] = GraspStyle.SIDE
        raw['grasp']['side_grasp_height'] = 0.30

    with pytest.raises(ValueError, match='side_grasp_height'):
        _task_with(tmp_path, raise_grasp)


# ------------------------------------------------------- placement marker --


def test_place_marker_matches_the_block_footprint():
    """The decal must show the block's own footprint, not an arbitrary patch."""
    task = load_task_config(TASK_CONFIG)
    assert task.place_marker.size == pytest.approx(
        (task.block.size[0], task.block.size[1])
    )


def test_place_marker_sits_on_the_platform():
    """The decal must be just above the surface to avoid z-fighting."""
    task = load_task_config(TASK_CONFIG)
    assert 0.0 < task.place_marker.height_above_surface <= 0.005


def test_place_marker_is_blue():
    """The marker must be visually distinct from the red block."""
    task = load_task_config(TASK_CONFIG)
    red, green, blue = task.place_marker.color
    assert blue > 0.5
    assert blue > red and blue > green


def test_place_marker_colour_channels_are_normalized(tmp_path):
    """USD display colour is 0-1, so an 8-bit value must be rejected."""
    def use_bytes(raw):
        raw['place_marker']['color'] = [13, 64, 230]

    with pytest.raises(ValueError, match='color channels'):
        _task_with(tmp_path, use_bytes)


def test_place_marker_tracks_the_sampled_place_pose():
    """The marker's target must be the same pose the episode places at."""
    task = load_task_config(TASK_CONFIG)
    for seed in range(10):
        _, (place_center, _) = sample_scene_layout(task, seed)
        waypoints = {
            phase: translation
            for phase, translation, _ in build_episode_waypoints(task, seed)
        }
        assert waypoints[EpisodePhase.PLACE][0] == pytest.approx(place_center[0])
        assert waypoints[EpisodePhase.PLACE][1] == pytest.approx(place_center[1])


# ------------------------------------------------------------------ block --


def test_block_footprint_fits_the_gripper():
    """The block must be narrower than the fully open fingertip aperture."""
    task = load_task_config(TASK_CONFIG)
    aperture = 2.0 * task.grasp.gripper_open
    assert max(task.block.size[0], task.block.size[1]) < aperture


def test_block_mass_is_declared_by_the_task():
    """The VLA workcell owns its block geometry, so it owns the mass too."""
    task = load_task_config(TASK_CONFIG)
    assert task.block.mass_kg > 0.0
    volume = task.block.size[0] * task.block.size[1] * task.block.size[2]
    density = task.block.mass_kg / volume
    # Plausible for a light solid: heavier than foam, lighter than hardwood.
    assert 50.0 < density < 1000.0


def test_block_mass_must_be_positive(tmp_path):
    """A zero-mass rigid body makes PhysX behaviour undefined."""
    def zero_mass(raw):
        raw['block']['mass_kg'] = 0.0

    with pytest.raises(ValueError, match='block.mass_kg'):
        _task_with(tmp_path, zero_mass)
