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

"""Validate the VLA task configuration and its geometry helpers."""

import copy
import math
from pathlib import Path

from arx_r5_isaac_sim_bringup.pose_math import (
    multiply_quaternions,
    quaternion_from_yaw,
    rotate_vector,
)
from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.collision_scene import render_collision_scene
from arx_r5_isaac_sim_bringup.vla.task_config import (
    assert_valid_against_scene,
    load_task_config,
    next_stable_sample_count,
    optical_frame_offsets,
    pose_world_to_base,
    sample_scene_layout,
    top_grasp_rotation,
    update_stable_pose_window,
    update_stable_translation_window,
    validate_against_scene,
    world_to_base,
)

import pytest

import yaml


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / 'config'
TASK_CONFIG = CONFIG_DIRECTORY / 'vla_task.yaml'
SCENE_CONFIG = CONFIG_DIRECTORY / 'vla_scene.yaml'
COLLISION_SCENE = CONFIG_DIRECTORY / 'vla_ground.scene'


@pytest.fixture(name='task')
def task_fixture():
    """Return the packaged VLA task configuration."""
    return load_task_config(TASK_CONFIG)


@pytest.fixture(name='scene')
def scene_fixture():
    """Return the packaged VLA USD scene contract."""
    return load_usd_scene_config(SCENE_CONFIG)


def _write_task(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding='utf-8'))
    mutate(raw)
    destination = tmp_path / 'vla_task.yaml'
    destination.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return destination


def test_packaged_configuration_is_self_consistent(task, scene):
    """Every randomized pose must be reachable and visible at startup."""
    assert validate_against_scene(task, scene) == ()
    assert scene.source_object_mass_kg == pytest.approx(task.block.mass_kg)


def test_scene_contract_declares_the_column_mount(scene):
    """The robot must sit 0.35 m up with a 15 degree forward pitch."""
    assert scene.expected_world_to_base_translation == (-0.2, 0.0, 0.35)
    rotation = scene.expected_world_to_base_rotation
    pitch_deg = math.degrees(2.0 * math.asin(rotation[1]))
    assert pitch_deg == pytest.approx(15.0, abs=1e-4)
    assert rotation[0] == pytest.approx(0.0)
    assert rotation[2] == pytest.approx(0.0)


def test_vla_scene_has_no_apriltag_textures(scene):
    """The collection workcell drops the tags the AprilTag demo needs."""
    assert not scene.has_tag_textures
    assert scene.source_tag_texture_prim is None


def test_legacy_scene_still_declares_tag_textures():
    """Relaxing the tags section must not break the AprilTag workcell."""
    legacy = load_usd_scene_config(CONFIG_DIRECTORY / 'arx_sim_usd_scene.yaml')
    assert legacy.has_tag_textures


def test_block_rests_on_the_platform(task):
    """The nominal block centre must sit exactly on the platform surface."""
    expected_z = task.workcell.platform.top_z + task.block.height / 2.0
    assert task.block.nominal_center[2] == pytest.approx(expected_z)
    assert task.workcell.platform.top_z == pytest.approx(0.15)


def test_release_clearance_exceeds_official_attachment_padding(task):
    """PLACE must not drive cuMotion's padded cuboid into the platform."""
    official_max_overshoot = 0.01
    xrdf_attached_object_buffer = 0.002
    assert task.placement.release_clearance > (
        official_max_overshoot + xrdf_attached_object_buffer
    )


def test_placement_validation_accepts_a_small_resting_pose_error(task):
    """Millimetre-scale PhysX settling must still count as on target."""
    target = task.block.nominal_center
    actual = (target[0] + 0.003, target[1] - 0.002, target[2] + 0.004)
    xy_error, z_error = task.placement.errors(actual, target)
    assert xy_error == pytest.approx(math.hypot(0.003, 0.002))
    assert z_error == pytest.approx(0.004)
    assert task.placement.accepts(actual, target)


def test_placement_validation_accepts_square_symmetric_yaw(task):
    """Quarter turns of the square block are equivalent target yaws."""
    target = task.block.nominal_center
    target_yaw = math.radians(17.0)
    actual_rotation = quaternion_from_yaw(
        target_yaw + math.radians(90.0 + 4.0)
    )
    errors = task.placement.errors(
        target,
        target,
        actual_rotation,
        target_yaw,
    )
    assert errors[2] == pytest.approx(0.0)
    assert errors[3] == pytest.approx(4.0)
    assert task.placement.accepts(
        target,
        target,
        actual_rotation,
        target_yaw,
    )


def test_placement_validation_rejects_wrong_yaw_modulo_symmetry(task):
    """A yaw beyond tolerance must fail even when translation is exact."""
    target = task.block.nominal_center
    target_yaw = math.radians(-8.0)
    actual_rotation = quaternion_from_yaw(
        target_yaw + math.radians(
            90.0 + task.placement.yaw_tolerance_deg + 1.0
        )
    )
    assert not task.placement.accepts(
        target,
        target,
        actual_rotation,
        target_yaw,
    )


def test_placement_validation_rejects_a_block_beside_the_target(task):
    """A detached block is not a successful place merely because it is free."""
    target = task.block.nominal_center
    actual = (
        target[0] + task.placement.xy_tolerance + 0.001,
        target[1],
        target[2],
    )
    assert not task.placement.accepts(actual, target)


def test_placement_validation_rejects_a_tipped_block(task):
    """The centre height catches a block resting on its long side."""
    target = task.block.nominal_center
    actual = (
        target[0],
        target[1],
        task.workcell.platform.top_z + task.block.size[0] / 2.0,
    )
    assert not task.placement.accepts(actual, target)


def test_placement_validation_rejects_tilt_at_nominal_centre_height(task):
    """Tilt must fail independently of the centre-height approximation."""
    target = task.block.nominal_center
    tilt = math.radians(task.placement.max_tilt_deg + 1.0)
    actual_rotation = multiply_quaternions(
        quaternion_from_yaw(math.radians(23.0)),
        (math.sin(tilt / 2.0), 0.0, 0.0, math.cos(tilt / 2.0)),
    )
    _, _, tilt_error, _ = task.placement.errors(
        target,
        target,
        actual_rotation,
        math.radians(23.0),
    )
    assert tilt_error == pytest.approx(task.placement.max_tilt_deg + 1.0)
    assert not task.placement.accepts(
        target,
        target,
        actual_rotation,
        math.radians(23.0),
    )


def test_placement_stability_requires_consecutive_nearby_samples(task):
    """A moving sample resets the settling gate instead of accumulating it."""
    tolerance = task.placement.stability_tolerance
    first = (0.0, 0.0, 0.0)
    nearby = (tolerance / 2.0, 0.0, 0.0)
    moving = (tolerance * 2.0, 0.0, 0.0)
    count = next_stable_sample_count(None, first, 0, tolerance)
    assert count == 1
    count = next_stable_sample_count(first, nearby, count, tolerance)
    assert count == 2
    count = next_stable_sample_count(nearby, moving, count, tolerance)
    assert count == 1


def test_placement_stability_is_anchored_to_the_window_first_sample(task):
    """Many tiny adjacent moves cannot hide cumulative block drift."""
    tolerance = task.placement.stability_tolerance
    window = update_stable_translation_window(
        None,
        (0.0, 0.0, 0.0),
        1_000_000_000,
        tolerance,
    )
    window = update_stable_translation_window(
        window,
        (0.75 * tolerance, 0.0, 0.0),
        1_200_000_000,
        tolerance,
    )
    assert window.sample_count == 2
    assert window.duration_sec == pytest.approx(0.2)

    # This is close to the preceding sample but outside the tolerance from
    # the original anchor, so it must begin a fresh stability window.
    window = update_stable_translation_window(
        window,
        (1.5 * tolerance, 0.0, 0.0),
        1_400_000_000,
        tolerance,
    )
    assert window.anchor == pytest.approx((1.5 * tolerance, 0.0, 0.0))
    assert window.sample_count == 1
    assert window.duration_sec == pytest.approx(0.0)


def test_placement_stability_rejects_rotation_while_center_is_static(task):
    """A spinning/tipping block cannot satisfy the stable settle window."""
    tolerance = task.placement.stability_tolerance
    orientation_tolerance = (
        task.placement.stability_orientation_tolerance_deg
    )
    window = update_stable_pose_window(
        None,
        (0.0, 0.0, 0.0),
        quaternion_from_yaw(0.0),
        1_000_000_000,
        tolerance,
        orientation_tolerance,
    )
    window = update_stable_pose_window(
        window,
        (0.0, 0.0, 0.0),
        quaternion_from_yaw(math.radians(orientation_tolerance + 1.0)),
        1_200_000_000,
        tolerance,
        orientation_tolerance,
    )
    assert window.sample_count == 1
    assert window.duration_sec == pytest.approx(0.0)


def test_placement_stability_requires_elapsed_simulation_time(task):
    """Sample count alone cannot satisfy the configured settling duration."""
    tolerance = task.placement.stability_tolerance
    window = None
    for index in range(20):
        window = update_stable_translation_window(
            window,
            (tolerance / 4.0, 0.0, 0.0),
            2_000_000_000 + index * 1_000_000,
            tolerance,
        )
    assert window.sample_count == 20
    assert window.duration_sec < task.placement.stable_duration_sec

    window = update_stable_translation_window(
        window,
        (tolerance / 5.0, 0.0, 0.0),
        2_000_000_000 + int(task.placement.stable_duration_sec * 1e9),
        tolerance,
    )
    assert window.duration_sec >= task.placement.stable_duration_sec


def test_min_separation_clears_the_block_footprint(task):
    """
    The target must never overlap where the block started.

    Both poses come from one shared region now, so disjoint bands no longer
    guarantee this; the separation constraint is what does.
    """
    # Worst case footprint is the 45 degree diagonal of the square block.
    diagonal = task.block.size[0] * math.sqrt(2.0)
    assert task.min_separation > diagonal


def test_sampled_layouts_always_clear_the_minimum_separation(task):
    """Rejection sampling must hold for every seed, not just most of them."""
    for seed in range(400):
        (block, _), (place, _) = sample_scene_layout(task, seed)
        separation = math.dist(block[:2], place[:2])
        assert separation >= task.min_separation - 1e-9, (seed, separation)


def test_carry_direction_is_not_one_sided(task):
    """A fixed carry direction is a shortcut a policy can learn instead of the task."""
    forward = backward = 0
    for seed in range(400):
        (block, _), (place, _) = sample_scene_layout(task, seed)
        if place[1] > block[1]:
            forward += 1
        else:
            backward += 1
    # Neither direction may dominate; an exact split is not expected.
    assert min(forward, backward) > 0.3 * (forward + backward)


def test_block_and_target_share_one_region(task):
    """Sampling both from one zone is what makes the poses interchangeable."""
    blocks = []
    places = []
    for seed in range(400):
        (block, _), (place, _) = sample_scene_layout(task, seed)
        blocks.append(block[1])
        places.append(place[1])
    # The two marginal distributions must overlap heavily, not occupy bands.
    assert min(blocks) < min(places) + 0.05
    assert max(places) > max(blocks) - 0.05


def test_grasp_rotation_points_the_tool_down(task):
    """link6 +X must point at the platform so grasp_frame lands on the block."""
    for yaw_deg in (-45.0, 0.0, 30.0, 45.0):
        rotation = top_grasp_rotation(math.radians(yaw_deg))
        tool_axis = rotate_vector(rotation, (1.0, 0.0, 0.0))
        assert tool_axis[2] == pytest.approx(-1.0, abs=1e-9)


def test_grasp_rotation_aligns_fingers_with_the_block(task):
    """The finger separation axis must follow the block's own yaw."""
    for yaw_deg in (-45.0, 0.0, 30.0):
        yaw = math.radians(yaw_deg)
        rotation = top_grasp_rotation(yaw)
        finger_axis = rotate_vector(rotation, (0.0, 1.0, 0.0))
        expected = (math.cos(yaw), math.sin(yaw), 0.0)
        for index in range(3):
            assert finger_axis[index] == pytest.approx(expected[index], abs=1e-9)


def test_grasp_pose_sits_above_the_block_centre(task):
    """The grasp keeps grasp_frame inside the block below its top face."""
    center = task.block.nominal_center
    translation, _ = task.grasp_pose_world(center, 0.0)
    grasp_point_z = translation[2] - task.grasp.grasp_frame_offset
    top_z = center[2] + task.block.height / 2.0
    assert grasp_point_z == pytest.approx(top_z - task.grasp.depth_below_top)
    assert grasp_point_z < top_z
    assert grasp_point_z > center[2] - task.block.height / 2.0


def test_top_grasp_clears_the_palm_and_keeps_finger_overlap(task):
    """The calibrated R5A geometry must not press the palm into the block."""
    # Measured along link6 +X from the checked-in collision meshes/URDF.
    palm_extent = 0.084
    fingertip_extent = 0.08657 + 0.071
    minimum_palm_clearance = 0.010
    minimum_finger_overlap = 0.030
    attachment_margin = 0.010

    center = task.block.nominal_center
    link6_pose, _ = task.grasp_pose_world(center, 0.0)
    top_z = center[2] + task.block.height / 2.0
    palm_bottom_z = link6_pose[2] - palm_extent
    fingertip_bottom_z = link6_pose[2] - fingertip_extent
    grasp_frame_z = link6_pose[2] - task.grasp.grasp_frame_offset

    assert palm_bottom_z - top_z >= minimum_palm_clearance
    assert top_z - fingertip_bottom_z >= minimum_finger_overlap
    assert abs(grasp_frame_z - center[2]) <= (
        task.grasp.attach_max_distance - attachment_margin
    )


def test_lift_offset_raises_vertically(task):
    """Lifting a grasped block is vertical whatever the grasp style."""
    center = task.block.nominal_center
    grasp, grasp_rotation = task.grasp_pose_world(center, 0.3)
    lifted, lifted_rotation = task.grasp_pose_world(
        center,
        0.3,
        lift=task.grasp.lift_height,
    )
    assert lifted[0] == pytest.approx(grasp[0])
    assert lifted[1] == pytest.approx(grasp[1])
    assert lifted[2] - grasp[2] == pytest.approx(task.grasp.lift_height)
    assert lifted_rotation == grasp_rotation


def test_top_grasp_retreat_is_vertical(task):
    """A top grasp backs off straight up, so retreat and lift coincide."""
    assert not task.grasp.is_side
    center = task.block.nominal_center
    grasp, _ = task.grasp_pose_world(center, 0.0)
    retreated, _ = task.grasp_pose_world(
        center,
        0.0,
        retreat=task.retreat_distance,
    )
    assert retreated[0] == pytest.approx(grasp[0])
    assert retreated[1] == pytest.approx(grasp[1])
    assert retreated[2] - grasp[2] == pytest.approx(task.retreat_distance)


def test_top_grasp_retreat_distance_is_the_approach_height(task):
    """The retreat distance must follow the configured style."""
    assert task.retreat_distance == pytest.approx(task.grasp.approach_height)


def test_world_to_base_round_trips(scene):
    """Mapping a point into base_link and back must be lossless."""
    translation = scene.expected_world_to_base_translation
    rotation = scene.expected_world_to_base_rotation
    point = (0.25, -0.03, 0.225)
    base_point = world_to_base(point, translation, rotation)
    restored = tuple(
        translation[index] + rotate_vector(rotation, base_point)[index]
        for index in range(3)
    )
    for index in range(3):
        assert restored[index] == pytest.approx(point[index], abs=1e-12)


def test_pose_world_to_base_preserves_the_tool_axis(task, scene):
    """The downward tool axis must survive the change of frame."""
    translation, rotation = task.grasp_pose_world(task.block.nominal_center, 0.0)
    base_translation, base_rotation = pose_world_to_base(
        translation,
        rotation,
        scene.expected_world_to_base_translation,
        scene.expected_world_to_base_rotation,
    )
    tool_axis_base = rotate_vector(base_rotation, (1.0, 0.0, 0.0))
    # base_link is pitched 15 degrees forward, so a world-down tool axis is
    # 15 degrees off -Z in base_link.
    assert tool_axis_base[2] == pytest.approx(-math.cos(math.radians(15.0)))
    assert base_translation == pytest.approx(
        world_to_base(
            translation,
            scene.expected_world_to_base_translation,
            scene.expected_world_to_base_rotation,
        )
    )


def test_nominal_block_is_on_the_camera_axis(task, scene):
    """The platform height was chosen to centre the block for the ZED X."""
    camera = scene.cameras[task.camera_visibility.camera]
    base_point = world_to_base(
        task.block.nominal_center,
        scene.expected_world_to_base_translation,
        scene.expected_world_to_base_rotation,
    )
    horizontal_deg, vertical_deg, depth = optical_frame_offsets(
        base_point,
        camera.expected_parent_to_optical_translation,
        camera.expected_optical_forward,
    )
    assert abs(vertical_deg) < 2.0
    assert abs(horizontal_deg) < 40.0
    assert depth > task.camera_visibility.min_depth


def test_sampling_is_deterministic(task):
    """The same seed must always produce the same layout."""
    assert sample_scene_layout(task, 11) == sample_scene_layout(task, 11)
    assert sample_scene_layout(task, 11) != sample_scene_layout(task, 12)


def test_sampled_layouts_stay_inside_their_zones(task):
    """Every sampled pose must land in the shared sampling region."""
    zone = task.sampling_zone
    for seed in range(60):
        (block, block_yaw), (place, place_yaw) = sample_scene_layout(task, seed)
        for center, yaw in (
            (block, block_yaw),
            (place, place_yaw),
        ):
            assert zone.x[0] <= center[0] <= zone.x[1]
            assert zone.y[0] <= center[1] <= zone.y[1]
            assert center[2] == pytest.approx(
                task.block.center_z_on(task.workcell.platform.top_z)
            )
            assert (
                math.radians(zone.yaw_deg[0])
                <= yaw
                <= math.radians(zone.yaw_deg[1])
            )


def test_sampled_layouts_are_reachable_and_visible(task, scene):
    """Corner validation must imply every interior sample is also valid."""
    camera = scene.cameras[task.camera_visibility.camera]
    for seed in range(40):
        (block, _), (place, _) = sample_scene_layout(task, seed)
        for center in (block, place):
            base_point = world_to_base(
                center,
                scene.expected_world_to_base_translation,
                scene.expected_world_to_base_rotation,
            )
            assert task.camera_visibility.violations(
                optical_frame_offsets(
                    base_point,
                    camera.expected_parent_to_optical_translation,
                    camera.expected_optical_forward,
                ),
                'sample',
            ) == ()
            for height in (0.0, task.grasp.approach_height, task.grasp.lift_height):
                link6, _ = task.grasp_pose_world(center, 0.0, retreat=height)
                assert task.reach_envelope.violations(
                    world_to_base(
                        link6,
                        scene.expected_world_to_base_translation,
                        scene.expected_world_to_base_rotation,
                    ),
                    'sample',
                ) == ()


def test_unreachable_zone_is_rejected(tmp_path, scene):
    """A zone beyond the measured envelope must fail at startup."""
    def push_out_of_reach(raw):
        raw['sampling_zone']['x'] = [0.60, 0.70]

    task = load_task_config(_write_task(tmp_path, push_out_of_reach))
    problems = validate_against_scene(task, scene)
    assert problems
    assert any('max_x' in problem for problem in problems)
    with pytest.raises(ValueError, match='geometrically inconsistent'):
        assert_valid_against_scene(task, scene)


def test_invisible_zone_is_rejected(tmp_path, scene):
    """A zone outside the ZED X frustum must fail at startup."""
    def push_out_of_view(raw):
        # Translate the zone rather than shrinking it, so this exercises the
        # frustum check instead of tripping the min_separation sanity check.
        raw['sampling_zone']['y'] = [-0.55, -0.25]

    task = load_task_config(_write_task(tmp_path, push_out_of_view))
    problems = validate_against_scene(task, scene)
    assert problems
    assert any('horizontal offset' in problem for problem in problems)


def test_min_separation_larger_than_the_zone_is_rejected(tmp_path):
    """A separation the zone cannot hold would make every draw fall back."""
    def shrink_zone(raw):
        raw['sampling_zone']['x'] = [0.23, 0.25]
        raw['sampling_zone']['y'] = [0.00, 0.04]

    with pytest.raises(ValueError, match='does not fit inside the sampling'):
        load_task_config(_write_task(tmp_path, shrink_zone))


def test_unknown_camera_is_rejected(tmp_path, scene):
    """Referring to a camera the USD does not declare must fail loudly."""
    def rename_camera(raw):
        raw['camera_visibility']['camera'] = 'nonexistent'

    task = load_task_config(_write_task(tmp_path, rename_camera))
    with pytest.raises(ValueError, match='not in the USD scene contract'):
        validate_against_scene(task, scene)


def test_gripper_commands_must_be_ordered(tmp_path):
    """A close command at or above the open command is a configuration error."""
    def invert_gripper(raw):
        raw['grasp']['gripper_close'] = 0.05

    with pytest.raises(ValueError, match='gripper_close'):
        load_task_config(_write_task(tmp_path, invert_gripper))


def test_placement_requires_multiple_stable_samples(tmp_path):
    """Legacy adjacent-frame configurations still reject one sample."""
    def use_one_sample(raw):
        raw['placement']['stable_samples'] = 1

    with pytest.raises(ValueError, match='stable_samples'):
        load_task_config(_write_task(tmp_path, use_one_sample))


def test_placement_stable_duration_must_fit_inside_timeout(tmp_path):
    """An impossible settling window must fail during configuration load."""
    def use_impossible_duration(raw):
        raw['placement']['stable_duration_sec'] = (
            raw['placement']['timeout_sec'] + 0.1
        )

    with pytest.raises(ValueError, match='stable_duration_sec'):
        load_task_config(_write_task(tmp_path, use_impossible_duration))


def test_placement_yaw_symmetry_order_must_be_integral(tmp_path):
    """Fractional symmetry orders do not describe a physical block."""
    def use_fractional_symmetry(raw):
        raw['placement']['yaw_symmetry_order'] = 4.5

    with pytest.raises(ValueError, match='yaw_symmetry_order'):
        load_task_config(_write_task(tmp_path, use_fractional_symmetry))


def test_placement_yaw_tolerance_cannot_cover_the_whole_period(tmp_path):
    """Tolerance above half a symmetry period would accept every yaw."""
    def use_unbounded_yaw(raw):
        raw['placement']['yaw_tolerance_deg'] = 46.0

    with pytest.raises(ValueError, match='yaw_tolerance_deg'):
        load_task_config(_write_task(tmp_path, use_unbounded_yaw))


def test_grasp_depth_must_be_inside_the_block(tmp_path):
    """A grasp deeper than the block would place the tool below it."""
    def deepen_grasp(raw):
        raw['grasp']['depth_below_top'] = 0.30

    with pytest.raises(ValueError, match='depth_below_top'):
        load_task_config(_write_task(tmp_path, deepen_grasp))


def test_schema_version_is_enforced(tmp_path):
    """Only schema_version 1 may load."""
    def bump_schema(raw):
        raw['schema_version'] = 2

    with pytest.raises(ValueError, match='schema_version 1'):
        load_task_config(_write_task(tmp_path, bump_schema))


def test_inverted_range_is_rejected(tmp_path):
    """A zone whose lower bound exceeds its upper bound must fail."""
    def invert_range(raw):
        raw['sampling_zone']['x'] = [0.30, 0.20]

    with pytest.raises(ValueError, match='lower bound'):
        load_task_config(_write_task(tmp_path, invert_range))


def test_generated_collision_scene_matches_the_configuration(task, scene):
    """The checked-in .scene file must match what the config renders."""
    rendered = render_collision_scene(task, scene)
    assert rendered == COLLISION_SCENE.read_text(encoding='utf-8')


def test_collision_scene_carries_ground_mount_and_platform(task, scene):
    """Planning runs with add_ground_plane=False, so all three must be listed."""
    rendered = render_collision_scene(task, scene)
    assert '* Ground' in rendered
    assert '* Mount' in rendered
    assert '* Platform' in rendered
    assert rendered.startswith('(noname)+\n')
    assert rendered.rstrip('\n').endswith('.')


def test_collision_boxes_are_expressed_in_base_link(task, scene):
    """Every box centre must equal the base-frame transform of its world pose."""
    lines = render_collision_scene(task, scene).splitlines()
    boxes = {
        'Ground': task.workcell.ground,
        'Mount': task.workcell.mount.collision_box,
        'Platform': task.workcell.platform.box,
    }
    for name, box in boxes.items():
        index = lines.index(f'* {name}')
        translation = [float(value) for value in lines[index + 1].split()]
        size = [float(value) for value in lines[index + 5].split()]
        expected = world_to_base(
            box.center,
            scene.expected_world_to_base_translation,
            scene.expected_world_to_base_rotation,
        )
        for axis in range(3):
            assert translation[axis] == pytest.approx(expected[axis], abs=1e-6)
            assert size[axis] == pytest.approx(box.size[axis])


def test_mount_collision_box_clears_the_robot_base(task):
    """A collision box reaching base_link would make every start state collide."""
    mount = task.workcell.mount
    assert mount.collision_box.top_z < mount.top_z
    assert mount.top_z == pytest.approx(0.35)
    assert mount.top_z - mount.collision_box.top_z >= 0.05


def test_configuration_is_not_mutated_by_validation(task, scene):
    """Validation must be a pure read of the configuration."""
    before = copy.deepcopy(task)
    validate_against_scene(task, scene)
    assert task == before
