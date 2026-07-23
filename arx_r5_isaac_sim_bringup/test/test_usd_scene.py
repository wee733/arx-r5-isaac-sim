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

"""Contract tests for the authored dual-camera USD workcell."""

from math import atan2, degrees
from pathlib import Path

from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / 'config' / 'arx_sim_usd_scene.yaml'


def test_authored_scene_has_two_unambiguous_camera_profiles():
    """ZED X and D455 must not share topics or implicit frame IDs."""
    config = load_usd_scene_config(CONFIG_PATH)

    assert tuple(config.cameras) == ('zedx', 'd455')
    assert config.cameras['zedx'].parent_frame == 'base_link'
    assert config.cameras['d455'].parent_frame == 'link6'
    assert config.cameras['zedx'].optical_frame == (
        'zed_x_left_camera_optical_frame'
    )
    assert config.cameras['d455'].optical_frame == 'd455_color_optical_frame'
    assert config.cameras['zedx'].color_image_topic.startswith('/zed_x/')
    assert config.cameras['d455'].color_image_topic.startswith('/d455/')
    assert (config.cameras['zedx'].width, config.cameras['zedx'].height) == (
        1920, 1200
    )


def test_zed_pitch_belongs_to_the_fixed_camera_not_the_robot_root():
    """The intended 10-degree look-down must not tilt the articulation."""
    config = load_usd_scene_config(CONFIG_PATH)
    zed = config.cameras['zedx']

    assert config.expected_world_to_base_rotation == pytest.approx(
        (0.0, 0.0, 0.0, 1.0)
    )
    pitch_degrees = degrees(atan2(
        -zed.expected_optical_forward[2],
        zed.expected_optical_forward[0],
    ))
    assert pitch_degrees == pytest.approx(10.0, abs=1e-6)
    assert zed.expected_parent_to_optical_translation == pytest.approx(
        (0.2896500463, 0.06, -0.0561502512)
    )


def test_camera_projection_is_normalized_without_changing_horizontal_fov():
    """The rendered aspect ratio and published pinhole intrinsics must agree."""
    simulation_path = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    )
    source = simulation_path.read_text(encoding='utf-8')

    assert 'vertical_aperture = horizontal_aperture * camera.height / camera.width' in (
        source
    )
    assert '_normalize_camera_projection(stage, camera)' in source


def test_reachable_layout_preserves_the_fixed_zed_world_pose():
    """Moving the fixed base closer must not move or unpitch the external camera."""
    config = load_usd_scene_config(CONFIG_PATH)
    layout = config.layouts['reachable']

    assert tuple(config.layouts) == ('as-authored', 'reachable')
    assert config.layouts['as-authored'].prim_translations == {}
    assert layout.prim_translations['/R5a'] == pytest.approx(
        (0.0, 0.0, 0.4)
    )
    assert layout.prim_translations['/R5a/base_link/ZED_X'] == pytest.approx(
        (-0.2760323015, 0.0, -0.0683176448)
    )

    # The source USD base is at x=-0.5483055088. Its original camera offset
    # plus that base translation equals the reachable session-layer offset.
    authored_world_translation = (
        -0.5483055088
        + config.cameras['zedx'].expected_parent_to_optical_translation[0],
        config.cameras['zedx'].expected_parent_to_optical_translation[1],
        0.4
        + config.cameras['zedx'].expected_parent_to_optical_translation[2],
    )
    reachable_world_translation = (
        layout.camera_parent_to_optical_translations['zedx'][0],
        layout.camera_parent_to_optical_translations['zedx'][1],
        0.4 + layout.camera_parent_to_optical_translations['zedx'][2],
    )
    assert reachable_world_translation == pytest.approx(
        authored_world_translation,
        abs=1e-9,
    )


def test_d455_contract_is_rigidly_attached_to_link6():
    """Only the upstream base-to-link6 edge may vary for eye-in-hand."""
    config = load_usd_scene_config(CONFIG_PATH)
    d455 = config.cameras['d455']

    assert d455.parent_frame == 'link6'
    assert d455.expected_parent_to_optical_translation == pytest.approx(
        (0.1044276157, -0.0115000179, 0.0767273606)
    )
    assert d455.expected_optical_forward == pytest.approx(
        (0.8660253514, 0.0, -0.5000000908)
    )


def test_authored_scene_disables_unjointed_sensor_rigid_bodies():
    """Nested sensor payloads must follow their articulation parent transforms."""
    config = load_usd_scene_config(CONFIG_PATH)

    assert config.sensor_rigid_body_paths == (
        '/R5a/base_link/ZED_X',
        '/R5a/link6/rsd455/RSD455',
    )


def test_authored_source_object_has_runtime_physics_contract():
    """The visual red block must become a collidable graspable rigid body."""
    config = load_usd_scene_config(CONFIG_PATH)

    assert config.source_object_prim_path == '/World/Workspace/TaggedCube'
    assert config.source_collision_prim_path.endswith('/TaggedCube/Body')
    assert config.grasp_body_prim_path == '/R5a/link6'
    assert config.grasp_frame_prim_path == '/R5a/link6/grasp_frame'
    assert config.source_object_mass_kg == pytest.approx(0.05)


def test_authored_source_object_stays_upright_until_physical_grasp():
    """The slender source block must not tip before perception stabilizes."""
    simulation_path = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    )
    source = simulation_path.read_text(encoding='utf-8')
    configure_start = source.index('def _configure_authored_source_object(')
    attachment_start = source.index('class AuthoredObjectAttachmentController:')
    configure_source = source[configure_start:attachment_start]
    create_joint_start = source.index('    def _create_joint(self) -> None:')
    remove_joint_start = source.index('    def _remove_joint(self) -> None:')
    create_joint_source = source[create_joint_start:remove_joint_start]

    assert 'CreateKinematicEnabledAttr(True)' in configure_source
    assert 'CreateEnableCCDAttr(False)' in configure_source
    dynamics_index = create_joint_source.index(
        'CreateKinematicEnabledAttr(False)'
    )
    ccd_index = create_joint_source.index('CreateEnableCCDAttr(True)')
    joint_index = create_joint_source.index('UsdPhysics.FixedJoint.Define(')
    assert dynamics_index < ccd_index < joint_index


def test_authored_gripper_collision_override_precedes_physx_parsing():
    """Finger concavities must survive instancing before PhysX sees them."""
    simulation_path = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    )
    source = simulation_path.read_text(encoding='utf-8')
    configure_start = source.index(
        'def _configure_authored_gripper_collisions(stage) -> None:'
    )
    validate_start = source.index(
        'def _validate_authored_gripper_collision_instances(stage) -> None:'
    )
    workspace_start = source.index(
        'def _configure_authored_workspace_contacts(stage) -> None:'
    )
    configure_source = source[configure_start:validate_start]
    validate_source = source[validate_start:workspace_start]
    attach_start = source.index('def _attach_prepared_authored_stage(')
    attachment_start = source.index('class AuthoredObjectAttachmentController:')
    attach_source = source[attach_start:attachment_start]

    for link_name in ('link7', 'link8'):
        assert (
            f"'/colliders/{link_name}/{link_name}/node_STL_BINARY_'"
            in source
        )
        assert (
            f"'/R5a/{link_name}/collisions/{link_name}/node_STL_BINARY_'"
            in source
        )
    assert 'UsdPhysics.Tokens.convexDecomposition' in configure_source
    assert 'CollisionAPI.Apply' not in configure_source
    assert 'collision_prim.IsInstanceProxy()' in validate_source
    configure_call = attach_source.index(
        '_configure_authored_gripper_collisions(stage)'
    )
    attach_call = attach_source.index('context.attach_stage_async(stage)')
    validate_call = attach_source.index(
        '_validate_authored_gripper_collision_instances(attached_stage)'
    )
    assert configure_call < attach_call < validate_call


def test_authored_workspace_contact_offsets_only_touch_existing_colliders():
    """Tags stay visual while table collision envelopes remain millimetric."""
    simulation_path = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    )
    source = simulation_path.read_text(encoding='utf-8')
    workspace_start = source.index(
        'def _configure_authored_workspace_contacts(stage) -> None:'
    )
    texture_start = source.index('def _repair_authored_tag_textures(')
    workspace_source = source[workspace_start:texture_start]
    attach_start = source.index('def _attach_prepared_authored_stage(')
    attachment_start = source.index('class AuthoredObjectAttachmentController:')
    attach_source = source[attach_start:attachment_start]

    assert "AUTHORED_WORKSPACE_PRIM_PATH = '/World/Workspace'" in source
    assert 'prim.HasAPI(UsdPhysics.CollisionAPI)' in workspace_source
    assert 'UsdPhysics.CollisionAPI.Apply' not in workspace_source
    assert 'CreateContactOffsetAttr(0.002)' in workspace_source
    assert 'CreateRestOffsetAttr(0.0)' in workspace_source
    normalize_call = attach_source.index(
        '_configure_authored_workspace_contacts(stage)'
    )
    attach_call = attach_source.index('context.attach_stage_async(stage)')
    assert normalize_call < attach_call


def test_authored_layout_is_session_only_and_cannot_overwrite_source_usd():
    """Runtime reachability changes must never become accidental asset edits."""
    simulation_path = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    )
    source = simulation_path.read_text(encoding='utf-8')

    assert 'stage.GetEditTarget().GetLayer() != stage.GetSessionLayer()' in source
    assert '--save-usd must not overwrite the selected authored USD' in source


def test_duplicate_camera_topics_are_rejected(tmp_path):
    """Two profiles may never silently publish the same ROS image stream."""
    source = CONFIG_PATH.read_text(encoding='utf-8')
    invalid = source.replace(
        '/d455/color/image_raw',
        '/zed_x/left/image_raw',
    )
    path = tmp_path / 'invalid.yaml'
    path.write_text(invalid, encoding='utf-8')

    with pytest.raises(ValueError, match='duplicate camera topics'):
        load_usd_scene_config(path)
