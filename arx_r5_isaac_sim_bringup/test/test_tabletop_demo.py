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

"""Contract tests for the tabletop AprilTag pick-and-place demo."""

from math import sqrt
from pathlib import Path
import struct

from arx_r5_isaac_sim_bringup.demo_config import (
    load_demo_config,
    validate_demo_dependency_configs,
)
from arx_r5_isaac_sim_bringup.pose_math import (
    compose_pose,
)

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG_PATH = PACKAGE_ROOT / 'config' / 'tabletop_apriltag_demo.yaml'
AUTHORED_DEMO_CONFIG_PATH = (
    PACKAGE_ROOT / 'config' / 'authored_usd_apriltag_demo.yaml'
)
FRONT_DEMO_CONFIG_PATH = (
    PACKAGE_ROOT / 'config' / 'front_demo_apriltag_demo.yaml'
)
CALIBRATION_PATH = PACKAGE_ROOT / 'config' / 'tabletop_camera_extrinsics.yaml'
GRASP_CONFIG_PATH = PACKAGE_ROOT / 'config' / 'tabletop_tagged_cube_grasps.yaml'
BLACKBOARD_PATH = PACKAGE_ROOT / 'config' / 'tabletop_apriltag_blackboard.yaml'
BEHAVIOR_TREE_PATH = (
    PACKAGE_ROOT / 'config' / 'tabletop_apriltag_behavior_tree.yaml'
)
CONTROLLERS_PATH = PACKAGE_ROOT / 'config' / 'ros2_controllers.yaml'


def _write_authored_demo_config(tmp_path, source_zone):
    raw = yaml.safe_load(
        AUTHORED_DEMO_CONFIG_PATH.read_text(encoding='utf-8')
    )
    for field in ('source_object', 'drop_target'):
        raw[field]['tag_texture'] = str(
            (AUTHORED_DEMO_CONFIG_PATH.parent / raw[field]['tag_texture'])
            .resolve()
        )
    raw['source_zone'] = source_zone
    path = tmp_path / 'demo.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _normalize(vector):
    magnitude = sqrt(sum(value * value for value in vector))
    return tuple(value / magnitude for value in vector)


def test_tabletop_geometry_and_tag_contract():
    """The source and destination tags must describe one reachable workcell."""
    config = load_demo_config(DEMO_CONFIG_PATH)
    table_surface = config.table.top_center[2] + config.table.top_size[2] / 2.0

    assert table_surface == pytest.approx(0.0)
    assert config.source_object.tag_id == 0
    assert config.drop_target.tag_id == 1
    assert config.source_object.tag_size == config.drop_target.tag_size
    assert config.source_object.center[2] == pytest.approx(
        table_surface + config.source_object.size[2] / 2.0
    )
    assert config.source_zone.enabled is False
    assert config.source_zone.frame == ''


def test_authored_source_zone_contains_pick_but_not_drop_workstation():
    """The continuous upstream tree must not rediscover a placed block."""
    config = load_demo_config(
        AUTHORED_DEMO_CONFIG_PATH
    )
    zone = config.source_zone
    source_in_base = (0.8918535, -0.180, -0.1727558)
    drop_in_base = (0.6976606, 0.1961878, -0.2826465)

    assert zone.enabled is True
    assert zone.frame == 'base_link'
    assert all(
        zone.minimum[index] <= source_in_base[index] <= zone.maximum[index]
        for index in range(3)
    )
    assert not all(
        zone.minimum[index] <= drop_in_base[index] <= zone.maximum[index]
        for index in range(3)
    )


def test_front_demo_source_zone_matches_the_translated_robot_root():
    """The opt-in root translation needs its own base-frame discovery zone."""
    config = load_demo_config(FRONT_DEMO_CONFIG_PATH)
    zone = config.source_zone
    source_in_base = (0.3518780, -0.180, -0.2679681)
    drop_in_base = (0.1576851, 0.1961878, -0.3778588)

    assert zone.enabled is True
    assert zone.frame == 'base_link'
    assert zone.minimum == pytest.approx((0.27, -0.30, -0.34))
    assert zone.maximum == pytest.approx((0.44, -0.08, -0.19))
    assert all(
        zone.minimum[index] <= source_in_base[index] <= zone.maximum[index]
        for index in range(3)
    )
    assert not all(
        zone.minimum[index] <= drop_in_base[index] <= zone.maximum[index]
        for index in range(3)
    )


@pytest.mark.parametrize('invalid_source_zone', (None, []))
def test_source_zone_must_be_a_mapping(tmp_path, invalid_source_zone):
    """Reject YAML null and sequence values with a schema-level error."""
    path = _write_authored_demo_config(tmp_path, invalid_source_zone)

    with pytest.raises(ValueError, match='source_zone must be a mapping'):
        load_demo_config(path)


def test_enabled_source_zone_requires_an_explicit_frame_and_bounds(tmp_path):
    """Enabled filtering must never guess its coordinate-frame contract."""
    bounds = {
        'min_xyz': [0.20, -0.30, -0.39],
        'max_xyz': [0.40, -0.08, -0.25],
    }
    path = _write_authored_demo_config(
        tmp_path,
        {'enabled': True, **bounds},
    )
    with pytest.raises(ValueError, match='source_zone.frame must be non-empty'):
        load_demo_config(path)

    path = _write_authored_demo_config(
        tmp_path,
        {'enabled': True, 'frame': 'base_link'},
    )
    with pytest.raises(ValueError, match='source_zone.min_xyz is required'):
        load_demo_config(path)


def test_apriltag_assets_have_the_expected_resolution():
    """Both committed marker textures must be complete 800-pixel PNG files."""
    config = load_demo_config(DEMO_CONFIG_PATH)
    for texture in (
        config.source_object.tag_texture,
        config.drop_target.tag_texture,
    ):
        data = texture.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II', data[16:24]) == (800, 800)


def test_camera_optical_frame_looks_at_the_workspace():
    """ROS optical +Z must point from the camera to its configured target."""
    config = load_demo_config(DEMO_CONFIG_PATH)
    rotated_forward, _ = compose_pose(
        (0.0, 0.0, 0.0),
        config.camera.optical_rotation,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    expected_forward = _normalize(tuple(
        config.camera.look_at[index] - config.camera.position[index]
        for index in range(3)
    ))
    assert rotated_forward == pytest.approx(expected_forward)


def test_camera_calibration_matches_the_rendered_camera():
    """The static TF and USD camera must share one source of geometric truth."""
    config = load_demo_config(DEMO_CONFIG_PATH)
    calibration = yaml.safe_load(CALIBRATION_PATH.read_text(encoding='utf-8'))
    transform = calibration['base_to_camera']

    assert calibration['publish'] is True
    assert transform['parent_frame'] == 'base_link'
    assert transform['child_frame'] == config.camera.optical_frame
    assert transform['translation'] == pytest.approx(config.camera.position)
    assert transform['rotation'] == pytest.approx(config.camera.optical_rotation)


def test_external_tag_and_camera_configs_match_the_scene(tmp_path):
    """Launch-time validation must prevent silent configuration divergence."""
    config = load_demo_config(DEMO_CONFIG_PATH)
    tag_config = {
        'schema_version': 1,
        'tag_family': config.tag_family,
        'tag_size': config.source_object.tag_size,
        'objects': {
            config.source_object.tag_id: {
                'class_id': 'tagged_cube',
                'dimensions': list(config.source_object.size),
                'tag_to_object': {
                    'translation': [
                        0.0,
                        0.0,
                        -config.source_object.size[2] / 2.0,
                    ],
                    'rotation': [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }
    camera_calibration = {
        'schema_version': 1,
        'publish': True,
        'base_to_camera': {
            'parent_frame': 'base_link',
            'child_frame': config.camera.optical_frame,
            'translation': list(config.camera.position),
            'rotation': list(config.camera.optical_rotation),
        },
    }
    tag_path = tmp_path / 'tag.yaml'
    camera_path = tmp_path / 'camera.yaml'
    tag_path.write_text(yaml.safe_dump(tag_config), encoding='utf-8')
    camera_path.write_text(
        yaml.safe_dump(camera_calibration),
        encoding='utf-8',
    )

    validate_demo_dependency_configs(config, tag_path, camera_path)
    tag_config['tag_size'] = config.source_object.tag_size * 2.0
    tag_path.write_text(yaml.safe_dump(tag_config), encoding='utf-8')
    with pytest.raises(ValueError, match='tag_size'):
        validate_demo_dependency_configs(config, tag_path, camera_path)


def test_destination_tag_maps_to_the_link6_drop_pose():
    """The full target-tag pose must produce the configured link6 goal."""
    config = load_demo_config(DEMO_CONFIG_PATH)
    translation, rotation = compose_pose(
        config.drop_target.center,
        (0.0, 0.0, 0.0, 1.0),
        config.drop_target.link6_offset_in_tag,
        config.drop_target.link6_rotation_in_tag,
    )

    assert translation == pytest.approx((0.30, 0.18, 0.1655))
    assert rotation == pytest.approx(
        (0.0, 2 ** -0.5, 0.0, 2 ** -0.5)
    )


def test_tabletop_grasp_is_a_single_reachable_top_down_seed():
    """The simulation should not offer the unreachable horizontal grasp seeds."""
    grasp_config = yaml.safe_load(GRASP_CONFIG_PATH.read_text(encoding='utf-8'))
    grasps = grasp_config['grasps']

    assert tuple(grasps) == ('grasp_top',)
    assert grasps['grasp_top']['position'] == pytest.approx([0.0, 0.0, 0.14])
    assert grasps['grasp_top']['orientation']['xyz'] == pytest.approx(
        [0.0, 2 ** -0.5, 0.0]
    )
    assert grasps['grasp_top']['orientation']['w'] == pytest.approx(2 ** -0.5)


def test_tabletop_blackboard_selects_the_simulation_grasp_file():
    """Only simulation input data should differ from the ARX behavior tree."""
    blackboard = yaml.safe_load(BLACKBOARD_PATH.read_text(encoding='utf-8'))
    tagged_cube = blackboard['blackboard_params']['supported_objects'][
        'tagged_cube'
    ]

    assert 'arx_r5_isaac_sim_bringup' in tagged_cube['grasp_file_path']
    assert tagged_cube['grasp_file_path'].endswith(
        '/config/tabletop_tagged_cube_grasps.yaml'
    )


def test_tabletop_behavior_tree_uses_reachable_clearance_offsets():
    """The simulation overrides data, not NVIDIA's behavior-tree code."""
    behavior_tree = yaml.safe_load(
        BEHAVIOR_TREE_PATH.read_text(encoding='utf-8')
    )
    plan_to_grasp = behavior_tree['behavior_tree_params'][
        'multi_object_pick_and_place'
    ]['plan_to_grasp']

    assert plan_to_grasp['grasp_approach_offset_distance'] == pytest.approx(
        [-0.065, 0.0, 0.0]
    )
    assert plan_to_grasp['retract_offset_distance'] == pytest.approx(
        [0.0, 0.0, 0.065]
    )


def test_sim_controller_does_not_relimit_cumotion_trajectory():
    """Delayed simulator feedback must not distort time-parameterized commands."""
    controllers = yaml.safe_load(CONTROLLERS_PATH.read_text(encoding='utf-8'))

    assert controllers['controller_manager']['ros__parameters'][
        'enforce_command_limits'
    ] is False


def test_aggregate_launch_does_not_duplicate_robot_or_motion_stack():
    """The demo must reuse the simulation stack and start only perception adapters."""
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'arx_r5a_apriltag_demo.launch.py'
    ).read_text(encoding='utf-8')

    assert "'start_camera': 'False'" in launch_source
    assert "'start_robot': 'False'" in launch_source
    assert "'start_motion_stack': 'False'" in launch_source
    assert "'start_orchestrator': 'False'" in launch_source
    assert "'start_apriltag_object_server': 'False'" in launch_source
    assert "'tag_detections_topic': tag_detections_topic" in launch_source
    assert "executable='apriltag_detection_retimer'" not in launch_source
    assert '/arx_r5_demo/tag_detections_raw' not in launch_source
    assert 'validate_demo_dependency_configs(' in launch_source
    assert "executable='apriltag_object_server'" in launch_source
    assert "'isaac_ros_cumotion_object_attachment'" in launch_source
    assert "executable='sim_pick_place_orchestrator'" in launch_source
    assert "'tabletop_apriltag_behavior_tree.yaml'" in launch_source
    assert "'tabletop_apriltag_blackboard.yaml'" in launch_source


def test_sim_orchestrator_accepts_ros_launch_arguments():
    """The wrapper must discard ROS remap arguments before parsing its own CLI."""
    source = (
        PACKAGE_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'sim_orchestrator.py'
    ).read_text(encoding='utf-8')

    assert 'remove_ros_args(args=sys.argv)[1:]' in source


def test_graph_check_requires_official_apriltag_cuda_output():
    """Runtime acceptance must sample detections and confirm the CUDA backend."""
    source = (PACKAGE_ROOT.parent / 'scripts' / 'check_ros_graph.sh').read_text(
        encoding='utf-8'
    )

    assert 'probe_topic /tag_detections' in source
    assert 'ros2 param get /camera_1/apriltag backends' in source
    assert 'AprilTag backend: CUDA (Isaac ROS cuAprilTag)' in source


def test_simulation_graph_publishes_camera_and_preserves_grasp_frame():
    """Camera topics and fixed-frame import safeguards must remain enabled."""
    source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    ).read_text(encoding='utf-8')

    assert 'import_config.merge_fixed_joints = True' in source
    assert 'ROS2CameraHelper' in source
    assert 'ROS2CameraInfoHelper' in source
    assert 'IsaacCreateRenderProduct' in source
    assert '_validate_grasp_frame_import(stage, robot_path)' in source

    scene_source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'tabletop_scene.py'
    ).read_text(encoding='utf-8')
    assert 'self._relative_position' in scene_source
    assert 'multiply_quaternions' in scene_source
