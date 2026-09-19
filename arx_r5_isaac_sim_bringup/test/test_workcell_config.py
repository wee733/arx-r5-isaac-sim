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

"""Contract tests for the authored ARX manipulation workcell."""

from pathlib import Path
import struct

from arx_r5_isaac_sim_bringup.demo_config import load_demo_config
from arx_r5_isaac_sim_bringup.pose_math import compose_pose

import pytest

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AUTHORED_CONFIG_PATH = (
    PACKAGE_ROOT / 'config' / 'authored_usd_apriltag_demo.yaml'
)
CONTROLLERS_PATH = PACKAGE_ROOT / 'config' / 'ros2_controllers.yaml'


def _write_authored_config(tmp_path, source_zone):
    raw = yaml.safe_load(AUTHORED_CONFIG_PATH.read_text(encoding='utf-8'))
    for field in ('source_object', 'drop_target'):
        raw[field]['tag_texture'] = str(
            (AUTHORED_CONFIG_PATH.parent / raw[field]['tag_texture']).resolve()
        )
    raw['source_zone'] = source_zone
    path = tmp_path / 'demo.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def test_authored_source_zone_contains_pick_but_not_drop_workstation():
    """Continuous discovery must not rediscover a placed block."""
    config = load_demo_config(AUTHORED_CONFIG_PATH)
    zone = config.source_zone
    source_in_base = (0.7405916395, -0.180, -0.1689645628)
    drop_in_base = (0.5463987272, 0.1961877804, -0.2788552953)

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


@pytest.mark.parametrize('invalid_source_zone', (None, []))
def test_source_zone_must_be_a_mapping(tmp_path, invalid_source_zone):
    """Reject YAML null and sequence values with a schema-level error."""
    path = _write_authored_config(tmp_path, invalid_source_zone)

    with pytest.raises(ValueError, match='source_zone must be a mapping'):
        load_demo_config(path)


def test_enabled_source_zone_requires_frame_and_bounds(tmp_path):
    """Enabled filtering must never guess its coordinate frame."""
    bounds = {
        'min_xyz': [0.20, -0.30, -0.39],
        'max_xyz': [0.40, -0.08, -0.25],
    }
    path = _write_authored_config(
        tmp_path,
        {'enabled': True, **bounds},
    )
    with pytest.raises(
        ValueError,
        match='source_zone.frame must be non-empty',
    ):
        load_demo_config(path)

    path = _write_authored_config(
        tmp_path,
        {'enabled': True, 'frame': 'base_link'},
    )
    with pytest.raises(
        ValueError,
        match='source_zone.min_xyz is required',
    ):
        load_demo_config(path)


def test_apriltag_assets_have_the_expected_resolution():
    """Both marker textures must be complete 800-pixel PNG files."""
    config = load_demo_config(AUTHORED_CONFIG_PATH)
    for texture in (
        config.source_object.tag_texture,
        config.drop_target.tag_texture,
    ):
        data = texture.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II', data[16:24]) == (800, 800)


def test_destination_tag_maps_to_the_configured_link6_offset():
    """The target tag must retain the configured tag-to-link6 transform."""
    config = load_demo_config(AUTHORED_CONFIG_PATH)
    translation, rotation = compose_pose(
        config.drop_target.center,
        (0.0, 0.0, 0.0, 1.0),
        config.drop_target.link6_offset_in_tag,
        config.drop_target.link6_rotation_in_tag,
    )

    assert translation == pytest.approx((0.089674989, 0.196187780, -0.3145))
    assert rotation == pytest.approx(
        (0.0, -(2 ** -0.5), 0.0, 2 ** -0.5)
    )


def test_workcell_configs_do_not_define_a_generated_camera():
    """Camera geometry belongs only to the authored USD sensor contract."""
    raw = yaml.safe_load(AUTHORED_CONFIG_PATH.read_text(encoding='utf-8'))
    assert 'camera' not in raw


def test_legacy_generated_camera_files_are_removed():
    """The obsolete Camera_1 workflow must not return accidentally."""
    removed_paths = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'tabletop_scene.py',
        PACKAGE_ROOT / 'launch' / 'arx_r5a_apriltag_demo.launch.py',
        PACKAGE_ROOT / 'config' / 'tabletop_camera_extrinsics.yaml',
        PACKAGE_ROOT / 'config' / 'tabletop_apriltag_demo.yaml',
        PACKAGE_ROOT / 'config' / 'tabletop_apriltag_behavior_tree.yaml',
        PACKAGE_ROOT / 'config' / 'tabletop_apriltag_blackboard.yaml',
        PACKAGE_ROOT / 'config' / 'tabletop_tagged_cube_grasps.yaml',
        PACKAGE_ROOT / 'config' / 'front_demo_apriltag_demo.yaml',
        PACKAGE_ROOT / 'config' / 'front_demo_table.scene',
        PACKAGE_ROOT / 'launch' / 'arx_r5a_zedx_front_demo.launch.py',
    )

    assert all(not path.exists() for path in removed_paths)


def test_sim_controller_does_not_relimit_cumotion_trajectory():
    """Simulator feedback must not distort time-parameterized commands."""
    controllers = yaml.safe_load(CONTROLLERS_PATH.read_text(encoding='utf-8'))

    assert controllers['controller_manager']['ros__parameters'][
        'enforce_command_limits'
    ] is False


def test_sim_orchestrator_accepts_ros_launch_arguments():
    """The wrapper must discard ROS remap arguments before parsing its CLI."""
    source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'sim_orchestrator.py'
    ).read_text(encoding='utf-8')

    assert 'remove_ros_args(args=sys.argv)[1:]' in source


def test_simulation_publishes_only_authored_camera_streams():
    """Camera helpers remain available without generating Camera_1."""
    source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    ).read_text(encoding='utf-8')

    assert 'ROS2CameraHelper' in source
    assert 'ROS2CameraInfoHelper' in source
    assert 'IsaacCreateRenderProduct' in source
    assert '_validate_grasp_frame_import(stage, robot_path)' in source
    assert 'create_tabletop_scene' not in source
    assert '_apply_authored_layout' not in source
    assert '--authored-layout' not in source
    assert "'camera_1'" not in source
    assert '/World/Sensors/Camera_1' not in source
