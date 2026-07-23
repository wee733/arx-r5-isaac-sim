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

"""Tests for the cross-process ARX R5A simulation contract."""

from pathlib import Path
import runpy
from unittest import mock
import xml.etree.ElementTree as ET

from arx_r5_isaac_sim_bringup.contracts import (
    ALL_JOINTS,
    ARM_JOINTS,
    DEFAULT_JOINT_POSITIONS,
    DESCRIPTION_PACKAGE,
    GRIPPER_COMMAND_JOINT,
    GRIPPER_MIMIC_JOINT,
    INDEPENDENT_JOINTS,
    JOINT_COMMANDS_TOPIC,
    JOINT_POSITION_LIMITS,
    JOINT_STATES_TOPIC,
    resolve_description_mesh_uris,
)
from arx_r5_isaac_sim_bringup.simulation import (
    _direction_error_degrees,
    _parse_position_vector,
    _quaternion_error_degrees,
    materialize_isaac_urdf,
)

import pytest

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
XACRO_NAMESPACE = {'xacro': 'http://www.ros.org/wiki/xacro'}


def _yaml(relative_path):
    return yaml.safe_load((PACKAGE_ROOT / relative_path).read_text(encoding='utf-8'))


def test_local_authored_tabletop_config_is_never_installed(monkeypatch):
    """A private editor snapshot must stay outside package artifacts."""
    with mock.patch('setuptools.setup') as setup_mock:
        monkeypatch.chdir(PACKAGE_ROOT)
        runpy.run_path(str(PACKAGE_ROOT / 'setup.py'), run_name='__setup_test__')

    data_files = setup_mock.call_args.kwargs['data_files']
    installed_configs = next(
        sources for destination, sources in data_files
        if destination.endswith('/config')
    )
    assert 'config/my_tabletop.yaml' not in installed_configs
    assert 'config/authored_usd_apriltag_demo.yaml' in installed_configs

    ignore_lines = {
        line.strip()
        for line in (PACKAGE_ROOT.parent / '.gitignore').read_text(
            encoding='utf-8'
        ).splitlines()
    }
    assert (
        '/arx_r5_isaac_sim_bringup/config/my_tabletop.yaml'
        in ignore_lines
    )


def test_ros2_control_uses_the_isaac_topic_bridge():
    """The sim xacro must use NVIDIA's expected topic pair."""
    root = ET.parse(
        PACKAGE_ROOT / 'urdf' / 'r5a.isaac_sim.ros2_control.xacro'
    ).getroot()
    plugin = root.find('.//hardware/plugin')
    parameters = {
        element.attrib['name']: element.text
        for element in root.findall('.//hardware/param')
    }

    assert plugin is not None
    assert plugin.text == 'topic_based_ros2_control/TopicBasedSystem'
    assert parameters['joint_commands_topic'] == f'/{JOINT_COMMANDS_TOPIC}'
    assert parameters['joint_states_topic'] == f'/{JOINT_STATES_TOPIC}'


def test_ros2_control_publishes_eight_position_commands_with_a_safe_mimic():
    """The transport must mirror joint7 without mismatched message arrays."""
    root = ET.parse(
        PACKAGE_ROOT / 'urdf' / 'r5a.isaac_sim.ros2_control.xacro'
    ).getroot()
    joints = root.findall('.//ros2_control/joint')
    assert tuple(joint.attrib['name'] for joint in joints) == ALL_JOINTS

    command_interfaces = {
        joint.attrib['name']: [
            interface.attrib['name']
            for interface in joint.findall('command_interface')
        ]
        for joint in joints
    }
    assert all(
        command_interfaces[name] == ['position'] for name in ALL_JOINTS
    )
    assert len(command_interfaces) == len(ALL_JOINTS) == 8

    mimic_joint = next(
        joint for joint in joints
        if joint.attrib['name'] == GRIPPER_MIMIC_JOINT
    )
    assert mimic_joint.attrib['mimic'] == 'false'
    mimic_parameters = {
        parameter.attrib['name']: parameter.text
        for parameter in mimic_joint.findall('param')
    }
    assert mimic_parameters['mimic'] == GRIPPER_COMMAND_JOINT
    assert float(mimic_parameters['multiplier']) == pytest.approx(1.0)
    assert [
        interface.attrib['name']
        for interface in mimic_joint.findall('state_interface')
    ] == ['position', 'velocity']


def test_ros2_control_position_limits_match_the_cumotion_urdf():
    """The simulator contract must publish the same hard bounds as PhysX."""
    root = ET.parse(
        PACKAGE_ROOT / 'urdf' / 'r5a.isaac_sim.ros2_control.xacro'
    ).getroot()
    joints = {
        joint.attrib['name']: joint for joint in root.findall('.//ros2_control/joint')
    }

    for joint_name in ALL_JOINTS:
        command_interface = joints[joint_name].find(
            './command_interface[@name="position"]'
        )
        limits = {
            parameter.attrib['name']: float(parameter.text)
            for parameter in command_interface.findall('param')
        }
        assert (limits['min'], limits['max']) == JOINT_POSITION_LIMITS[joint_name]


def test_controller_and_moveit_arm_joint_order_match():
    """The two controller layers must interpolate the same six arm joints."""
    ros2_control = _yaml('config/ros2_controllers.yaml')
    moveit = _yaml('config/moveit_controllers.yaml')

    assert ros2_control['controller_manager']['ros__parameters'][
        'enforce_command_limits'
    ] is False
    assert tuple(
        ros2_control['manipulator_controller']['ros__parameters']['joints']
    ) == ARM_JOINTS
    assert tuple(
        moveit['moveit_simple_controller_manager']['manipulator_controller'][
            'joints'
        ]
    ) == ARM_JOINTS
    gripper_joint = ros2_control['gripper_controller']['ros__parameters'][
        'joint'
    ]
    assert gripper_joint == GRIPPER_COMMAND_JOINT
    assert ARM_JOINTS + (gripper_joint,) == INDEPENDENT_JOINTS
    assert GRIPPER_MIMIC_JOINT not in ARM_JOINTS + (gripper_joint,)


def test_initial_positions_cover_every_articulation_joint():
    """Startup state must cover all DOFs and keep the mimic pair equal."""
    initial_positions = _yaml('config/initial_positions.yaml')['initial_positions']
    assert tuple(initial_positions) == ALL_JOINTS
    assert initial_positions == DEFAULT_JOINT_POSITIONS
    assert initial_positions['joint8'] == initial_positions['joint7']


def test_wrapper_imports_velocity_limited_cumotion_urdf():
    """The sim wrapper must avoid the stale repository-root URDF."""
    root = ET.parse(
        PACKAGE_ROOT / 'urdf' / 'r5a.isaac_sim.urdf.xacro'
    ).getroot()
    includes = [
        element.attrib['filename']
        for element in root.findall('xacro:include', XACRO_NAMESPACE)
    ]
    assert any(path.endswith('/urdf/r5a_cumotion.urdf') for path in includes)
    assert not any(path.endswith('/R5a.urdf') for path in includes)


def test_mesh_uri_resolution_is_package_scoped(tmp_path):
    """Only ARX description URIs are rewritten for the importer."""
    description_share = tmp_path / DESCRIPTION_PACKAGE
    description_share.mkdir()
    text = (
        '<mesh filename="package://'
        f'{DESCRIPTION_PACKAGE}/meshes/visual/link1.STL"/>'
    )
    resolved = resolve_description_mesh_uris(text, description_share)
    assert 'package://' not in resolved
    assert str(description_share / 'meshes' / 'visual' / 'link1.STL') in resolved


def test_materialized_urdf_is_importer_safe(tmp_path):
    """The generated importer input must not retain package mesh URIs."""
    description_share = tmp_path / DESCRIPTION_PACKAGE
    description_share.mkdir()
    source = tmp_path / 'source.urdf'
    destination = tmp_path / 'resolved.urdf'
    source.write_text(
        '<robot name="R5a"><link name="base_link"><visual><geometry>'
        f'<mesh filename="package://{DESCRIPTION_PACKAGE}/meshes/base.STL"/>'
        '</geometry></visual></link></robot>',
        encoding='utf-8',
    )

    materialize_isaac_urdf(source, description_share, destination)
    resolved = destination.read_text(encoding='utf-8')
    assert f'{description_share}/meshes/base.STL' in resolved


def test_mimic_startup_positions_cannot_diverge():
    """The two gripper fingers cannot start with conflicting positions."""
    with pytest.raises(ValueError, match='joint8 must equal'):
        _parse_position_vector('0,1,1.5,0,0,0,0.044,0.0')


def test_startup_positions_must_respect_urdf_limits():
    """Reject startup targets that the imported articulation would clamp."""
    with pytest.raises(ValueError, match='exceed URDF limits'):
        _parse_position_vector('0,1,1.5,0,0,0,0.05,0.05')


def test_authored_scene_rotation_checks_handle_direction_and_quaternion_sign():
    """Scene validation must measure pitch and treat q/-q as equivalent."""
    assert _direction_error_degrees(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ) == pytest.approx(90.0)
    assert _quaternion_error_degrees(
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, -1.0),
    ) == pytest.approx(0.0)
