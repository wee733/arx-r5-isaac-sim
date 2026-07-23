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

"""Launch the ARX tabletop AprilTag pick-and-place demo."""

import os

from ament_index_python.packages import get_package_share_directory
from arx_r5_isaac_sim_bringup.demo_config import (
    load_demo_config,
    validate_demo_dependency_configs,
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = 'arx_r5_isaac_sim_bringup'
MANIPULATION_BRINGUP = 'isaac_ros_manipulation_arx_r5a_bringup'
MANIPULATOR_CONTAINER = 'manipulator_container'


def _value(context, name: str) -> str:
    return context.perform_substitution(LaunchConfiguration(name))


def _bool(context, name: str) -> bool:
    value = _value(context, name).strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Launch argument {name} must be boolean, got {value!r}')


def _include(package_name: str, relative_path: str, arguments=None):
    launch_path = os.path.join(
        get_package_share_directory(package_name),
        relative_path,
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments=(arguments or {}).items(),
    )


def launch_setup(context, *args, **kwargs):
    """Compose simulation control, perception, and official orchestration once."""
    demo_config_path = _value(context, 'demo_config_file')
    demo_config = load_demo_config(demo_config_path)
    tag_detections_topic = '/tag_detections'
    manipulation_share = get_package_share_directory(MANIPULATION_BRINGUP)
    behavior_tree_config = os.path.join(
        get_package_share_directory(PACKAGE_NAME),
        'config',
        'tabletop_apriltag_behavior_tree.yaml',
    )
    blackboard_config = os.path.join(
        get_package_share_directory(PACKAGE_NAME),
        'config',
        'tabletop_apriltag_blackboard.yaml',
    )
    tag_config = os.path.join(
        manipulation_share,
        'config',
        'apriltag',
        'tagged_cube.yaml',
    )
    camera_calibration_file = _value(context, 'camera_calibration_file')
    validate_demo_dependency_configs(
        demo_config,
        tag_config,
        camera_calibration_file,
    )

    simulation_motion_stack = _include(
        PACKAGE_NAME,
        'launch/arx_r5a_isaac_sim.launch.py',
        {
            'use_sim_time': 'True',
            'start_cumotion': 'True',
            'start_rviz': _value(context, 'start_rviz'),
            'read_esdf_world': 'False',
            'cumotion_time_dilation_factor': _value(
                context,
                'cumotion_time_dilation_factor',
            ),
            'log_level': _value(context, 'log_level'),
        },
    )

    perception_launch = _include(
        MANIPULATION_BRINGUP,
        'launch/arx_r5a_apriltag_pick_and_place.launch.py',
        {
            'use_sim_time': 'True',
            'headless': _value(context, 'headless'),
            'log_level': _value(context, 'log_level'),
            'start_camera': 'False',
            'start_apriltag': 'True',
            'start_apriltag_object_server': 'False',
            'start_object_selection_server': 'True',
            'start_robot': 'False',
            'start_motion_stack': 'False',
            'start_rviz': 'False',
            'start_orchestrator': 'False',
            'enable_nvblox': 'False',
            'camera_width': str(demo_config.camera.width),
            'camera_height': str(demo_config.camera.height),
            'color_image_topic': demo_config.camera.color_image_topic,
            'color_camera_info_topic': demo_config.camera.color_info_topic,
            'camera_optical_frame': demo_config.camera.optical_frame,
            'tag_config_file': tag_config,
            'camera_calibration_file': camera_calibration_file,
            'tag_detections_topic': tag_detections_topic,
            'min_tag_edge_px': '24.0',
            'min_stable_frames': '5',
            'pose_ttl_sec': '1.0',
        },
    )
    perception_stack = GroupAction(actions=[
        # Isaac ROS 4.5 RectifyNode subscribes to ``image_raw``. The ARX
        # overlay currently exposes an ``image`` remap, so adapt that concrete
        # namespaced input without changing the reusable manipulation repo.
        SetRemap(
            src='/camera_1/image_raw',
            dst=demo_config.camera.color_image_topic,
        ),
        perception_launch,
    ])
    object_server = Node(
        package='isaac_ros_manipulation_arx_r5a_apriltag',
        executable='apriltag_object_server',
        name='apriltag_object_server',
        output='screen',
        parameters=[{
            'tag_config_file': tag_config,
            'detections_topic': tag_detections_topic,
            'expected_camera_frame': demo_config.camera.optical_frame,
            'pose_ttl_sec': 2.0,
            # DDS delivery of /clock and camera frames can differ by one or
            # two simulation ticks. Accept bounded skew without rewriting the
            # official Isaac ROS detection timestamp.
            'future_tolerance_sec': 0.2,
            'min_stable_frames': 5,
            'min_tag_edge_px': 24.0,
            'max_translation_jump_m': 0.02,
            'max_rotation_jump_deg': 5.0,
            'use_sim_time': True,
        }],
    )

    attachment_actions = []
    if _bool(context, 'start_object_attachment'):
        attachment_actions.append(TimerAction(
            period=2.0,
            actions=[_include(
                'isaac_ros_cumotion_object_attachment',
                'launch/object_attachment.launch.py',
                {
                    'object_attachment.clear_esdf_on_attach': 'False',
                    'object_attachment.esdf_reference_frame': 'base_link',
                    'object_attachment.container_name': MANIPULATOR_CONTAINER,
                },
            )],
        ))

    orchestrator_arguments = [
        '--behavior-tree-config-file',
        behavior_tree_config,
        '--blackboard-config-file',
        blackboard_config,
        '--log-level',
        _value(context, 'log_level'),
        '--use-sim-time',
    ]
    if _bool(context, 'print_ascii_tree'):
        orchestrator_arguments.append('--print-ascii-tree')
    orchestrator = TimerAction(
        period=8.0,
        actions=[Node(
            package=PACKAGE_NAME,
            executable='sim_pick_place_orchestrator',
            name='multi_object_pick_and_place_orchestrator',
            output='screen',
            arguments=orchestrator_arguments,
            condition=IfCondition(LaunchConfiguration('start_orchestrator')),
        )],
    )

    drop_target = demo_config.drop_target
    goal_client = TimerAction(
        period=10.0,
        actions=[Node(
            package=PACKAGE_NAME,
            executable='apriltag_pick_place_goal_client',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'detections_topic': tag_detections_topic,
                'action_name': '/multi_object_pick_and_place',
                'base_frame': 'base_link',
                'tag_family': demo_config.tag_family,
                'source_tag_id': demo_config.source_object.tag_id,
                'target_tag_id': drop_target.tag_id,
                'stable_frames': 5,
                'transform_timeout_sec': 0.1,
                'auto_start': ParameterValue(
                    LaunchConfiguration('auto_start'),
                    value_type=bool,
                ),
                'link6_offset_in_tag': list(
                    drop_target.link6_offset_in_tag
                ),
                'link6_rotation_in_tag': list(
                    drop_target.link6_rotation_in_tag
                ),
            }],
            condition=IfCondition(LaunchConfiguration('start_goal_client')),
        )],
    )

    return [
        simulation_motion_stack,
        perception_stack,
        object_server,
        *attachment_actions,
        orchestrator,
        goal_client,
    ]


def generate_launch_description():
    """Declare demo arguments and assemble the aggregate launch."""
    package_share = FindPackageShare(PACKAGE_NAME)
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='False'),
        DeclareLaunchArgument('start_rviz', default_value='True'),
        DeclareLaunchArgument('start_orchestrator', default_value='True'),
        DeclareLaunchArgument('start_goal_client', default_value='True'),
        DeclareLaunchArgument('start_object_attachment', default_value='True'),
        DeclareLaunchArgument('auto_start', default_value='True'),
        DeclareLaunchArgument('print_ascii_tree', default_value='False'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'cumotion_time_dilation_factor',
            default_value='1.0',
            description=(
                'Default for the cuMotion server and MoveIt plugin. The '
                'automatic behavior tree sends its own value from '
                'tabletop_apriltag_behavior_tree.yaml.'
            ),
        ),
        DeclareLaunchArgument(
            'demo_config_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'tabletop_apriltag_demo.yaml',
            ]),
        ),
        DeclareLaunchArgument(
            'camera_calibration_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'tabletop_camera_extrinsics.yaml',
            ]),
        ),
        OpaqueFunction(function=launch_setup),
    ])
