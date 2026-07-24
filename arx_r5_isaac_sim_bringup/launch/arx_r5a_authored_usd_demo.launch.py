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

"""Launch one authored-USD camera through the official manipulation stack."""

import os

from ament_index_python.packages import get_package_share_directory

from arx_r5_isaac_sim_bringup.demo_config import load_demo_config
from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = 'arx_r5_isaac_sim_bringup'
MANIPULATION_BRINGUP = 'isaac_ros_manipulation_arx_r5a_bringup'
MANIPULATOR_CONTAINER = 'manipulator_container'
DETECTION_TOPICS = {
    'zedx': '/zed_x/tag_detections',
    'd455': '/d455/tag_detections',
}
RAW_DETECTION_TOPICS = {
    'zedx': '/zed_x/tag_detections_raw',
    'd455': '/d455/tag_detections_raw',
}
PERCEPTION_NAMESPACES = {
    'zedx': 'zed_x',
    'd455': 'd455',
}
RAW_PLANNING_SCENE_TOPIC = '/cumotion/static_planning_scene_raw'
PLANNING_SCENE_TOPIC = '/planning_scene'
DEFAULT_DEMO_CONFIG = 'authored_usd_apriltag_demo.yaml'
DEFAULT_COLLISION_SCENE = 'authored_usd_table.scene'


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
    """Resolve one camera profile and compose the unchanged upstream stack."""
    camera_profile = _value(context, 'camera_profile')
    scene_config = load_usd_scene_config(
        _value(context, 'usd_scene_config_file')
    )
    if camera_profile not in scene_config.cameras:
        available = ', '.join(scene_config.cameras)
        raise ValueError(
            f'Unknown camera profile {camera_profile!r}; '
            f'available: {available}'
        )
    camera = scene_config.cameras[camera_profile]
    package_share = get_package_share_directory(PACKAGE_NAME)
    demo_config_path = _value(context, 'demo_config_file').strip()
    if not demo_config_path:
        demo_config_path = os.path.join(
            package_share,
            'config',
            DEFAULT_DEMO_CONFIG,
        )
    demo_config = load_demo_config(demo_config_path)
    if (
        demo_config.source_zone.enabled and
        demo_config.source_zone.frame != scene_config.base_frame
    ):
        raise ValueError(
            'source_zone.frame must match the USD scene base_frame: '
            f'{demo_config.source_zone.frame!r} != '
            f'{scene_config.base_frame!r}'
        )
    initial_positions_file = os.path.join(
        package_share,
        'config',
        (
            'd455_observation_positions.yaml'
            if camera_profile == 'd455'
            else 'initial_positions.yaml'
        ),
    )
    collision_scene_file = _value(context, 'collision_scene_file').strip()
    if not collision_scene_file:
        collision_scene_file = os.path.join(
            package_share,
            'config',
            DEFAULT_COLLISION_SCENE,
        )
    detections_override = _value(context, 'tag_detections_topic').strip()
    tag_detections_topic = (
        detections_override or DETECTION_TOPICS[camera_profile]
    )
    raw_tag_detections_topic = RAW_DETECTION_TOPICS[camera_profile]
    perception_namespace = PERCEPTION_NAMESPACES[camera_profile]
    rectified_image_topic = (
        f'/{perception_namespace}/apriltag/image_rect'
    )
    rectified_camera_info_topic = (
        f'/{perception_namespace}/apriltag/camera_info_rect'
    )

    simulation_motion_stack = _include(
        PACKAGE_NAME,
        'launch/arx_r5a_isaac_sim.launch.py',
        {
            'use_sim_time': 'True',
            'start_cumotion': 'True',
            'start_rviz': _value(context, 'start_rviz'),
            'publish_world_to_base_transform': 'False',
            'read_esdf_world': 'False',
            'collision_scene_file': collision_scene_file,
            'cumotion_static_planning_scene_topic': (
                RAW_PLANNING_SCENE_TOPIC
            ),
            'initial_positions_file_path': initial_positions_file,
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
            'start_apriltag': 'True',
            'start_apriltag_object_server': 'False',
            'start_object_selection_server': 'True',
            'start_robot': 'False',
            'start_motion_stack': 'False',
            'start_rviz': 'False',
            'start_orchestrator': 'False',
            'enable_nvblox': 'False',
            'camera_width': str(camera.width),
            'camera_height': str(camera.height),
            'color_image_topic': camera.color_image_topic,
            'color_camera_info_topic': camera.color_info_topic,
            'camera_optical_frame': camera.optical_frame,
            'perception_namespace': perception_namespace,
            'apriltag_backends': 'CUDA',
            'tag_config_file': _value(context, 'tag_config_file'),
            'camera_calibration_file': _value(
                context,
                'camera_calibration_file',
            ),
            'tag_detections_topic': raw_tag_detections_topic,
            'min_tag_edge_px': '24.0',
            'min_stable_frames': '5',
            'pose_ttl_sec': '1.0',
        },
    )
    pose_refiner = Node(
        package=PACKAGE_NAME,
        executable='apriltag_pose_refiner',
        name=f'{camera_profile}_apriltag_pose_refiner',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': raw_tag_detections_topic,
            'output_topic': tag_detections_topic,
            'camera_info_topic': rectified_camera_info_topic,
            'image_topic': rectified_image_topic,
            'expected_frame': camera.optical_frame,
            'apriltag_backend': 'CUDA',
            'tag_size': demo_config.source_object.tag_size,
            'maximum_reprojection_error_px': 2.0,
        }],
    )

    planning_scene_adapter = Node(
        package=PACKAGE_NAME,
        executable='planning_scene_frame_adapter',
        name='planning_scene_frame_adapter',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': RAW_PLANNING_SCENE_TOPIC,
            'output_topic': PLANNING_SCENE_TOPIC,
            'source_frame': 'world',
            'target_frame': scene_config.base_frame,
        }],
    )

    # GetObjectPose has no header in the official interface. The ARX adapter
    # therefore transforms each timestamped AprilTag detection to base_link
    # before filtering/caching, which is required for a moving wrist camera.
    object_server = Node(
        package='isaac_ros_manipulation_arx_r5a_apriltag',
        executable='apriltag_object_server',
        name=f'{camera_profile}_apriltag_object_server',
        output='screen',
        parameters=[{
            'tag_config_file': _value(context, 'tag_config_file'),
            'detections_topic': tag_detections_topic,
            'expected_camera_frame': camera.optical_frame,
            'output_frame': scene_config.base_frame,
            'transform_timeout_sec': 0.5,
            'pose_ttl_sec': 2.0,
            'future_tolerance_sec': 0.5,
            'min_stable_frames': 5,
            'min_tag_edge_px': 24.0,
            'max_translation_jump_m': 0.02,
            'max_rotation_jump_deg': 5.0,
            # The upstream behavior tree intentionally keeps discovering
            # objects. Exclude the destination workcell after a successful
            # drop while retaining the cached pose for the active task.
            'source_zone_enabled': demo_config.source_zone.enabled,
            'source_zone_min_xyz': list(demo_config.source_zone.minimum),
            'source_zone_max_xyz': list(demo_config.source_zone.maximum),
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
                    'object_attachment.esdf_reference_frame': (
                        scene_config.base_frame
                    ),
                    'object_attachment.max_overshoot': '0.01',
                    'object_attachment.container_name': MANIPULATOR_CONTAINER,
                },
            )],
        ))

    orchestrator_arguments = [
        '--behavior-tree-config-file',
        _value(context, 'behavior_tree_config_file'),
        '--blackboard-config-file',
        _value(context, 'blackboard_config_file'),
        '--log-level',
        _value(context, 'log_level'),
        '--use-sim-time',
    ]
    if _bool(context, 'print_ascii_tree'):
        orchestrator_arguments.append('--print-ascii-tree')
    if _bool(context, 'quiesce_on_terminal'):
        orchestrator_arguments.append('--quiesce-on-terminal')
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
            name=f'{camera_profile}_apriltag_pick_place_goal_client',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'detections_topic': tag_detections_topic,
                'action_name': '/multi_object_pick_and_place',
                'base_frame': scene_config.base_frame,
                'expected_camera_frame': camera.optical_frame,
                'tag_family': demo_config.tag_family,
                'source_tag_id': demo_config.source_object.tag_id,
                'target_tag_id': drop_target.tag_id,
                'stable_frames': 5,
                # Exact TF can arrive several rendered frames after the RGB
                # message. The client waits asynchronously, so this timeout
                # never blocks AprilTag or TF callbacks.
                'transform_timeout_sec': 2.0,
                'drop_pose_ttl_sec': float(
                    _value(context, 'drop_pose_ttl_sec')
                ),
                'max_translation_spread_m': float(
                    _value(
                        context,
                        'drop_pose_max_translation_spread_m',
                    )
                ),
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
        perception_launch,
        pose_refiner,
        planning_scene_adapter,
        object_server,
        *attachment_actions,
        orchestrator,
        goal_client,
    ]


def generate_launch_description():
    """Declare the shared authored-USD workflow arguments."""
    package_share = FindPackageShare(PACKAGE_NAME)
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_profile',
            choices=['zedx', 'd455'],
            description='Authored USD camera used by Isaac ROS AprilTag.',
        ),
        DeclareLaunchArgument('headless', default_value='False'),
        DeclareLaunchArgument('start_rviz', default_value='True'),
        DeclareLaunchArgument('start_orchestrator', default_value='True'),
        DeclareLaunchArgument('start_goal_client', default_value='True'),
        DeclareLaunchArgument('start_object_attachment', default_value='True'),
        DeclareLaunchArgument('auto_start', default_value='True'),
        DeclareLaunchArgument(
            'quiesce_on_terminal',
            default_value='True',
            description=(
                'Pause official behavior-tree ticks after the one-shot demo '
                'reaches a terminal workflow status.'
            ),
        ),
        DeclareLaunchArgument('drop_pose_ttl_sec', default_value='0.5'),
        DeclareLaunchArgument(
            'drop_pose_max_translation_spread_m',
            default_value='0.01',
        ),
        DeclareLaunchArgument('print_ascii_tree', default_value='False'),
        DeclareLaunchArgument('tag_detections_topic', default_value=''),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'cumotion_time_dilation_factor',
            default_value='1.0',
        ),
        DeclareLaunchArgument(
            'usd_scene_config_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'arx_sim_usd_scene.yaml',
            ]),
        ),
        DeclareLaunchArgument(
            'demo_config_file',
            default_value='',
            description=(
                'Optional override; empty selects the authored workcell '
                'configuration.'
            ),
        ),
        DeclareLaunchArgument(
            'collision_scene_file',
            default_value='',
            description=(
                'Optional MoveIt scene override; empty selects the authored '
                'workcell collision scene.'
            ),
        ),
        DeclareLaunchArgument(
            'tag_config_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'authored_usd_tagged_cube.yaml',
            ]),
        ),
        DeclareLaunchArgument(
            'camera_calibration_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'authored_usd_camera_tf_disabled.yaml',
            ]),
        ),
        DeclareLaunchArgument(
            'behavior_tree_config_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'authored_usd_apriltag_behavior_tree.yaml',
            ]),
        ),
        DeclareLaunchArgument(
            'blackboard_config_file',
            default_value=PathJoinSubstitution([
                package_share,
                'config',
                'authored_usd_apriltag_blackboard.yaml',
            ]),
        ),
        OpaqueFunction(function=launch_setup),
    ])
