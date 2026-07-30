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
Bring up the ROS side of the dual-camera VLA data-collection workcell.

Deliberately narrower than ``arx_r5a_authored_usd_demo.launch.py``. That file
starts the AprilTag pipeline, the object server, the behavior-tree orchestrator
and the goal client, and each of those claims a global name (``/get_objects``,
``/multi_object_pick_and_place``, a hard-coded adapter node name), which is
exactly why it can only ever serve one camera at a time.

Collection needs none of them: the object pose is known from the episode seed,
and the waypoints are scripted. Dropping that layer is what lets ZED X and D455
stream together, so both cameras' images and ``tf_static`` come straight from
Isaac Sim with no extra ROS nodes.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PACKAGE_NAME = 'arx_r5_isaac_sim_bringup'
RAW_PLANNING_SCENE_TOPIC = '/cumotion/static_planning_scene_raw'
PLANNING_SCENE_TOPIC = '/planning_scene'
DEFAULT_TASK_CONFIG = 'vla_task.yaml'
DEFAULT_SCENE_CONFIG = 'vla_scene.yaml'
DEFAULT_COLLISION_SCENE = 'vla_ground.scene'
DEFAULT_INITIAL_POSITIONS = 'vla_initial_positions.yaml'


def _value(context, name: str) -> str:
    return context.perform_substitution(LaunchConfiguration(name))


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
    """Compose the single-instance motion stack with the collection nodes."""
    package_share = get_package_share_directory(PACKAGE_NAME)

    def config_path(name: str, default_filename: str) -> str:
        override = _value(context, name).strip()
        if override:
            return override
        return os.path.join(package_share, 'config', default_filename)

    task_config_file = config_path('task_config_file', DEFAULT_TASK_CONFIG)
    scene_config_file = config_path('scene_config_file', DEFAULT_SCENE_CONFIG)
    collision_scene_file = config_path(
        'collision_scene_file',
        DEFAULT_COLLISION_SCENE,
    )
    initial_positions_file = config_path(
        'initial_positions_file',
        DEFAULT_INITIAL_POSITIONS,
    )

    motion_stack = _include(
        PACKAGE_NAME,
        'launch/arx_r5a_isaac_sim.launch.py',
        {
            'use_sim_time': 'True',
            'start_cumotion': 'True',
            'start_rviz': _value(context, 'start_rviz'),
            # Isaac Sim publishes the authored world -> base_link pose from the
            # USD, so the legacy identity transform must stay off.
            'publish_world_to_base_transform': 'False',
            'read_esdf_world': 'False',
            'collision_scene_file': collision_scene_file,
            'cumotion_static_planning_scene_topic': RAW_PLANNING_SCENE_TOPIC,
            'initial_positions_file_path': initial_positions_file,
            'cumotion_time_dilation_factor': _value(
                context,
                'cumotion_time_dilation_factor',
            ),
            'log_level': _value(context, 'log_level'),
        },
    )

    # Keep the official cuMotion collision model synchronized with the
    # simulator's PhysX FixedJoint.  The VLA driver calls /attach_object only
    # after its physical lift and calls it again after physical release.  VLA
    # does not run nvblox, so ESDF clearing must stay disabled; otherwise the
    # upstream node would wait for an unavailable nvblox service and abort.
    object_attachment = TimerAction(
        period=2.0,
        actions=[_include(
            'isaac_ros_cumotion_object_attachment',
            'launch/object_attachment.launch.py',
            {
                'object_attachment.clear_esdf_on_attach': 'False',
                'object_attachment.esdf_reference_frame': 'base_link',
                'object_attachment.max_overshoot': '0.01',
                # arx_r5a_isaac_sim.launch.py creates this official container.
                'object_attachment.container_name': 'cumotion_container',
            },
        )],
        condition=IfCondition(LaunchConfiguration('start_object_attachment')),
    )

    # Isaac ROS 4.5 stamps every collision object in the .scene file with the
    # `world` frame although the numbers are authored in base_link.
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
            'target_frame': 'base_link',
        }],
    )

    recorder = Node(
        package=PACKAGE_NAME,
        executable='vla_recorder',
        name='vla_recorder',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'scene_config_file': scene_config_file,
            'task_config_file': task_config_file,
            'record_depth': ParameterValue(
                LaunchConfiguration('record_depth'),
                value_type=bool,
            ),
            'terminal_drain_sec': ParameterValue(
                LaunchConfiguration('recorder_terminal_drain_sec'),
                value_type=float,
            ),
            'episode_heartbeat_timeout_sec': ParameterValue(
                LaunchConfiguration('episode_heartbeat_timeout_sec'),
                value_type=float,
            ),
            'max_episode_frames': ParameterValue(
                LaunchConfiguration('max_episode_frames'),
                value_type=int,
            ),
        }],
        condition=IfCondition(LaunchConfiguration('start_recorder')),
    )

    # The driver blocks on cuMotion and MoveIt action servers, so it starts
    # after the controllers and move_group have had time to come up.
    episode_driver = TimerAction(
        period=10.0,
        actions=[Node(
            package=PACKAGE_NAME,
            executable='vla_episode_driver',
            name='vla_episode_driver',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'package_share_directory': package_share,
                'task_config_file': task_config_file,
                'scene_config_file': scene_config_file,
                'episode_count': ParameterValue(
                    LaunchConfiguration('episode_count'),
                    value_type=int,
                ),
                'start_seed': ParameterValue(
                    LaunchConfiguration('start_seed'),
                    value_type=int,
                ),
                'auto_start': ParameterValue(
                    LaunchConfiguration('auto_start'),
                    value_type=bool,
                ),
                'dry_run': ParameterValue(
                    LaunchConfiguration('dry_run'),
                    value_type=bool,
                ),
                'use_object_attachment': ParameterValue(
                    LaunchConfiguration('start_object_attachment'),
                    value_type=bool,
                ),
                'object_attachment_timeout_sec': ParameterValue(
                    LaunchConfiguration('object_attachment_timeout_sec'),
                    value_type=float,
                ),
                'time_dilation_factor': ParameterValue(
                    LaunchConfiguration('cumotion_time_dilation_factor'),
                    value_type=float,
                ),
                'gripper_close_timeout_sec': ParameterValue(
                    LaunchConfiguration('gripper_close_timeout_sec'),
                    value_type=float,
                ),
                'episode_heartbeat_period_sec': ParameterValue(
                    LaunchConfiguration('episode_heartbeat_period_sec'),
                    value_type=float,
                ),
                'scene_reset_wall_timeout_sec': ParameterValue(
                    LaunchConfiguration('scene_reset_wall_timeout_sec'),
                    value_type=float,
                ),
                'placement_wall_timeout_sec': ParameterValue(
                    LaunchConfiguration('placement_wall_timeout_sec'),
                    value_type=float,
                ),
                'execute_timeout_min_sec': ParameterValue(
                    LaunchConfiguration('execute_timeout_min_sec'),
                    value_type=float,
                ),
                'execute_timeout_scale': ParameterValue(
                    LaunchConfiguration('execute_timeout_scale'),
                    value_type=float,
                ),
                'execute_timeout_margin_sec': ParameterValue(
                    LaunchConfiguration('execute_timeout_margin_sec'),
                    value_type=float,
                ),
                'execute_wall_timeout_sec': ParameterValue(
                    LaunchConfiguration('execute_wall_timeout_sec'),
                    value_type=float,
                ),
            }],
            condition=IfCondition(LaunchConfiguration('start_episode_driver')),
        )],
    )

    return [
        motion_stack,
        object_attachment,
        planning_scene_adapter,
        recorder,
        episode_driver,
    ]


def generate_launch_description():
    """Declare the collection workflow arguments."""
    return LaunchDescription([
        DeclareLaunchArgument('start_rviz', default_value='True'),
        DeclareLaunchArgument('start_object_attachment', default_value='True'),
        DeclareLaunchArgument('start_recorder', default_value='True'),
        DeclareLaunchArgument('start_episode_driver', default_value='True'),
        DeclareLaunchArgument(
            'auto_start',
            default_value='False',
            description=(
                'Begin collecting as soon as the action servers appear. When '
                'False, call the vla_episode_driver/collect service.'
            ),
        ),
        DeclareLaunchArgument('episode_count', default_value='5'),
        DeclareLaunchArgument('start_seed', default_value='0'),
        DeclareLaunchArgument(
            'dry_run',
            default_value='False',
            description=(
                'Plan every waypoint without executing. Use this to validate '
                'reachability before the first real collection run.'
            ),
        ),
        DeclareLaunchArgument(
            'record_depth',
            default_value='False',
            description='Also subscribe to both aligned depth streams.',
        ),
        DeclareLaunchArgument(
            'recorder_terminal_drain_sec',
            default_value='0.5',
            description=(
                'Wall-clock drain window for camera samples captured no '
                'later than the terminal simulation timestamp.'
            ),
        ),
        DeclareLaunchArgument(
            'episode_heartbeat_period_sec',
            default_value='1.0',
            description='Steady-clock period for active-episode heartbeats.',
        ),
        DeclareLaunchArgument(
            'episode_heartbeat_timeout_sec',
            default_value='5.0',
            description='Recorder lease timeout when heartbeats stop.',
        ),
        DeclareLaunchArgument(
            'max_episode_frames',
            default_value='100000',
            description=(
                'Hard per-episode frame cap; reaching it stops the batch '
                'before stamp uniqueness can be weakened.'
            ),
        ),
        DeclareLaunchArgument(
            'cumotion_time_dilation_factor',
            default_value='0.2',
        ),
        DeclareLaunchArgument(
            'gripper_close_timeout_sec',
            default_value='15.0',
            description=(
                'Wall-clock timeout for the physical attachment '
                'acknowledgement after closing the gripper.'
            ),
        ),
        DeclareLaunchArgument(
            'object_attachment_timeout_sec',
            default_value='30.0',
            description=(
                'Timeout warning threshold for the official cuMotion '
                'attach/detach action. The client drains late results.'
            ),
        ),
        DeclareLaunchArgument(
            'scene_reset_wall_timeout_sec',
            default_value='10.0',
            description=(
                'Wall watchdog around reset ACK and simulation settling.'
            ),
        ),
        DeclareLaunchArgument(
            'placement_wall_timeout_sec',
            default_value='10.0',
            description=(
                'Wall watchdog for placement when simulation time pauses.'
            ),
        ),
        DeclareLaunchArgument(
            'execute_timeout_min_sec',
            default_value='60.0',
            description=(
                'Minimum default trajectory execution timeout in simulation '
                'seconds.'
            ),
        ),
        DeclareLaunchArgument(
            'execute_timeout_scale',
            default_value='1.5',
            description=(
                'Scale applied to trajectory duration in simulation seconds.'
            ),
        ),
        DeclareLaunchArgument(
            'execute_timeout_margin_sec',
            default_value='10.0',
            description=(
                'Fixed margin in simulation seconds added to trajectory '
                'execution timeout.'
            ),
        ),
        DeclareLaunchArgument(
            'execute_wall_timeout_sec',
            default_value='10.0',
            description=(
                'Wall watchdog for execution when simulation time stops '
                'advancing; it is not a total wall-time limit.'
            ),
        ),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'task_config_file',
            default_value='',
            description='Optional override; empty selects vla_task.yaml.',
        ),
        DeclareLaunchArgument(
            'scene_config_file',
            default_value='',
            description='Optional override; empty selects vla_scene.yaml.',
        ),
        DeclareLaunchArgument(
            'collision_scene_file',
            default_value='',
            description='Optional override; empty selects vla_ground.scene.',
        ),
        DeclareLaunchArgument(
            'initial_positions_file',
            default_value='',
            description=(
                'Optional override; empty selects vla_initial_positions.yaml.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
