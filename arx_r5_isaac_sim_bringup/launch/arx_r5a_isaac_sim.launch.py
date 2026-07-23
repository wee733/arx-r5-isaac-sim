# Copyright 2026 wee733
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This launch file was originally derived from
# https://github.com/ros-planning/moveit2_tutorials/blob/efef1d3/doc/how_to_guides/isaac_panda/launch/isaac_demo.launch.py  # noqa
#
# BSD 3-Clause License
#
# Copyright (c) 2008-2013, Willow Garage, Inc. All rights reserved.
# Copyright (c) 2015-2023, PickNik, LLC. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Launch ROS 2 control, MoveIt, cuMotion, and RViz for the Isaac Sim R5A."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

import yaml


BRINGUP_PACKAGE = 'arx_r5_isaac_sim_bringup'
DESCRIPTION_PACKAGE = 'isaac_ros_manipulation_arx_r5a_robot_description'


def _value(context, name):
    return context.perform_substitution(LaunchConfiguration(name))


def _as_bool(value):
    return value.lower() in ('1', 'true', 'yes', 'on')


def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def launch_setup(context, *args, **kwargs):
    """Build the runtime graph after all launch paths are resolved."""
    urdf_path = _value(context, 'urdf_path')
    srdf_path = _value(context, 'srdf_path')
    initial_positions_path = _value(context, 'initial_positions_file_path')
    kinematics_path = _value(context, 'kinematics_file_path')
    joint_limits_path = _value(context, 'joint_limits_file_path')
    moveit_controllers_path = _value(context, 'moveit_controllers_file_path')
    ros2_controllers_path = _value(context, 'ros2_controllers_file_path')
    rviz_config_path = _value(context, 'rviz_config_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    start_cumotion = _as_bool(_value(context, 'start_cumotion'))

    moveit_config = (
        MoveItConfigsBuilder('r5a', package_name=DESCRIPTION_PACKAGE)
        .robot_description(
            file_path=urdf_path,
            mappings={'initial_positions_file': initial_positions_path},
        )
        .robot_description_semantic(file_path=srdf_path)
        .robot_description_kinematics(file_path=kinematics_path)
        .joint_limits(file_path=joint_limits_path)
        .trajectory_execution(file_path=moveit_controllers_path)
        .planning_pipelines(
            default_planning_pipeline='ompl',
            pipelines=['ompl'],
        )
        .to_moveit_configs()
    )

    if start_cumotion:
        cumotion_pipeline_path = os.path.join(
            get_package_share_directory('isaac_ros_cumotion_moveit'),
            'config',
            'isaac_ros_cumotion_planning.yaml',
        )
        cumotion_pipeline = _load_yaml(cumotion_pipeline_path)
        moveit_config.planning_pipelines['planning_pipelines'].insert(
            0,
            'isaac_ros_cumotion',
        )
        moveit_config.planning_pipelines[
            'isaac_ros_cumotion'
        ] = cumotion_pipeline
        moveit_config.planning_pipelines[
            'default_planning_pipeline'
        ] = 'isaac_ros_cumotion'

    moveit_config.trajectory_execution['trajectory_execution'][
        'allowed_start_tolerance'
    ] = 0.1

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/joint_states', '/isaac_joint_states')],
    )

    world_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_arx_r5a_base',
        output='log',
        arguments=['--frame-id', 'world', '--child-frame-id', 'base_link'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(
            LaunchConfiguration('publish_world_to_base_transform')
        ),
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            ParameterFile(ros2_controllers_path, allow_substs=True),
            {'use_sim_time': use_sim_time},
        ],
    )

    def spawner(controller_name, condition=None):
        return Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[
                controller_name,
                '--controller-manager',
                '/controller_manager',
                '--controller-manager-timeout',
                _value(context, 'controller_manager_timeout'),
            ],
            condition=condition,
        )

    joint_state_spawner = spawner('joint_state_broadcaster')
    arm_controller_spawner = spawner('manipulator_controller')
    gripper_controller_spawner = spawner(
        'gripper_controller',
        condition=IfCondition(LaunchConfiguration('start_gripper_controller')),
    )

    delayed_joint_state_spawner = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[TimerAction(period=3.0, actions=[joint_state_spawner])],
        )
    )
    delayed_trajectory_spawners = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                TimerAction(
                    period=5.0,
                    actions=[
                        arm_controller_spawner,
                        gripper_controller_spawner,
                    ],
                )
            ],
        )
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': use_sim_time},
        ],
        arguments=[
            '--ros-args',
            '--log-level',
            _value(context, 'log_level'),
        ],
    )
    delayed_move_group = TimerAction(period=6.0, actions=[move_group])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_moveit',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration('start_rviz')),
    )

    actions = [
        robot_state_publisher,
        world_to_base,
        ros2_control_node,
        delayed_joint_state_spawner,
        delayed_trajectory_spawners,
        delayed_move_group,
        rviz,
    ]

    if start_cumotion:
        cumotion_launch = os.path.join(
            get_package_share_directory('isaac_ros_cumotion'),
            'launch',
            'isaac_ros_cumotion.launch.py',
        )
        cumotion_arguments = {
            'cumotion_action_server.xrdf_file_path': _value(
                context,
                'cumotion_xrdf_file_path',
            ),
            'cumotion_action_server.urdf_file_path': _value(
                context,
                'cumotion_urdf_file_path',
            ),
            'cumotion_action_server.tool_frame': _value(
                context,
                'cumotion_tool_frame',
            ),
            'cumotion_action_server.time_dilation_factor': _value(
                context,
                'cumotion_time_dilation_factor',
            ),
            'cumotion_action_server.read_esdf_world': _value(
                context,
                'read_esdf_world',
            ),
            'cumotion_action_server.moveit_collision_objects_scene_file': (
                _value(context, 'collision_scene_file')
            ),
            'cumotion_action_server.add_ground_plane': 'False',
            'cumotion_action_server.override_moveit_scaling_factors': 'False',
        }
        actions.append(
            GroupAction(actions=[
                SetRemap(
                    src='/planning_scene',
                    dst=_value(
                        context,
                        'cumotion_static_planning_scene_topic',
                    ),
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(cumotion_launch),
                    launch_arguments=cumotion_arguments.items(),
                ),
            ])
        )

    return actions


def generate_launch_description():
    """Declare paths separately so the package remains overlay-friendly."""
    bringup_share = FindPackageShare(BRINGUP_PACKAGE)
    description_share = FindPackageShare(DESCRIPTION_PACKAGE)

    arguments = [
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('start_cumotion', default_value='True'),
        DeclareLaunchArgument('start_rviz', default_value='True'),
        DeclareLaunchArgument(
            'start_gripper_controller',
            default_value='True',
        ),
        DeclareLaunchArgument(
            'publish_world_to_base_transform',
            default_value='True',
            description=(
                'Publish the legacy identity world -> base_link transform. '
                'Disable this when Isaac Sim publishes an authored USD pose.'
            ),
        ),
        DeclareLaunchArgument('read_esdf_world', default_value='False'),
        DeclareLaunchArgument(
            'cumotion_static_planning_scene_topic',
            default_value='/planning_scene',
            description=(
                'PlanningScene topic published by the official cuMotion '
                'StaticPlanningSceneServer. Leave canonical unless a frame '
                'adapter consumes a private raw topic.'
            ),
        ),
        DeclareLaunchArgument(
            'collision_scene_file',
            default_value='',
            description=(
                'Optional MoveIt .scene file loaded by the official cuMotion '
                'StaticPlanningSceneServer.'
            ),
        ),
        DeclareLaunchArgument(
            'controller_manager_timeout',
            default_value='60',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            choices=['debug', 'info', 'warn', 'error'],
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value=PathJoinSubstitution(
                [bringup_share, 'urdf', 'r5a.isaac_sim.urdf.xacro']
            ),
        ),
        DeclareLaunchArgument(
            'srdf_path',
            default_value=PathJoinSubstitution(
                [description_share, 'srdf', 'r5a.srdf']
            ),
        ),
        DeclareLaunchArgument(
            'initial_positions_file_path',
            default_value=PathJoinSubstitution(
                [bringup_share, 'config', 'initial_positions.yaml']
            ),
        ),
        DeclareLaunchArgument(
            'kinematics_file_path',
            default_value=PathJoinSubstitution(
                [description_share, 'config', 'kinematics.yaml']
            ),
        ),
        DeclareLaunchArgument(
            'joint_limits_file_path',
            default_value=PathJoinSubstitution(
                [description_share, 'config', 'joint_limits.yaml']
            ),
        ),
        DeclareLaunchArgument(
            'moveit_controllers_file_path',
            default_value=PathJoinSubstitution(
                [bringup_share, 'config', 'moveit_controllers.yaml']
            ),
        ),
        DeclareLaunchArgument(
            'ros2_controllers_file_path',
            default_value=PathJoinSubstitution(
                [bringup_share, 'config', 'ros2_controllers.yaml']
            ),
        ),
        DeclareLaunchArgument(
            'rviz_config_file',
            default_value=PathJoinSubstitution(
                [description_share, 'config', 'moveit.rviz']
            ),
        ),
        DeclareLaunchArgument(
            'cumotion_urdf_file_path',
            default_value=PathJoinSubstitution(
                [description_share, 'urdf', 'r5a_cumotion.urdf']
            ),
        ),
        DeclareLaunchArgument(
            'cumotion_xrdf_file_path',
            default_value=PathJoinSubstitution(
                [description_share, 'xrdf', 'r5a.xrdf']
            ),
        ),
        DeclareLaunchArgument('cumotion_tool_frame', default_value='link6'),
        DeclareLaunchArgument(
            'cumotion_time_dilation_factor',
            default_value='1.0',
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
