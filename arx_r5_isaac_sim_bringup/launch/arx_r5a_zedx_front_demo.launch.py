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

"""Select the opt-in front-ZED authored-USD manipulation layout."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = 'arx_r5_isaac_sim_bringup'


def generate_launch_description():
    """Include the ZED workflow with the front-demo layout contract."""
    shared_launch = PathJoinSubstitution([
        FindPackageShare(PACKAGE_NAME),
        'launch',
        'arx_r5a_authored_usd_demo.launch.py',
    ])
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(shared_launch),
            launch_arguments={
                'camera_profile': 'zedx',
                'layout_profile': 'front-demo',
            }.items(),
        ),
    ])
