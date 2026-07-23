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

"""Install the ARX R5A Isaac Sim ROS 2 package."""

from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'arx_r5_isaac_sim_bringup'
local_config_basenames = {'my_tabletop.yaml'}
config_files = sorted(
    path
    for path in glob('config/*.yaml') + glob('config/*.scene')
    if os.path.basename(path) not in local_config_basenames
)


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (
            'share/' + package_name,
            ['package.xml', '../LICENSE', '../NOTICE.md'],
        ),
        (
            os.path.join('share', package_name, 'LICENSES'),
            ['../LICENSES/BSD-3-Clause.txt'],
        ),
        (
            os.path.join('share', package_name, 'config'),
            config_files,
        ),
        (
            os.path.join('share', package_name, 'assets', 'apriltag'),
            glob('assets/apriltag/*.png'),
        ),
        (
            os.path.join('share', package_name, 'assets', 'scenes'),
            glob('assets/scenes/*.usd'),
        ),
        (
            os.path.join('share', package_name, 'assets', 'meshes'),
            glob('assets/meshes/*.obj'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wee733',
    maintainer_email='wee733@users.noreply.github.com',
    description='Isaac Sim execution bridge and cuMotion bringup for ARX R5A.',
    license='Apache-2.0 AND BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start_arx_r5a_sim = arx_r5_isaac_sim_bringup.simulation:main',
            'apriltag_pick_place_goal_client = '
            'arx_r5_isaac_sim_bringup.tag_goal_client:main',
            'sim_pick_place_orchestrator = '
            'arx_r5_isaac_sim_bringup.sim_orchestrator:main',
            'planning_scene_frame_adapter = '
            'arx_r5_isaac_sim_bringup.planning_scene_frame_adapter:main',
            'apriltag_pose_refiner = '
            'arx_r5_isaac_sim_bringup.apriltag_pose_refiner:main',
        ],
    },
)
