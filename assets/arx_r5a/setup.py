# Copyright (c) 2026 wee733
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

from glob import glob

from setuptools import setup


package_name = 'isaac_ros_manipulation_arx_r5a_robot_description'


setup(
    name=package_name,
    version='0.3.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml', '.setup_assistant']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/meshes/collision', glob('meshes/collision/*')),
        ('share/' + package_name + '/meshes/visual', glob('meshes/visual/*')),
        ('share/' + package_name + '/srdf', glob('srdf/*')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/xrdf', glob('xrdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wee733',
    maintainer_email='wee733@users.noreply.github.com',
    description=(
        'ARX R5A description and configuration for MoveIt 2, '
        'ros2_control, and cuMotion.'
    ),
    license='BSD-3-Clause',
    tests_require=['pytest'],
)
