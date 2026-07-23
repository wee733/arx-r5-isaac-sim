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

"""Static contracts for the two user-facing camera demo scripts."""

from pathlib import Path
import re

import pytest

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPOSITORY_ROOT / 'scripts'


@pytest.mark.parametrize(
    (
        'script_name',
        'launch_file',
        'camera_label',
        'image_topic',
        'camera_info_topic',
        'camera_mode',
    ),
    (
        (
            'run_zedx_demo.sh',
            'arx_r5a_zedx_eye_to_hand.launch.py',
            'ZED X eye-to-hand camera',
            '/zed_x/left/image_raw',
            '/zed_x/left/camera_info',
            'zedx',
        ),
        (
            'run_d455_demo.sh',
            'arx_r5a_d455_eye_in_hand.launch.py',
            'D455 eye-in-hand camera',
            '/d455/color/image_raw',
            '/d455/color/camera_info',
            'd455',
        ),
    ),
)
def test_camera_script_selects_one_matching_pipeline(
    script_name,
    launch_file,
    camera_label,
    image_topic,
    camera_info_topic,
    camera_mode,
):
    """Each wrapper must select matching Isaac Sim and ROS camera profiles."""
    script = (SCRIPTS_ROOT / script_name).read_text(encoding='utf-8')

    assert f"ARX_DEMO_LAUNCH_FILE='{launch_file}'" in script
    assert f"ARX_DEMO_CAMERA_LABEL='{camera_label}'" in script
    assert f"ARX_DEMO_COLOR_IMAGE_TOPIC='{image_topic}'" in script
    assert f"ARX_DEMO_COLOR_INFO_TOPIC='{camera_info_topic}'" in script
    assert f'run_{camera_mode}_sim.sh' in script
    assert 'run_apriltag_demo.sh' in script

    sim_script = (SCRIPTS_ROOT / f'run_{camera_mode}_sim.sh').read_text(
        encoding='utf-8'
    )
    assert '--usd' in sim_script
    assert f'--camera-mode {camera_mode}' in sim_script
    assert '--authored-layout reachable' in sim_script
    assert '--reset-usd-joints' in sim_script


def test_d455_sim_and_ros_start_from_the_validated_observation_pose():
    """The moving camera and ros2_control must agree on the initial state."""
    config = yaml.safe_load((
        REPOSITORY_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'config'
        / 'd455_observation_positions.yaml'
    ).read_text(encoding='utf-8'))
    positions = config['initial_positions']
    assert tuple(positions) == (
        'joint1', 'joint2', 'joint3', 'joint4',
        'joint5', 'joint6', 'joint7', 'joint8',
    )
    validated_positions = (
        -1.3919817209, 1.8859872818, 0.5746622086, -0.6957563162,
        -0.6273930669, -1.9993159771, 0.044, 0.044,
    )
    assert tuple(positions.values()) == pytest.approx(validated_positions)

    script = (SCRIPTS_ROOT / 'run_d455_sim.sh').read_text(encoding='utf-8')
    match = re.search(r'--initial-positions=([^\s\\]+)', script)
    assert match is not None
    sim_positions = tuple(float(value) for value in match.group(1).split(','))
    assert sim_positions == pytest.approx(validated_positions)


def test_graph_checker_verifies_both_camera_contracts_and_cuda_backend():
    """The read-only checker must distinguish both official CUDA pipelines."""
    script = (SCRIPTS_ROOT / 'check_ros_graph.sh').read_text(
        encoding='utf-8'
    )

    for option in ('--zedx', '--d455'):
        assert option in script
    for topic in (
        '/zed_x/left/image_raw',
        '/zed_x/left/camera_info',
        '/zed_x/tag_detections_raw',
        '/zed_x/tag_detections',
        '/d455/color/image_raw',
        '/d455/color/camera_info',
        '/d455/tag_detections_raw',
        '/d455/tag_detections',
    ):
        assert topic in script
    assert 'ros2 param get /camera_1/apriltag backends' in script
    assert 'AprilTag backend: CUDA (Isaac ROS cuAprilTag)' in script


@pytest.mark.parametrize('readme_name', ('README.md', 'README.en.md'))
def test_readme_documents_two_terminal_commands(readme_name):
    """Both readers must get complete commands for both camera modes."""
    readme = (REPOSITORY_ROOT / readme_name).read_text(encoding='utf-8')

    for camera_mode in ('zedx', 'd455'):
        assert f'run_{camera_mode}_sim.sh' in readme
        assert f'run_{camera_mode}_demo.sh auto_start:=False' in readme
        assert f'check_ros_graph.sh --{camera_mode}' in readme
    assert 'nvidia::isaac_ros::apriltag::AprilTagNode' in readme
