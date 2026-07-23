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

"""Static contracts for the authored-USD manipulation launch layer."""

import ast
from pathlib import Path

import pytest

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / 'config'
LAUNCH_ROOT = PACKAGE_ROOT / 'launch'
SHARED_LAUNCH = LAUNCH_ROOT / 'arx_r5a_authored_usd_demo.launch.py'
MOTION_LAUNCH = LAUNCH_ROOT / 'arx_r5a_isaac_sim.launch.py'
POSE_REFINER = (
    PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'apriltag_pose_refiner.py'
)
SIM_ORCHESTRATOR = (
    PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'sim_orchestrator.py'
)


def _load_terminal_workflow_quiescence():
    """Load the dependency-free callback class without a ROS installation."""
    source = SIM_ORCHESTRATOR.read_text(encoding='utf-8')
    parsed = ast.parse(source, filename=str(SIM_ORCHESTRATOR))
    class_node = next(
        node for node in parsed.body
        if isinstance(node, ast.ClassDef)
        and node.name == '_TerminalWorkflowQuiescence'
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    namespace = {
        '_TERMINAL_WORKFLOW_STATUSES': frozenset({0, 1, 2, 3}),
    }
    exec(compile(module, str(SIM_ORCHESTRATOR), 'exec'), namespace)
    return namespace['_TerminalWorkflowQuiescence']


def test_authored_scene_disables_the_legacy_identity_world_transform():
    """The authored USD pose must be the sole world-to-base TF authority."""
    motion_source = MOTION_LAUNCH.read_text(encoding='utf-8')
    shared_source = SHARED_LAUNCH.read_text(encoding='utf-8')

    assert "'publish_world_to_base_transform'" in motion_source
    assert (
        "LaunchConfiguration('publish_world_to_base_transform')"
        in motion_source
    )
    assert "'publish_world_to_base_transform': 'False'" in shared_source


def test_shared_launch_selects_unique_camera_contracts():
    """One aggregate launch must serve both cameras without topic aliasing."""
    source = SHARED_LAUNCH.read_text(encoding='utf-8')

    assert "'zedx': '/zed_x/tag_detections'" in source
    assert "'d455': '/d455/tag_detections'" in source
    assert "'zedx': '/zed_x/tag_detections_raw'" in source
    assert "'d455': '/d455/tag_detections_raw'" in source
    assert "src='/camera_1/image_raw'" in source
    assert 'dst=camera.color_image_topic' in source
    assert "'expected_camera_frame': camera.optical_frame" in source


def test_authored_workflow_discovers_objects_only_in_the_source_zone():
    """A placed tag must not start a second official behavior-tree cycle."""
    source = SHARED_LAUNCH.read_text(encoding='utf-8')
    config = yaml.safe_load((
        CONFIG_ROOT / 'authored_usd_apriltag_demo.yaml'
    ).read_text(encoding='utf-8'))
    source_zone = config['source_zone']

    assert source_zone['enabled'] is True
    assert source_zone['frame'] == 'base_link'
    assert source_zone['min_xyz'] == pytest.approx([0.20, -0.30, -0.39])
    assert source_zone['max_xyz'] == pytest.approx([0.40, -0.08, -0.25])
    assert 'demo_config.source_zone.frame != scene_config.base_frame' in source
    assert 'source_zone.frame must match the USD scene base_frame' in source
    assert "'source_zone_enabled': demo_config.source_zone.enabled" in source
    assert (
        "'source_zone_min_xyz': list(demo_config.source_zone.minimum)"
        in source
    )
    assert (
        "'source_zone_max_xyz': list(demo_config.source_zone.maximum)"
        in source
    )


def test_drop_pose_requires_a_low_spread_base_frame_window():
    """The destination goal must reject a 20 mm PnP depth swing."""
    source = SHARED_LAUNCH.read_text(encoding='utf-8')

    assert "'max_translation_spread_m': float(" in source
    assert "'drop_pose_max_translation_spread_m'," in source
    assert (
        "'drop_pose_max_translation_spread_m',\n"
        "            default_value='0.01'"
    ) in source
    assert "'output_frame': scene_config.base_frame" in source
    assert "'transform_timeout_sec': 0.5" in source
    assert "'future_tolerance_sec': 0.5" in source
    assert "'d455_observation_positions.yaml'" in source
    assert "'enable_nvblox': 'False'" in source
    assert "executable='apriltag_pose_refiner'" in source
    assert "'camera_info_topic': '/camera_1/apriltag/camera_info_rect'" in (
        source
    )
    assert "'image_topic': '/camera_1/apriltag/image_rect'" in source
    assert "'apriltag_backends': 'CUDA'" in source
    assert "'apriltag_backend': 'CUDA'" in source
    assert "'expected_camera_frame': camera.optical_frame" in source


def test_pose_refiner_uses_roi_only_and_never_mixes_native_poses():
    """The manipulation stream must contain only exact-stamp refined poses."""
    source = POSE_REFINER.read_text(encoding='utf-8')

    assert 'validate_image_message(message)' in source
    assert 'refine_corners_subpixel_from_image(' in source
    assert 'image_to_grayscale' not in source
    assert '_publisher.publish(deepcopy(message))' not in source
    assert 'Dropping raw cuAprilTag sample from refined output' in source
    assert 'self._last_image_stamp = None' in source
    assert 'self._last_detection_stamp = None' in source
    assert 'self._images.clear()' in source
    assert 'self._pending_detections.clear()' in source


def test_shared_launch_reuses_official_components():
    """The simulation launch may configure, but not fork, upstream stacks."""
    source = SHARED_LAUNCH.read_text(encoding='utf-8')

    assert (
        "MANIPULATION_BRINGUP = 'isaac_ros_manipulation_arx_r5a_bringup'"
        in source
    )
    assert "'launch/arx_r5a_apriltag_pick_and_place.launch.py'" in source
    assert "'launch/arx_r5a_isaac_sim.launch.py'" in source
    assert "'isaac_ros_cumotion_object_attachment'" in source
    assert "executable='sim_pick_place_orchestrator'" in source


def test_authored_demo_enables_simulation_only_terminal_quiescence():
    """The one-shot launch must pause without replacing the official tree."""
    launch_source = SHARED_LAUNCH.read_text(encoding='utf-8')
    orchestrator_source = SIM_ORCHESTRATOR.read_text(encoding='utf-8')
    callback_source = orchestrator_source.split(
        'class _TerminalWorkflowQuiescence:',
        maxsplit=1,
    )[1].split('\ndef main()', maxsplit=1)[0]

    assert "'quiesce_on_terminal'," in launch_source
    assert "default_value='True'" in launch_source.split(
        "'quiesce_on_terminal',",
        maxsplit=1,
    )[1].split('),', maxsplit=1)[0]
    assert "if _bool(context, 'quiesce_on_terminal'):" in launch_source
    assert "orchestrator_arguments.append('--quiesce-on-terminal')" in (
        launch_source
    )
    assert "parser.add_argument('--quiesce-on-terminal'" in orchestrator_source
    assert 'orchestrator.tree.add_post_tick_handler(' in orchestrator_source
    for status_name in (
        'FAILED',
        'SUCCESS',
        'PARTIAL_SUCCESS',
        'INCOMPLETE',
    ):
        assert f'MultiObjectPickAndPlace.Result.{status_name}' in (
            orchestrator_source
        )

    assert 'self._orchestrator.is_orchestrator_busy' in callback_source
    assert 'blackboard.workflow_status' in callback_source
    assert 'tree.timer.cancel()' in callback_source
    assert 'blackboard.workflow_status =' not in callback_source
    assert '.shutdown()' not in callback_source


@pytest.mark.parametrize('terminal_status', [0, 1, 2, 3])
def test_terminal_quiescence_waits_for_a_goal_and_cancels_once(
    terminal_status,
):
    """Ignore pre-goal status and pause once after an accepted workflow."""
    quiescence_type = _load_terminal_workflow_quiescence()

    class FakeBlackboard:
        def __init__(self):
            self.workflow_status = terminal_status

        @staticmethod
        def exists(name):
            return name == 'workflow_status'

    class FakeLogger:
        def __init__(self):
            self.calls = []

        def info(self, message):
            self.calls.append(message)

    class FakeOrchestrator:
        def __init__(self):
            self.is_orchestrator_busy = False
            self.blackboard = FakeBlackboard()
            self.logger = FakeLogger()

    class FakeTimer:
        def __init__(self):
            self.cancel_count = 0

        def cancel(self):
            self.cancel_count += 1

    class FakeTree:
        def __init__(self):
            self.timer = FakeTimer()

    orchestrator = FakeOrchestrator()
    tree = FakeTree()
    quiescence = quiescence_type(orchestrator)

    quiescence(tree)
    assert tree.timer.cancel_count == 0

    orchestrator.blackboard.workflow_status = 4
    orchestrator.is_orchestrator_busy = True
    quiescence(tree)
    assert tree.timer.cancel_count == 0

    orchestrator.is_orchestrator_busy = False
    orchestrator.blackboard.workflow_status = terminal_status
    quiescence(tree)
    quiescence(tree)

    assert tree.timer.cancel_count == 1
    assert len(orchestrator.logger.calls) == 1
    assert str(terminal_status) in orchestrator.logger.calls[0]


def test_authored_table_is_loaded_by_the_official_cumotion_scene_server():
    """The physical table and the planner collision table must stay aligned."""
    shared_source = SHARED_LAUNCH.read_text(encoding='utf-8')
    motion_source = MOTION_LAUNCH.read_text(encoding='utf-8')
    scene_path = CONFIG_ROOT / 'authored_usd_table.scene'

    assert scene_path.is_file()
    assert "'collision_scene_file': collision_scene_file" in shared_source
    assert "'authored_usd_table.scene'" in shared_source
    assert (
        "'cumotion_action_server.moveit_collision_objects_scene_file'"
        in motion_source
    )

    scene_source = scene_path.read_text(encoding='utf-8')
    for leg_center in (
        '-0.17 -0.32 -0.76',
        '-0.17 0.32 -0.76',
        '0.67 -0.32 -0.76',
        '0.67 0.32 -0.76',
    ):
        assert leg_center in scene_source


def test_official_world_scene_is_normalized_to_the_arx_planning_frame():
    """The official collision scene must reach MoveIt in base_link."""
    shared_source = SHARED_LAUNCH.read_text(encoding='utf-8')
    motion_source = MOTION_LAUNCH.read_text(encoding='utf-8')
    adapter_source = (
        PACKAGE_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'planning_scene_frame_adapter.py'
    ).read_text(encoding='utf-8')

    assert 'RAW_PLANNING_SCENE_TOPIC' in shared_source
    assert '/cumotion/static_planning_scene_raw' in shared_source
    assert 'PLANNING_SCENE_TOPIC' in shared_source
    assert "executable='planning_scene_frame_adapter'" in shared_source
    assert "'output_topic': PLANNING_SCENE_TOPIC" in shared_source
    assert "'cumotion_static_planning_scene_topic': (" in shared_source
    assert "src='/planning_scene'" in motion_source
    assert "'cumotion_static_planning_scene_topic'" in (
        motion_source
    )
    assert "('/planning_scene'," not in motion_source
    assert 'remappings=[' not in shared_source.split(
        "executable='sim_pick_place_orchestrator'",
        maxsplit=1,
    )[1].split('condition=', maxsplit=1)[0]
    assert 'collision_object.header.frame_id = self._target_frame' in (
        adapter_source
    )
    assert '/cumotion/static_planning_scene_raw' in adapter_source
    assert 'output_topic' in adapter_source


@pytest.mark.parametrize(
    ('filename', 'profile'),
    (
        ('arx_r5a_zedx_eye_to_hand.launch.py', 'zedx'),
        ('arx_r5a_d455_eye_in_hand.launch.py', 'd455'),
    ),
)
def test_camera_entry_points_are_thin_includes(filename, profile):
    """Each user-facing launch must only select the shared camera profile."""
    source = (LAUNCH_ROOT / filename).read_text(encoding='utf-8')

    assert "'arx_r5a_authored_usd_demo.launch.py'" in source
    assert (
        f"launch_arguments={{'camera_profile': '{profile}'}}.items()"
        in source
    )
    assert 'ComposableNode' not in source
    assert 'apriltag_object_server' not in source


def test_authored_behavior_tree_consumes_base_frame_poses():
    """Timestamped adapter output must not be transformed a second time."""
    path = CONFIG_ROOT / 'authored_usd_apriltag_behavior_tree.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    pose_estimation = config['behavior_tree_params'][
        'multi_object_pick_and_place'
    ]['pose_estimation']

    assert pose_estimation['base_frame_id'] == 'base_link'
    assert pose_estimation['camera_frame_id'] == 'base_link'


def test_authored_camera_tf_is_owned_by_isaac_sim():
    """The reusable ARX launch must not duplicate the USD-derived TF."""
    path = CONFIG_ROOT / 'authored_usd_camera_tf_disabled.yaml'
    calibration = yaml.safe_load(path.read_text(encoding='utf-8'))

    assert calibration['publish'] is False


def test_side_tag_maps_to_the_block_center():
    """The authored side-mounted tag must match the tall red block."""
    path = CONFIG_ROOT / 'authored_usd_tagged_cube.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    object_config = config['objects'][0]
    transform = object_config['tag_to_object']

    assert object_config['class_id'] == 'tagged_block'
    assert object_config['dimensions'] == pytest.approx([0.05, 0.05, 0.15])
    assert transform['translation'] == pytest.approx([0.0, 0.0, 0.026])
    assert transform['rotation'] == pytest.approx(
        [0.0, -0.7071068, 0.0, 0.7071068]
    )


def test_authored_drop_goal_is_tag_relative_and_clears_the_detected_plane():
    """The held block must clear the detected tag plane without world truth."""
    demo_path = CONFIG_ROOT / 'authored_usd_apriltag_demo.yaml'
    config = yaml.safe_load(demo_path.read_text(encoding='utf-8'))
    drop = config['drop_target']
    grasp_path = CONFIG_ROOT / 'authored_usd_tagged_block_grasps.yaml'
    grasp_config = yaml.safe_load(grasp_path.read_text(encoding='utf-8'))
    link6_to_object_center = grasp_config['grasps']['grasp_top']['position'][2]
    block_half_height = config['source_object']['size'][2] / 2.0
    offset = drop['link6_offset_in_tag']
    clearance = -offset[2] - link6_to_object_center - block_half_height
    object_attachment_max_overshoot = 0.010
    attached_object_xrdf_buffer = 0.002
    maximum_pnp_depth_error = 0.010
    safety_margin = 0.005
    required_clearance = (
        object_attachment_max_overshoot +
        attached_object_xrdf_buffer +
        maximum_pnp_depth_error +
        safety_margin
    )
    launch_source = SHARED_LAUNCH.read_text(encoding='utf-8')

    assert 'plane_z' not in drop
    assert offset[:2] == pytest.approx([0.0, 0.0])
    assert clearance == pytest.approx(0.035)
    assert clearance >= required_clearance
    assert "'object_attachment.max_overshoot': '0.01'" in launch_source
    assert drop['link6_rotation_in_tag'] == pytest.approx(
        [0.0, -0.7071068, 0.0, 0.7071068]
    )


def test_drop_goal_waits_for_the_exact_detection_timestamp_without_blocking():
    """Eye-in-hand TF must not pair an old image with the newest wrist pose."""
    source = (
        PACKAGE_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'tag_goal_client.py'
    ).read_text(encoding='utf-8')

    assert 'wait_for_transform_async(' in source
    assert 'Time.from_msg(header.stamp)' in source
    assert '_pending_transform' in source
    assert 'lookup_transform(' not in source


def test_d455_eye_in_hand_chain_uses_dynamic_exact_time_tf_and_single_bin():
    """The wrist camera must never use a latest-TF or fixed-camera shortcut."""
    scene = yaml.safe_load((
        CONFIG_ROOT / 'arx_sim_usd_scene.yaml'
    ).read_text(encoding='utf-8'))
    shared_source = SHARED_LAUNCH.read_text(encoding='utf-8')
    goal_source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'tag_goal_client.py'
    ).read_text(encoding='utf-8')
    simulation_source = (
        PACKAGE_ROOT / 'arx_r5_isaac_sim_bringup' / 'simulation.py'
    ).read_text(encoding='utf-8')

    d455 = scene['cameras']['d455']
    assert d455['parent_frame'] == 'link6'
    assert d455['optical_frame'] == 'd455_color_optical_frame'
    assert d455['color_image_topic'] == '/d455/color/image_raw'
    assert d455['color_info_topic'] == '/d455/color/camera_info'
    assert "'d455': '/d455/tag_detections_raw'" in shared_source
    assert "'d455': '/d455/tag_detections'" in shared_source
    assert 'dst=camera.color_image_topic' in shared_source
    assert "'expected_camera_frame': camera.optical_frame" in shared_source
    assert "'output_frame': scene_config.base_frame" in shared_source
    assert "'d455_observation_positions.yaml'" in shared_source

    assert 'Buffer(node=self)' in goal_source
    assert 'wait_for_transform_async(' in goal_source
    assert 'Time.from_msg(header.stamp)' in goal_source
    assert 'lookup_transform(' not in goal_source
    assert 'goal.mode = MultiObjectPickAndPlace.Goal.SINGLE_BIN' in goal_source
    assert 'goal.target_poses.poses = [self._drop_pose.pose]' in goal_source
    assert 'goal.class_ids = []' in goal_source

    assert 'parent_frame=camera.parent_frame' in simulation_source
    assert "(f'{publisher}.inputs:staticPublisher', True)" in simulation_source
    assert 'camera_config.optical_frame' in simulation_source


def test_authored_blackboard_uses_the_matching_block_mesh_and_grasps():
    """Collision and grasp data must match the authored geometry."""
    path = CONFIG_ROOT / 'authored_usd_apriltag_blackboard.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    tagged_block = config['blackboard_params']['supported_objects'][
        'tagged_block'
    ]

    assert tagged_block['grasp_file_path'].endswith(
        '/config/authored_usd_tagged_block_grasps.yaml'
    )
    assert tagged_block['mesh_file_path'].endswith(
        '/assets/meshes/tagged_block.obj'
    )


def test_authored_grasp_keeps_the_link6_goal_inside_the_reachable_workspace():
    """The top-down link6 seed must retain margin from the IK boundary."""
    path = CONFIG_ROOT / 'authored_usd_tagged_block_grasps.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    grasps = config['grasps']

    assert tuple(grasps) == ('grasp_top',)
    assert grasps['grasp_top']['position'] == pytest.approx(
        [0.0, 0.0, 0.205]
    )
    assert grasps['grasp_top']['orientation']['xyz'] == pytest.approx(
        [0.0, 0.7071068, 0.0]
    )
    assert grasps['grasp_top']['orientation']['w'] == pytest.approx(
        0.7071068
    )


def test_physical_attachment_triggers_at_the_block_contact_aperture():
    """A colliding 50 mm block must attach before a zero close command."""
    path = CONFIG_ROOT / 'authored_usd_apriltag_demo.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    attachment = config['attachment']

    assert attachment['close_threshold'] == pytest.approx(0.028)
    assert attachment['open_threshold'] == pytest.approx(0.035)
    assert attachment['close_threshold'] < attachment['open_threshold']
