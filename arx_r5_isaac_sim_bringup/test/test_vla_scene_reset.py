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
Guard the episode-reset contract in the simulation process.

Every rule covered here fails silently in Isaac Sim rather than raising, which
is why they are pinned as source-level assertions (instantiating the controller
needs a live stage):

* A kinematic rigid body follows its PhysX kinematic target, so a direct
  world-pose write is reverted on the next step -- the reset is logged but
  never appears. The VLA block must therefore stay dynamic.
* PhysX refuses velocity writes on a kinematic body and logs the refusal from
  C++ without raising.
* ``RigidPrim._on_post_reset`` unconditionally writes velocities and restores
  the default pose, so the block must NOT be registered with ``world.scene``.
* ``ROS2Subscriber`` reports 0 for an ``Int32`` output before any message
  arrives, so the wire protocol must not use 0 to mean "seed 0".
"""

from pathlib import Path

from arx_r5_isaac_sim_bringup.vla.episode_events import (
    NO_SCENE_COMMAND,
    scene_command_for_request,
    scene_command_for_seed,
    seed_from_scene_command,
)


SIMULATION_SOURCE = (
    Path(__file__).resolve().parents[1]
    / 'arx_r5_isaac_sim_bringup'
    / 'simulation.py'
).read_text(encoding='utf-8')
DRIVER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / 'arx_r5_isaac_sim_bringup'
    / 'vla'
    / 'episode_driver.py'
).read_text(encoding='utf-8')


def _apply_seed_source() -> str:
    start = SIMULATION_SOURCE.index('    def apply_seed(self, seed: int, command: int)')
    end = SIMULATION_SOURCE.index('    def update(self) -> None:', start)
    return SIMULATION_SOURCE[start:end]


def test_vla_block_is_configured_dynamic():
    """A kinematic block would revert every teleport on the next step."""
    assert 'kinematic_object=task_config is None' in SIMULATION_SOURCE
    assert 'def _configure_authored_source_object(' in SIMULATION_SOURCE
    assert 'rigid_body.CreateKinematicEnabledAttr(kinematic)' \
        in SIMULATION_SOURCE


def test_apriltag_workcell_keeps_its_kinematic_block():
    """The 4:1 AprilTag block still needs the kinematic staging behaviour."""
    assert 'kinematic: bool = True' in SIMULATION_SOURCE
    assert 'kinematic_object: bool = True' in SIMULATION_SOURCE


def test_ccd_follows_the_kinematic_flag():
    """Continuous collision detection is invalid on a kinematic body."""
    assert 'physx_body.CreateEnableCCDAttr(not kinematic)' in SIMULATION_SOURCE


def test_reset_does_not_toggle_the_kinematic_flag():
    """Flipping to kinematic mid-reset is what made the teleport invisible."""
    source = _apply_seed_source()
    assert 'CreateKinematicEnabledAttr' not in source


def test_velocities_are_zeroed_before_the_pose_is_written():
    """Momentum carried across a teleport would launch the block."""
    source = _apply_seed_source()
    velocity_index = source.index('set_linear_velocity')
    pose_index = source.index('set_world_pose')
    assert velocity_index < pose_index


def test_block_is_not_registered_with_the_world_scene():
    """
    Scene registration puts the block under RigidPrim._on_post_reset.

    That hook unconditionally writes velocities and restores the default pose,
    which both errors on a kinematic body and undoes every scene reset. Only
    the articulation belongs in world.scene.
    """
    registrations = [
        line.strip()
        for line in SIMULATION_SOURCE.splitlines()
        if 'world.scene.add(' in line or 'scene.add(' in line
    ]
    assert not any('block' in line.lower() for line in registrations), \
        registrations


def test_block_wrapper_is_created_lazily():
    """
    The wrapper must be built while physics is running, not before reset.

    RigidPrim._on_physics_ready gives it a tensor handle at construction time
    when the simulation is already live, which is what makes the pose write go
    through PhysX instead of falling back to USD.
    """
    assert 'def _rigid_prim(self):' in SIMULATION_SOURCE
    assert 'if self._block_prim is None:' in SIMULATION_SOURCE


def test_reset_releases_any_existing_grasp():
    """A new episode must not begin with the previous block still attached."""
    source = _apply_seed_source()
    assert 'self._attachment.release()' in source


def test_scene_command_is_decoded_through_the_shared_protocol():
    """The simulator must not reinterpret the wire value on its own."""
    assert 'scene_request_from_command(value)' in SIMULATION_SOURCE
    assert 'from arx_r5_isaac_sim_bringup.vla.episode_events import' \
        in SIMULATION_SOURCE


def test_startup_default_does_not_request_a_reset():
    """The value the graph reports before any message must mean 'no command'."""
    assert seed_from_scene_command(NO_SCENE_COMMAND) is None
    assert scene_command_for_seed(0) != NO_SCENE_COMMAND


def test_same_seed_request_is_not_deduplicated_and_is_acknowledged():
    """The request token, not the seed, controls reset deduplication."""
    source = SIMULATION_SOURCE[
        SIMULATION_SOURCE.index('class VlaSceneController:'):
        SIMULATION_SOURCE.index('def _create_ros_action_graph(')
    ]
    assert 'self._last_command' in source
    assert 'if command == self._last_command:' in source
    assert '_set_scene_reset_ack(command)' in source
    assert scene_command_for_request(3, 1) != scene_command_for_request(3, 2)


def test_restarted_driver_does_not_always_repeat_token_one():
    """ROS may restart while Isaac Sim keeps its last applied command."""
    assert 'secrets.randbelow(SCENE_MAX_REQUEST_TOKEN) + 1' in DRIVER_SOURCE


def test_reset_ack_has_its_own_int32_publisher():
    """Attachment state must never be used as the reset completion signal."""
    assert 'SCENE_RESET_ACK_TOPIC,' in SIMULATION_SOURCE
    assert "SCENE_RESET_ACK_NODE = 'PublishSceneResetAck'" in SIMULATION_SOURCE
    assert "SCENE_RESET_ACK_NODE}.inputs:messageName', 'Int32'" in SIMULATION_SOURCE
    assert 'def _connect_scene_reset_ack_graph()' in SIMULATION_SOURCE


def test_reset_refuses_to_run_without_a_physics_handle():
    """Silently writing USD would look like success while changing nothing."""
    source = _apply_seed_source()
    assert 'if not self._require_physics_handle():' in source
    assert 'is_physics_handle_valid()' in SIMULATION_SOURCE


def test_reset_moves_the_placement_marker():
    """The visual target must follow the seed, not stay at the nominal pose."""
    source = _apply_seed_source()
    assert 'self._move_place_marker(place_center, place_yaw)' in source
    assert 'def _move_place_marker(' in SIMULATION_SOURCE


def test_marker_move_precedes_the_block_write():
    """Both must land before the log line reports the reset as done."""
    source = _apply_seed_source()
    marker = source.index('_move_place_marker(')
    log = source.index("f'[arx-r5-sim] scene reset seed=")
    assert marker < log


def test_block_mass_comes_from_the_task_config():
    """The VLA workcell owns its block geometry, so it owns the mass too."""
    assert 'object_mass_kg=(' in SIMULATION_SOURCE
    assert 'task_config.block.mass_kg' in SIMULATION_SOURCE
    assert 'config.source_object_mass_kg if mass_kg is None else mass_kg' \
        in SIMULATION_SOURCE


def test_reset_reports_the_pose_it_actually_achieved():
    """Logging the requested pose would hide a write that did not take."""
    source = _apply_seed_source()
    read_back = source.index('prim.get_world_pose()')
    log = source.index("f'[arx-r5-sim] scene reset seed=")
    assert read_back < log
    assert 'actual_position[0]' in source
    assert 'WARNING: block settled' in source
