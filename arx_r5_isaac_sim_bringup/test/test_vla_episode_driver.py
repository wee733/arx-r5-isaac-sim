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
Test the episode waypoint plan and the recorder event contract.

Only the ROS-free logic is covered: waypoint generation, the event payload and
the home-pose loader. Anything that needs a live cuMotion or MoveIt server
belongs in an integration run against Isaac Sim.
"""

import json
import math
from pathlib import Path

from arx_r5_isaac_sim_bringup.contracts import ARM_JOINTS
from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.episode_events import (
    ATTACHMENT_STATE_TOPIC,
    EPISODE_EVENT_TOPIC,
    EPISODE_HEARTBEAT_TOPIC,
    EpisodeEvent,
    EpisodeHeartbeat,
    EpisodePhase,
    EpisodeStatus,
    NO_SCENE_COMMAND,
    RECORDER_FAULT_TOPIC,
    RecorderFault,
    scene_command_for_request,
    scene_command_for_seed,
    SCENE_COMMAND_TOPIC,
    scene_request_from_command,
    seed_from_scene_command,
    tf_frame_from_prim_path,
)
from arx_r5_isaac_sim_bringup.vla.episode_plan import (
    build_episode_waypoints,
    load_home_positions,
    phase_rotation_candidates,
)
from arx_r5_isaac_sim_bringup.vla.task_config import (
    load_task_config,
    sample_scene_layout,
    world_to_base,
)

import pytest


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / 'config'
TASK_CONFIG = CONFIG_DIRECTORY / 'vla_task.yaml'
SCENE_CONFIG = CONFIG_DIRECTORY / 'vla_scene.yaml'
INITIAL_POSITIONS = CONFIG_DIRECTORY / 'vla_initial_positions.yaml'


@pytest.fixture(name='task')
def task_fixture():
    """Return the packaged VLA task configuration."""
    return load_task_config(TASK_CONFIG)


@pytest.fixture(name='scene')
def scene_fixture():
    """Return the packaged VLA USD scene contract."""
    return load_usd_scene_config(SCENE_CONFIG)


def test_waypoints_follow_the_pick_and_place_order(task):
    """The plan must approach, grasp, lift, transfer, place, then retreat."""
    phases = [phase for phase, _, _ in build_episode_waypoints(task, 3)]
    assert phases == [
        EpisodePhase.APPROACH,
        EpisodePhase.GRASP,
        EpisodePhase.LIFT,
        EpisodePhase.TRANSFER,
        EpisodePhase.PLACE,
        EpisodePhase.RETREAT,
    ]


def test_waypoints_are_deterministic(task):
    """The same seed must produce byte-identical waypoints."""
    assert build_episode_waypoints(task, 5) == \
        build_episode_waypoints(task, 5)
    assert build_episode_waypoints(task, 5) != \
        build_episode_waypoints(task, 6)


def test_waypoints_match_the_sampled_layout(task):
    """Waypoints must sit above the block and the place target the seed chose."""
    seed = 17
    (block_center, _), (place_center, _) = sample_scene_layout(task, seed)
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(task, seed)
    }
    for phase in (EpisodePhase.APPROACH, EpisodePhase.GRASP, EpisodePhase.LIFT):
        assert waypoints[phase][0] == pytest.approx(block_center[0])
        assert waypoints[phase][1] == pytest.approx(block_center[1])
    for phase in (
        EpisodePhase.TRANSFER,
        EpisodePhase.PLACE,
        EpisodePhase.RETREAT,
    ):
        assert waypoints[phase][0] == pytest.approx(place_center[0])
        assert waypoints[phase][1] == pytest.approx(place_center[1])


def test_approach_and_lift_clear_the_grasp(task):
    """Standoff waypoints must be strictly above the grasp they bracket."""
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(task, 21)
    }
    assert waypoints[EpisodePhase.APPROACH][2] - waypoints[EpisodePhase.GRASP][2] \
        == pytest.approx(task.grasp.approach_height)
    assert waypoints[EpisodePhase.LIFT][2] - waypoints[EpisodePhase.GRASP][2] \
        == pytest.approx(task.grasp.lift_height)
    assert waypoints[EpisodePhase.RETREAT][2] \
        - waypoints[EpisodePhase.PLACE][2] \
        == pytest.approx(
            task.grasp.lift_height - task.placement.release_clearance
        )


def test_transfer_keeps_the_block_lifted(task):
    """The transfer must happen at lift height, not by dragging the block."""
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(task, 9)
    }
    assert waypoints[EpisodePhase.TRANSFER][2] == pytest.approx(
        waypoints[EpisodePhase.LIFT][2]
    )


def test_transfer_offers_every_equivalent_source_grasp_rotation(task):
    """A lateral transfer may need another square-grasp IK branch."""
    waypoints = {
        phase: rotation
        for phase, _, rotation in build_episode_waypoints(task, 3)
    }
    source_rotation = task.equivalent_grasp_rotations(
        waypoints[EpisodePhase.GRASP]
    )[0]

    candidates = phase_rotation_candidates(
        task,
        EpisodePhase.TRANSFER,
        waypoints[EpisodePhase.TRANSFER],
        source_rotation,
        None,
    )

    assert candidates == task.equivalent_grasp_rotations(source_rotation)
    assert candidates[0] == source_rotation
    assert len(candidates) == 4


def test_acquisition_keeps_the_selected_grasp_rotation(task):
    """GRASP and LIFT must not rotate the wrist around a contacted block."""
    waypoints = {
        phase: rotation
        for phase, _, rotation in build_episode_waypoints(task, 3)
    }
    selected = task.equivalent_grasp_rotations(
        waypoints[EpisodePhase.APPROACH]
    )[2]

    for phase in (EpisodePhase.GRASP, EpisodePhase.LIFT):
        assert phase_rotation_candidates(
            task,
            phase,
            waypoints[phase],
            selected,
            None,
        ) == (selected,)


def test_top_grasp_carry_height_stays_inside_the_proven_ik_envelope(task):
    """Palm clearance must not push LIFT back above the solved R5A pose."""
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(task, 0)
    }

    assert waypoints[EpisodePhase.LIFT][2] == pytest.approx(0.525)
    assert waypoints[EpisodePhase.TRANSFER][2] == pytest.approx(0.525)
    assert waypoints[EpisodePhase.RETREAT][2] == pytest.approx(0.525)


def test_place_releases_above_the_support_collision_envelope(task):
    """The inflated official attached-object model must clear the platform."""
    waypoints = {
        phase: translation
        for phase, translation, _ in build_episode_waypoints(task, 9)
    }
    _, (place_center, place_yaw) = sample_scene_layout(task, 9)
    resting_pose, _ = task.grasp_pose_world(place_center, place_yaw)

    assert waypoints[EpisodePhase.PLACE][2] - resting_pose[2] == \
        pytest.approx(task.placement.release_clearance)
    assert task.placement.release_clearance >= 0.02
    assert waypoints[EpisodePhase.RETREAT][2] > \
        waypoints[EpisodePhase.PLACE][2]


def test_every_waypoint_orientation_is_a_unit_quaternion(task):
    """Malformed quaternions would be silently renormalized by the planner."""
    for _, _, rotation in build_episode_waypoints(task, 4):
        magnitude = math.sqrt(sum(value * value for value in rotation))
        assert magnitude == pytest.approx(1.0, abs=1e-9)


def test_all_waypoints_stay_inside_the_reach_envelope(task, scene):
    """No planned waypoint may sit beyond where cuMotion is known to solve."""
    for seed in range(25):
        for _, translation, _ in build_episode_waypoints(task, seed):
            base_point = world_to_base(
                translation,
                scene.expected_world_to_base_translation,
                scene.expected_world_to_base_rotation,
            )
            assert task.reach_envelope.violations(base_point, 'waypoint') == ()


def test_home_positions_cover_every_arm_joint():
    """The home pose must define all six arm joints in canonical order."""
    positions = load_home_positions(INITIAL_POSITIONS)
    assert len(positions) == len(ARM_JOINTS)
    assert all(math.isfinite(value) for value in positions)


def test_home_positions_reject_an_incomplete_file(tmp_path):
    """A truncated home pose must fail loudly rather than plan half an arm."""
    incomplete = tmp_path / 'incomplete.yaml'
    incomplete.write_text(
        'initial_positions:\n  joint1: 0.0\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='missing arm joints'):
        load_home_positions(incomplete)


def test_episode_event_round_trips():
    """Recorders must be able to parse exactly what the driver publishes."""
    event = EpisodeEvent(
        session_id='session-a',
        run_id='run-a',
        episode_id=2,
        seed=42,
        phase=EpisodePhase.GRASP,
        status=EpisodeStatus.RUNNING,
        instruction='pick up the block',
        stamp_sec=12.5,
        detail='',
        extra={'world_translation': [0.25, -0.03, 0.375]},
    )
    restored = EpisodeEvent.from_json(event.to_json())
    assert restored == event
    assert not restored.is_terminal


def test_heartbeat_round_trips_the_complete_episode_identity():
    """A lease renewal cannot accidentally keep a different seed alive."""
    heartbeat = EpisodeHeartbeat(
        session_id='session-a',
        run_id='run-a',
        episode_id=2,
        seed=42,
        stamp_sec=12.5,
    )

    assert EpisodeHeartbeat.from_json(heartbeat.to_json()) == heartbeat
    assert heartbeat.identity == ('session-a', 'run-a', 2, 42)


def test_recorder_fault_round_trips_writer_detail_and_identity():
    """The driver receives enough context to stop only the matching run."""
    fault = RecorderFault(
        recorder_name='raw_writer',
        session_id='session-a',
        run_id='run-a',
        episode_id=2,
        seed=42,
        stamp_sec=12.5,
        detail='disk full',
    )

    assert RecorderFault.from_json(fault.to_json()) == fault
    assert fault.identity == ('session-a', 'run-a', 2, 42)


@pytest.mark.parametrize(
    'payload_type',
    (EpisodeHeartbeat, RecorderFault),
)
def test_lease_and_fault_payloads_reject_non_finite_timestamps(payload_type):
    """Non-finite timestamps must not escape onto control topics."""
    fields = {
        'session_id': 'session-a',
        'run_id': 'run-a',
        'episode_id': 2,
        'seed': 42,
        'stamp_sec': float('nan'),
    }
    if payload_type is RecorderFault:
        fields.update(recorder_name='writer', detail='failed')
    with pytest.raises(ValueError, match='stamp_sec'):
        payload_type(**fields)


def test_episode_identity_includes_the_scene_seed():
    """A mismatched deterministic layout cannot join the active episode."""
    event = EpisodeEvent(
        session_id='session-a',
        run_id='run-a',
        episode_id=2,
        seed=42,
        phase=EpisodePhase.OBSERVE,
        status=EpisodeStatus.RUNNING,
        instruction='pick up the block',
        stamp_sec=12.5,
    )
    changed_seed = EpisodeEvent(
        session_id=event.session_id,
        run_id=event.run_id,
        episode_id=event.episode_id,
        seed=43,
        phase=event.phase,
        status=event.status,
        instruction=event.instruction,
        stamp_sec=event.stamp_sec,
    )

    assert event.identity == ('session-a', 'run-a', 2, 42)
    assert changed_seed.identity != event.identity


def test_episode_event_reports_terminal_status():
    """Recorders close an episode on the terminal event, so it must be flagged."""
    for status in (
        EpisodeStatus.SUCCEEDED,
        EpisodeStatus.FAILED,
        EpisodeStatus.ABORTED,
    ):
        event = EpisodeEvent(
            session_id='session-a',
            run_id='run-a',
            episode_id=0,
            seed=0,
            phase=EpisodePhase.HOME,
            status=status,
            instruction='x',
            stamp_sec=0.0,
        )
        assert event.is_terminal


def test_episode_event_rejects_incomplete_payloads():
    """A malformed event must not silently produce a half-populated record."""
    with pytest.raises(ValueError, match='missing fields'):
        EpisodeEvent.from_json(json.dumps({'episode_id': 1}))
    with pytest.raises(ValueError, match='must be a JSON object'):
        EpisodeEvent.from_json(json.dumps([1, 2, 3]))


@pytest.mark.parametrize(
    'payload',
    (
        {
            'session_id': 's',
            'run_id': 'r',
            'episode_id': 0,
            'seed': 0,
            'phase': EpisodePhase.OBSERVE,
            'status': EpisodeStatus.RUNNING,
            'instruction': 'pick',
            'stamp_sec': float('nan'),
        },
        {
            'session_id': 's',
            'run_id': 'r',
            'episode_id': 0,
            'seed': 0,
            'phase': 'not-a-phase',
            'status': EpisodeStatus.RUNNING,
            'instruction': 'pick',
            'stamp_sec': 1.0,
        },
        {
            'session_id': 's',
            'run_id': 'r',
            'episode_id': 0,
            'seed': 0,
            'phase': EpisodePhase.OBSERVE,
            'status': 'not-a-status',
            'instruction': 'pick',
            'stamp_sec': 1.0,
        },
    ),
)
def test_episode_event_rejects_non_finite_or_unknown_fields(payload):
    """Malformed lifecycle messages must not reach recorder hooks."""
    with pytest.raises(ValueError):
        EpisodeEvent.from_json(json.dumps(payload, allow_nan=True))


def test_episode_event_rejects_non_object_extra():
    """The optional metadata field must remain a JSON object."""
    payload = {
        'session_id': 's',
        'run_id': 'r',
        'episode_id': 0,
        'seed': 0,
        'phase': EpisodePhase.OBSERVE,
        'status': EpisodeStatus.RUNNING,
        'instruction': 'pick',
        'stamp_sec': 1.0,
        'extra': [],
    }
    with pytest.raises(ValueError, match='extra'):
        EpisodeEvent.from_json(json.dumps(payload))


def test_event_topics_are_absolute():
    """Relative names would land under each node's namespace."""
    assert SCENE_COMMAND_TOPIC.startswith('/')
    assert EPISODE_EVENT_TOPIC.startswith('/')
    assert EPISODE_HEARTBEAT_TOPIC.startswith('/')
    assert RECORDER_FAULT_TOPIC.startswith('/')
    assert ATTACHMENT_STATE_TOPIC.startswith('/')


def test_no_scene_command_is_distinguishable_from_seed_zero():
    """
    Isaac Sim's subscriber reports 0 before any message arrives.

    Without the +1 offset the simulator would apply seed 0's layout at startup
    and then ignore a genuine seed-0 command as a duplicate -- and 0 is the
    default start seed.
    """
    assert seed_from_scene_command(NO_SCENE_COMMAND) is None
    assert scene_command_for_seed(0) != NO_SCENE_COMMAND
    assert seed_from_scene_command(scene_command_for_seed(0)) == 0


def test_scene_command_round_trips_every_seed():
    """Whatever the driver publishes must decode to the seed it ran."""
    for seed in range(0, 200):
        command = scene_command_for_seed(seed)
        assert command > NO_SCENE_COMMAND
        assert seed_from_scene_command(command) == seed


def test_consecutive_seeds_produce_distinct_commands():
    """Adjacent episodes must not be dropped as duplicate wire values."""
    commands = [scene_command_for_seed(seed) for seed in range(10)]
    assert len(set(commands)) == len(commands)


def test_same_seed_with_new_request_token_is_a_new_command():
    """A retry must reach Isaac Sim even when its deterministic layout repeats."""
    first = scene_command_for_request(12, 41)
    retry = scene_command_for_request(12, 42)
    assert first != retry
    assert scene_request_from_command(first) == (12, 41)
    assert scene_request_from_command(retry) == (12, 42)


def test_invalid_packed_request_is_not_a_reset():
    """A missing token or seed field must remain distinguishable from a request."""
    assert scene_request_from_command(1) is None
    assert scene_request_from_command(1 << 16) is None


def test_negative_seeds_are_rejected():
    """A negative seed would encode into the reserved no-command value."""
    with pytest.raises(ValueError, match='must not be negative'):
        scene_command_for_seed(-1)


def test_stale_or_unset_commands_decode_to_nothing():
    """Any value at or below the sentinel means no reset was requested."""
    for command in (NO_SCENE_COMMAND, -1, -100):
        assert seed_from_scene_command(command) is None


def test_phase_order_covers_every_phase():
    """EpisodePhase.ORDER is the documented lifecycle; keep it complete."""
    declared = {
        value
        for name, value in vars(EpisodePhase).items()
        if not name.startswith('_') and isinstance(value, str)
    }
    assert declared == set(EpisodePhase.ORDER)


def test_observe_and_verify_place_bound_the_recorded_episode():
    """Observation starts immediately and placement is verified after retreat."""
    assert EpisodePhase.ORDER.index(EpisodePhase.OBSERVE) < \
        EpisodePhase.ORDER.index(EpisodePhase.APPROACH)
    assert EpisodePhase.ORDER.index(EpisodePhase.RETREAT) < \
        EpisodePhase.ORDER.index(EpisodePhase.VERIFY_PLACE)


@pytest.mark.parametrize(
    ('prim_path', 'expected'),
    (
        ('/World/Workspace/Block', 'Block'),
        ('/World/Objects/RedBlock', 'RedBlock'),
        ('/World/Objects/RedBlock/', 'RedBlock'),
    ),
)
def test_tf_frame_is_derived_from_the_object_prim(prim_path, expected):
    """Ground-truth TF follows the configured object, not a Block constant."""
    assert tf_frame_from_prim_path(prim_path) == expected


@pytest.mark.parametrize('prim_path', ('', '/', 'Block', '///'))
def test_tf_frame_rejects_paths_without_a_leaf(prim_path):
    """A malformed object path must fail at startup rather than lose TF."""
    with pytest.raises(ValueError):
        tf_frame_from_prim_path(prim_path)
