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
Test dual-camera frame pairing.

Both Isaac Sim cameras are driven by the same OnPhysicsStep node and stamp with
simulation time, so their images arrive with identical timestamps. The pairer
turns that into complete frames; getting it wrong would silently record
observations where the two views are from different instants.
"""

from dataclasses import dataclass
from pathlib import Path

from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.frame_sync import (
    canonical_joint_positions,
    CausalSampleBuffer,
    DEPTH_MODALITY,
    LATEST_JOINT_COMMAND_POLICY,
    LATEST_JOINT_STATE_POLICY,
    stamp_ns,
    StampPairer,
)

import pytest


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / 'config'
SCENE_CONFIG = CONFIG_DIRECTORY / 'vla_scene.yaml'
CAMERAS = ('zedx', 'd455')


@dataclass
class _Stamp:
    sec: int
    nanosec: int


@dataclass
class _Header:
    stamp: _Stamp


@dataclass
class _StampedMessage:
    """Stand-in for sensor_msgs/Image; the pairer only reads header.stamp."""

    header: _Header


@dataclass
class _JointSample:
    """Stand-in for sensor_msgs/JointState used by causal tests."""

    header: _Header
    name: list
    position: list


def _image(timestamp_ns: int) -> _StampedMessage:
    return _StampedMessage(header=_Header(stamp=_Stamp(
        sec=timestamp_ns // 1_000_000_000,
        nanosec=timestamp_ns % 1_000_000_000,
    )))


def _joint_sample(timestamp_ns: int, offset: float = 0.0) -> _JointSample:
    """Create a deliberately non-canonical publisher order."""
    return _JointSample(
        header=_image(timestamp_ns).header,
        name=[
            'joint3', 'joint1', 'joint8', 'joint7',
            'joint2', 'joint4', 'joint6', 'joint5',
        ],
        position=[
            3.0 + offset, 1.0 + offset, 8.0 + offset, 7.0 + offset,
            2.0 + offset, 4.0 + offset, 6.0 + offset, 5.0 + offset,
        ],
    )


def test_frame_completes_only_when_both_cameras_report():
    """A single camera must never produce a frame on its own."""
    pairer = StampPairer(CAMERAS)
    assert pairer.add('zedx', _image(1_000_000_000)) is None
    frame = pairer.add('d455', _image(1_000_000_000))
    assert frame is not None
    assert sorted(frame.images) == ['d455', 'zedx']
    assert frame.stamp_ns == 1_000_000_000


def test_frames_pair_by_stamp_not_arrival_order():
    """Interleaved arrivals must still pair by their timestamps."""
    pairer = StampPairer(CAMERAS)
    assert pairer.add('zedx', _image(2_000_000_000)) is None
    assert pairer.add('zedx', _image(3_000_000_000)) is None
    frame = pairer.add('d455', _image(3_000_000_000))
    assert frame is not None
    assert frame.stamp_ns == 3_000_000_000


def test_stamps_within_tolerance_still_pair():
    """A non-simulation camera pair may use an explicit tolerance."""
    pairer = StampPairer(CAMERAS, tolerance_ns=1_000_000)
    assert pairer.add('zedx', _image(6_000_000_000)) is None
    frame = pairer.add('d455', _image(6_000_500_000))
    assert frame is not None
    assert frame.stamp_ns == 6_000_000_000


def test_stamps_beyond_tolerance_do_not_pair():
    """Genuinely different instants must not be merged into one frame."""
    pairer = StampPairer(CAMERAS, tolerance_ns=1_000_000)
    assert pairer.add('zedx', _image(7_000_000_000)) is None
    assert pairer.add('d455', _image(7_100_000_000)) is None


def test_older_bucket_is_retained_after_newer_frame_completes():
    """Late partners may complete an older bucket before the pending limit."""
    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(4_000_000_000))
    pairer.add('zedx', _image(5_000_000_000))
    frame = pairer.add('d455', _image(5_000_000_000))
    assert frame is not None
    assert pairer.pending_buckets == 1
    older_frame = pairer.add('d455', _image(4_000_000_000))
    assert older_frame is not None
    assert older_frame.stamp_ns == 4_000_000_000


def test_pending_buckets_are_bounded():
    """A stopped camera must not let the pairer grow without limit."""
    pairer = StampPairer(CAMERAS, max_pending=4)
    for index in range(20):
        assert pairer.add('zedx', _image(index * 1_000_000_000)) is None
    assert pairer.dropped >= 15
    assert pairer.pending_buckets == 4


def test_pairer_statistics_reset_at_episode_boundary():
    """A new episode cannot report drops or pairs from its predecessor."""
    pairer = StampPairer(CAMERAS, max_pending=1)
    pairer.add('zedx', _image(1_000_000_000))
    pairer.add('zedx', _image(2_000_000_000))
    assert pairer.overflow_buckets == 1
    assert pairer.received_by_camera['zedx'] == 2
    pairer.reset()
    assert pairer.paired_frames == 0
    assert pairer.overflow_buckets == 0
    assert pairer.pending_buckets == 0
    assert pairer.received_by_camera == {'zedx': 0, 'd455': 0}


def test_duplicate_image_is_counted_per_camera():
    """Duplicate delivery is visible without preventing the eventual pair."""
    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(1_000_000_000))
    pairer.add('zedx', _image(1_000_000_000))
    frame = pairer.add('d455', _image(1_000_000_000))
    assert frame is not None
    assert pairer.paired_frames == 1
    assert pairer.duplicate_by_camera == {'zedx': 1, 'd455': 0}


def test_duplicate_delivery_after_frame_completion_is_not_emitted_again():
    """A fully duplicated DDS sample cannot create a second training row."""
    pairer = StampPairer(CAMERAS)
    timestamp = 1_500_000_000
    assert pairer.add('zedx', _image(timestamp)) is None
    assert pairer.add('d455', _image(timestamp)) is not None
    assert pairer.add('zedx', _image(timestamp)) is None
    assert pairer.add('d455', _image(timestamp)) is None
    assert pairer.paired_frames == 1
    assert pairer.duplicate_by_camera == {'zedx': 1, 'd455': 1}


def test_exact_completed_lookup_does_not_iterate_the_completed_set():
    """The 30 Hz exact path must remain constant-time as an episode grows."""
    class NoIterationSet(set):
        def __iter__(self):
            raise AssertionError('exact completed lookup iterated the set')

    pairer = StampPairer(CAMERAS)
    timestamp = 1_600_000_000
    pairer._completed_stamps = NoIterationSet({timestamp})

    assert pairer.add('zedx', _image(timestamp)) is None
    assert pairer.duplicate_by_camera['zedx'] == 1


def test_pairer_reports_out_of_order_completed_frames():
    """Late camera partners must be visible to the eventual writer."""
    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(2_000_000_000))
    pairer.add('zedx', _image(3_000_000_000))
    assert pairer.add('d455', _image(3_000_000_000)) is not None
    assert pairer.add('d455', _image(2_000_000_000)) is not None
    assert pairer.out_of_order_frames == 1


def test_pairer_reports_stream_rate_and_nearest_camera_skew():
    """Diagnostics distinguish cadence mismatch from a fixed stamp offset."""
    pairer = StampPairer(CAMERAS)
    for index in range(4):
        pairer.add('zedx', _image(index * 100_000_000))
        pairer.add('d455', _image(index * 100_000_000 + 2_000_000))

    assert pairer.stream_rate_hz('zedx') == pytest.approx(10.0)
    summary = pairer.nearest_camera_skew_summary()[('zedx', 'd455')]
    assert summary['median_ns'] == pytest.approx(2_000_000)
    assert summary['max_ns'] == pytest.approx(2_000_000)


def test_simulation_default_requires_exact_stamp():
    """The collection default must not silently approximate sim timestamps."""
    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(1_000_000_000))
    assert pairer.add('d455', _image(1_000_000_001)) is None


def test_rgbd_frame_completes_only_after_all_four_streams_arrive():
    """Both depth images participate in the same barrier as both RGBs."""
    pairer = StampPairer(CAMERAS, modalities=('color', 'depth'))
    timestamp = 8_000_000_000
    assert pairer.add('zedx', _image(timestamp)) is None
    assert pairer.add('d455', _image(timestamp)) is None
    assert pairer.add(
        'zedx', _image(timestamp), modality=DEPTH_MODALITY
    ) is None
    frame = pairer.add(
        'd455', _image(timestamp), modality=DEPTH_MODALITY
    )
    assert frame is not None
    assert sorted(frame.images) == ['d455', 'zedx']
    assert sorted(frame.depth_images) == ['d455', 'zedx']
    assert {
        stamp_ns(message)
        for message in (*frame.images.values(), *frame.depth_images.values())
    } == {timestamp}


def test_rgbd_pairer_never_reuses_latest_depth_from_another_stamp():
    """Depth from a previous tick cannot leak into a newer RGB frame."""
    pairer = StampPairer(CAMERAS, modalities=('color', 'depth'))
    old_stamp = 9_000_000_000
    new_stamp = 10_000_000_000
    for camera in CAMERAS:
        assert pairer.add(
            camera, _image(old_stamp), modality=DEPTH_MODALITY
        ) is None
    for camera in CAMERAS:
        assert pairer.add(camera, _image(new_stamp)) is None
    assert pairer.paired_frames == 0
    assert pairer.pending_buckets == 2
    assert pairer.pending_images == 4


def test_rgbd_pairer_rejects_approximate_timestamp_configuration():
    """Collection must not weaken the four-stream exact-stamp contract."""
    with pytest.raises(ValueError, match='exact timestamps'):
        StampPairer(
            CAMERAS,
            tolerance_ns=1,
            modalities=('color', 'depth'),
        )


def test_rgbd_reset_discards_pending_color_and_depth_streams():
    """Episode STARTED can clear every partial multimodal observation."""
    pairer = StampPairer(CAMERAS, modalities=('color', 'depth'))
    pairer.add('zedx', _image(11_000_000_000))
    pairer.add(
        'd455',
        _image(11_000_000_000),
        modality=DEPTH_MODALITY,
    )
    assert pairer.pending_images == 2
    pairer.reset()
    assert pairer.pending_buckets == 0
    assert pairer.pending_images == 0
    assert all(count == 0 for count in pairer.received_by_stream.values())


def test_frame_marks_joint_state_as_latest_not_exactly_synchronized():
    """Dataset writers can distinguish latest-state sampling from sync."""
    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(12_000_000_000))
    frame = pairer.add('d455', _image(12_000_000_000))
    assert frame is not None
    frame.attach_latest_joint_state(_image(11_999_000_000))
    assert frame.joint_state_sync_policy == LATEST_JOINT_STATE_POLICY
    assert frame.joint_state_stamp_ns == 11_999_000_000
    assert frame.joint_state_time_offset_ns == -1_000_000


def test_frame_exposes_the_absolute_controller_command_as_action():
    """GR00T action comes from /isaac_joint_commands, not feedback state."""
    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(12_000_000_000))
    frame = pairer.add('d455', _image(12_000_000_000))
    assert frame is not None
    frame.attach_latest_joint_command(_joint_sample(11_999_000_000))
    assert frame.joint_command_sync_policy == LATEST_JOINT_COMMAND_POLICY
    assert frame.joint_command_stamp_ns == 11_999_000_000
    assert frame.joint_command_time_offset_ns == -1_000_000


def test_causal_buffer_never_selects_future_feedback_or_action():
    """A later controller callback cannot leak into an earlier image row."""
    buffer = CausalSampleBuffer()
    buffer.add(_joint_sample(140))
    buffer.add(_joint_sample(90))
    buffer.add(_joint_sample(120))

    assert stamp_ns(buffer.latest_at(130)) == 120
    assert buffer.latest_at(80) is None

    pairer = StampPairer(CAMERAS)
    pairer.add('zedx', _image(130))
    frame = pairer.add('d455', _image(130))
    with pytest.raises(ValueError, match='newer than the camera frame'):
        frame.attach_latest_joint_command(_joint_sample(140))


def test_causal_buffer_prefers_latest_arrival_at_the_same_timestamp():
    """Sequence order deterministically breaks equal-stamp ties."""
    buffer = CausalSampleBuffer()
    first = _joint_sample(100, offset=1.0)
    second = _joint_sample(100, offset=2.0)
    buffer.add(first)
    buffer.add(second)

    assert buffer.latest_at(100) is second


def test_causal_buffer_trims_the_oldest_stamps_after_out_of_order_insert():
    """Capacity keeps the newest causal samples, not the newest arrivals."""
    buffer = CausalSampleBuffer(max_samples=3)
    for timestamp in (30, 10, 40, 20):
        buffer.add(_joint_sample(timestamp))

    assert buffer.latest_at(19) is None
    assert stamp_ns(buffer.latest_at(20)) == 20
    assert stamp_ns(buffer.latest_at(100)) == 40


def test_diagnostic_stamp_history_is_bounded_and_counts_truncation():
    """Rate/skew diagnostics cannot grow without a frame-count bound."""
    pairer = StampPairer(CAMERAS, max_stamp_history=3)
    for timestamp in range(5):
        pairer.add('zedx', _image(timestamp))

    stream = ('zedx', 'color')
    assert tuple(pairer._stamp_history_by_stream[stream]) == (2, 3, 4)
    assert pairer.stamp_history_truncations_by_stream[stream] == 2


def test_joint_samples_are_reordered_to_the_canonical_six_plus_one_vector():
    """Names, not publisher array order, define state/action dimensions."""
    assert canonical_joint_positions(_joint_sample(1)) == pytest.approx(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    )


def test_joint_sample_validation_rejects_missing_or_non_finite_values():
    """Malformed feedback cannot silently enter a training episode."""
    missing = _joint_sample(1)
    missing.name.remove('joint7')
    missing.position.pop(3)
    with pytest.raises(ValueError, match='missing required joints'):
        canonical_joint_positions(missing)

    non_finite = _joint_sample(1)
    non_finite.position[0] = float('nan')
    with pytest.raises(ValueError, match='NaN or Inf'):
        canonical_joint_positions(non_finite)


def test_recorder_subscribes_to_post_controller_absolute_commands():
    """The recorder interface must contain the model's required q_cmd."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'recorder_base.py'
    ).read_text(encoding='utf-8')
    assert "ABSOLUTE_JOINT_COMMANDS_TOPIC = f'/{JOINT_COMMANDS_TOPIC}'" in source
    assert 'self._on_joint_command' in source
    assert 'frame.attach_latest_joint_command(' in source


def test_unknown_camera_is_rejected():
    """A topic wired to the wrong camera name must fail loudly."""
    pairer = StampPairer(CAMERAS)
    with pytest.raises(KeyError):
        pairer.add('unknown', _image(0))


def test_unknown_modality_is_rejected():
    """A wrongly wired image stream must fail loudly."""
    pairer = StampPairer(CAMERAS)
    with pytest.raises(KeyError, match='unknown modality'):
        pairer.add('zedx', _image(0), modality='depth')


def test_pairer_requires_at_least_one_camera():
    """An empty camera set would emit frames with no observations at all."""
    with pytest.raises(ValueError, match='at least one camera'):
        StampPairer(())


def test_pairer_requires_color_modality():
    """Frame.images cannot be constructed without an RGB stream."""
    with pytest.raises(ValueError, match='color modality'):
        StampPairer(CAMERAS, modalities=('depth',))


def test_pairer_rejects_modalities_it_cannot_return():
    """A required stream must not be silently dropped from the Frame."""
    with pytest.raises(ValueError, match='unsupported modalities'):
        StampPairer(CAMERAS, modalities=('color', 'infrared'))


def test_recorder_gates_images_before_the_pairer():
    """Inter-episode images must not enter pending synchronization buckets."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'recorder_base.py'
    ).read_text(encoding='utf-8')
    callback_start = source.index('    def _add_camera_sample')
    callback_end = source.index('    def _on_joint_state', callback_start)
    callback = source[callback_start:callback_end]
    assert callback.index('if not self._episode_open:') < callback.index(
        'self._pairer.add(camera_name, message, modality=modality)'
    )


def test_recorder_resets_pairer_on_episode_start():
    """STARTED begins both frame and synchronization accounting anew."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'recorder_base.py'
    ).read_text(encoding='utf-8')
    handler_start = source.index('    def _on_episode_event')
    handler_end = source.index(
        '    # -------------------------------------------------------------- '
        'hooks',
        handler_start,
    )
    handler = source[handler_start:handler_end]
    started = handler.index('if event.status == EpisodeStatus.STARTED:')
    reset = handler.index('self._pairer.reset()', started)
    frames = handler.index('self.frames_recorded = 0', started)
    assert reset < frames


def test_recorder_rejects_images_older_than_started_event():
    """Queued setup frames cannot leak across the STARTED boundary."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'recorder_base.py'
    ).read_text(encoding='utf-8')
    callback_start = source.index('    def _add_camera_sample')
    callback_end = source.index('    def _on_joint_state', callback_start)
    callback = source[callback_start:callback_end]
    assert 'self._episode_start_stamp_ns' in callback
    assert 'stamp_ns(message) <= self._episode_start_stamp_ns' in callback


def test_recorder_labels_frames_from_event_timestamp_history():
    """Queued images use their simulation time, not callback arrival order."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'recorder_base.py'
    ).read_text(encoding='utf-8')
    assert 'frame.event = self._event_for_stamp(frame.stamp_ns)' in source
    assert 'if event_stamp_ns <= frame_stamp_ns:' in source
    assert 'self._event_history.append((event_stamp_ns, event))' in source


def test_recorder_drains_late_tail_frames_before_terminal_close():
    """Terminal callback arrival must not immediately discard camera tail."""
    source = (
        Path(__file__).resolve().parents[1]
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'recorder_base.py'
    ).read_text(encoding='utf-8')
    terminal = source.index('self._terminal_event = event')
    deadline = source.index('self._terminal_drain_deadline = (', terminal)
    close = source.index('def _finish_terminal_drain', terminal)
    assert terminal < deadline < close
    assert 'stamp_ns(message) > self._terminal_stamp_ns' in source
    assert 'self._post_terminal_dropped_images += 1' in source
    assert 'ClockType.STEADY_TIME' in source
    assert 'def finalize_active_episode_on_shutdown' in source
    assert 'node.finalize_active_episode_on_shutdown()' in source


def test_scene_contract_declares_both_collection_cameras():
    """Collection records two views; the contract must provide both."""
    scene = load_usd_scene_config(SCENE_CONFIG)
    assert sorted(scene.cameras) == ['d455', 'zedx']
    topics = [
        camera.color_image_topic for camera in scene.cameras.values()
    ]
    assert len(set(topics)) == len(topics)


def test_camera_frame_skip_counts_match():
    """Mismatched frame skipping would break the shared-tick guarantee."""
    scene = load_usd_scene_config(SCENE_CONFIG)
    skip_counts = {
        camera.frame_skip_count for camera in scene.cameras.values()
    }
    assert len(skip_counts) == 1


def test_vla_collection_publishes_the_declared_30_hz_camera_rate():
    """A synchronized 10 Hz stream is still the wrong training contract."""
    scene = load_usd_scene_config(SCENE_CONFIG)
    assert all(
        camera.frame_skip_count == 0
        for camera in scene.cameras.values()
    )
