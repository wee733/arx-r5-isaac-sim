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

"""Test the persistent ARX raw-v1 VLA recorder and its ROS adapter."""

import json
from pathlib import Path
from types import SimpleNamespace

from arx_r5_isaac_sim_bringup.vla.episode_events import (
    EpisodeEvent,
    EpisodePhase,
    EpisodeStatus,
)
from arx_r5_isaac_sim_bringup.vla.frame_sync import Frame
from arx_r5_isaac_sim_bringup.vla.raw_v1_recorder import (
    image_to_rgb,
    RawV1Recorder,
)
from arx_r5_isaac_sim_bringup.vla.raw_v1_writer import ArxRawRecorder

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _EncoderInput:
    """Collect raw RGB bytes and materialize a stand-in video on close."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.data = bytearray()
        self.closed = False

    def write(self, payload):
        if self.closed:
            raise BrokenPipeError('encoder input is closed')
        self.data.extend(payload)
        return len(payload)

    def close(self):
        if self.closed:
            return
        self.output_path.write_bytes(bytes(self.data))
        self.closed = True


class _EncoderProcess:
    """Provide the subprocess methods exercised by ArxRawRecorder."""

    def __init__(self, output_path):
        self.stdin = _EncoderInput(output_path)
        self.return_code = None

    def wait(self, timeout):
        del timeout
        self.return_code = 0
        return self.return_code

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = -15

    def kill(self):
        self.return_code = -9


class _Logger:
    """Accept adapter lifecycle logs without constructing a ROS node."""

    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class _RawLifecycleHarness:
    """Record which raw writer lifecycle method the adapter selects."""

    def __init__(self, root):
        self.root = Path(root)
        self.active = False
        self.active_frame_count = 0
        self.started = []
        self.ended = []
        self.aborted = 0

    def start_episode(self, task, *, metadata):
        self.active = True
        self.started.append((task, metadata))
        return 7

    def end_episode(self, *, notes):
        self.active = False
        self.ended.append(notes)
        return 7

    def abort_episode(self):
        self.active = False
        self.aborted += 1


class _AdapterLifecycleHarness(RawV1Recorder):
    """Bypass Node initialization while exercising adapter hooks."""

    def __init__(self, raw):
        self._raw = raw
        self._raw_episode_index = None
        self._logger = _Logger()

    def get_logger(self):
        return self._logger


@pytest.fixture
def fake_encoder(monkeypatch):
    """Replace only the external encoder while retaining writer behavior."""
    monkeypatch.setattr(
        ArxRawRecorder,
        '_resolve_encoder',
        lambda _self, _requested: 'libx264',
    )

    def start_encoder(recorder, camera, _height, _width):
        output = recorder._active['temp_dir'] / f'{camera}.mp4'
        return _EncoderProcess(output)

    monkeypatch.setattr(ArxRawRecorder, '_start_encoder', start_encoder)


def _image_message(rgb, encoding, *, stamp_ns=2_000_000_000, padding=3):
    """Encode RGB pixels as one padded sensor_msgs/Image stand-in."""
    if encoding == 'rgb8':
        encoded = rgb
    elif encoding == 'bgr8':
        encoded = rgb[..., ::-1]
    elif encoding == 'rgba8':
        alpha = np.full((*rgb.shape[:2], 1), 123, dtype=np.uint8)
        encoded = np.concatenate((rgb, alpha), axis=2)
    elif encoding == 'bgra8':
        alpha = np.full((*rgb.shape[:2], 1), 123, dtype=np.uint8)
        encoded = np.concatenate((rgb[..., ::-1], alpha), axis=2)
    else:  # pragma: no cover - test helper guard
        raise AssertionError(encoding)
    packed = np.ascontiguousarray(encoded)
    row_bytes = packed.shape[1] * packed.shape[2]
    rows = np.full(
        (packed.shape[0], row_bytes + padding),
        255,
        dtype=np.uint8,
    )
    rows[:, :row_bytes] = packed.reshape(packed.shape[0], row_bytes)
    stamp = SimpleNamespace(
        sec=stamp_ns // 1_000_000_000,
        nanosec=stamp_ns % 1_000_000_000,
    )
    return SimpleNamespace(
        encoding=encoding,
        width=packed.shape[1],
        height=packed.shape[0],
        step=row_bytes + padding,
        data=rows.tobytes(),
        header=SimpleNamespace(stamp=stamp),
    )


def _joint_message(offset=0.0):
    """Return an eight-joint sample in deliberately scrambled order."""
    return SimpleNamespace(
        name=[
            'joint3', 'joint8', 'joint1', 'joint7',
            'joint5', 'joint2', 'joint6', 'joint4',
        ],
        position=[
            3.0 + offset, 800.0 + offset, 1.0 + offset, 0.04 + offset,
            5.0 + offset, 2.0 + offset, 6.0 + offset, 4.0 + offset,
        ],
    )


def _event(status=EpisodeStatus.STARTED):
    """Return one valid episode event for adapter lifecycle tests."""
    return EpisodeEvent(
        session_id='session-a',
        run_id='run-a',
        episode_id=4,
        seed=17,
        phase=EpisodePhase.RESET,
        status=status,
        instruction='pick up the red block',
        stamp_sec=1.0,
        detail='verified' if status == EpisodeStatus.SUCCEEDED else '',
        extra={'layout': 'left'},
    )


@pytest.mark.parametrize('encoding', ('rgb8', 'bgr8', 'rgba8', 'bgra8'))
def test_image_adapter_decodes_supported_encodings_and_row_padding(encoding):
    """Every Isaac ROS color encoding becomes contiguous RGB pixels."""
    expected = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    result = image_to_rgb(_image_message(expected, encoding))

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous


def test_adapter_extracts_named_joint7_and_discards_mimic_joint8():
    """Publisher array order and joint8 can never redefine model columns."""
    result = RawV1Recorder._positions(_joint_message(), 'joint command')

    np.testing.assert_allclose(
        result,
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.04]),
    )
    assert result.dtype == np.float32


def test_adapter_appends_rgb_state_action_and_original_camera_stamps():
    """A synchronized Frame is translated without substituting feedback."""
    calls = []
    adapter = RawV1Recorder.__new__(RawV1Recorder)
    adapter._front_camera = 'zedx'
    adapter._wrist_camera = 'd455'
    adapter._raw = SimpleNamespace(append=lambda **kwargs: calls.append(kwargs))
    front_rgb = np.full((2, 3, 3), 10, dtype=np.uint8)
    wrist_rgb = np.full((2, 3, 3), 20, dtype=np.uint8)
    frame = Frame(
        stamp_ns=2_000_000_000,
        images={
            'zedx': _image_message(
                front_rgb,
                'rgba8',
                stamp_ns=2_000_000_000,
            ),
            'd455': _image_message(
                wrist_rgb,
                'bgr8',
                stamp_ns=2_000_500_000,
            ),
        },
        joint_state=_joint_message(),
        joint_command=_joint_message(offset=10.0),
    )

    adapter.on_frame(frame)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]['front_rgb'], front_rgb)
    np.testing.assert_array_equal(calls[0]['wrist_rgb'], wrist_rgb)
    np.testing.assert_allclose(
        calls[0]['state'],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.04],
    )
    np.testing.assert_allclose(
        calls[0]['action'],
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 10.04],
    )
    assert calls[0]['timestamp'] == pytest.approx(2.0)
    assert calls[0]['front_timestamp'] == pytest.approx(2.0)
    assert calls[0]['wrist_timestamp'] == pytest.approx(2.0005)


@pytest.mark.parametrize('missing_field', ('joint_state', 'joint_command'))
def test_adapter_rejects_frames_without_state_or_action(missing_field):
    """An incomplete training row faults instead of writing guessed values."""
    adapter = RawV1Recorder.__new__(RawV1Recorder)
    adapter._front_camera = 'zedx'
    adapter._wrist_camera = 'd455'
    adapter._raw = SimpleNamespace(append=lambda **_kwargs: None)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    values = {
        'joint_state': _joint_message(),
        'joint_command': _joint_message(),
    }
    values[missing_field] = None
    frame = Frame(
        stamp_ns=2_000_000_000,
        images={
            'zedx': _image_message(rgb, 'rgb8'),
            'd455': _image_message(rgb, 'rgb8'),
        },
        **values,
    )

    with pytest.raises(ValueError, match='unavailable'):
        adapter.on_frame(frame)


def test_adapter_commits_only_succeeded_episode_status(tmp_path):
    """Success commits while every non-success terminal status is discarded."""
    raw = _RawLifecycleHarness(tmp_path / 'arx_raw')
    adapter = _AdapterLifecycleHarness(raw)
    started = _event()
    adapter.on_episode_start(started)
    raw.active_frame_count = 16

    adapter.on_episode_end(_event(EpisodeStatus.SUCCEEDED), 16)

    assert raw.started[0][0] == started.instruction
    assert raw.started[0][1]['source_episode_id'] == started.episode_id
    assert raw.started[0][1]['seed'] == started.seed
    assert raw.ended == ['verified']
    assert raw.aborted == 0

    adapter.on_episode_start(started)
    raw.active_frame_count = 3
    adapter.on_episode_end(_event(EpisodeStatus.ABORTED), 3)
    assert raw.aborted == 1
    assert raw.ended == ['verified']


def test_writer_atomically_commits_complete_raw_v1_episode(
    tmp_path,
    fake_encoder,
):
    """A success produces the exact portable dataset tree and array schema."""
    del fake_encoder
    writer = ArxRawRecorder(tmp_path / 'arx_raw', fps=30)
    assert writer.start_episode(
        'pick up the red block',
        metadata={'session_id': 'session-a', 'seed': 17},
    ) == 0
    temporary = writer.root / 'episodes' / '.episode_000000.recording'
    final = writer.root / 'episodes' / 'episode_000000'
    assert temporary.is_dir()
    assert not final.exists()

    for index in range(16):
        timestamp = 100.0 + index / 30.0
        writer.append(
            front_rgb=np.full((4, 6, 3), index, dtype=np.uint8),
            wrist_rgb=np.full((2, 4, 3), index + 1, dtype=np.uint8),
            state=np.arange(7, dtype=np.float32) + index,
            action=np.arange(7, dtype=np.float32) + index + 0.5,
            timestamp=timestamp,
            front_timestamp=timestamp + 0.001,
            wrist_timestamp=timestamp + 0.002,
        )

    assert writer.end_episode(notes='verified placement') == 0
    assert not temporary.exists()
    assert final.is_dir()
    assert {path.name for path in final.iterdir()} == {
        'episode.json',
        'front.mp4',
        'trajectory.npz',
        'wrist.mp4',
    }

    dataset = json.loads((writer.root / 'dataset.json').read_text())
    assert dataset['schema_version'] == 'arx-raw-v1'
    assert dataset['joint_names'] == [
        'joint1', 'joint2', 'joint3', 'joint4',
        'joint5', 'joint6', 'gripper',
    ]
    assert dataset['state_units'] == ['rad'] * 6 + ['m']
    assert dataset['action_semantics'] == 'absolute_command_target'
    episode = json.loads((final / 'episode.json').read_text())
    assert episode['success'] is True
    assert episode['length'] == 16
    assert episode['seed'] == 17
    assert episode['notes'] == 'verified placement'

    with np.load(final / 'trajectory.npz') as trajectory:
        assert trajectory['timestamp'].shape == (16,)
        assert trajectory['timestamp'].dtype == np.float64
        assert trajectory['front_timestamp'].shape == (16,)
        assert trajectory['wrist_timestamp'].shape == (16,)
        assert trajectory['state'].shape == (16, 7)
        assert trajectory['state'].dtype == np.float32
        assert trajectory['action'].shape == (16, 7)
        assert trajectory['action'].dtype == np.float32
        assert trajectory['timestamp'][0] == pytest.approx(0.0)
        assert trajectory['front_timestamp'][0] == pytest.approx(0.001)
        assert trajectory['wrist_timestamp'][0] == pytest.approx(0.002)
        np.testing.assert_allclose(
            trajectory['action'][15],
            np.arange(7, dtype=np.float32) + 15.5,
        )

    assert (final / 'front.mp4').stat().st_size == 16 * 4 * 6 * 3
    assert (final / 'wrist.mp4').stat().st_size == 16 * 2 * 4 * 3


def test_writer_aborts_without_publishing_partial_episode(
    tmp_path,
    fake_encoder,
):
    """ABORTED/FAILED trajectories leave neither a final nor temp episode."""
    del fake_encoder
    writer = ArxRawRecorder(tmp_path / 'arx_raw')
    writer.start_episode('pick up the red block')
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    writer.append(
        front_rgb=rgb,
        wrist_rgb=rgb,
        state=np.zeros(7, dtype=np.float32),
        action=np.zeros(7, dtype=np.float32),
    )

    writer.abort_episode()

    episodes = writer.root / 'episodes'
    assert not (episodes / 'episode_000000').exists()
    assert not (episodes / '.episode_000000.recording').exists()
    assert not writer.active


def test_writer_end_episode_success_false_keeps_the_public_recorder_api(
    tmp_path,
    fake_encoder,
):
    """The standalone handoff API can explicitly discard a failed episode."""
    del fake_encoder
    writer = ArxRawRecorder(tmp_path / 'arx_raw')
    writer.start_episode('failed task')

    assert writer.end_episode(success=False) is None
    assert not list((writer.root / 'episodes').iterdir())


def test_writer_rejects_short_success_and_removes_temporary_data(
    tmp_path,
    fake_encoder,
):
    """A trajectory shorter than the 16-step action horizon is unusable."""
    del fake_encoder
    writer = ArxRawRecorder(tmp_path / 'arx_raw')
    writer.start_episode('too short')

    with pytest.raises(ValueError, match='at least 16 frames'):
        writer.end_episode()

    assert not writer.active
    assert not list((writer.root / 'episodes').iterdir())


def test_installed_collection_entrypoint_uses_the_persistent_recorder():
    """The production executable cannot silently regress to NullRecorder."""
    setup_source = (PACKAGE_ROOT / 'setup.py').read_text(encoding='utf-8')
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'arx_r5a_vla_collect.launch.py'
    ).read_text(encoding='utf-8')

    assert (
        "'arx_r5_isaac_sim_bringup.vla.raw_v1_recorder:main'"
        in setup_source
    )
    assert "'output_dir': LaunchConfiguration('recorder_output_dir')" in (
        launch_source
    )
    assert "on_exit=Shutdown(reason='VLA recorder exited')" in launch_source
