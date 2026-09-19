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

"""Test recorder writer faults, leases and frame-count safety boundaries."""

from types import SimpleNamespace

from arx_r5_isaac_sim_bringup.vla import recorder_base as recorder_module
from arx_r5_isaac_sim_bringup.vla.episode_events import (
    EpisodeEvent,
    EpisodeHeartbeat,
    EpisodePhase,
    EpisodeStatus,
    RecorderFault,
)
from arx_r5_isaac_sim_bringup.vla.frame_sync import (
    CausalSampleBuffer,
    COLOR_MODALITY,
    StampPairer,
)
from arx_r5_isaac_sim_bringup.vla.recorder_base import RecorderBase

from std_msgs.msg import String


class _Logger:
    """Collect recorder diagnostics without constructing a ROS node."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def info(self, _message):
        pass


class _Publisher:
    """Collect std_msgs/String publications."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Clock:
    """Expose a controllable ROS-time nanosecond value."""

    def __init__(self):
        self.nanoseconds = 0

    def now(self):
        return SimpleNamespace(nanoseconds=self.nanoseconds)


class _RecorderHarness(RecorderBase):
    """Exercise RecorderBase state transitions without Node initialization."""

    def __init__(
        self,
        *,
        fail_start=False,
        fail_frame=False,
        fail_end_calls=0,
        max_episode_frames=100,
        terminal_drain_sec=0.0,
    ):
        self.logger = _Logger()
        self.clock = _Clock()
        self.fault_publisher = _Publisher()
        self.fail_start = bool(fail_start)
        self.fail_frame = bool(fail_frame)
        self.fail_end_calls = int(fail_end_calls)
        self.start_events = []
        self.frames = []
        self.end_events = []

        self._pairer = StampPairer(('left', 'right'))
        self._joint_states = CausalSampleBuffer()
        self._joint_commands = CausalSampleBuffer()
        self._invalid_joint_states = 0
        self._invalid_joint_commands = 0
        self._event = None
        self._event_history = []
        self._episode_open = False
        self._active_identity = None
        self._episode_start_stamp_ns = None
        self._terminal_event = None
        self._terminal_stamp_ns = None
        self._terminal_drain_deadline = None
        self._last_event_stamp_ns = None
        self._boundary_dropped_images = 0
        self._post_terminal_dropped_images = 0
        self.frames_recorded = 0
        self._lease_deadline = None
        self._writer_healthy = True
        self._writer_fault_detail = ''
        self._episode_heartbeat_timeout_sec = 5.0
        self._max_episode_frames = int(max_episode_frames)
        self._terminal_drain_sec = float(terminal_drain_sec)
        self._fault_publisher = self.fault_publisher

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return self.clock

    def get_name(self):
        return 'test_recorder'

    def _lookup_object_pose(self):
        return None

    def on_episode_start(self, event):
        self.start_events.append(event)
        if self.fail_start:
            raise RuntimeError('start failed')

    def on_frame(self, frame):
        if self.fail_frame:
            raise RuntimeError('frame failed')
        self.frames.append(frame)

    def on_episode_end(self, event, _frame_count):
        self.end_events.append(event)
        if self.fail_end_calls > 0:
            self.fail_end_calls -= 1
            raise RuntimeError('end failed')


def _event(
    *,
    run_id='run-a',
    episode_id=1,
    seed=2,
    status=EpisodeStatus.STARTED,
    stamp_sec=1.0,
):
    """Return one valid event for recorder transition tests."""
    return EpisodeEvent(
        session_id='session-a',
        run_id=run_id,
        episode_id=episode_id,
        seed=seed,
        phase=EpisodePhase.RESET,
        status=status,
        instruction='pick',
        stamp_sec=stamp_sec,
    )


def _message(payload) -> String:
    return String(data=payload.to_json())


def _image(timestamp_ns):
    stamp = SimpleNamespace(
        sec=timestamp_ns // 1_000_000_000,
        nanosec=timestamp_ns % 1_000_000_000,
    )
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp))


def _faults(recorder):
    return tuple(
        RecorderFault.from_json(message.data)
        for message in recorder.fault_publisher.messages
    )


def test_start_failure_aborts_latches_unhealthy_and_rejects_next_started():
    """A partially opened writer cannot silently accept another episode."""
    recorder = _RecorderHarness(fail_start=True)
    recorder._on_episode_event(_message(_event()))

    assert not recorder._writer_healthy
    assert not recorder._episode_open
    assert recorder.end_events[-1].status == EpisodeStatus.ABORTED
    assert 'start failed' in _faults(recorder)[-1].detail

    recorder._on_episode_event(_message(_event(
        run_id='run-b',
        episode_id=2,
        seed=3,
        stamp_sec=2.0,
    )))
    assert len(recorder.start_events) == 1
    assert _faults(recorder)[-1].run_id == 'run-b'


def test_frame_failure_does_not_increment_count_and_aborts_episode():
    """Only observations durably accepted by the writer count as frames."""
    recorder = _RecorderHarness(fail_frame=True)
    recorder._on_episode_event(_message(_event()))
    recorder._add_camera_sample('left', COLOR_MODALITY, _image(2_000_000_000))
    recorder._add_camera_sample('right', COLOR_MODALITY, _image(2_000_000_000))

    assert recorder.frames_recorded == 0
    assert recorder.end_events[-1].status == EpisodeStatus.ABORTED
    assert not recorder._writer_healthy
    assert 'frame failed' in _faults(recorder)[-1].detail


def test_end_failure_is_retried_as_aborted_and_faults_the_driver():
    """A failed commit cannot leave a successful dataset outcome behind."""
    recorder = _RecorderHarness(fail_end_calls=1)
    recorder._on_episode_event(_message(_event()))
    recorder._on_episode_event(_message(_event(
        status=EpisodeStatus.SUCCEEDED,
        stamp_sec=2.0,
    )))

    assert [event.status for event in recorder.end_events] == [
        EpisodeStatus.SUCCEEDED,
        EpisodeStatus.ABORTED,
    ]
    assert not recorder._writer_healthy
    assert 'end failed' in _faults(recorder)[-1].detail


def test_matching_heartbeat_renews_lease_and_expiry_allows_new_run(
    monkeypatch,
):
    """Wrong identities are ignored and driver loss aborts only one episode."""
    recorder = _RecorderHarness()
    wall_time = [0.0]
    monkeypatch.setattr(
        recorder_module.time,
        'monotonic',
        lambda: wall_time[0],
    )
    started = _event()
    recorder._on_episode_event(_message(started))
    assert recorder._lease_deadline == 5.0

    wall_time[0] = 4.0
    wrong = EpisodeHeartbeat(
        session_id=started.session_id,
        run_id='wrong-run',
        episode_id=started.episode_id,
        seed=started.seed,
        stamp_sec=4.0,
    )
    recorder._on_episode_heartbeat(_message(wrong))
    assert recorder._lease_deadline == 5.0

    matching = EpisodeHeartbeat(
        session_id=started.session_id,
        run_id=started.run_id,
        episode_id=started.episode_id,
        seed=started.seed,
        stamp_sec=4.0,
    )
    recorder._on_episode_heartbeat(_message(matching))
    assert recorder._lease_deadline == 9.0

    wall_time[0] = 9.1
    recorder._check_episode_lease()
    assert recorder.end_events[-1].status == EpisodeStatus.ABORTED
    assert recorder._writer_healthy

    recorder._on_episode_event(_message(_event(
        run_id='run-b',
        episode_id=2,
        seed=3,
        stamp_sec=10.0,
    )))
    assert recorder._episode_open
    assert len(recorder.start_events) == 2


def test_terminal_drain_is_not_rewritten_by_heartbeat_watchdog(monkeypatch):
    """A proven terminal status remains authoritative during DDS draining."""
    recorder = _RecorderHarness(terminal_drain_sec=1.0)
    wall_time = [0.0]
    monkeypatch.setattr(
        recorder_module.time,
        'monotonic',
        lambda: wall_time[0],
    )
    recorder._on_episode_event(_message(_event()))
    wall_time[0] = 1.0
    recorder._on_episode_event(_message(_event(
        status=EpisodeStatus.SUCCEEDED,
        stamp_sec=2.0,
    )))
    assert recorder._lease_deadline is None

    wall_time[0] = 10.0
    recorder._check_episode_lease()
    assert recorder._episode_open
    recorder._finish_terminal_drain()
    assert recorder.end_events[-1].status == EpisodeStatus.SUCCEEDED


def test_frame_limit_stops_batch_without_evicting_completed_stamps():
    """The hard cap preserves duplicate detection for every accepted frame."""
    recorder = _RecorderHarness(max_episode_frames=1)
    recorder._on_episode_event(_message(_event()))
    recorder._add_camera_sample('left', COLOR_MODALITY, _image(2_000_000_000))
    recorder._add_camera_sample('right', COLOR_MODALITY, _image(2_000_000_000))

    assert recorder.frames_recorded == 1
    assert recorder._pairer._completed_stamps == {2_000_000_000}
    assert recorder.end_events[-1].status == EpisodeStatus.ABORTED
    assert not recorder._writer_healthy
    assert 'maximum episode frame count' in _faults(recorder)[-1].detail


def test_shutdown_aborts_writer_and_faults_a_still_running_driver():
    """A recorder-only shutdown cannot leave robot motion unrecorded."""
    recorder = _RecorderHarness()
    recorder._on_episode_event(_message(_event()))

    recorder.finalize_active_episode_on_shutdown()

    assert recorder.end_events[-1].status == EpisodeStatus.ABORTED
    assert not recorder._episode_open
    assert not recorder._writer_healthy
    assert 'recorder shut down' in _faults(recorder)[-1].detail


def test_recorder_signals_finalize_writer_before_ros_shutdown(monkeypatch):
    """SIGINT and SIGTERM both stop callbacks before writer finalization."""
    for selected_signal in (
        recorder_module.signal.SIGINT,
        recorder_module.signal.SIGTERM,
    ):
        trace = []
        handlers = {}
        context_ok = [False]
        init_options = []

        class FakeNode:
            def finalize_active_episode_on_shutdown(self):
                assert context_ok[0]
                trace.append('finalize_writer')

            def destroy_node(self):
                assert context_ok[0]
                trace.append('destroy_node')

        class FakeExecutor:
            def add_node(self, _node):
                trace.append('add_node')

            def spin_once(self, timeout_sec):
                trace.append(('spin_once', timeout_sec))
                handlers[selected_signal](selected_signal, None)

            def shutdown(self, timeout_sec):
                assert context_ok[0]
                trace.append(('executor_shutdown', timeout_sec))

            def remove_node(self, _node):
                trace.append('remove_node')

        def init(*, args, signal_handler_options):
            context_ok[0] = True
            init_options.append((args, signal_handler_options))

        def shutdown():
            trace.append('ros_shutdown')
            context_ok[0] = False

        def install_handler(signum, handler):
            previous = handlers.get(signum, object())
            handlers[signum] = handler
            return previous

        monkeypatch.setattr(recorder_module, 'NullRecorder', FakeNode)
        monkeypatch.setattr(
            recorder_module,
            'SingleThreadedExecutor',
            FakeExecutor,
        )
        monkeypatch.setattr(recorder_module.rclpy, 'init', init)
        monkeypatch.setattr(
            recorder_module.rclpy,
            'ok',
            lambda: context_ok[0],
        )
        monkeypatch.setattr(recorder_module.rclpy, 'shutdown', shutdown)
        monkeypatch.setattr(
            recorder_module.signal,
            'signal',
            install_handler,
        )

        assert recorder_module.main([]) == 0

        assert init_options[0][1] is recorder_module.SignalHandlerOptions.NO
        executor_index = next(
            index for index, item in enumerate(trace)
            if isinstance(item, tuple) and item[0] == 'executor_shutdown'
        )
        assert executor_index < trace.index('finalize_writer')
        assert trace.index('finalize_writer') < trace.index('destroy_node')
        assert trace.index('destroy_node') < trace.index('ros_shutdown')
