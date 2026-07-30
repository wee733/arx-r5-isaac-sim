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

"""Test transactional episode-boundary publication in the driver."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _install_attachment_stub():
    """Provide the optional manipulation action in minimal test images."""
    try:
        import isaac_ros_manipulation_interfaces.action  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    package = types.ModuleType('isaac_ros_manipulation_interfaces')
    package.__path__ = []
    action = types.ModuleType('isaac_ros_manipulation_interfaces.action')

    class AttachObject:
        """Minimal generated-action stand-in used during module import."""

        class Goal:
            pass

    action.AttachObject = AttachObject
    package.action = action
    sys.modules[package.__name__] = package
    sys.modules[action.__name__] = action


_install_attachment_stub()

driver_module = importlib.import_module(
    'arx_r5_isaac_sim_bringup.vla.episode_driver'
)
events_module = importlib.import_module(
    'arx_r5_isaac_sim_bringup.vla.episode_events'
)
EpisodeDriver = driver_module.EpisodeDriver
MotionError = driver_module.MotionError
EpisodeEvent = events_module.EpisodeEvent
EpisodePhase = events_module.EpisodePhase
EpisodeStatus = events_module.EpisodeStatus
RecorderFault = events_module.RecorderFault


class _Logger:
    """Collect lifecycle diagnostics."""

    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(str(message))

    def warning(self, _message):
        pass


class _Publisher:
    """Inject DDS publication failures deterministically."""

    def __init__(self):
        self.messages = []
        self.failures = 0

    def publish(self, message):
        if self.failures:
            self.failures -= 1
            raise RuntimeError('publisher failed')
        self.messages.append(message)


class _Clock:
    """Expose a controllable ROS timestamp."""

    def __init__(self, nanoseconds=1_000_000_000):
        self.nanoseconds = nanoseconds

    def now(self):
        return SimpleNamespace(nanoseconds=self.nanoseconds)


def _driver():
    """Construct only the state needed by EpisodeDriver._publish_event."""
    driver = EpisodeDriver.__new__(EpisodeDriver)
    driver._episode_state_lock = __import__('threading').RLock()
    driver._current_run_id = 'run-a'
    driver._session_id = 'session-a'
    driver._active_episode = None
    driver._terminal_event_sent = False
    driver._current_episode_phase = EpisodePhase.RESET
    driver._task = SimpleNamespace(instruction='pick')
    driver._clock = _Clock()
    driver._event_publisher = _Publisher()
    driver._logger = _Logger()
    driver.get_clock = lambda: driver._clock
    driver.get_logger = lambda: driver._logger
    return driver


def test_started_and_terminal_publication_commit_only_after_publish():
    """A failed DDS call leaves both transitions retryable."""
    driver = _driver()
    driver._event_publisher.failures = 1

    assert not driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    assert driver._active_episode is None
    assert not driver._terminal_event_sent

    assert driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    identity = ('session-a', 'run-a', 1, 2)
    assert driver._active_episode == identity

    driver._event_publisher.failures = 1
    assert not driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.ABORTED,
    )
    assert driver._active_episode == identity
    assert not driver._terminal_event_sent

    assert driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.ABORTED,
    )
    assert driver._active_episode is None
    assert driver._terminal_event_sent


def test_event_validation_failure_does_not_poison_started_state():
    """Non-finite timestamps fail explicitly without committing boundaries."""
    driver = _driver()
    driver._clock.nanoseconds = float('nan')

    assert not driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    assert driver._active_episode is None


def test_serialization_failure_can_be_retried(monkeypatch):
    """A JSON encoder exception must not block a later STARTED retry."""
    driver = _driver()
    original = EpisodeEvent.to_json

    def fail_once(self):
        del self
        raise TypeError('serialization failed')

    monkeypatch.setattr(EpisodeEvent, 'to_json', fail_once)
    assert not driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    assert driver._active_episode is None

    monkeypatch.setattr(EpisodeEvent, 'to_json', original)
    assert driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )


def test_state_conflict_is_explicit_and_does_not_move_the_robot():
    """A second STARTED boundary is rejected without mutating identity."""
    driver = _driver()
    assert driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    assert not driver._publish_event(
        2,
        3,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    assert driver._active_episode == ('session-a', 'run-a', 1, 2)
    assert driver._terminal_event_sent is False


def test_matching_recorder_fault_stops_run_and_publishes_aborted():
    """Writer failure is propagated into both motion stop and episode state."""
    driver = _driver()
    threading = __import__('threading')
    driver._run_lock = threading.Lock()
    driver._stop_event = threading.Event()
    driver._recorder_faulted = False
    driver._recorder_fault_detail = ''
    assert driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    fault = RecorderFault(
        recorder_name='writer',
        session_id='session-a',
        run_id='run-a',
        episode_id=1,
        seed=2,
        stamp_sec=2.0,
        detail='disk full',
    )

    driver._on_recorder_fault(SimpleNamespace(data=fault.to_json()))

    assert driver._stop_event.is_set()
    assert driver._recorder_faulted
    assert driver._active_episode is None
    terminal = EpisodeEvent.from_json(
        driver._event_publisher.messages[-1].data
    )
    assert terminal.status == EpisodeStatus.ABORTED
    assert 'disk full' in terminal.detail


def test_fault_for_another_run_is_ignored():
    """A transient-local fault from an old run cannot stop a new identity."""
    driver = _driver()
    threading = __import__('threading')
    driver._run_lock = threading.Lock()
    driver._stop_event = threading.Event()
    driver._recorder_faulted = False
    driver._recorder_fault_detail = ''
    fault = RecorderFault(
        recorder_name='writer',
        session_id='session-a',
        run_id='old-run',
        episode_id=1,
        seed=2,
        stamp_sec=2.0,
        detail='old fault',
    )

    driver._on_recorder_fault(SimpleNamespace(data=fault.to_json()))

    assert not driver._stop_event.is_set()
    assert not driver._recorder_faulted


@pytest.mark.parametrize('status', EpisodeStatus.TERMINAL)
def test_required_event_helper_stops_on_a_false_result(status):
    """Callers cannot continue motion after a required boundary fails."""
    driver = _driver()
    driver._stop_event = __import__('threading').Event()
    with pytest.raises(MotionError):
        driver._publish_event_or_stop(
            1,
            2,
            EpisodePhase.RESET,
            status,
        )
    assert driver._stop_event.is_set()


def _clock_wait_driver():
    """Return a driver harness for reset and placement wait loops."""
    driver = _driver()
    threading = __import__('threading')
    driver._stop_event = threading.Event()
    driver._scene_reset_ack_lock = threading.Lock()
    driver._scene_reset_ack = 123
    driver._scene_reset_ack_generation = 2
    driver._scene_reset_wall_timeout_sec = 1.0
    driver._placement_wall_timeout_sec = 1.0
    return driver


def test_scene_reset_waits_for_ack_and_full_sim_settle_after_clock_jump(
    monkeypatch,
):
    """A backward simulation jump restarts settling under one wall budget."""
    driver = _clock_wait_driver()
    driver._clock.nanoseconds = 1_000_000_000
    wall_time = [0.0]
    sim_schedule = iter((
        1_200_000_000,
        100_000_000,
        400_000_000,
        700_000_000,
    ))
    sleep_calls = [0]

    def advance(_duration):
        wall_time[0] += 0.1
        driver._clock.nanoseconds = next(sim_schedule)
        sleep_calls[0] += 1

    monkeypatch.setattr(driver_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(
        driver_module.time,
        'monotonic',
        lambda: wall_time[0],
    )
    monkeypatch.setattr(driver_module.time, 'sleep', advance)

    driver._wait_for_scene_reset(
        command=123,
        baseline_generation=1,
        start_sim_ns=1_000_000_000,
        settle_sec=0.5,
        wall_started_at=0.0,
    )
    assert sleep_calls[0] == 4


def test_scene_reset_wall_watchdog_handles_a_paused_clock(monkeypatch):
    """An ACK cannot hide a simulator that stopped advancing /clock."""
    driver = _clock_wait_driver()
    driver._clock.nanoseconds = 0
    wall_time = [0.0]

    def advance(_duration):
        wall_time[0] += 0.25

    monkeypatch.setattr(driver_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(
        driver_module.time,
        'monotonic',
        lambda: wall_time[0],
    )
    monkeypatch.setattr(driver_module.time, 'sleep', advance)

    with pytest.raises(MotionError, match='wall-clock watchdog'):
        driver._wait_for_scene_reset(
            command=123,
            baseline_generation=1,
            start_sim_ns=0,
            settle_sec=0.5,
            wall_started_at=0.0,
        )


def test_scene_reset_settle_interval_starts_after_exact_ack(monkeypatch):
    """Time spent waiting for ACK cannot count as post-reset settling."""
    driver = _clock_wait_driver()
    driver._clock.nanoseconds = 0
    driver._scene_reset_ack_generation = 1
    wall_time = [0.0]
    sleep_calls = [0]

    def advance(_duration):
        wall_time[0] += 0.1
        driver._clock.nanoseconds += 200_000_000
        sleep_calls[0] += 1
        if sleep_calls[0] == 2:
            with driver._scene_reset_ack_lock:
                driver._scene_reset_ack_generation = 2

    monkeypatch.setattr(driver_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(
        driver_module.time,
        'monotonic',
        lambda: wall_time[0],
    )
    monkeypatch.setattr(driver_module.time, 'sleep', advance)

    driver._wait_for_scene_reset(
        command=123,
        baseline_generation=1,
        start_sim_ns=0,
        settle_sec=0.5,
        wall_started_at=0.0,
    )

    assert sleep_calls[0] == 5
    assert driver._clock.nanoseconds == 1_000_000_000


@pytest.mark.parametrize(
    ('advance_sim', 'expected'),
    (
        (True, 'simulation-time timeout'),
        (False, 'wall-clock watchdog'),
    ),
)
def test_placement_has_simulation_and_wall_timeouts(
    monkeypatch,
    advance_sim,
    expected,
):
    """Placement cannot hang when simulated time runs or pauses."""
    driver = _clock_wait_driver()
    driver._task = SimpleNamespace(
        placement=SimpleNamespace(timeout_sec=0.5)
    )
    driver._placement_wall_timeout_sec = 0.5
    driver._latest_object_pose = lambda: None
    wall_time = [0.0]

    def advance(_duration):
        wall_time[0] += 0.1
        if advance_sim:
            driver._clock.nanoseconds += 200_000_000

    monkeypatch.setattr(driver_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(
        driver_module.time,
        'monotonic',
        lambda: wall_time[0],
    )
    monkeypatch.setattr(driver_module.time, 'sleep', advance)

    with pytest.raises(MotionError, match=expected):
        driver._wait_for_placement((0.0, 0.0, 0.0))


def test_worker_finally_clears_run_state_when_abort_publish_fails():
    """A failed shutdown boundary cannot leave the driver permanently busy."""
    driver = _driver()
    threading = __import__('threading')
    driver._run_lock = threading.Lock()
    driver._worker = threading.current_thread()
    driver._current_run_start_seed = 2
    driver._server_timeout_sec = 1.0
    driver._motion = SimpleNamespace(wait_for_servers=lambda **_kwargs: None)
    driver._object_attachment = None
    driver._stop_event = threading.Event()
    assert driver._publish_event(
        1,
        2,
        EpisodePhase.RESET,
        EpisodeStatus.STARTED,
    )
    driver._event_publisher.failures = 1
    driver._stop_event.set()

    driver._run('run-a', 2)

    assert driver._worker is None
    assert driver._current_run_id is None
    assert driver._current_run_start_seed is None
    assert driver._active_episode == ('session-a', 'run-a', 1, 2)
    assert not driver._terminal_event_sent


@pytest.mark.parametrize(
    'signum',
    (driver_module.signal.SIGINT, driver_module.signal.SIGTERM),
)
def test_driver_signal_stops_worker_before_ros_shutdown(monkeypatch, signum):
    """Both process signals preserve ROS until stop and ABORT publication."""
    trace = []
    handlers = {}
    context_ok = [False]
    init_options = []

    class FakeLogger:
        def error(self, message):
            trace.append(('error', str(message)))

    class FakeNode:
        def stop_run(self, join_timeout_sec, detail):
            assert context_ok[0]
            trace.append(('stop_run', join_timeout_sec, detail))
            return True

        def get_logger(self):
            return FakeLogger()

        def destroy_node(self):
            assert context_ok[0]
            trace.append('destroy_node')

    class FakeExecutor:
        def add_node(self, _node):
            trace.append('add_node')

        def spin(self):
            handlers[signum](signum, None)
            trace.append('spin_returned')

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

    def install_handler(installed_signum, handler):
        previous = handlers.get(installed_signum, object())
        handlers[installed_signum] = handler
        return previous

    monkeypatch.setattr(driver_module, 'EpisodeDriver', FakeNode)
    monkeypatch.setattr(
        driver_module,
        'MultiThreadedExecutor',
        FakeExecutor,
    )
    monkeypatch.setattr(driver_module.rclpy, 'init', init)
    monkeypatch.setattr(driver_module.rclpy, 'ok', lambda: context_ok[0])
    monkeypatch.setattr(driver_module.rclpy, 'shutdown', shutdown)
    monkeypatch.setattr(driver_module.signal, 'signal', install_handler)

    assert driver_module.main([]) == 0

    assert init_options[0][1] is driver_module.SignalHandlerOptions.NO
    stop_index = next(
        index for index, item in enumerate(trace)
        if isinstance(item, tuple) and item[0] == 'stop_run'
    )
    executor_index = next(
        index for index, item in enumerate(trace)
        if isinstance(item, tuple) and item[0] == 'executor_shutdown'
    )
    assert stop_index < executor_index
    assert executor_index < trace.index('destroy_node')
    assert trace.index('destroy_node') < trace.index('ros_shutdown')
    assert str(signum) in trace[stop_index][2]
