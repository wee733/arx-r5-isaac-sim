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

"""Guard physical gripper completion and action cancellation contracts."""

import importlib.util
from pathlib import Path
import sys
import threading
import time
import types

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOTION_SOURCE = (
    PACKAGE_ROOT
    / 'arx_r5_isaac_sim_bringup'
    / 'vla'
    / 'motion_client.py'
).read_text(encoding='utf-8')
DRIVER_SOURCE = (
    PACKAGE_ROOT
    / 'arx_r5_isaac_sim_bringup'
    / 'vla'
    / 'episode_driver.py'
).read_text(encoding='utf-8')
SIMULATION_SOURCE = (
    PACKAGE_ROOT
    / 'arx_r5_isaac_sim_bringup'
    / 'simulation.py'
).read_text(encoding='utf-8')
RUN_SCRIPT = (
    PACKAGE_ROOT.parent / 'scripts' / 'run_vla_collect.sh'
).read_text(encoding='utf-8')
LAUNCH_SOURCE = (
    PACKAGE_ROOT / 'launch' / 'arx_r5a_vla_collect.launch.py'
).read_text(encoding='utf-8')


def _method_source(source: str, name: str, next_name: str) -> str:
    start = source.index(f'    def {name}(')
    end = source.index(f'    def {next_name}(', start)
    return source[start:end]


def _install_module(monkeypatch, name: str, **members):
    """Install one lightweight generated-message/module stand-in."""
    module = types.ModuleType(name)
    module.__dict__.update(members)
    if '.' not in name:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)
    return module


@pytest.fixture(name='motion_module')
def motion_module_fixture(monkeypatch):
    """Import motion_client with pure-Python ROS message/action stubs."""
    class GoalStatus:
        STATUS_SUCCEEDED = 4
        STATUS_CANCELED = 5
        STATUS_ABORTED = 6

    class CancelGoal:
        class Response:
            ERROR_NONE = 0

    class Message:
        class Goal:
            pass

    class Pose:
        pass

    class PoseArray:
        pass

    class MoveItErrorCodes:
        SUCCESS = 1

    class PlanningScene:
        pass

    class Node:
        pass

    _install_module(monkeypatch, 'action_msgs')
    _install_module(monkeypatch, 'action_msgs.msg', GoalStatus=GoalStatus)
    _install_module(monkeypatch, 'action_msgs.srv', CancelGoal=CancelGoal)
    _install_module(monkeypatch, 'control_msgs')
    _install_module(monkeypatch, 'control_msgs.action', GripperCommand=Message)
    _install_module(monkeypatch, 'geometry_msgs')
    _install_module(
        monkeypatch,
        'geometry_msgs.msg',
        Pose=Pose,
        PoseArray=PoseArray,
    )
    _install_module(monkeypatch, 'isaac_ros_cumotion_interfaces')
    _install_module(
        monkeypatch,
        'isaac_ros_cumotion_interfaces.action',
        MotionPlan=Message,
    )
    _install_module(
        monkeypatch,
        'isaac_ros_cumotion_interfaces.srv',
        PublishStaticPlanningScene=Message,
    )
    _install_module(monkeypatch, 'moveit_msgs')
    _install_module(
        monkeypatch,
        'moveit_msgs.action',
        ExecuteTrajectory=Message,
    )
    _install_module(
        monkeypatch,
        'moveit_msgs.msg',
        MoveItErrorCodes=MoveItErrorCodes,
        PlanningScene=PlanningScene,
    )
    _install_module(monkeypatch, 'rclpy', ok=lambda: True)
    _install_module(monkeypatch, 'rclpy.action', ActionClient=object)
    _install_module(
        monkeypatch,
        'rclpy.callback_groups',
        MutuallyExclusiveCallbackGroup=object,
    )
    _install_module(monkeypatch, 'rclpy.node', Node=Node)
    _install_module(monkeypatch, 'sensor_msgs')
    _install_module(monkeypatch, 'sensor_msgs.msg', JointState=Message)
    _install_module(monkeypatch, 'std_msgs')
    _install_module(monkeypatch, 'std_msgs.msg', Header=Message)

    module_name = '_vla_motion_client_test_double'
    module_path = (
        PACKAGE_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'motion_client.py'
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Collect warnings emitted by action cleanup paths."""

    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(str(message))


class _FakeClock:
    """Return a deterministic sequence of ROS-time nanosecond values."""

    def __init__(self, values=(0,), on_read=None):
        self.values = tuple(int(value) for value in values)
        self.on_read = on_read
        self.reads = 0

    def now(self):
        index = min(self.reads, len(self.values) - 1)
        value = self.values[index]
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads, value)
        return types.SimpleNamespace(nanoseconds=value)


class _FakeNode:
    """Expose only the logger used by CumotionMotionClient."""

    def __init__(self, clock=None):
        self.logger = _FakeLogger()
        self.clock = clock or _FakeClock()

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return self.clock


_PENDING = object()


class _FakeFuture:
    """Small rclpy Future stand-in with deterministic callback delivery."""

    def __init__(self, result=_PENDING, exception=None, on_result=None):
        self._done = result is not _PENDING or exception is not None
        self._result = result
        self._exception = exception
        self._on_result = on_result
        self._callbacks = []
        self.cancel_calls = 0
        self.result_calls = 0

    def done(self):
        return self._done

    def result(self):
        self.result_calls += 1
        if self._on_result is not None:
            callback, self._on_result = self._on_result, None
            callback()
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, callback):
        self._callbacks.append(callback)
        if self._done:
            callback(self)

    def set_result(self, result):
        self._result = result
        self._done = True
        for callback in tuple(self._callbacks):
            callback(self)

    def cancel(self):
        self.cancel_calls += 1


class _FakeGoalId:
    """UUID-shaped action goal identifier."""

    def __init__(self, value=1):
        self.uuid = [int(value)] * 16


class _FakeGoalInfo:
    """CancelGoal response entry."""

    def __init__(self, goal_id):
        self.goal_id = goal_id


class _FakeCancelResponse:
    """Subset of action_msgs/srv/CancelGoal used by the client."""

    def __init__(self, return_code, goal_id=None):
        self.return_code = int(return_code)
        self.goals_canceling = (
            [] if goal_id is None else [_FakeGoalInfo(goal_id)]
        )


class _FakeResultResponse:
    """GetResult response carrying a goal status and action result."""

    def __init__(self, status, result='result'):
        self.status = int(status)
        self.result = result


class _FakeGoalHandle:
    """Accepted action goal with controllable cancel and result futures."""

    def __init__(self, cancel_future, result_future, goal_id=None):
        self.accepted = True
        self.goal_id = goal_id or _FakeGoalId()
        self.cancel_future = cancel_future
        self.result_future = result_future
        self.cancel_calls = 0
        self.result_requests = 0

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel_future

    def get_result_async(self):
        self.result_requests += 1
        return self.result_future


class _FakeActionClient:
    """Return one configured send-goal future."""

    def __init__(self, send_future):
        self.send_future = send_future

    def send_goal_async(self, _goal):
        return self.send_future


def _motion_client_without_ros(motion_module, clock=None):
    client = motion_module.CumotionMotionClient.__new__(
        motion_module.CumotionMotionClient
    )
    client._node = _FakeNode(clock=clock)
    return client


def test_send_timeout_cancels_and_drains_a_late_accepted_goal(
    motion_module,
    monkeypatch,
):
    """Retain a timed-out request, then cancel its late server goal."""
    send_future = _FakeFuture()
    result_future = _FakeFuture()
    goal_id = _FakeGoalId(7)
    cancel_response = _FakeCancelResponse(0, goal_id)
    cancel_future = _FakeFuture(cancel_response)
    goal_handle = _FakeGoalHandle(
        cancel_future,
        result_future,
        goal_id=goal_id,
    )
    client = _motion_client_without_ros(motion_module)

    def time_out_send(_node, future, _timeout_sec):
        assert future is send_future
        raise motion_module.MotionError('send response timeout')

    monkeypatch.setattr(motion_module, '_spin_until_complete', time_out_send)

    with pytest.raises(
        motion_module.MotionError,
        match='send response timeout',
    ):
        client._send(
            _FakeActionClient(send_future),
            object(),
            1.0,
            'gripper',
        )

    assert send_future.cancel_calls == 0
    assert goal_handle.cancel_calls == 0

    send_future.set_result(goal_handle)

    assert goal_handle.result_requests == 1
    assert goal_handle.cancel_calls == 1
    assert cancel_future.result_calls == 1

    result_future.set_result(
        _FakeResultResponse(motion_module.GoalStatus.STATUS_CANCELED)
    )
    assert result_future.result_calls == 1


def test_physical_completion_cancels_and_waits_for_terminal_result(
    motion_module,
):
    """Cancel remotely and confirm terminal state after attachment ACK."""
    result_future = _FakeFuture()
    goal_id = _FakeGoalId(8)

    def finish_canceled_goal():
        result_future.set_result(
            _FakeResultResponse(
                motion_module.GoalStatus.STATUS_CANCELED,
                result='physically attached',
            )
        )

    cancel_future = _FakeFuture(
        _FakeCancelResponse(0, goal_id),
        on_result=finish_canceled_goal,
    )
    goal_handle = _FakeGoalHandle(
        cancel_future,
        result_future,
        goal_id=goal_id,
    )
    send_future = _FakeFuture(goal_handle)
    client = _motion_client_without_ros(motion_module)

    result = client._send(
        _FakeActionClient(send_future),
        object(),
        1.0,
        'gripper',
        completion_predicate=lambda: True,
    )

    assert result == 'physically attached'
    assert goal_handle.cancel_calls == 1
    assert result_future.done()
    assert result_future.result_calls >= 1


def test_physical_completion_can_outlive_controller_terminal_result(
    motion_module,
):
    """A confirmed PhysX grasp proceeds while the old result is drained."""
    result_future = _FakeFuture()
    goal_handle = _FakeGoalHandle(
        _FakeFuture(),
        result_future,
    )
    client = _motion_client_without_ros(motion_module)

    def late_terminal(*_args, **_kwargs):
        result_future.add_done_callback(client._consume_late_result)
        raise motion_module.MotionError('terminal result timeout')

    client._cancel_goal = late_terminal
    result = client._send(
        _FakeActionClient(_FakeFuture(goal_handle)),
        object(),
        1.0,
        'gripper',
        completion_predicate=lambda: True,
    )

    assert result is None
    assert client._node.logger.warnings
    result_future.set_result(
        _FakeResultResponse(motion_module.GoalStatus.STATUS_CANCELED)
    )
    assert result_future.result_calls == 1


def test_collection_stop_cancels_an_accepted_action(motion_module):
    """Recorder failure or shutdown must not wait for natural completion."""
    result_future = _FakeFuture()
    goal_id = _FakeGoalId(9)

    def finish_canceled_goal():
        result_future.set_result(_FakeResultResponse(
            motion_module.GoalStatus.STATUS_CANCELED
        ))

    goal_handle = _FakeGoalHandle(
        _FakeFuture(
            _FakeCancelResponse(0, goal_id),
            on_result=finish_canceled_goal,
        ),
        result_future,
        goal_id=goal_id,
    )
    client = _motion_client_without_ros(motion_module)
    client._stop_predicate = lambda: True

    with pytest.raises(
        motion_module.MotionError,
        match='interrupted by a collection stop request',
    ):
        client._send(
            _FakeActionClient(_FakeFuture(goal_handle)),
            object(),
            1.0,
            'execute trajectory',
        )

    assert goal_handle.cancel_calls == 1
    assert result_future.done()


def test_execute_default_timeout_tracks_trajectory_duration(motion_module):
    """Long dilated plans receive simulation-time trajectory margin."""
    client = _motion_client_without_ros(motion_module)
    client._execute_timeout_min_sec = 60.0
    client._execute_timeout_scale = 1.5
    client._execute_timeout_margin_sec = 10.0
    client._execute_wall_timeout_sec = 10.0
    client._execute_client = object()
    captured = {}

    def send(_action_client, _goal, timeout_sec, _label, **kwargs):
        captured['timeout_sec'] = timeout_sec
        captured.update(kwargs)
        return types.SimpleNamespace(
            error_code=types.SimpleNamespace(
                val=motion_module.MoveItErrorCodes.SUCCESS
            )
        )

    client._send = send
    duration = types.SimpleNamespace(sec=50, nanosec=0)
    trajectory = types.SimpleNamespace(
        joint_trajectory=types.SimpleNamespace(
            header=None,
            points=[types.SimpleNamespace(time_from_start=duration)],
        ),
        multi_dof_joint_trajectory=types.SimpleNamespace(
            header=None,
            points=[],
        ),
    )

    client.execute(trajectory)
    assert captured['timeout_sec'] == pytest.approx(85.0)
    assert captured['result_clock'] is client._node.clock
    assert captured['result_wall_timeout_sec'] == pytest.approx(10.0)

    client.execute(trajectory, timeout_sec=7.0)
    assert captured['timeout_sec'] == pytest.approx(7.0)


def test_execute_slow_rtf_does_not_consume_simulation_timeout(
    motion_module,
    monkeypatch,
):
    """Clock progress keeps execution alive beyond the old wall deadline."""
    result_future = _FakeFuture()

    def finish_after_slow_progress(read_count, _value):
        if read_count == 5:
            result_future.set_result(_FakeResultResponse(
                motion_module.GoalStatus.STATUS_SUCCEEDED,
                result='executed',
            ))

    clock = _FakeClock(
        (0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000),
        on_read=finish_after_slow_progress,
    )
    goal_handle = _FakeGoalHandle(_FakeFuture(), result_future)
    client = _motion_client_without_ros(motion_module, clock=clock)
    wall_now = 0.0

    def slow_wall_clock():
        nonlocal wall_now
        wall_now += 20.0
        return wall_now

    monkeypatch.setattr(motion_module.time, 'monotonic', slow_wall_clock)
    monkeypatch.setattr(motion_module.time, 'sleep', lambda _seconds: None)

    result = client._send(
        _FakeActionClient(_FakeFuture(goal_handle)),
        object(),
        60.0,
        'execute trajectory',
        result_clock=clock,
        result_wall_timeout_sec=30.0,
    )

    assert result == 'executed'
    assert wall_now > 60.0
    assert goal_handle.cancel_calls == 0


def test_execute_simulation_timeout_cancels_and_drains_goal(
    motion_module,
    monkeypatch,
):
    """Execution timeout is measured by ROS simulation time."""
    result_future = _FakeFuture()
    goal_id = _FakeGoalId(10)

    def finish_canceled_goal():
        result_future.set_result(_FakeResultResponse(
            motion_module.GoalStatus.STATUS_CANCELED
        ))

    goal_handle = _FakeGoalHandle(
        _FakeFuture(
            _FakeCancelResponse(0, goal_id),
            on_result=finish_canceled_goal,
        ),
        result_future,
        goal_id=goal_id,
    )
    clock = _FakeClock((0, 500_000_000, 1_100_000_000))
    client = _motion_client_without_ros(motion_module, clock=clock)
    wall_now = 0.0

    def steady_wall_clock():
        nonlocal wall_now
        wall_now += 0.1
        return wall_now

    monkeypatch.setattr(motion_module.time, 'monotonic', steady_wall_clock)
    monkeypatch.setattr(motion_module.time, 'sleep', lambda _seconds: None)

    with pytest.raises(
        motion_module.MotionError,
        match='simulation-time timeout',
    ):
        client._send(
            _FakeActionClient(_FakeFuture(goal_handle)),
            object(),
            1.0,
            'execute trajectory',
            result_clock=clock,
            result_wall_timeout_sec=5.0,
        )

    assert goal_handle.cancel_calls == 1
    assert result_future.done()


def test_execute_wall_watchdog_handles_a_stopped_simulation_clock(
    motion_module,
    monkeypatch,
):
    """A paused /clock still bounds the wait and drains the remote goal."""
    result_future = _FakeFuture()
    goal_id = _FakeGoalId(11)

    def finish_canceled_goal():
        result_future.set_result(_FakeResultResponse(
            motion_module.GoalStatus.STATUS_CANCELED
        ))

    goal_handle = _FakeGoalHandle(
        _FakeFuture(
            _FakeCancelResponse(0, goal_id),
            on_result=finish_canceled_goal,
        ),
        result_future,
        goal_id=goal_id,
    )
    clock = _FakeClock((0,))
    client = _motion_client_without_ros(motion_module, clock=clock)
    wall_now = 0.0

    def stalled_wall_clock():
        nonlocal wall_now
        wall_now += 0.6
        return wall_now

    monkeypatch.setattr(motion_module.time, 'monotonic', stalled_wall_clock)
    monkeypatch.setattr(motion_module.time, 'sleep', lambda _seconds: None)

    with pytest.raises(
        motion_module.MotionError,
        match='wall-clock watchdog expired',
    ):
        client._send(
            _FakeActionClient(_FakeFuture(goal_handle)),
            object(),
            60.0,
            'execute trajectory',
            result_clock=clock,
            result_wall_timeout_sec=1.0,
        )

    assert goal_handle.cancel_calls == 1
    assert result_future.done()


def test_execute_clock_rewind_restarts_the_simulation_window(
    motion_module,
    monkeypatch,
):
    """A reset-induced ROS-time rewind cannot inherit the old deadline."""
    result_future = _FakeFuture()

    def finish_after_rewind(read_count, _value):
        if read_count == 4:
            result_future.set_result(_FakeResultResponse(
                motion_module.GoalStatus.STATUS_SUCCEEDED,
                result='executed after reset',
            ))

    clock = _FakeClock(
        (5_000_000_000, 8_000_000_000, 1_000_000_000, 3_000_000_000),
        on_read=finish_after_rewind,
    )
    goal_handle = _FakeGoalHandle(_FakeFuture(), result_future)
    client = _motion_client_without_ros(motion_module, clock=clock)
    wall_now = 0.0

    def steady_wall_clock():
        nonlocal wall_now
        wall_now += 0.1
        return wall_now

    monkeypatch.setattr(motion_module.time, 'monotonic', steady_wall_clock)
    monkeypatch.setattr(motion_module.time, 'sleep', lambda _seconds: None)

    result = client._send(
        _FakeActionClient(_FakeFuture(goal_handle)),
        object(),
        4.0,
        'execute trajectory',
        result_clock=clock,
        result_wall_timeout_sec=1.0,
    )

    assert result == 'executed after reset'
    assert any(
        'simulation clock moved backward' in warning
        for warning in client._node.logger.warnings
    )


def test_execute_wall_watchdog_is_exposed_by_launch():
    """Operators can tune the stopped-clock guard without changing code."""
    assert "'execute_wall_timeout_sec': ParameterValue(" in LAUNCH_SOURCE
    assert "'execute_wall_timeout_sec',\n            default_value='10.0'" in (
        LAUNCH_SOURCE
    )


def test_joint_state_refresh_waits_for_updates_after_the_baseline(
    motion_module,
):
    """Attachment hot reload must be followed by genuinely new state data."""
    client = _motion_client_without_ros(motion_module)
    client._joint_state_condition = threading.Condition()
    client._joint_state_generation = 4
    client._on_joint_state(types.SimpleNamespace(
        name=['joint1'],
        position=[0.0],
    ))
    client._on_joint_state(types.SimpleNamespace(
        name=list(motion_module.ARM_JOINTS),
        position=[float('nan')] * len(motion_module.ARM_JOINTS),
    ))
    assert client.joint_state_generation == 4
    message = types.SimpleNamespace(
        name=list(motion_module.ARM_JOINTS),
        position=[0.0] * len(motion_module.ARM_JOINTS),
    )

    def publish_updates():
        time.sleep(0.01)
        client._on_joint_state(message)
        client._on_joint_state(message)

    publisher = threading.Thread(target=publish_updates)
    publisher.start()
    generation = client.wait_for_joint_state_updates(
        after_generation=4,
        minimum_updates=2,
        timeout_sec=0.5,
    )
    publisher.join()

    assert generation == 6


def test_aborted_action_exposes_the_server_result_message(motion_module):
    """Planner diagnostics must not be reduced to an opaque status number."""
    result = types.SimpleNamespace(message='No valid joint state available')
    result_future = _FakeFuture(
        _FakeResultResponse(
            motion_module.GoalStatus.STATUS_ABORTED,
            result=result,
        )
    )
    goal_handle = _FakeGoalHandle(_FakeFuture(), result_future)
    client = _motion_client_without_ros(motion_module)

    with pytest.raises(
        motion_module.MotionError,
        match='No valid joint state available',
    ):
        client._send(
            _FakeActionClient(_FakeFuture(goal_handle)),
            object(),
            0.5,
            'cuMotion motion plan',
        )


def test_cancel_goal_raises_when_no_terminal_result_is_confirmed(
    motion_module,
    monkeypatch,
):
    """Never treat a rejected cancel plus pending result as stopped."""
    result_future = _FakeFuture()
    cancel_future = _FakeFuture(_FakeCancelResponse(1))
    goal_handle = _FakeGoalHandle(cancel_future, result_future)
    client = _motion_client_without_ros(motion_module)

    def finish_only_cancel(_node, future, _timeout_sec):
        if future is cancel_future:
            return cancel_future.result()
        assert future is result_future
        raise motion_module.MotionError('terminal result timeout')

    monkeypatch.setattr(
        motion_module,
        '_spin_until_complete',
        finish_only_cancel,
    )

    with pytest.raises(
        motion_module.MotionError,
        match='did not reach a terminal result.*return code 1',
    ):
        client._cancel_goal(goal_handle, result_future, 'gripper')

    assert goal_handle.cancel_calls == 1
    assert not result_future.done()


def test_close_waits_for_a_fresh_physical_attachment_ack():
    """Joint stall alone cannot prove that the object was grasped."""
    close_source = _method_source(
        DRIVER_SOURCE,
        '_close_gripper',
        '_open_gripper',
    )

    assert 'completion_predicate=' in close_source
    assert '_attachment_snapshot_since(' in close_source
    assert 'True,' in close_source


def test_physical_grasp_ack_survives_a_late_gripper_action_timeout():
    """A real FixedJoint must not be discarded by controller stall timing."""
    close_source = _method_source(
        DRIVER_SOURCE,
        '_close_gripper',
        '_open_gripper',
    )

    assert 'except MotionError as error:' in close_source
    assert 'deadline = time.monotonic() + 0.5' in close_source
    assert 'physical_attachment_confirmed()' in close_source
    assert 'already acknowledged a physical grasp' in close_source


def test_open_waits_for_the_physical_release_ack():
    """Retreat must not start while the FixedJoint still holds the block."""
    open_source = _method_source(
        DRIVER_SOURCE,
        '_open_gripper',
        '_publish_event',
    )

    assert '_wait_for_attachment_state(' in open_source
    assert 'False,' in open_source
    assert 'did not acknowledge ' in open_source
    assert "'object release'" in open_source


def test_driver_publishes_explicit_close_and_open_intent():
    """Action cancellation/hold positions must not define grasp semantics."""
    close_source = _method_source(
        DRIVER_SOURCE,
        '_close_gripper',
        '_open_gripper',
    )
    open_source = _method_source(
        DRIVER_SOURCE,
        '_open_gripper',
        '_latest_object_pose',
    )

    assert close_source.index(
        'self._publish_gripper_intent(True)'
    ) < close_source.index('self._motion.command_gripper(')
    assert open_source.index(
        'self._publish_gripper_intent(False)'
    ) < open_source.index('self._motion.command_gripper(')


def test_driver_establishes_ack_freshness_before_publishing_intent():
    """An immediate simulator ACK must not fall before the time barrier."""
    close_source = _method_source(
        DRIVER_SOURCE,
        '_close_gripper',
        '_open_gripper',
    )
    open_source = _method_source(
        DRIVER_SOURCE,
        '_open_gripper',
        '_latest_object_pose',
    )

    assert close_source.index(
        'started_at = time.monotonic()'
    ) < close_source.index('self._publish_gripper_intent(True)')
    assert close_source.index(
        'baseline_generation = self._attachment_generation'
    ) < close_source.index('self._publish_gripper_intent(True)')
    assert open_source.index(
        'started_at = time.monotonic()'
    ) < open_source.index('self._publish_gripper_intent(False)')
    assert open_source.index(
        'baseline_generation = self._attachment_generation'
    ) < open_source.index('self._publish_gripper_intent(False)')


def test_carry_phases_recheck_physical_attachment():
    """A dropped or briefly reattached block invalidates the demonstration."""
    assert 'self._attachment_loss_generation += 1' in DRIVER_SOURCE
    assert (
        "self._require_attachment(phase, 'before planning')" in DRIVER_SOURCE
    )
    assert (
        "self._require_attachment(phase, 'after planning')" in DRIVER_SOURCE
    )
    assert (
        "self._require_attachment(phase, 'after execution')" in DRIVER_SOURCE
    )


def test_first_attachment_ack_latches_the_loss_generation():
    """A drop/re-attach during action cancellation must remain a failure."""
    close_source = _method_source(
        DRIVER_SOURCE,
        '_close_gripper',
        '_open_gripper',
    )

    latch = close_source.index(
        'self._carry_attachment_loss_generation = ('
    )
    command = close_source.index('self._motion.command_gripper(')
    final_check = close_source.index(
        "self._require_attachment('close_gripper', 'after action completion')"
    )
    assert latch < final_check
    assert command < final_check


def test_failed_cleanup_stops_the_remaining_seed_batch():
    """Unknown physical/XRDF state cannot be reused by the next episode."""
    run_start = DRIVER_SOURCE.index('    def _run(self, run_id: str')
    run_end = DRIVER_SOURCE.index('    def start_run', run_start)
    run_source = DRIVER_SOURCE[run_start:run_end]

    assert 'if not self._cleanup_failed_episode():' in run_source
    assert 'state indeterminate; stopping this batch' in run_source


def test_failed_carry_keeps_the_gripper_closed_instead_of_dropping():
    """A transfer/place failure must stop safely with the object held."""
    cleanup_source = _method_source(
        DRIVER_SOURCE,
        '_cleanup_failed_episode',
        '_latest_object_pose',
    )

    carry_check = cleanup_source.index('if physically_carrying:')
    stop = cleanup_source.index('return False', carry_check)
    open_gripper = cleanup_source.index('self._open_gripper()', stop)
    assert carry_check < stop < open_gripper
    assert 'releasing in mid-air' in cleanup_source


def test_attachment_hot_reload_waits_for_fresh_joint_states():
    """Attach and detach must not race the replacement RobotManager."""
    attach_source = _method_source(
        DRIVER_SOURCE,
        '_attach_object_to_cumotion',
        '_detach_object_from_cumotion',
    )
    detach_source = _method_source(
        DRIVER_SOURCE,
        '_detach_object_from_cumotion',
        '_cleanup_failed_episode',
    )

    assert attach_source.index(
        'self._object_attachment.attach('
    ) < attach_source.index(
        "self._wait_for_cumotion_robot_manager('object attach')"
    )
    assert detach_source.index(
        'self._object_attachment.detach('
    ) < detach_source.index(
        "self._wait_for_cumotion_robot_manager('object detach')"
    )
    assert 'ROBOT_MANAGER_REFRESH_UPDATES = 2' in DRIVER_SOURCE
    retry_source = _method_source(
        DRIVER_SOURCE,
        '_plan_after_cumotion_reload',
        '_lookup_attachment_transform',
    )
    assert 'retrying once' in retry_source
    assert 'minimum_updates=1' in retry_source


def test_simulator_publishes_the_fixed_joint_state_as_ros_bool():
    """The physical controller and episode driver need an explicit bridge."""
    assert "ATTACHMENT_STATE_TOPIC = '/vla/object_attached'" in (
        PACKAGE_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'episode_events.py'
    ).read_text(encoding='utf-8')
    assert "'isaacsim.ros2.bridge.ROS2Publisher'" in SIMULATION_SOURCE
    assert "'omni.graph.nodes.ConstantBool'" in SIMULATION_SOURCE
    assert "inputs:messageName',\n                'Bool'" in SIMULATION_SOURCE
    assert 'ATTACHMENT_STATE_VALUE_NODE}.inputs:value' in SIMULATION_SOURCE
    assert (
        'ATTACHMENT_STATE_VALUE_NODE}.outputs:value' not in SIMULATION_SOURCE
    )
    assert '_connect_attachment_state_graph()' in SIMULATION_SOURCE
    assert 'authored_attachment.attached' in SIMULATION_SOURCE


def test_simulator_subscribes_to_explicit_gripper_intent():
    """VLA attachment must ignore action-controller hold positions."""
    events_source = (
        PACKAGE_ROOT
        / 'arx_r5_isaac_sim_bringup'
        / 'vla'
        / 'episode_events.py'
    ).read_text(encoding='utf-8')
    assert (
        "GRIPPER_INTENT_TOPIC = '/vla/gripper_close_intent'"
        in events_source
    )
    assert "GRIPPER_INTENT_NODE = 'SubscribeGripperIntent'" in (
        SIMULATION_SOURCE
    )
    assert 'close_requested = self._intent.close_requested()' in (
        SIMULATION_SOURCE
    )
    assert 'gripper_intent_topic=(' in SIMULATION_SOURCE
    assert 'GRIPPER_INTENT_TOPIC if task_config is not None' in (
        SIMULATION_SOURCE
    )


def test_ros_launch_refuses_a_sim_without_the_attachment_topic():
    """An old simulator must fail before collection instead of hanging."""
    assert '/vla/object_attached' in RUN_SCRIPT
    assert '/vla/gripper_close_intent' in RUN_SCRIPT


def test_vla_runner_rejects_stale_singleton_ros_graph_entities():
    """A second action server would make results nondeterministic."""
    assert 'stale_nodes=(' in RUN_SCRIPT
    assert '/controller_manager' in RUN_SCRIPT
    assert '/move_group' in RUN_SCRIPT
    assert 'stale_actions=(' in RUN_SCRIPT
    assert '/gripper_controller/gripper_cmd' in RUN_SCRIPT
    assert 'ros2 service list' in RUN_SCRIPT
    assert '--include-hidden-services' in RUN_SCRIPT
    assert 'node_list=' in RUN_SCRIPT
    assert 'ros2 action list --no-daemon' not in RUN_SCRIPT
    assert 'ROS graph conflict:' in RUN_SCRIPT


def test_vla_sim_uses_a_critically_damped_drive_with_tracking_margin():
    """Do not hide joint3 gravity error by loosening controller tolerance."""
    sim_script = (
        PACKAGE_ROOT.parent / 'scripts' / 'run_vla_sim.sh'
    ).read_text(encoding='utf-8')
    assert 'ARX_VLA_DRIVE_STIFFNESS:-1600.0' in sim_script
    assert 'ARX_VLA_DRIVE_DAMPING:-80.0' in sim_script
