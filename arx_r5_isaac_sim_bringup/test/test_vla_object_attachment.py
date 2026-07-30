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

"""Pure-Python contract tests for the official cuMotion attachment client."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = (
    PACKAGE_ROOT
    / 'arx_r5_isaac_sim_bringup'
    / 'vla'
    / 'object_attachment_client.py'
)
LAUNCH_PATH = PACKAGE_ROOT / 'launch' / 'arx_r5a_vla_collect.launch.py'
DRIVER_PATH = (
    PACKAGE_ROOT
    / 'arx_r5_isaac_sim_bringup'
    / 'vla'
    / 'episode_driver.py'
)


def _install_module(monkeypatch, name: str, **members):
    """Install a lightweight generated-message/module stand-in."""
    module = ModuleType(name)
    module.__dict__.update(members)
    if '.' not in name:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)
    return module


class _Vector3:
    """Minimal geometry vector."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Quaternion(_Vector3):
    """Minimal geometry quaternion."""

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        super().__init__(x, y, z)
        self.w = float(w)


class _Marker:
    """Subset of visualization_msgs/Marker used by the client."""

    SPHERE = 2
    CUBE = 1
    MESH_RESOURCE = 10
    ADD = 0

    def __init__(self):
        self.header = SimpleNamespace(frame_id='', stamp=None)
        self.type = 0
        self.action = 0
        self.ns = ''
        self.id = 0
        self.pose = SimpleNamespace(
            position=_Vector3(),
            orientation=_Quaternion(),
        )
        self.scale = _Vector3()
        self.frame_locked = False
        self.color = SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
        self.mesh_resource = ''


class _AttachObject:
    """Generated action stand-in."""

    class Goal:
        def __init__(self):
            self.attach_object = False
            self.object_config = _Marker()


class _GoalStatus:
    """Action status constants used by rclpy clients."""

    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


@pytest.fixture(name='client_module')
def client_module_fixture(monkeypatch):
    """Import the client with ROS generated types replaced by test doubles."""
    class _ActionClient:
        def __init__(self, *_args, **_kwargs):
            pass

    _install_module(monkeypatch, 'action_msgs')
    _install_module(monkeypatch, 'action_msgs.msg', GoalStatus=_GoalStatus)
    _install_module(monkeypatch, 'isaac_ros_cumotion_interfaces')
    _install_module(
        monkeypatch,
        'isaac_ros_cumotion_interfaces.action',
        AttachObject=_AttachObject,
    )
    _install_module(monkeypatch, 'rclpy', ok=lambda: True)
    _install_module(monkeypatch, 'rclpy.action', ActionClient=_ActionClient)
    _install_module(
        monkeypatch,
        'rclpy.callback_groups',
        MutuallyExclusiveCallbackGroup=object,
    )
    _install_module(monkeypatch, 'rclpy.node', Node=object)
    _install_module(monkeypatch, 'visualization_msgs')
    _install_module(monkeypatch, 'visualization_msgs.msg', Marker=_Marker)

    module_name = '_vla_object_attachment_client_test_double'
    spec = importlib.util.spec_from_file_location(module_name, CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _Logger:
    """Capture warnings without requiring rclpy logging."""

    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(str(message))


class _Node:
    """Minimal node exposing a logger."""

    def __init__(self):
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


class _Future:
    """Deterministic future that can be completed by a fake clock."""

    def __init__(self, value=None, done=False):
        self._value = value
        self._done = bool(done)

    def done(self):
        return self._done

    def result(self):
        return self._value

    def set_result(self, value):
        self._value = value
        self._done = True


class _GoalHandle:
    """Accepted action goal with a terminal result future."""

    def __init__(self, result_future, accepted=True):
        self.accepted = accepted
        self._result_future = result_future

    def get_result_async(self):
        return self._result_future


class _ActionClientDouble:
    """Action client with controllable send and result futures."""

    def __init__(self, send_future):
        self.send_future = send_future
        self.goals = []

    def wait_for_server(self, timeout_sec):
        del timeout_sec
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.send_future


def _make_marker(module, frame='grasp_frame'):
    """Build a valid marker without needing generated ROS messages."""
    marker = module.Marker()
    marker.header.frame_id = frame
    marker.type = module.Marker.CUBE
    marker.scale = _Vector3(0.04, 0.04, 0.15)
    marker.pose.orientation = _Quaternion()
    return marker


def _make_transform(module, frame='grasp_frame'):
    """Build T_grasp_frame_block for marker construction tests."""
    del module
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame, stamp='sim-stamp'),
        transform=SimpleNamespace(
            translation=_Vector3(0.01, -0.02, 0.04),
            rotation=_Quaternion(0.0, 0.0, 0.5, 0.8660254037844386),
        ),
    )


def _make_wrapper(module, fake_action_client):
    """Construct a wrapper without invoking the real ActionClient."""
    wrapper = module.CumotionObjectAttachmentClient.__new__(
        module.CumotionObjectAttachmentClient
    )
    wrapper._node = _Node()
    wrapper._attachment_frame = 'grasp_frame'
    wrapper._client = fake_action_client
    wrapper._known_attached = None
    return wrapper


def test_cuboid_marker_uses_object_pose_in_grasp_frame(client_module):
    """The Marker pose and full dimensions follow the official contract."""
    marker = client_module.cuboid_marker_from_transform(
        _make_transform(client_module),
        (0.04, 0.04, 0.15),
    )

    assert marker.header.frame_id == 'grasp_frame'
    assert marker.header.stamp == 'sim-stamp'
    assert marker.type == client_module.Marker.CUBE
    assert marker.scale.x == pytest.approx(0.04)
    assert marker.scale.y == pytest.approx(0.04)
    assert marker.scale.z == pytest.approx(0.15)
    assert marker.pose.position.x == pytest.approx(0.01)
    assert marker.pose.position.y == pytest.approx(-0.02)
    assert marker.pose.position.z == pytest.approx(0.04)
    assert marker.pose.orientation.z == pytest.approx(0.5)
    assert marker.pose.orientation.w == pytest.approx(0.8660254038)


def test_marker_rejects_pose_in_world_frame(client_module):
    """The client must not silently treat a world pose as grasp-relative."""
    with pytest.raises(ValueError, match='target frame'):
        client_module.cuboid_marker_from_transform(
            _make_transform(client_module, frame='world'),
            (0.04, 0.04, 0.15),
        )


def test_marker_validation_rejects_non_unit_quaternion(client_module):
    """The official server rejects malformed marker orientations."""
    marker = _make_marker(client_module)
    marker.pose.orientation.w = 2.0
    with pytest.raises(ValueError, match='normalized quaternion'):
        client_module.validate_attachment_marker(marker)


def test_timeout_is_soft_and_terminal_result_is_drained(
    client_module,
    monkeypatch,
):
    """Drain a late result before returning, without a local cancel."""
    send_future = _Future(done=False)
    result_future = _Future(done=False)
    result = SimpleNamespace(outcome='Object successfully attached')
    result_response = SimpleNamespace(
        status=_GoalStatus.STATUS_SUCCEEDED,
        result=result,
    )
    handle = _GoalHandle(result_future)
    action_client = _ActionClientDouble(send_future)
    wrapper = _make_wrapper(client_module, action_client)

    clock = {'now': 0.0, 'sleep_count': 0}

    def monotonic():
        return clock['now']

    def sleep(seconds):
        clock['now'] += seconds
        clock['sleep_count'] += 1
        if clock['sleep_count'] == 2:
            send_future.set_result(handle)
        if clock['sleep_count'] == 4:
            result_future.set_result(result_response)

    monkeypatch.setattr(
        client_module,
        'time',
        SimpleNamespace(monotonic=monotonic, sleep=sleep),
    )

    returned = wrapper.attach(_make_marker(client_module), timeout_sec=0.01)

    assert returned is result
    assert wrapper.known_attached is True
    assert action_client.goals[0].attach_object is True
    assert wrapper._node.logger.warnings


def test_ensure_detached_accepts_official_already_detached_rejection(
    client_module,
):
    """A restart can safely normalize a planner that is already detached."""
    result_future = _Future(done=True)
    send_future = _Future(
        value=_GoalHandle(result_future, accepted=False),
        done=True,
    )
    wrapper = _make_wrapper(client_module, _ActionClientDouble(send_future))

    assert wrapper.ensure_detached() is False
    assert wrapper.known_attached is False


def test_ensure_detached_does_not_repeat_a_proven_false_state(client_module):
    """Normal place->detach must not spam a rejected goal next episode."""
    action_client = _ActionClientDouble(_Future(done=True))
    wrapper = _make_wrapper(client_module, action_client)
    wrapper._known_attached = False

    assert wrapper.ensure_detached() is False
    assert action_client.goals == []


def test_attach_rejects_invalid_scale_before_sending(client_module):
    """No malformed goal should reach the official action server."""
    send_future = _Future(done=True)
    action_client = _ActionClientDouble(send_future)
    wrapper = _make_wrapper(client_module, action_client)
    marker = _make_marker(client_module)
    marker.scale.x = 0.0

    with pytest.raises(ValueError, match='positive'):
        wrapper.attach(marker)
    assert action_client.goals == []


def test_vla_launch_starts_official_attachment_component():
    """The VLA launch must expose the upstream component, not a mock server."""
    source = LAUNCH_PATH.read_text(encoding='utf-8')
    assert "'isaac_ros_cumotion_object_attachment'" in source
    assert "'launch/object_attachment.launch.py'" in source
    assert "'object_attachment.clear_esdf_on_attach': 'False'" in source
    assert "'object_attachment.container_name': 'cumotion_container'" in source
    assert (
        "DeclareLaunchArgument('start_object_attachment', "
        "default_value='True')" in source
    )


def test_driver_attaches_only_after_lift_and_detaches_after_release():
    """Mirror NVIDIA's safe support-surface attachment ordering."""
    source = DRIVER_PATH.read_text(encoding='utf-8')
    lift_branch = source.index('elif phase == EpisodePhase.LIFT:')
    attach = source.index('self._attach_object_to_cumotion()', lift_branch)
    place_branch = source.index('elif phase == EpisodePhase.PLACE:', attach)
    open_gripper = source.index('self._open_gripper()', place_branch)
    detach = source.index(
        'self._detach_object_from_cumotion()',
        open_gripper,
    )

    assert lift_branch < attach < place_branch
    assert place_branch < open_gripper < detach


def test_driver_normalizes_stale_attachment_before_homing():
    """A driver-only restart must not home with a ghost attached object."""
    source = DRIVER_PATH.read_text(encoding='utf-8')
    run_episode = source[source.index('    def _run_episode('):]
    setup = run_episode.index('# Setup, deliberately outside')
    ensure_detached = run_episode.index(
        'self._ensure_cumotion_detached()',
        setup,
    )
    open_gripper = run_episode.index('self._open_gripper()', setup)
    go_home = run_episode.index('self._go_home()', setup)

    assert ensure_detached < open_gripper < go_home


def test_transfer_and_place_require_official_attachment_state():
    """Physical FixedJoint alone is insufficient for collision-aware plans."""
    source = DRIVER_PATH.read_text(encoding='utf-8')
    assert 'phase in (EpisodePhase.TRANSFER, EpisodePhase.PLACE)' in source
    assert 'self._official_attached is not True' in source
    assert 'cuMotion attached-object model is not active' in source


def test_package_declares_direct_attachment_message_dependencies():
    """The new client must not rely on transitive ROS package dependencies."""
    source = (PACKAGE_ROOT / 'package.xml').read_text(encoding='utf-8')
    assert '<exec_depend>isaac_ros_cumotion_interfaces</exec_depend>' in source
    assert '<exec_depend>visualization_msgs</exec_depend>' in source
