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
Client contract for the official cuMotion attached-object model.

Isaac Sim's PhysX ``FixedJoint`` and cuMotion's collision attachment are two
separate states.  The former makes the block move with the wrist; the latter
adds collision spheres below the XRDF ``attached_object`` frame so future
``MotionPlan`` requests account for the carried geometry.

The NVIDIA action server accepts cancellation but does not inspect the
canceling state while it updates the robot description.  Consequently this
client treats ``timeout_sec`` as a warning threshold after a goal has been
sent: it never abandons the request or cancels only the local future, and it
continues draining the terminal result before returning.  That prevents a
late background service update from racing the next planning request.
"""

import math
import time
from typing import Optional, Sequence

from action_msgs.msg import GoalStatus

from isaac_ros_cumotion_interfaces.action import AttachObject

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from visualization_msgs.msg import Marker


DEFAULT_ATTACH_ACTION = '/attach_object'
DEFAULT_ATTACHMENT_FRAME = 'grasp_frame'
QUATERNION_NORM_TOLERANCE = 1e-3
SUPPORTED_MARKER_TYPES = (
    Marker.SPHERE,
    Marker.CUBE,
    Marker.MESH_RESOURCE,
)


class ObjectAttachmentError(RuntimeError):
    """Raised when the official cuMotion attachment state is not confirmed."""


def validate_attachment_marker(
    marker: Marker,
    expected_frame: str = DEFAULT_ATTACHMENT_FRAME,
) -> None:
    """
    Validate the subset of ``Marker`` consumed by NVIDIA's action server.

    The server does not transform ``marker.pose`` from an arbitrary header
    frame.  It directly writes generated sphere centres under the XRDF
    ``attached_object`` link, which is coincident with ``grasp_frame`` for the
    ARX model.  Requiring that frame here turns an otherwise silent collision
    geometry error into an immediate exception.
    """
    frame_id = str(marker.header.frame_id)
    if frame_id != expected_frame:
        raise ValueError(
            'attached-object marker pose must be expressed in '
            f'{expected_frame!r}, got {frame_id!r}'
        )
    if int(marker.type) not in SUPPORTED_MARKER_TYPES:
        raise ValueError(
            'attached-object marker type must be SPHERE, CUBE, or '
            'MESH_RESOURCE'
        )

    scale = (marker.scale.x, marker.scale.y, marker.scale.z)
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in scale
    ):
        raise ValueError(
            'attached-object marker scale must contain three positive values'
        )

    orientation = marker.pose.orientation
    quaternion = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    if any(not math.isfinite(value) for value in quaternion):
        raise ValueError('attached-object marker orientation must be finite')
    squared_norm = sum(value * value for value in quaternion)
    if abs(squared_norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            'attached-object marker orientation must be a normalized '
            'quaternion'
        )

    if (
        int(marker.type) == Marker.MESH_RESOURCE
        and not str(marker.mesh_resource).strip()
    ):
        raise ValueError(
            'MESH_RESOURCE attachment requires marker.mesh_resource'
        )


def cuboid_marker_from_transform(
    transform,
    size: Sequence[float],
    frame_id: str = DEFAULT_ATTACHMENT_FRAME,
) -> Marker:
    """
    Build a cuboid marker from an object transform in ``grasp_frame``.

    ``transform`` should be the result of
    ``lookup_transform('grasp_frame', object_frame, Time())``.  ROS expresses
    that transform as the pose of the object frame in ``grasp_frame``, exactly
    the convention required by the cuMotion attachment node.  ``size`` is the
    cuboid's full x/y/z extent in metres, not half extents.
    """
    dimensions = tuple(float(value) for value in size)
    if len(dimensions) != 3:
        raise ValueError(
            'cuboid attachment size must contain exactly three values'
        )
    source_frame = str(transform.header.frame_id)
    if source_frame != frame_id:
        raise ValueError(
            f'transform target frame must be {frame_id!r}, '
            f'got {source_frame!r}'
        )

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = transform.header.stamp
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.ns = 'vla_attached_object'
    marker.id = 0

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    marker.pose.position.x = float(translation.x)
    marker.pose.position.y = float(translation.y)
    marker.pose.position.z = float(translation.z)
    marker.pose.orientation.x = float(rotation.x)
    marker.pose.orientation.y = float(rotation.y)
    marker.pose.orientation.z = float(rotation.z)
    marker.pose.orientation.w = float(rotation.w)
    marker.scale.x, marker.scale.y, marker.scale.z = dimensions
    marker.frame_locked = True
    marker.color.r = 1.0
    marker.color.a = 1.0

    validate_attachment_marker(marker, expected_frame=frame_id)
    return marker


class CumotionObjectAttachmentClient:
    """Synchronously manage the official cuMotion attached-object action."""

    def __init__(
        self,
        node: Node,
        action_name: str = DEFAULT_ATTACH_ACTION,
        attachment_frame: str = DEFAULT_ATTACHMENT_FRAME,
    ) -> None:
        """Create one action client owned by a multi-threaded ROS node."""
        self._node = node
        self._attachment_frame = str(attachment_frame)
        if not self._attachment_frame:
            raise ValueError('attachment_frame must not be empty')
        self._client = ActionClient(
            node,
            AttachObject,
            action_name,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._known_attached: Optional[bool] = None

    @property
    def known_attached(self) -> Optional[bool]:
        """
        Return the last state proven by this client.

        ``None`` means that no terminal action result has established the
        current planner state.
        """
        return self._known_attached

    def wait_for_server(self, timeout_sec: float = 60.0) -> None:
        """Wait for ``/attach_object`` before any goal can be emitted."""
        timeout = _positive_timeout(timeout_sec)
        if not self._client.wait_for_server(timeout_sec=timeout):
            raise ObjectAttachmentError(
                'cuMotion object attachment action server did not appear '
                f'within {timeout:.1f} s'
            )

    def attach(self, marker: Marker, timeout_sec: float = 30.0):
        """Attach a marker to the cuMotion model and confirm success."""
        validate_attachment_marker(
            marker,
            expected_frame=self._attachment_frame,
        )
        goal = AttachObject.Goal()
        goal.attach_object = True
        goal.object_config = marker
        result = self._send_and_drain(
            goal,
            timeout_sec=timeout_sec,
            label='cuMotion object attach',
            allow_rejected=False,
        )
        outcome = str(getattr(result, 'outcome', ''))
        if 'attached' not in outcome.lower():
            self._known_attached = None
            raise ObjectAttachmentError(
                f'cuMotion attach returned an unexpected outcome: {outcome!r}'
            )
        self._known_attached = True
        return result

    def detach(
        self,
        timeout_sec: float = 30.0,
        allow_already_detached: bool = False,
    ):
        """
        Restore the original XRDF and confirm the object was detached.

        When ``allow_already_detached`` is true, a rejected detach goal is
        interpreted using the official server contract: detach is rejected
        only when its internal state already contains no attached object.
        """
        goal = AttachObject.Goal()
        goal.attach_object = False
        result = self._send_and_drain(
            goal,
            timeout_sec=timeout_sec,
            label='cuMotion object detach',
            allow_rejected=allow_already_detached,
        )
        if result is None:
            self._known_attached = False
            return None
        outcome = str(getattr(result, 'outcome', ''))
        if 'detached' not in outcome.lower():
            self._known_attached = None
            raise ObjectAttachmentError(
                f'cuMotion detach returned an unexpected outcome: {outcome!r}'
            )
        self._known_attached = False
        return result

    def ensure_detached(self, timeout_sec: float = 30.0) -> bool:
        """
        Make the planner bare, including after a driver-only restart.

        Returns ``True`` when a real detach action ran and ``False`` when the
        official server rejected the goal because it was already detached.
        """
        if self._known_attached is False:
            return False
        result = self.detach(
            timeout_sec=timeout_sec,
            allow_already_detached=True,
        )
        return result is not None

    def _wait_for_future(self, future, timeout_sec: float, label: str):
        """Wait through a soft timeout without orphaning a server mutation."""
        timeout = _positive_timeout(timeout_sec)
        deadline = time.monotonic() + timeout
        warned = False
        while rclpy.ok() and not future.done():
            if not warned and time.monotonic() > deadline:
                warned = True
                self._node.get_logger().warning(
                    f'{label} exceeded {timeout:.1f} s; continuing to drain '
                    'its '
                    'response because the official server does not stop its '
                    'robot-description update when cancellation is requested'
                )
            time.sleep(0.02)
        if not future.done():
            self._known_attached = None
            raise ObjectAttachmentError(
                f'{label} was interrupted before its state became terminal; '
                'cuMotion attachment state is indeterminate'
            )
        try:
            return future.result()
        except Exception as error:
            self._known_attached = None
            raise ObjectAttachmentError(
                f'{label} request failed: {error}'
            ) from error

    def _send_and_drain(
        self,
        goal,
        timeout_sec: float,
        label: str,
        allow_rejected: bool,
    ):
        """Send one goal and retain every pending request until terminal."""
        timeout = _positive_timeout(timeout_sec)
        if not self._client.wait_for_server(timeout_sec=timeout):
            raise ObjectAttachmentError(
                f'{label} action server is unavailable'
            )

        send_future = self._client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            send_future,
            timeout,
            f'{label} goal response',
        )
        if (
            goal_handle is None
            or not bool(getattr(goal_handle, 'accepted', False))
        ):
            if allow_rejected:
                return None
            self._known_attached = None
            raise ObjectAttachmentError(f'{label} goal was rejected')

        result_response = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout,
            f'{label} terminal result',
        )
        if result_response is None:
            self._known_attached = None
            raise ObjectAttachmentError(f'{label} returned no result')
        if int(result_response.status) != int(GoalStatus.STATUS_SUCCEEDED):
            self._known_attached = None
            raise ObjectAttachmentError(
                f'{label} finished with goal status {result_response.status}'
            )
        return result_response.result


def _positive_timeout(value: float) -> float:
    """Return one finite positive timeout value."""
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError('timeout_sec must be finite and positive')
    return timeout
