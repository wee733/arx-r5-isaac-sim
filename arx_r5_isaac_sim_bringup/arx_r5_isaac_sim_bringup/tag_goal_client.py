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

"""Convert a destination AprilTag into an official workflow goal."""

from math import isfinite
from time import monotonic

from geometry_msgs.msg import PoseStamped

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray

from isaac_ros_manipulation_interfaces.action import MultiObjectPickAndPlace

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from tf2_ros import Buffer, TransformException, TransformListener

from .pose_math import compose_pose, normalize_quaternion
from .tag_goal_policy import TargetSequenceGate, TranslationMedianGate


def _vector_parameter(node: Node, name: str, default, size: int) -> tuple:
    parameter = node.declare_parameter(name, default)
    values = tuple(float(value) for value in parameter.value)
    if len(values) != size or not all(isfinite(value) for value in values):
        raise ValueError(f'{name} must contain {size} finite values')
    return values


def _stamp_to_nanoseconds(header) -> int:
    """Return an exact integer timestamp from a ROS Header."""
    return (
        int(header.stamp.sec) * 1_000_000_000 +
        int(header.stamp.nanosec)
    )


class AprilTagPickPlaceGoalClient(Node):
    """Send one workflow goal after both tags become stable."""

    def __init__(self) -> None:
        """Configure exact-time TF and destination-pose freshness policy."""
        super().__init__('apriltag_pick_place_goal_client')
        self._detections_topic = str(
            self.declare_parameter('detections_topic', '/tag_detections').value
        )
        self._action_name = str(
            self.declare_parameter(
                'action_name',
                '/multi_object_pick_and_place',
            ).value
        )
        self._base_frame = str(
            self.declare_parameter('base_frame', 'base_link').value
        ).strip()
        self._expected_camera_frame = str(
            self.declare_parameter('expected_camera_frame', '').value
        ).strip()
        self._tag_family = str(
            self.declare_parameter('tag_family', 'tag36h11').value
        )
        self._source_tag_id = int(
            self.declare_parameter('source_tag_id', 0).value
        )
        self._target_tag_id = int(
            self.declare_parameter('target_tag_id', 1).value
        )
        self._stable_frames = int(
            self.declare_parameter('stable_frames', 5).value
        )
        self._max_translation_spread_m = float(
            self.declare_parameter(
                'max_translation_spread_m',
                0.01,
            ).value
        )
        self._auto_start = bool(
            self.declare_parameter('auto_start', True).value
        )
        self._transform_timeout_sec = float(
            self.declare_parameter('transform_timeout_sec', 2.0).value
        )
        if (
            not isfinite(self._transform_timeout_sec)
            or self._transform_timeout_sec <= 0.0
        ):
            raise ValueError('transform_timeout_sec must be positive')
        self._drop_pose_ttl_sec = float(
            self.declare_parameter('drop_pose_ttl_sec', 0.5).value
        )
        if (
            not isfinite(self._drop_pose_ttl_sec) or
            self._drop_pose_ttl_sec <= 0.0
        ):
            raise ValueError('drop_pose_ttl_sec must be positive')
        self._drop_pose_ttl_ns = int(
            self._drop_pose_ttl_sec * 1_000_000_000
        )
        self._target_offset = _vector_parameter(
            self,
            'link6_offset_in_tag',
            [0.0, 0.0, 0.165],
            3,
        )
        self._target_rotation = normalize_quaternion(_vector_parameter(
            self,
            'link6_rotation_in_tag',
            [0.0, 0.7071068, 0.0, 0.7071068],
            4,
        ))
        if self._source_tag_id == self._target_tag_id:
            raise ValueError('source_tag_id and target_tag_id must differ')
        if not self._base_frame:
            raise ValueError('base_frame must be non-empty')
        if not self._expected_camera_frame:
            raise ValueError('expected_camera_frame must be non-empty')
        if self._stable_frames < 1:
            raise ValueError('stable_frames must be at least one')
        if (
            not isfinite(self._max_translation_spread_m) or
            self._max_translation_spread_m <= 0.0
        ):
            raise ValueError(
                'max_translation_spread_m must be finite and positive'
            )

        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action_client = ActionClient(
            self,
            MultiObjectPickAndPlace,
            self._action_name,
        )
        self._drop_pose_publisher = self.create_publisher(
            PoseStamped,
            '/arx_r5_demo/drop_pose',
            1,
        )
        self._subscription = self.create_subscription(
            AprilTagDetectionArray,
            self._detections_topic,
            self._on_detections,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(0.5, self._maybe_send_goal)
        self._source_frames = 0
        self._target_sequence = TargetSequenceGate()
        self._translation_gate = TranslationMedianGate(
            window_size=self._stable_frames,
            max_spread_m=self._max_translation_spread_m,
        )
        self._queued_target = None
        self._pending_transform = None
        self._pending_transform_started = None
        self._drop_pose = None
        self._goal_pending = False
        self._goal_sent = False
        self.get_logger().info(
            f'Waiting for {self._tag_family}:{self._source_tag_id} and '
            f'{self._tag_family}:{self._target_tag_id} on '
            f'{self._detections_topic}'
        )

    def _invalidate_target_sequence(self) -> None:
        """Discard work and poses from a broken visible sequence."""
        self._target_sequence.miss_target()
        self._translation_gate.reset()
        self._queued_target = None
        pending = self._pending_transform
        self._pending_transform = None
        self._pending_transform_started = None
        self._drop_pose = None
        if pending is not None:
            pending[0].cancel()

    def _on_detections(self, message: AprilTagDetectionArray) -> None:
        if message.header.frame_id != self._expected_camera_frame:
            self.get_logger().error(
                'Rejecting AprilTag array in frame '
                f'{message.header.frame_id!r}; expected '
                f'{self._expected_camera_frame!r}'
            )
            self._source_frames = 0
            self._invalidate_target_sequence()
            return
        source_seen = False
        target_detection = None
        for detection in message.detections:
            if detection.family != self._tag_family:
                continue
            if int(detection.id) == self._source_tag_id:
                source_seen = True
            elif int(detection.id) == self._target_tag_id:
                target_detection = detection

        self._source_frames = self._source_frames + 1 if source_seen else 0
        if target_detection is None:
            self._invalidate_target_sequence()
            return
        generation = self._target_sequence.observe_target()
        self._queued_target = (
            generation,
            message.header,
            target_detection.pose.pose.pose,
        )
        self._start_transform_request()

    def _start_transform_request(self) -> None:
        """Wait asynchronously for the TF matching one camera frame."""
        if self._pending_transform is not None or self._queued_target is None:
            return

        generation, header, target_pose = self._queued_target
        self._queued_target = None
        try:
            future = self._tf_buffer.wait_for_transform_async(
                self._base_frame,
                header.frame_id,
                Time.from_msg(header.stamp),
            )
        except (TransformException, TypeError, ValueError) as error:
            self.get_logger().debug(
                f'Waiting for timestamped camera TF: {error}'
            )
            return

        self._pending_transform = (future, generation, header, target_pose)
        self._pending_transform_started = monotonic()
        future.add_done_callback(self._on_transform_ready)

    def _on_transform_ready(self, future) -> None:
        pending = self._pending_transform
        if pending is None or pending[0] is not future:
            return
        _, generation, header, target_pose = pending
        self._pending_transform = None
        self._pending_transform_started = None
        if future.cancelled():
            self._start_transform_request()
            return
        try:
            transform = future.result()
        except (TransformException, TypeError, ValueError) as error:
            self.get_logger().debug(
                f'Waiting for timestamped camera TF: {error}'
            )
            self._start_transform_request()
            return

        if generation == self._target_sequence.generation:
            self._publish_drop_pose(
                transform,
                generation,
                header,
                target_pose,
            )
        self._start_transform_request()

    def _expire_transform_request(self) -> None:
        """Discard a camera frame whose exact TF never entered the buffer."""
        if (
            self._pending_transform is None or
            self._pending_transform_started is None or
            monotonic() - self._pending_transform_started
            <= self._transform_timeout_sec
        ):
            return
        future = self._pending_transform[0]
        self._pending_transform = None
        self._pending_transform_started = None
        future.cancel()
        self._start_transform_request()

    def _publish_drop_pose(
        self,
        transform,
        generation,
        header,
        target_pose,
    ) -> None:
        """Compose and publish the destination pose in the robot base frame."""
        transform_translation = (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        )
        transform_rotation = normalize_quaternion((
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ))
        tag_translation = (
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        )
        tag_rotation = normalize_quaternion((
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w,
        ))
        base_tag_translation, base_tag_rotation = compose_pose(
            transform_translation,
            transform_rotation,
            tag_translation,
            tag_rotation,
        )
        drop_translation, drop_rotation = compose_pose(
            base_tag_translation,
            base_tag_rotation,
            self._target_offset,
            self._target_rotation,
        )

        stamp_ns = _stamp_to_nanoseconds(header)
        stable_translation = self._translation_gate.observe(
            generation=generation,
            stamp_ns=stamp_ns,
            translation=drop_translation,
        )
        if stable_translation is None:
            # Never retain an older stable estimate while the newest complete
            # window says that the target is moving or the PnP depth is noisy.
            self._drop_pose = None
            if self._translation_gate.spread_m is not None:
                self.get_logger().debug(
                    'Destination translation is not stable: window spread '
                    f'{self._translation_gate.spread_m:.4f} m exceeds '
                    f'{self._max_translation_spread_m:.4f} m'
                )
            return

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = header.stamp
        pose_stamped.header.frame_id = self._base_frame
        pose_stamped.pose.position.x = stable_translation[0]
        pose_stamped.pose.position.y = stable_translation[1]
        pose_stamped.pose.position.z = stable_translation[2]
        pose_stamped.pose.orientation.x = drop_rotation[0]
        pose_stamped.pose.orientation.y = drop_rotation[1]
        pose_stamped.pose.orientation.z = drop_rotation[2]
        pose_stamped.pose.orientation.w = drop_rotation[3]
        if not self._target_sequence.accept_pose(
            generation,
            stamp_ns,
        ):
            return
        self._drop_pose = pose_stamped
        self._drop_pose_publisher.publish(pose_stamped)

    def _maybe_send_goal(self) -> None:
        self._expire_transform_request()
        now_ns = self.get_clock().now().nanoseconds
        ready = (
            self._source_frames >= self._stable_frames
            and self._drop_pose is not None
            and self._target_sequence.ready(
                required_frames=self._stable_frames,
                now_ns=now_ns,
                ttl_ns=self._drop_pose_ttl_ns,
            )
        )
        if (
            not ready or not self._auto_start or
            self._goal_pending or self._goal_sent
        ):
            return
        if not self._action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().debug(
                f'Waiting for {self._action_name} action server'
            )
            return

        goal = MultiObjectPickAndPlace.Goal()
        goal.mode = MultiObjectPickAndPlace.Goal.SINGLE_BIN
        goal.target_poses.header = self._drop_pose.header
        goal.target_poses.poses = [self._drop_pose.pose]
        goal.class_ids = []
        self._goal_pending = True
        future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._on_feedback,
        )
        future.add_done_callback(self._on_goal_response)
        self.get_logger().info(
            'Both AprilTags are stable; sent the official '
            'MultiObjectPickAndPlace goal'
        )

    def _on_goal_response(self, future) -> None:
        self._goal_pending = False
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Pick-and-place goal was rejected')
            return
        self._goal_sent = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_message) -> None:
        message = feedback_message.feedback.message
        if message:
            self.get_logger().info(f'Workflow: {message}')

    def _on_result(self, future) -> None:
        wrapped_result = future.result()
        result = wrapped_result.result
        self.get_logger().info(
            'Pick-and-place finished: '
            f'status={result.workflow_status}, '
            f'summary={result.workflow_summary}'
        )


def main() -> None:
    """Run the destination-tag action adapter."""
    rclpy.init()
    node = AprilTagPickPlaceGoalClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
