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

"""Refine cuAprilTag poses from exact-time rectified camera images."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from time import monotonic

import cv2

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray

import numpy as np

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CameraInfo, Image

from .apriltag_pose_math import (
    is_stamp_rewind,
    refine_corners_subpixel_from_image,
    refine_pose,
    validate_image_message,
)


@dataclass
class _CachedImage:
    """One untouched image or an explicit unusable-image marker."""

    message: object
    rejection_reason: str = ''


def _stamp_key(message) -> tuple:
    """Return an exact ROS header stamp key without float conversion."""
    return (
        int(message.header.stamp.sec),
        int(message.header.stamp.nanosec),
    )


class AprilTagPoseRefiner(Node):
    """Replace only cuAprilTag's pose while preserving its CUDA contract."""

    def __init__(self) -> None:
        """Configure exact-stamp image/detection synchronization."""
        super().__init__('apriltag_pose_refiner')
        self._input_topic = str(
            self.declare_parameter(
                'input_topic',
                '/tag_detections_raw',
            ).value
        ).strip()
        self._output_topic = str(
            self.declare_parameter(
                'output_topic',
                '/tag_detections',
            ).value
        ).strip()
        self._camera_info_topic = str(
            self.declare_parameter(
                'camera_info_topic',
                '/camera_1/apriltag/camera_info_rect',
            ).value
        ).strip()
        self._image_topic = str(
            self.declare_parameter(
                'image_topic',
                '/camera_1/apriltag/image_rect',
            ).value
        ).strip()
        self._expected_frame = str(
            self.declare_parameter('expected_frame', '').value
        ).strip()
        self._apriltag_backend = str(
            self.declare_parameter('apriltag_backend', 'CUDA').value
        ).strip().upper()
        self._tag_size = float(
            self.declare_parameter('tag_size', 0.04).value
        )
        self._maximum_error = float(
            self.declare_parameter(
                'maximum_reprojection_error_px',
                2.0,
            ).value
        )
        self._sync_cache_size = int(
            self.declare_parameter('sync_cache_size', 16).value
        )
        self._image_wait_timeout_sec = float(
            self.declare_parameter('image_wait_timeout_sec', 0.25).value
        )
        if not self._input_topic or not self._output_topic:
            raise ValueError('detection topics must be non-empty')
        if self._input_topic == self._output_topic:
            raise ValueError('detection input and output topics must differ')
        if not self._camera_info_topic:
            raise ValueError('camera_info_topic must be non-empty')
        if not self._image_topic:
            raise ValueError('image_topic must be non-empty')
        if self._apriltag_backend != 'CUDA':
            raise ValueError(
                'apriltag_pose_refiner supports only the Isaac ROS CUDA '
                'corner-order contract'
            )
        if not isfinite(self._tag_size) or self._tag_size <= 0.0:
            raise ValueError('tag_size must be finite and positive')
        if not isfinite(self._maximum_error) or self._maximum_error <= 0.0:
            raise ValueError(
                'maximum_reprojection_error_px must be finite and positive'
            )
        if self._sync_cache_size <= 0:
            raise ValueError('sync_cache_size must be positive')
        if (
            not isfinite(self._image_wait_timeout_sec) or
            self._image_wait_timeout_sec <= 0.0
        ):
            raise ValueError(
                'image_wait_timeout_sec must be finite and positive'
            )

        self._camera_matrix = None
        self._camera_frame = ''
        self._reported_tags = set()
        self._reported_drops = set()
        self._images = OrderedDict()
        self._pending_detections = OrderedDict()
        self._last_image_stamp = None
        self._last_detection_stamp = None

        output_qos = QoSProfile(depth=10)
        output_qos.reliability = ReliabilityPolicy.RELIABLE
        self._publisher = self.create_publisher(
            AprilTagDetectionArray,
            self._output_topic,
            output_qos,
        )
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self._image_subscription = self.create_subscription(
            Image,
            self._image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._detection_subscription = self.create_subscription(
            AprilTagDetectionArray,
            self._input_topic,
            self._on_detections,
            qos_profile_sensor_data,
        )
        self._timeout_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._timeout_timer = self.create_timer(
            min(0.05, self._image_wait_timeout_sec),
            self._expire_unmatched_detections,
            clock=self._timeout_clock,
        )
        self.get_logger().info(
            f'Refining official {self._apriltag_backend} cuAprilTag corners '
            f'from {self._input_topic} using exact-time images on '
            f'{self._image_topic}; publishing {self._output_topic}'
        )

    def _on_camera_info(self, message: CameraInfo) -> None:
        matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        if (
            not np.all(np.isfinite(matrix)) or
            matrix[0, 0] <= 0.0 or
            matrix[1, 1] <= 0.0
        ):
            self.get_logger().error('Ignoring invalid rectified CameraInfo')
            return
        self._camera_matrix = matrix
        self._camera_frame = message.header.frame_id

    def _remember_image(self, key: tuple, sample: _CachedImage) -> None:
        self._images[key] = sample
        self._images.move_to_end(key)
        while len(self._images) > self._sync_cache_size:
            self._images.popitem(last=False)

    def _observe_stamp(self, stream: str, key: tuple) -> None:
        """Clear cross-stream caches when one simulator clock rewinds."""
        attribute = f'_last_{stream}_stamp'
        previous = getattr(self, attribute)
        if is_stamp_rewind(previous, key):
            self._images.clear()
            self._pending_detections.clear()
            self._last_image_stamp = None
            self._last_detection_stamp = None
            self.get_logger().warning(
                f'ROS timestamp rewound on the {stream} stream from '
                f'{previous[0]}.{previous[1]:09d} to '
                f'{key[0]}.{key[1]:09d}; cleared exact-stamp caches'
            )
        setattr(self, attribute, key)

    def _warn_once(self, key, message: str) -> None:
        """Report a repeated drop condition once without flooding logs."""
        if key in self._reported_drops:
            return
        self.get_logger().warning(message)
        self._reported_drops.add(key)

    def _drop_detection(self, reason_key, reason: str) -> None:
        """Drop one raw sample so native and refined poses never interleave."""
        self._warn_once(
            ('detection', reason_key),
            f'Dropping raw cuAprilTag sample from refined output: {reason}',
        )

    def _on_image(self, message: Image) -> None:
        key = _stamp_key(message)
        self._observe_stamp('image', key)
        try:
            # This checks only metadata and buffer length. Pixel conversion is
            # deferred until a detection with this exact stamp arrives.
            validate_image_message(message)
        except (TypeError, ValueError) as exception:
            sample = _CachedImage(
                None,
                f'exact-stamp image is unusable: {exception}',
            )
        else:
            sample = _CachedImage(message)

        pending = self._pending_detections.pop(key, None)
        if pending is not None:
            self._publish_paired_detection(pending[0], sample)
            return
        self._remember_image(key, sample)

    def _on_detections(self, message: AprilTagDetectionArray) -> None:
        key = _stamp_key(message)
        self._observe_stamp('detection', key)
        if self._expected_frame and message.header.frame_id != self._expected_frame:
            self.get_logger().error(
                f'Refusing detections in {message.header.frame_id!r}; '
                f'expected {self._expected_frame!r}'
            )
            return

        if key in self._images:
            sample = self._images.pop(key)
            self._publish_paired_detection(message, sample)
            return

        previous = self._pending_detections.pop(key, None)
        if previous is not None:
            self._drop_detection(
                'duplicate_stamp',
                'duplicate raw detection stamp replaced before image arrived',
            )
        self._pending_detections[key] = (message, monotonic())
        while len(self._pending_detections) > self._sync_cache_size:
            self._pending_detections.popitem(last=False)
            self._drop_detection(
                'cache_full',
                'exact-time image did not arrive before the sync cache filled',
            )

    def _expire_unmatched_detections(self) -> None:
        now = monotonic()
        expired_keys = [
            key for key, (_, started) in self._pending_detections.items()
            if now - started > self._image_wait_timeout_sec
        ]
        for key in expired_keys:
            self._pending_detections.pop(key)
            self._drop_detection(
                'image_timeout',
                'timed out waiting for the exact-stamp rectified image',
            )

    def _publish_paired_detection(
        self,
        message: AprilTagDetectionArray,
        sample: _CachedImage,
    ) -> None:
        if sample.message is None:
            self._drop_detection('invalid_image', sample.rejection_reason)
            return
        image_frame = sample.message.header.frame_id
        if image_frame != message.header.frame_id:
            self._drop_detection(
                'image_frame_mismatch',
                f'exact-stamp image frame {image_frame!r} differs from '
                f'detection frame {message.header.frame_id!r}',
            )
            return
        if self._camera_matrix is None:
            self._drop_detection(
                'missing_camera_info',
                f'CameraInfo has not arrived on {self._camera_info_topic}',
            )
            return
        if self._camera_frame and message.header.frame_id != self._camera_frame:
            self._drop_detection(
                'camera_info_frame_mismatch',
                f'detection frame {message.header.frame_id!r} differs from '
                f'CameraInfo frame {self._camera_frame!r}',
            )
            return

        refined_message = deepcopy(message)
        refined_detections = []
        for detection in refined_message.detections:
            if len(detection.corners) != 4:
                self._warn_once(
                    ('invalid_corners', detection.family, int(detection.id)),
                    f'Dropping {detection.family}:{detection.id} from refined '
                    'output because it does not contain four CUDA corners',
                )
                continue
            pose = detection.pose.pose.pose
            native_translation = np.asarray([
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ])
            native_quaternion = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            official_corners = [
                (corner.x, corner.y) for corner in detection.corners
            ]
            try:
                subpixel_corners = refine_corners_subpixel_from_image(
                    sample.message,
                    official_corners,
                    window=(4, 4),
                )
                translation, quaternion, error = refine_pose(
                    subpixel_corners,
                    self._camera_matrix,
                    self._tag_size,
                    native_quaternion,
                )
            except (RuntimeError, ValueError, cv2.error) as exception:
                self._warn_once(
                    ('refinement_failed', detection.family, int(detection.id)),
                    f'Dropping {detection.family}:{detection.id} from refined '
                    f'output because subpixel/IPPE refinement failed: {exception}',
                )
                continue
            if error > self._maximum_error:
                self._warn_once(
                    ('reprojection_error', detection.family, int(detection.id)),
                    f'Dropping {detection.family}:{detection.id} from refined '
                    f'output: IPPE reprojection error {error:.3f} px exceeds '
                    f'{self._maximum_error:.3f} px',
                )
                continue

            pose.position.x, pose.position.y, pose.position.z = translation
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = quaternion
            tag_key = (detection.family, int(detection.id))
            if tag_key not in self._reported_tags:
                correction = float(
                    np.linalg.norm(np.asarray(translation) - native_translation)
                )
                self.get_logger().info(
                    f'Refined {detection.family}:{detection.id} from '
                    f'subpixel corners with {error:.3f} px reprojection '
                    f'error; native translation correction={correction:.4f} m'
                )
                self._reported_tags.add(tag_key)
            refined_detections.append(detection)

        if not refined_detections:
            return
        # These deep copies still carry the official IDs, detection header,
        # source timestamp, and corner coordinates. The output intentionally
        # contains only detections whose pose fields were refined above.
        refined_message.detections = refined_detections
        self._publisher.publish(refined_message)


def main() -> None:
    """Run the AprilTag planar-pose refinement adapter."""
    rclpy.init()
    node = AprilTagPoseRefiner()
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
