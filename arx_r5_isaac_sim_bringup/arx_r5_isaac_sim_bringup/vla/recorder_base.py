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
Common episode-boundary and synchronization machinery for VLA recorders.

Subclasses implement :meth:`RecorderBase.on_episode_start`,
:meth:`RecorderBase.on_frame` and :meth:`RecorderBase.on_episode_end` for a
specific on-disk format.  The production ``vla_recorder`` entry point uses the
ARX raw-v1 implementation; :class:`NullRecorder` remains available only for
plumbing diagnostics.

Camera synchronization lives in :mod:`frame_sync`.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
import signal
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from arx_r5_isaac_sim_bringup.contracts import JOINT_COMMANDS_TOPIC
from arx_r5_isaac_sim_bringup.usd_scene import (
    CameraPublisherConfig,
    load_usd_scene_config,
)
from arx_r5_isaac_sim_bringup.vla.episode_events import (
    EPISODE_EVENT_TOPIC,
    EPISODE_HEARTBEAT_TOPIC,
    EpisodeEvent,
    EpisodeHeartbeat,
    EpisodePhase,
    EpisodeStatus,
    RECORDER_FAULT_TOPIC,
    RecorderFault,
    tf_frame_from_prim_path,
)
from arx_r5_isaac_sim_bringup.vla.frame_sync import (
    canonical_joint_positions,
    CausalSampleBuffer,
    COLOR_MODALITY,
    DEFAULT_STAMP_TOLERANCE_NS,
    DEPTH_MODALITY,
    Frame,
    stamp_ns,
    StampPairer,
)
from arx_r5_isaac_sim_bringup.vla.task_config import load_task_config

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions

from sensor_msgs.msg import Image, JointState

from std_msgs.msg import String

import tf2_ros


JOINT_STATES_TOPIC = '/joint_states'
ABSOLUTE_JOINT_COMMANDS_TOPIC = f'/{JOINT_COMMANDS_TOPIC}'


class RecorderBase(Node, ABC):
    """
    Subscribe to both cameras and hand complete frames to a subclass.

    Recording is gated on episode events: nothing is captured until the driver
    announces an episode, and the episode is closed out on its terminal event.
    """

    def __init__(self, node_name: str = 'vla_recorder') -> None:
        """Load the camera contract and subscribe to every recorded stream."""
        super().__init__(node_name)

        scene_config_file = self.declare_parameter(
            'scene_config_file',
            '',
        ).get_parameter_value().string_value
        task_config_file = self.declare_parameter(
            'task_config_file',
            '',
        ).get_parameter_value().string_value
        if not scene_config_file:
            raise ValueError('scene_config_file parameter is required')
        if not task_config_file:
            raise ValueError('task_config_file parameter is required')
        self._scene = load_usd_scene_config(scene_config_file)
        task = load_task_config(task_config_file)
        self._object_tf_frame = tf_frame_from_prim_path(task.block.prim_path)
        self._record_depth = self.declare_parameter(
            'record_depth',
            False,
        ).get_parameter_value().bool_value
        stamp_tolerance_ns = self.declare_parameter(
            'stamp_tolerance_ns',
            DEFAULT_STAMP_TOLERANCE_NS,
        ).get_parameter_value().integer_value
        self._terminal_drain_sec = self.declare_parameter(
            'terminal_drain_sec',
            0.5,
        ).get_parameter_value().double_value
        self._episode_heartbeat_timeout_sec = self.declare_parameter(
            'episode_heartbeat_timeout_sec',
            5.0,
        ).get_parameter_value().double_value
        self._max_episode_frames = self.declare_parameter(
            'max_episode_frames',
            100_000,
        ).get_parameter_value().integer_value
        if self._terminal_drain_sec < 0.0:
            raise ValueError('terminal_drain_sec must not be negative')
        if self._episode_heartbeat_timeout_sec <= 0.0:
            raise ValueError(
                'episode_heartbeat_timeout_sec must be positive'
            )
        if self._max_episode_frames < 1:
            raise ValueError('max_episode_frames must be at least one')

        self._cameras: Dict[str, CameraPublisherConfig] = dict(
            self._scene.cameras
        )
        self._pairer = StampPairer(
            tuple(self._cameras),
            tolerance_ns=stamp_tolerance_ns,
            modalities=(
                (COLOR_MODALITY, DEPTH_MODALITY)
                if self._record_depth else (COLOR_MODALITY,)
            ),
        )
        self._joint_states = CausalSampleBuffer()
        self._joint_commands = CausalSampleBuffer()
        self._invalid_joint_states = 0
        self._invalid_joint_commands = 0
        self._event: Optional[EpisodeEvent] = None
        self._event_history: List[Tuple[int, EpisodeEvent]] = []
        self._episode_open = False
        self._active_identity: Optional[tuple[str, str, int, int]] = None
        self._episode_start_stamp_ns: Optional[int] = None
        self._terminal_event: Optional[EpisodeEvent] = None
        self._terminal_stamp_ns: Optional[int] = None
        self._terminal_drain_deadline: Optional[float] = None
        self._last_event_stamp_ns: Optional[int] = None
        self._boundary_dropped_images = 0
        self._post_terminal_dropped_images = 0
        self.frames_recorded = 0
        self._lease_deadline: Optional[float] = None
        self._writer_healthy = True
        self._writer_fault_detail = ''

        heartbeat_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        fault_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._fault_publisher = self.create_publisher(
            String,
            RECORDER_FAULT_TOPIC,
            fault_qos,
        )

        for camera_name, camera in self._cameras.items():
            self.create_subscription(
                Image,
                camera.color_image_topic,
                self._make_image_callback(camera_name),
                qos_profile_sensor_data,
            )
            if self._record_depth:
                self.create_subscription(
                    Image,
                    camera.depth_image_topic,
                    self._make_depth_callback(camera_name),
                    qos_profile_sensor_data,
                )
        self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self._on_joint_state,
            10,
        )
        self.create_subscription(
            JointState,
            ABSOLUTE_JOINT_COMMANDS_TOPIC,
            self._on_joint_command,
            10,
        )
        self.create_subscription(
            String,
            EPISODE_EVENT_TOPIC,
            self._on_episode_event,
            10,
        )
        self.create_subscription(
            String,
            EPISODE_HEARTBEAT_TOPIC,
            self._on_episode_heartbeat,
            heartbeat_qos,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._terminal_drain_timer = self.create_timer(
            0.05,
            self._finish_terminal_drain,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self._lease_timer = self.create_timer(
            min(0.5, self._episode_heartbeat_timeout_sec / 5.0),
            self._check_episode_lease,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        self.get_logger().info(
            'recorder subscribed to '
            + ', '.join(
                f'{name}:{camera.color_image_topic}'
                for name, camera in self._cameras.items()
            )
            + f'; object_tf_frame={self._object_tf_frame}'
        )

    # ------------------------------------------------------------ callbacks --

    def _make_image_callback(self, camera_name: str):
        def callback(message: Image) -> None:
            self._add_camera_sample(camera_name, COLOR_MODALITY, message)

        return callback

    def _make_depth_callback(self, camera_name: str):
        def callback(message: Image) -> None:
            self._add_camera_sample(camera_name, DEPTH_MODALITY, message)

        return callback

    def _add_camera_sample(
        self,
        camera_name: str,
        modality: str,
        message: Image,
    ) -> None:
        # Do not let setup, homing, or inter-episode images contaminate
        # either pending buckets or the next episode's counters.  RGB and
        # depth use the same barrier, so reset() clears every pending stream.
        if not self._episode_open:
            return
        # DDS delivery can leave a setup image queued when STARTED arrives.
        # Use the event's simulation timestamp as a second boundary so an old
        # image cannot become the first observation of the next episode.  An
        # equal-stamp image is also ambiguous (the STARTED event can be
        # published between two callbacks on the same physics tick), so the
        # first accepted image must be strictly newer.
        if (
            self._episode_start_stamp_ns is not None and
            stamp_ns(message) <= self._episode_start_stamp_ns
        ):
            self._boundary_dropped_images += 1
            return
        if (
            self._terminal_stamp_ns is not None and
            stamp_ns(message) > self._terminal_stamp_ns
        ):
            self._post_terminal_dropped_images += 1
            return
        frame = self._pairer.add(camera_name, message, modality=modality)
        if frame is None:
            return
        state = self._joint_states.latest_at(frame.stamp_ns)
        command = self._joint_commands.latest_at(frame.stamp_ns)
        frame.attach_latest_joint_state(state)
        frame.attach_latest_joint_command(command)
        if state is not None:
            frame.extra['state_positions'] = canonical_joint_positions(state)
        if command is not None:
            frame.extra['action_positions'] = canonical_joint_positions(
                command
            )
        frame.object_pose = self._lookup_object_pose()
        frame.event = self._event_for_stamp(frame.stamp_ns)
        try:
            self.on_frame(frame)
        except Exception as error:  # pragma: no cover - writer-specific
            self._handle_writer_failure('frame', error)
            return
        self.frames_recorded += 1
        if self.frames_recorded >= self._max_episode_frames:
            self._handle_recorder_limit()

    def _on_joint_state(self, message: JointState) -> None:
        try:
            canonical_joint_positions(message)
        except (AttributeError, TypeError, ValueError) as error:
            self._invalid_joint_states += 1
            self.get_logger().warning(f'ignoring invalid joint state: {error}')
            return
        self._joint_states.add(message)

    def _on_joint_command(self, message: JointState) -> None:
        """Cache the absolute post-controller target sent to Isaac Sim."""
        try:
            canonical_joint_positions(message)
        except (AttributeError, TypeError, ValueError) as error:
            self._invalid_joint_commands += 1
            self.get_logger().warning(
                f'ignoring invalid absolute joint command: {error}'
            )
            return
        self._joint_commands.add(message)

    def _event_for_stamp(self, frame_stamp_ns: int) -> Optional[EpisodeEvent]:
        """Return the last phase transition no newer than this frame."""
        for event_stamp_ns, event in reversed(self._event_history):
            if event_stamp_ns <= frame_stamp_ns:
                return event
        return None

    def _lookup_object_pose(self):
        try:
            return self._tf_buffer.lookup_transform(
                self._scene.world_frame,
                self._object_tf_frame,
                rclpy.time.Time(),
            )
        except tf2_ros.TransformException:
            return None

    def _now_sec(self) -> float:
        """Return the node's ROS time as protocol seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def _aborted_event(
        self,
        event: EpisodeEvent,
        detail: str,
    ) -> EpisodeEvent:
        """Return an ABORTED terminal event for recorder-side closure."""
        return replace(
            event,
            status=EpisodeStatus.ABORTED,
            stamp_sec=self._now_sec(),
            detail=str(detail),
        )

    def _publish_recorder_fault(
        self,
        event: EpisodeEvent,
        detail: str,
    ) -> bool:
        """Publish one identity-scoped writer fault without masking cleanup."""
        try:
            fault = RecorderFault(
                recorder_name=self.get_name(),
                session_id=event.session_id,
                run_id=event.run_id,
                episode_id=event.episode_id,
                seed=event.seed,
                stamp_sec=self._now_sec(),
                detail=str(detail),
            )
            payload = fault.to_json()
            self._fault_publisher.publish(String(data=payload))
        except Exception as error:  # pragma: no cover - ROS runtime
            self.get_logger().error(
                f'could not publish recorder fault: {error}'
            )
            return False
        return True

    def _latch_recorder_fault(
        self,
        event: EpisodeEvent,
        detail: str,
    ) -> None:
        """Make writer failure permanent for this recorder process."""
        self._writer_healthy = False
        self._writer_fault_detail = str(detail)
        self._publish_recorder_fault(event, detail)

    def _handle_writer_failure(self, stage: str, error: Exception) -> None:
        """Abort the active episode and report one failed writer hook."""
        event = self._event
        if event is None:
            self.get_logger().error(
                f'episode writer failed during {stage}: {error}'
            )
            return
        detail = (
            f'episode writer failed during {stage}: '
            f'{type(error).__name__}: {error}'
        )
        self.get_logger().error(detail)
        self._latch_recorder_fault(event, detail)
        if self._episode_open:
            self._close_episode(self._aborted_event(event, detail))

    def _handle_recorder_limit(self) -> None:
        """Stop the batch before completed-stamp history can be truncated."""
        event = self._event
        if event is None or not self._episode_open:
            return
        detail = (
            'maximum episode frame count reached '
            f'({self._max_episode_frames}); stopping collection before '
            'timestamp uniqueness can be weakened'
        )
        self.get_logger().error(detail)
        self._latch_recorder_fault(event, detail)
        self._close_episode(self._aborted_event(event, detail))

    def _refresh_episode_lease(self) -> None:
        """Renew the active driver lease using the process steady clock."""
        self._lease_deadline = (
            time.monotonic() + self._episode_heartbeat_timeout_sec
        )

    def _on_episode_heartbeat(self, message: String) -> None:
        """Renew only the lease for the exact active episode identity."""
        try:
            heartbeat = EpisodeHeartbeat.from_json(message.data)
        except ValueError as error:
            self.get_logger().warning(
                f'ignoring malformed episode heartbeat: {error}'
            )
            return
        if (
            self._episode_open
            and heartbeat.identity == self._active_identity
            and self._terminal_event is None
        ):
            self._refresh_episode_lease()

    def _check_episode_lease(self) -> None:
        """Abort when an active driver stops renewing its recorder lease."""
        if (
            not self._episode_open
            or self._terminal_event is not None
            or self._lease_deadline is None
            or time.monotonic() < self._lease_deadline
            or self._event is None
        ):
            return
        detail = (
            'episode heartbeat lease expired after '
            f'{self._episode_heartbeat_timeout_sec:.1f} s'
        )
        self.get_logger().error(detail)
        self._close_episode(self._aborted_event(self._event, detail))

    def _on_episode_event(self, message: String) -> None:
        try:
            event = EpisodeEvent.from_json(message.data)
        except ValueError as error:
            self.get_logger().warning(
                f'ignoring malformed episode event: {error}'
            )
            return
        event_stamp_ns = int(round(event.stamp_sec * 1e9))
        if event.status == EpisodeStatus.STARTED:
            if not self._writer_healthy:
                detail = (
                    'recorder is unhealthy after a previous failure and '
                    'must be restarted before accepting another episode: '
                    f'{self._writer_fault_detail}'
                )
                self.get_logger().error(detail)
                self._publish_recorder_fault(event, detail)
                return
            if self._episode_open:
                if event.identity == self._active_identity:
                    self.get_logger().warning(
                        f'ignoring duplicate STARTED for episode '
                        f'{event.episode_id}'
                    )
                    return
                if (
                    self._last_event_stamp_ns is not None
                    and event_stamp_ns <= self._last_event_stamp_ns
                ):
                    self.get_logger().warning(
                        'ignoring stale STARTED from another driver/run while '
                        'an episode is active'
                    )
                    return
                if self._terminal_event is not None:
                    # The normal inter-episode setup is much longer than the
                    # drain window, but finalize the proven terminal outcome
                    # rather than rewriting it as ABORTED if a newer driver
                    # starts unusually quickly.
                    self._close_episode(self._terminal_event)
                else:
                    # A newer driver can take over after the old one crashed
                    # without publishing a terminal event. Keep the
                    # interrupted partial episode explicitly failed instead
                    # of silently merging two sessions into one directory.
                    previous = self._event
                    if previous is not None:
                        aborted = replace(
                            previous,
                            status=EpisodeStatus.ABORTED,
                            stamp_sec=event.stamp_sec,
                            detail=(
                                'superseded by a newer driver/run STARTED '
                                'event'
                            ),
                        )
                        self._close_episode(aborted)
                if not self._writer_healthy:
                    detail = (
                        'recorder became unhealthy while closing the previous '
                        'episode and cannot accept this STARTED event'
                    )
                    self._publish_recorder_fault(event, detail)
                    return
            self._pairer.reset()
            self.frames_recorded = 0
            self._episode_start_stamp_ns = event_stamp_ns
            self._terminal_event = None
            self._terminal_stamp_ns = None
            self._terminal_drain_deadline = None
            self._last_event_stamp_ns = event_stamp_ns
            self._boundary_dropped_images = 0
            self._post_terminal_dropped_images = 0
            self._invalid_joint_states = 0
            self._invalid_joint_commands = 0
            self._event_history = [(event_stamp_ns, event)]
            self._episode_open = True
            self._active_identity = event.identity
            self._event = event
            self._refresh_episode_lease()
            try:
                self.on_episode_start(event)
            except Exception as error:  # pragma: no cover - writer-specific
                self._handle_writer_failure('start', error)
            return
        if not self._episode_open:
            return
        if event.identity != self._active_identity:
            self.get_logger().warning(
                f'ignoring event for inactive identity {event.identity}; '
                f'active identity is {self._active_identity}'
            )
            return
        if (
            self._episode_start_stamp_ns is not None
            and event_stamp_ns < self._episode_start_stamp_ns
        ):
            self.get_logger().warning(
                f'ignoring event older than episode {event.episode_id} start'
            )
            return
        if (
            event.is_terminal
            and self._last_event_stamp_ns is not None
            and event_stamp_ns < self._last_event_stamp_ns
        ):
            self.get_logger().warning(
                'ignoring terminal event older than the latest accepted '
                'phase event'
            )
            return
        self._last_event_stamp_ns = max(
            event_stamp_ns,
            self._last_event_stamp_ns or event_stamp_ns,
        )
        if not event.is_terminal:
            if self._terminal_event is not None:
                if event_stamp_ns > (self._terminal_stamp_ns or 0):
                    self.get_logger().warning(
                        'ignoring non-terminal event newer than the active '
                        'terminal boundary'
                    )
                    return
            self._event = event
            self._event_history.append((event_stamp_ns, event))
            self._event_history.sort(
                key=lambda item: (
                    item[0],
                    EpisodePhase.ORDER.index(item[1].phase),
                )
            )
            return
        if self._terminal_event is not None:
            self.get_logger().warning(
                f'ignoring duplicate terminal event for episode '
                f'{event.episode_id}'
            )
            return
        self._terminal_event = event
        self._event = event
        self._terminal_stamp_ns = event_stamp_ns
        self._lease_deadline = None
        self._terminal_drain_deadline = (
            time.monotonic() + self._terminal_drain_sec
        )
        self._event_history.append((event_stamp_ns, event))
        self._event_history.sort(
            key=lambda item: (
                item[0],
                EpisodePhase.ORDER.index(item[1].phase),
            )
        )
        if self._terminal_drain_sec == 0.0:
            self._close_episode(event)

    def _finish_terminal_drain(self) -> None:
        """Close after late pre-terminal DDS samples had time to arrive."""
        if (
            not self._episode_open
            or self._terminal_event is None
            or self._terminal_drain_deadline is None
            or time.monotonic() < self._terminal_drain_deadline
        ):
            return
        self._close_episode(self._terminal_event)

    def _close_episode(self, event: EpisodeEvent) -> None:
        """Finalize and clear the recorder state for one active identity."""
        self._episode_open = False
        self._lease_deadline = None
        final_event = event
        try:
            self.on_episode_end(event, self.frames_recorded)
        except Exception as error:  # pragma: no cover - writer-specific
            detail = (
                'episode writer failed during end: '
                f'{type(error).__name__}: {error}'
            )
            self.get_logger().error(detail)
            final_event = self._aborted_event(event, detail)
            self._latch_recorder_fault(event, detail)
            if event.status != EpisodeStatus.ABORTED:
                try:
                    # A writer may have failed after persisting the requested
                    # terminal status. Give it one best-effort cleanup call
                    # that explicitly records the episode as unusable.
                    self.on_episode_end(final_event, self.frames_recorded)
                except Exception as cleanup_error:
                    self.get_logger().error(
                        'episode writer also failed while marking the end '
                        f'ABORTED: {cleanup_error}'
                    )
        finally:
            self._event = final_event
            self._episode_start_stamp_ns = None
            self._terminal_event = None
            self._terminal_stamp_ns = None
            self._terminal_drain_deadline = None
            self._last_event_stamp_ns = None
            self._active_identity = None
            self._event_history.clear()

    def finalize_active_episode_on_shutdown(self) -> None:
        """Commit a proven terminal or explicitly abort an open writer."""
        if not self._episode_open:
            return
        if self._terminal_event is not None:
            self._close_episode(self._terminal_event)
            return
        if self._event is None:
            return
        detail = 'recorder shut down before a terminal episode event'
        # A recorder can be terminated independently of the driver. Publish
        # the same identity-scoped fault used for writer failures so the
        # still-running driver cancels motion and closes its event stream.
        self._latch_recorder_fault(self._event, detail)
        aborted = self._aborted_event(
            self._event,
            detail,
        )
        self._close_episode(aborted)

    # -------------------------------------------------------------- hooks --

    @abstractmethod
    def on_episode_start(self, event: EpisodeEvent) -> None:
        """Open storage for a new episode."""

    @abstractmethod
    def on_frame(self, frame: Frame) -> None:
        """Persist one synchronized observation."""

    @abstractmethod
    def on_episode_end(self, event: EpisodeEvent, frame_count: int) -> None:
        """Finalize the episode, successful or not."""


class NullRecorder(RecorderBase):
    """
    Count frames without writing anything.

    This explicit diagnostic recorder is not the production collection
    executable; use ``vla_null_recorder`` when only ROS plumbing is needed.
    """

    def on_episode_start(self, event: EpisodeEvent) -> None:
        """Log the start of an episode."""
        self.get_logger().info(
            f'episode {event.episode_id} started (seed {event.seed}): '
            f'{event.instruction!r}'
        )

    def on_frame(self, frame: Frame) -> None:
        """Log every 50th frame so the rate is visible without flooding."""
        if self.frames_recorded % 50 != 0:
            return
        cameras = ', '.join(sorted(frame.images))
        has_pose = 'yes' if frame.object_pose is not None else 'no'
        has_action = 'yes' if frame.joint_command is not None else 'no'
        self.get_logger().info(
            f'frame {self.frames_recorded} stamp={frame.stamp_ns} '
            f'cameras=[{cameras}] object_pose={has_pose} '
            f'action={has_action} '
            f'phase={frame.event.phase if frame.event else "?"}'
        )

    def on_episode_end(self, event: EpisodeEvent, frame_count: int) -> None:
        """Report the episode outcome and how much was captured."""
        stream_diagnostics = []
        for stream, received in sorted(
            self._pairer.received_by_stream.items()
        ):
            rate = self._pairer.stream_rate_hz(*stream)
            stream_diagnostics.append(
                f'{stream[0]}/{stream[1]}: received={received}, '
                f'rate={rate:.2f}Hz' if rate is not None else
                f'{stream[0]}/{stream[1]}: received={received}, rate=n/a'
            )
            stream_diagnostics[-1] += (
                f', duplicates={self._pairer.duplicate_by_stream[stream]}, '
                f'non_monotonic='
                f'{self._pairer.non_monotonic_by_stream[stream]}, '
                f'evicted={self._pairer.overflow_by_stream[stream]}, '
                f'history_truncated='
                f'{self._pairer.stamp_history_truncations_by_stream[stream]}'
            )
        skew_diagnostics = [
            f'{left}<->{right}: median={summary["median_ns"] / 1e6:.3f}ms, '
            f'max={summary["max_ns"] / 1e6:.3f}ms'
            for (left, right), summary in sorted(
                self._pairer.nearest_camera_skew_summary().items()
            )
        ]
        self.get_logger().info(
            f'episode {event.episode_id} {event.status}: '
            f'{frame_count} paired frames, '
            f'{self._boundary_dropped_images} pre-start images dropped, '
            f'{self._post_terminal_dropped_images} post-terminal images '
            'dropped, '
            f'{self._invalid_joint_states} invalid joint states and '
            f'{self._invalid_joint_commands} invalid joint commands ignored, '
            f'{self._pairer.pending_buckets} incomplete timestamp buckets '
            f'({self._pairer.pending_images} images retained), '
            f'{self._pairer.overflow_buckets} buckets evicted at the pending '
            f'limit, {self._pairer.out_of_order_frames} completed frames '
            f'arrived out of order, max paired skew '
            f'{self._pairer.max_completed_frame_skew_ns / 1e6:.3f} ms; '
            f'max_episode_frames={self._max_episode_frames}; '
            f'streams=[{"; ".join(stream_diagnostics)}]; '
            f'nearest camera skew=[{"; ".join(skew_diagnostics)}]'
        )


def run_recorder(
    node_factory: Callable[[], RecorderBase],
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Spin one recorder implementation with ordered signal cleanup."""
    rclpy.init(
        args=argv,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    executor = None
    shutdown_requested = threading.Event()
    previous_signal_handlers = {}

    def request_shutdown(_signum, _frame) -> None:
        # Do not raise asynchronously through a writer callback. Let the
        # current callback return, stop scheduling new work, then close the
        # writer while the ROS context is still available.
        shutdown_requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signum] = signal.signal(
            signum,
            request_shutdown,
        )
    try:
        node = node_factory()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok() and not shutdown_requested.is_set():
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        shutdown_requested.set()
        if executor is not None:
            executor.shutdown(timeout_sec=5.0)
        if node is not None:
            node.finalize_active_episode_on_shutdown()
            if executor is not None:
                executor.remove_node(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Spin the explicit no-op recorder used for plumbing diagnostics."""
    return run_recorder(NullRecorder, argv)


if __name__ == '__main__':
    raise SystemExit(main())
