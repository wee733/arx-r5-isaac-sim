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
Drive scripted pick-and-place episodes for VLA data collection.

Replaces the official behavior tree for collection runs. The tree exists to
discover objects and recover from perception failures; a demonstration
generator instead wants a fixed, reproducible waypoint sequence whose object
layout is known exactly, which is what makes the resulting episodes usable as
expert data.

Each episode is driven by a single integer seed. The seed goes to Isaac Sim on
``/vla/scene_command``, and both sides feed it to the same deterministic
sampler in :mod:`task_config`, so the driver knows where the block is without
any perception at all.
"""

import argparse
from pathlib import Path
import secrets
import signal
import threading
import time
import traceback
from typing import Optional, Sequence
import uuid

from arx_r5_isaac_sim_bringup.contracts import ARM_JOINTS
from arx_r5_isaac_sim_bringup.usd_scene import load_usd_scene_config
from arx_r5_isaac_sim_bringup.vla.episode_events import (
    ATTACHMENT_STATE_TOPIC,
    EPISODE_EVENT_TOPIC,
    EPISODE_HEARTBEAT_TOPIC,
    EpisodeEvent,
    EpisodeHeartbeat,
    EpisodePhase,
    EpisodeStatus,
    GRIPPER_INTENT_TOPIC,
    RECORDER_FAULT_TOPIC,
    RecorderFault,
    scene_command_for_request,
    SCENE_COMMAND_TOPIC,
    SCENE_MAX_REQUEST_TOKEN,
    SCENE_RESET_ACK_TOPIC,
    tf_frame_from_prim_path,
)
from arx_r5_isaac_sim_bringup.vla.episode_plan import (
    build_episode_waypoints,
    load_home_positions,
    phase_rotation_candidates,
)
from arx_r5_isaac_sim_bringup.vla.motion_client import (
    CumotionMotionClient,
    MotionError,
)
from arx_r5_isaac_sim_bringup.vla.object_attachment_client import (
    cuboid_marker_from_transform,
    CumotionObjectAttachmentClient,
    DEFAULT_ATTACHMENT_FRAME,
    ObjectAttachmentError,
)
from arx_r5_isaac_sim_bringup.vla.task_config import (
    assert_valid_against_scene,
    load_task_config,
    pose_world_to_base,
    sample_scene_layout,
    StablePoseWindow,
    update_stable_pose_window,
)

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions

from std_msgs.msg import Bool, Int32, String

from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformException, TransformListener


ROBOT_MANAGER_REFRESH_UPDATES = 2
ROBOT_MANAGER_REFRESH_TIMEOUT_SEC = 2.0


class EpisodeDriver(Node):
    """Run scripted cuMotion episodes and announce their boundaries."""

    def __init__(self) -> None:
        """Load the task, wire the publishers and prepare the motion client."""
        super().__init__('vla_episode_driver')

        package_share = self.declare_parameter(
            'package_share_directory',
            '',
        ).get_parameter_value().string_value
        task_config_file = self.declare_parameter(
            'task_config_file',
            '',
        ).get_parameter_value().string_value
        scene_config_file = self.declare_parameter(
            'scene_config_file',
            '',
        ).get_parameter_value().string_value
        self._episode_count = self.declare_parameter(
            'episode_count',
            5,
        ).get_parameter_value().integer_value
        self._start_seed = self.declare_parameter(
            'start_seed',
            0,
        ).get_parameter_value().integer_value
        self._auto_start = self.declare_parameter(
            'auto_start',
            False,
        ).get_parameter_value().bool_value
        time_dilation_factor = self.declare_parameter(
            'time_dilation_factor',
            0.2,
        ).get_parameter_value().double_value
        self._server_timeout_sec = self.declare_parameter(
            'server_timeout_sec',
            120.0,
        ).get_parameter_value().double_value
        self._dry_run = self.declare_parameter(
            'dry_run',
            False,
        ).get_parameter_value().bool_value
        self._use_object_attachment = self.declare_parameter(
            'use_object_attachment',
            True,
        ).get_parameter_value().bool_value
        self._object_attachment_timeout_sec = self.declare_parameter(
            'object_attachment_timeout_sec',
            30.0,
        ).get_parameter_value().double_value
        self._gripper_close_timeout_sec = self.declare_parameter(
            'gripper_close_timeout_sec',
            15.0,
        ).get_parameter_value().double_value
        self._episode_heartbeat_period_sec = self.declare_parameter(
            'episode_heartbeat_period_sec',
            1.0,
        ).get_parameter_value().double_value
        self._scene_reset_wall_timeout_sec = self.declare_parameter(
            'scene_reset_wall_timeout_sec',
            10.0,
        ).get_parameter_value().double_value
        self._placement_wall_timeout_sec = self.declare_parameter(
            'placement_wall_timeout_sec',
            10.0,
        ).get_parameter_value().double_value
        execute_timeout_min_sec = self.declare_parameter(
            'execute_timeout_min_sec',
            60.0,
        ).get_parameter_value().double_value
        execute_timeout_scale = self.declare_parameter(
            'execute_timeout_scale',
            1.5,
        ).get_parameter_value().double_value
        execute_timeout_margin_sec = self.declare_parameter(
            'execute_timeout_margin_sec',
            10.0,
        ).get_parameter_value().double_value
        execute_wall_timeout_sec = self.declare_parameter(
            'execute_wall_timeout_sec',
            10.0,
        ).get_parameter_value().double_value
        if self._gripper_close_timeout_sec <= 0.0:
            raise ValueError('gripper_close_timeout_sec must be positive')
        if self._object_attachment_timeout_sec <= 0.0:
            raise ValueError('object_attachment_timeout_sec must be positive')
        if self._episode_heartbeat_period_sec <= 0.0:
            raise ValueError('episode_heartbeat_period_sec must be positive')
        if self._scene_reset_wall_timeout_sec <= 0.0:
            raise ValueError('scene_reset_wall_timeout_sec must be positive')
        if self._placement_wall_timeout_sec <= 0.0:
            raise ValueError('placement_wall_timeout_sec must be positive')
        if self._episode_count < 1:
            raise ValueError('episode_count must be at least one')
        if self._start_seed < 0:
            raise ValueError('start_seed must not be negative')

        share = Path(package_share) if package_share else None
        if not task_config_file:
            raise ValueError('task_config_file parameter is required')
        if not scene_config_file:
            raise ValueError('scene_config_file parameter is required')
        self._task = load_task_config(task_config_file)
        self._scene = load_usd_scene_config(scene_config_file)
        self._object_tf_frame = tf_frame_from_prim_path(
            self._task.block.prim_path
        )
        # Fail here rather than mid-run: a zone corner outside the reach
        # envelope or the camera frustum would only surface as a planner error
        # several episodes in.
        assert_valid_against_scene(self._task, self._scene)

        home_file = Path(self._task.episode.home_positions_file)
        if not home_file.is_absolute():
            base = (
                share if share is not None
                else Path(task_config_file).parent
            )
            home_file = base / 'config' / home_file.name
            if not home_file.is_file():
                home_file = Path(task_config_file).parent / home_file.name
        self._home_positions = load_home_positions(home_file)

        self._scene_command_publisher = self.create_publisher(
            Int32,
            SCENE_COMMAND_TOPIC,
            10,
        )
        self._scene_reset_ack_lock = threading.Lock()
        self._scene_reset_ack = 0
        self._scene_reset_ack_generation = 0
        # Isaac Sim commonly stays open while the ROS launch is restarted.
        # Starting every new driver at token 1 would repeat the exact command
        # from a previous one-episode seed-0 run, which the simulator correctly
        # treats as an already-applied request.  Randomize the session's first
        # token, then increment deterministically within that process.
        self._next_scene_request_token = (
            secrets.randbelow(SCENE_MAX_REQUEST_TOKEN) + 1
        )
        self.create_subscription(
            Int32,
            SCENE_RESET_ACK_TOPIC,
            self._on_scene_reset_ack,
            10,
        )
        self._event_publisher = self.create_publisher(
            String,
            EPISODE_EVENT_TOPIC,
            10,
        )
        self._gripper_intent_publisher = self.create_publisher(
            Bool,
            GRIPPER_INTENT_TOPIC,
            10,
        )
        self._attachment_lock = threading.Lock()
        self._attachment_state = False
        self._attachment_state_received_at = 0.0
        self._attachment_generation = 0
        self._attachment_loss_generation = 0
        self._carry_attachment_loss_generation: Optional[int] = None
        self._official_attached: Optional[bool] = None
        self.create_subscription(
            Bool,
            ATTACHMENT_STATE_TOPIC,
            self._on_attachment_state,
            10,
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
        )
        self._stop_event = threading.Event()
        self._motion = CumotionMotionClient(
            self,
            link_name=self._scene.base_frame,
            time_dilation_factor=time_dilation_factor,
            stop_predicate=self._stop_event.is_set,
            execute_timeout_min_sec=execute_timeout_min_sec,
            execute_timeout_scale=execute_timeout_scale,
            execute_timeout_margin_sec=execute_timeout_margin_sec,
            execute_wall_timeout_sec=execute_wall_timeout_sec,
        )
        self._object_attachment = (
            CumotionObjectAttachmentClient(self)
            if self._use_object_attachment else None
        )
        # Consumed by the first plan after an attach/detach hot reload.  The
        # joint-state barrier is necessary but two DDS subscribers can still
        # process the same messages in different orders, so that one plan gets
        # one bounded retry as the downstream readiness probe.
        self._cumotion_reload_pending = False
        self._run_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._next_episode_id = 0
        # A session ID survives only for this driver process; a new run gets a
        # separate ID.  Recorders use both values so a late message from an
        # older driver cannot close an episode opened by a newer one.
        self._session_id = uuid.uuid4().hex
        self._current_run_id: Optional[str] = None
        self._current_run_start_seed: Optional[int] = None
        self._next_seed = self._start_seed
        self._episode_state_lock = threading.RLock()
        self._active_episode: Optional[tuple[str, str, int, int]] = None
        self._terminal_event_sent = False
        self._recorder_faulted = False
        self._recorder_fault_detail = ''
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
        self._heartbeat_publisher = self.create_publisher(
            String,
            EPISODE_HEARTBEAT_TOPIC,
            heartbeat_qos,
        )
        self.create_subscription(
            String,
            RECORDER_FAULT_TOPIC,
            self._on_recorder_fault,
            fault_qos,
        )
        self._heartbeat_timer = self.create_timer(
            self._episode_heartbeat_period_sec,
            self._publish_heartbeat,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        # The event stream is not necessarily updated when a plan/action
        # raises (for example, planning can fail before its RUNNING event is
        # published).  Keep the operation currently being attempted so the
        # terminal failure event identifies the real phase instead of always
        # claiming ``retreat``.
        self._current_episode_phase = EpisodePhase.RESET

        self.create_service(Trigger, '~/collect', self._on_collect_request)
        self.get_logger().info(
            f'episode driver ready: instruction={self._task.instruction!r}, '
            f'episodes={self._episode_count}, start_seed={self._start_seed}, '
            f'auto_start={self._auto_start}, dry_run={self._dry_run}, '
            f'object_attachment={self._use_object_attachment}, '
            f'attachment_topic={ATTACHMENT_STATE_TOPIC}, '
            f'gripper_intent_topic={GRIPPER_INTENT_TOPIC}, '
            f'scene_reset_ack_topic={SCENE_RESET_ACK_TOPIC}, '
            f'episode_heartbeat_topic={EPISODE_HEARTBEAT_TOPIC}, '
            f'recorder_fault_topic={RECORDER_FAULT_TOPIC}, '
            f'object_tf_frame={self._object_tf_frame}'
        )
        if self._auto_start:
            self.start_run()

    # ------------------------------------------------------------- events --

    def _publish_heartbeat(self) -> None:
        """Renew the recorder lease for the exact active identity."""
        with self._episode_state_lock:
            identity = self._active_episode
        if identity is None:
            return
        session_id, run_id, episode_id, seed = identity
        try:
            heartbeat = EpisodeHeartbeat(
                session_id=session_id,
                run_id=run_id,
                episode_id=episode_id,
                seed=seed,
                stamp_sec=self.get_clock().now().nanoseconds / 1e9,
            )
            payload = heartbeat.to_json()
            self._heartbeat_publisher.publish(String(data=payload))
        except Exception as error:  # pragma: no cover - ROS runtime
            detail = f'episode heartbeat publication failed: {error}'
            self.get_logger().error(detail)
            self._stop_event.set()
            self._abort_active_episode(detail)

    def _on_recorder_fault(self, message: String) -> None:
        """Stop a matching batch as soon as its recorder becomes unhealthy."""
        try:
            fault = RecorderFault.from_json(message.data)
        except ValueError as error:
            self.get_logger().warning(
                f'ignoring malformed recorder fault: {error}'
            )
            return
        with self._run_lock:
            current_run_id = self._current_run_id
            matches_run = (
                fault.session_id == self._session_id
                and current_run_id is not None
                and fault.run_id == current_run_id
            )
            if not matches_run:
                return
            self._recorder_faulted = True
            self._recorder_fault_detail = (
                f'{fault.recorder_name}: {fault.detail}'
            )
        detail = (
            f'recorder fault for episode {fault.episode_id}, seed '
            f'{fault.seed}: {self._recorder_fault_detail}'
        )
        self.get_logger().error(detail)
        # The motion client's stop predicate observes this event and remotely
        # cancels any accepted planning, execution or gripper action.
        self._stop_event.set()
        self._abort_active_episode(detail)

    def _on_attachment_state(self, message: Bool) -> None:
        """Cache the simulator's physical attach/detach acknowledgement."""
        with self._attachment_lock:
            state = bool(message.data)
            if self._attachment_state and not state:
                self._attachment_loss_generation += 1
            self._attachment_state = state
            self._attachment_state_received_at = time.monotonic()
            self._attachment_generation += 1

    def _publish_gripper_intent(self, close: bool) -> None:
        """Tell Isaac Sim whether a close/hold or open/release is intended."""
        self._gripper_intent_publisher.publish(Bool(data=bool(close)))

    def _require_attachment(self, phase: str, stage: str) -> None:
        """Fail if the grasp was lost after close acknowledgement."""
        with self._attachment_lock:
            if not self._attachment_state:
                raise MotionError(
                    f'object is not physically attached during {phase} '
                    f'({stage})'
                )
            baseline = self._carry_attachment_loss_generation
            if (
                baseline is not None and
                self._attachment_loss_generation != baseline
            ):
                raise MotionError(
                    f'object attachment was lost during {phase} ({stage})'
                )

    def _on_scene_reset_ack(self, message: Int32) -> None:
        """Cache simulator ACKs for exact reset-request matching."""
        with self._scene_reset_ack_lock:
            self._scene_reset_ack = int(message.data)
            self._scene_reset_ack_generation += 1

    def _wait_for_scene_reset(
        self,
        command: int,
        baseline_generation: int,
        start_sim_ns: int,
        settle_sec: float,
        wall_started_at: float,
    ) -> None:
        """Wait for exact ACK and simulated settling under one watchdog."""
        settle_ns = int(round(float(settle_sec) * 1e9))
        last_sim_ns = int(start_sim_ns)
        settle_started_ns: Optional[int] = None
        while rclpy.ok() and not self._stop_event.is_set():
            with self._scene_reset_ack_lock:
                ack_received = (
                    self._scene_reset_ack == command
                    and self._scene_reset_ack_generation > baseline_generation
                )
            sim_now_ns = self.get_clock().now().nanoseconds
            if sim_now_ns < last_sim_ns:
                self.get_logger().warning(
                    'simulation clock moved backward while waiting for scene '
                    'reset; restarting the settle interval'
                )
                if settle_started_ns is not None:
                    settle_started_ns = sim_now_ns
            last_sim_ns = sim_now_ns
            if ack_received and settle_started_ns is None:
                # The ACK is published only after Isaac Sim writes the new
                # block pose. Settlement must therefore begin at the first
                # observation of that ACK, not while the old scene is still
                # waiting to be reset.
                settle_started_ns = sim_now_ns
            sim_elapsed_ns = (
                0 if settle_started_ns is None else
                max(0, sim_now_ns - settle_started_ns)
            )
            if ack_received and sim_elapsed_ns >= settle_ns:
                return
            wall_elapsed = time.monotonic() - wall_started_at
            if wall_elapsed >= self._scene_reset_wall_timeout_sec:
                raise MotionError(
                    'scene reset wall-clock watchdog expired after '
                    f'{wall_elapsed:.1f} s '
                    f'(exact_ack={ack_received}, '
                    f'sim_settle={sim_elapsed_ns / 1e9:.3f}/'
                    f'{settle_sec:.3f} s)'
                )
            time.sleep(0.02)
        raise MotionError('scene reset interrupted before ACK and settling')

    def _attachment_seen_since(
        self,
        started_at: float,
        attached: bool,
        min_generation: int = 0,
    ) -> bool:
        """Return whether a fresh simulator state matches ``attached``."""
        return self._attachment_snapshot_since(
            started_at,
            attached,
            min_generation,
        ) is not None

    def _attachment_snapshot_since(
        self,
        started_at: float,
        attached: bool,
        min_generation: int = 0,
    ) -> Optional[tuple[int, int]]:
        """Return fresh state and loss generations under one lock."""
        with self._attachment_lock:
            if not (
                self._attachment_state == attached
                and self._attachment_state_received_at >= started_at
                and self._attachment_generation > min_generation
            ):
                return None
            return (
                self._attachment_generation,
                self._attachment_loss_generation,
            )

    def _wait_for_attachment_state(
        self,
        started_at: float,
        attached: bool,
        timeout_sec: float = 2.0,
        min_generation: int = 0,
    ) -> bool:
        """Wait briefly for the state publisher to reflect a release/reset."""
        deadline = time.monotonic() + timeout_sec
        while (
            rclpy.ok()
            and not self._stop_event.is_set()
            and time.monotonic() <= deadline
        ):
            if self._attachment_seen_since(
                started_at,
                attached,
                min_generation,
            ):
                return True
            time.sleep(0.02)
        return False

    def _close_gripper(self) -> None:
        """Close until Isaac Sim confirms the physical FixedJoint."""
        # Establish both freshness barriers before publishing the request.
        # Isaac Sim can process the Bool and publish its attachment ACK within
        # one ROS executor turn; taking ``started_at`` after publication can
        # incorrectly classify that valid ACK as stale.
        started_at = time.monotonic()
        with self._attachment_lock:
            if self._attachment_state:
                raise MotionError(
                    'cannot close: simulator still reports the previous '
                    'object as attached'
                )
            baseline_generation = self._attachment_generation
            self._carry_attachment_loss_generation = None
        latched_loss_generation: Optional[int] = None

        def physical_attachment_confirmed() -> bool:
            nonlocal latched_loss_generation
            snapshot = self._attachment_snapshot_since(
                started_at,
                True,
                baseline_generation,
            )
            if snapshot is None:
                return False
            if latched_loss_generation is None:
                _, latched_loss_generation = snapshot
                # Latch the first confirmed grasp before action cancellation
                # drains. A True -> False -> True sequence during that drain
                # must remain a lost-grasp failure, not become a new baseline.
                with self._attachment_lock:
                    self._carry_attachment_loss_generation = (
                        latched_loss_generation
                    )
            return True

        self._publish_gripper_intent(True)
        try:
            self._motion.command_gripper(
                self._task.grasp.gripper_close,
                timeout_sec=self._gripper_close_timeout_sec,
                completion_predicate=physical_attachment_confirmed,
            )
        except MotionError as error:
            # DDS delivery and ros2_control cancellation can race the PhysX
            # acknowledgement by one executor turn. If the simulator has
            # already proved a bilateral grasp, do not discard that grasp just
            # because the controller's stall action timed out.
            deadline = time.monotonic() + 0.5
            while (
                rclpy.ok()
                and not self._stop_event.is_set()
                and time.monotonic() <= deadline
                and not physical_attachment_confirmed()
            ):
                time.sleep(0.02)
            if not physical_attachment_confirmed():
                raise
            self.get_logger().warning(
                f'gripper action reported {error}, but the simulator '
                'already acknowledged a physical grasp; continuing'
            )
        # The action cancellation used to return as soon as the simulator
        # reported True.  Check once more after the cancel/result drain, then
        # remember the loss generation so a transient drop during carry cannot
        # be hidden by a later re-attach.
        self._require_attachment('close_gripper', 'after action completion')
        if latched_loss_generation is None:
            raise MotionError(
                'gripper action completed without a latched physical grasp'
            )
        self.get_logger().info(
            'simulator attachment acknowledged; continuing to lift'
        )

    def _open_gripper(self) -> None:
        """Open the fingers, then wait for the simulator's release ack."""
        # As with close, the time and generation barriers must predate the
        # request so an immediate release ACK remains observable.
        started_at = time.monotonic()
        with self._attachment_lock:
            baseline_generation = self._attachment_generation
        self._publish_gripper_intent(False)
        self._motion.command_gripper(self._task.grasp.gripper_open)
        if not self._wait_for_attachment_state(
            started_at,
            False,
            min_generation=baseline_generation,
        ):
            raise MotionError(
                'gripper action finished but simulator did not acknowledge '
                'object release'
            )
        with self._attachment_lock:
            self._carry_attachment_loss_generation = None

    def _ensure_cumotion_detached(self) -> None:
        """Remove stale planner attachment, including after driver restart."""
        if self._object_attachment is None:
            return
        try:
            changed = self._object_attachment.ensure_detached(
                timeout_sec=self._object_attachment_timeout_sec,
            )
        except ObjectAttachmentError as error:
            self._official_attached = None
            raise MotionError(
                f'could not normalize cuMotion attachment state: {error}'
            ) from error
        self._official_attached = False
        if changed:
            self._wait_for_cumotion_robot_manager('stale-object detach')
            self.get_logger().warning(
                'removed a stale cuMotion attached object before episode setup'
            )

    def _wait_for_cumotion_robot_manager(self, operation: str) -> None:
        """Wait until the hot-reloaded planner has a fresh robot state."""
        baseline = self._motion.joint_state_generation
        try:
            self._motion.wait_for_joint_state_updates(
                baseline,
                minimum_updates=ROBOT_MANAGER_REFRESH_UPDATES,
                timeout_sec=ROBOT_MANAGER_REFRESH_TIMEOUT_SEC,
            )
        except MotionError as error:
            raise MotionError(
                f'cuMotion RobotManager did not refresh after {operation}: '
                f'{error}'
            ) from error
        self._cumotion_reload_pending = True
        self.get_logger().info(
            f'cuMotion RobotManager received fresh joint state after '
            f'{operation}'
        )

    def _plan_after_cumotion_reload(self, plan, label: str):
        """Run one plan, retrying once only after a description hot reload."""
        reload_pending = self._cumotion_reload_pending
        self._cumotion_reload_pending = False
        try:
            return plan()
        except MotionError as first_error:
            if not reload_pending:
                raise
            self.get_logger().warning(
                f'{label} failed on the first plan after a cuMotion robot '
                f'description reload: {first_error}; waiting for one more '
                'joint-state update and retrying once'
            )
            baseline = self._motion.joint_state_generation
            try:
                self._motion.wait_for_joint_state_updates(
                    baseline,
                    minimum_updates=1,
                    timeout_sec=ROBOT_MANAGER_REFRESH_TIMEOUT_SEC,
                )
            except MotionError as refresh_error:
                raise MotionError(
                    f'{label} failed after cuMotion reload and its retry '
                    f'barrier did not refresh: {refresh_error}'
                ) from first_error
            try:
                return plan()
            except MotionError as retry_error:
                raise MotionError(
                    f'{label} failed after the one allowed post-reload '
                    f'retry: {retry_error}'
                ) from retry_error

    def _lookup_attachment_transform(self, timeout_sec: float = 2.0):
        """Return the measured Block pose expressed in ``grasp_frame``."""
        deadline = time.monotonic() + timeout_sec
        last_error = None
        while (
            rclpy.ok()
            and not self._stop_event.is_set()
            and time.monotonic() <= deadline
        ):
            try:
                return self._tf_buffer.lookup_transform(
                    DEFAULT_ATTACHMENT_FRAME,
                    self._object_tf_frame,
                    rclpy.time.Time(),
                )
            except TransformException as error:
                last_error = error
                time.sleep(0.02)
        detail = f': {last_error}' if last_error is not None else ''
        raise MotionError(
            'no measured grasp_frame-to-object transform was available for '
            f'cuMotion attachment{detail}'
        )

    def _attach_object_to_cumotion(self) -> None:
        """Add the physically carried block to subsequent cuMotion plans."""
        if self._object_attachment is None:
            return
        self._require_attachment(EpisodePhase.LIFT, 'before cuMotion attach')
        transform = self._lookup_attachment_transform()
        marker = cuboid_marker_from_transform(
            transform,
            self._task.block.size,
            frame_id=DEFAULT_ATTACHMENT_FRAME,
        )
        try:
            self._object_attachment.attach(
                marker,
                timeout_sec=self._object_attachment_timeout_sec,
            )
        except (ObjectAttachmentError, ValueError) as error:
            self._official_attached = None
            raise MotionError(
                f'cuMotion could not attach the carried object: {error}'
            ) from error
        self._official_attached = True
        self._wait_for_cumotion_robot_manager('object attach')
        self.get_logger().info(
            'carried block added to the official cuMotion collision model'
        )

    def _detach_object_from_cumotion(self) -> None:
        """Restore the bare-robot XRDF after physical release."""
        if self._object_attachment is None:
            return
        try:
            self._object_attachment.detach(
                timeout_sec=self._object_attachment_timeout_sec,
            )
        except ObjectAttachmentError as error:
            self._official_attached = None
            raise MotionError(
                f'cuMotion could not detach the released object: {error}'
            ) from error
        self._official_attached = False
        self._wait_for_cumotion_robot_manager('object detach')
        self.get_logger().info(
            'released block removed from the official cuMotion collision '
            'model'
        )

    def _cleanup_failed_episode(self) -> bool:
        """Restore state and return whether another episode is safe."""
        with self._attachment_lock:
            physically_carrying = self._attachment_state
        if physically_carrying:
            # Never turn a planner/controller error into an uncontrolled
            # mid-air drop.  Keep both the physical grasp and cuMotion's
            # attached-object model intact, then stop the batch.  A later
            # operator-triggered run performs the normal open/detach setup.
            self.get_logger().error(
                'episode failed while the block is still attached; keeping '
                'the gripper closed and stopping the batch instead of '
                'releasing in mid-air'
            )
            return False
        restored = True
        try:
            self._open_gripper()
        except MotionError as error:
            restored = False
            self.get_logger().warning(
                f'could not reopen the gripper after a failed episode: {error}'
            )
        if (
            self._object_attachment is not None
            and self._official_attached is not False
        ):
            try:
                self._ensure_cumotion_detached()
            except MotionError as error:
                restored = False
                self.get_logger().warning(
                    'could not restore the bare cuMotion model after a failed '
                    f'episode: {error}'
                )
        return restored

    def _latest_object_pose(self):
        """Return the newest world-frame block pose and TF timestamp."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._scene.world_frame,
                self._object_tf_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        stamp = transform.header.stamp
        return (
            (
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ),
            (
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
            int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
        )

    def _latest_object_translation(self):
        """
        Return the newest world-frame block centre and TF timestamp.

        Kept as a small compatibility adapter for tools that only need the
        translation.  Placement verification uses :meth:`_latest_object_pose`
        so it can validate tilt and yaw as well.
        """
        sample = self._latest_object_pose()
        if sample is None:
            return None
        translation, _, stamp = sample
        return translation, stamp

    def _wait_for_placement(self, target_translation, target_yaw=None):
        """Require a stable, upright block pose on the sampled target."""
        policy = self._task.placement
        wall_started_at = time.monotonic()
        sim_started_ns = self.get_clock().now().nanoseconds
        last_sim_now_ns = sim_started_ns
        stability_window: Optional[StablePoseWindow] = None
        last_pose = None
        timeout_reason = ''
        sim_elapsed_sec = 0.0
        wall_elapsed_sec = 0.0

        # ``target_yaw`` is optional for compatibility with older callers;
        # the episode path always sets it from the same seeded layout.
        if target_yaw is None:
            target_yaw = getattr(self, '_placement_target_yaw', 0.0)

        while rclpy.ok() and not self._stop_event.is_set():
            sim_now_ns = self.get_clock().now().nanoseconds
            if sim_now_ns < last_sim_now_ns:
                self.get_logger().warning(
                    'simulation clock moved backward during placement '
                    'verification; restarting timeout and stability windows'
                )
                sim_started_ns = sim_now_ns
                stability_window = None
                last_pose = None
            last_sim_now_ns = sim_now_ns
            sim_elapsed_sec = max(0, sim_now_ns - sim_started_ns) / 1e9
            wall_elapsed_sec = time.monotonic() - wall_started_at
            if sim_elapsed_sec >= policy.timeout_sec:
                timeout_reason = (
                    'simulation-time timeout '
                    f'({sim_elapsed_sec:.3f}/{policy.timeout_sec:.3f} s)'
                )
                break
            if wall_elapsed_sec >= self._placement_wall_timeout_sec:
                timeout_reason = (
                    'wall-clock watchdog '
                    f'({wall_elapsed_sec:.3f}/'
                    f'{self._placement_wall_timeout_sec:.3f} s, '
                    f'simulation elapsed {sim_elapsed_sec:.3f} s)'
                )
                break
            sample = self._latest_object_pose()
            if sample is None:
                time.sleep(0.05)
                continue
            translation, rotation, stamp = sample
            if (
                stability_window is not None
                and stamp <= stability_window.latest_stamp_ns
            ):
                time.sleep(0.05)
                continue

            stability_window = update_stable_pose_window(
                stability_window,
                translation,
                rotation,
                stamp,
                policy.stability_tolerance,
                policy.stability_orientation_tolerance_deg,
            )
            last_pose = (translation, rotation, stamp)

            if stability_window.duration_sec >= policy.stable_duration_sec:
                xy_error, z_error, tilt_error, yaw_error = policy.errors(
                    translation,
                    target_translation,
                    rotation,
                    target_yaw,
                )
                if not policy.accepts(
                    translation,
                    target_translation,
                    rotation,
                    target_yaw,
                ):
                    raise MotionError(
                        'released object settled outside the target: '
                        f'xy_error={xy_error:.4f} m '
                        f'(limit {policy.xy_tolerance:.4f}), '
                        f'z_error={z_error:.4f} m '
                        f'(limit {policy.z_tolerance:.4f}), '
                        f'tilt={tilt_error:.2f} deg '
                        f'(limit {policy.max_tilt_deg:.2f}), '
                        f'yaw_error={yaw_error:.2f} deg '
                        f'(limit {policy.yaw_tolerance_deg:.2f})'
                    )
                self.get_logger().info(
                    'placement verified from simulator ground truth: '
                    f'xy_error={xy_error:.4f} m, '
                    f'z_error={z_error:.4f} m, '
                    f'tilt={tilt_error:.2f} deg, '
                    f'yaw_error={yaw_error:.2f} deg, '
                    f'stable_duration={stability_window.duration_sec:.3f} s'
                )
                return {
                    'target_translation': list(target_translation),
                    'object_translation': list(translation),
                    'placement_xy_error': xy_error,
                    'placement_z_error': z_error,
                    'placement_tilt_deg': tilt_error,
                    'placement_yaw_error_deg': yaw_error,
                    'placement_stable_duration_sec':
                        stability_window.duration_sec,
                    'placement_stable_samples': stability_window.sample_count,
                }
            time.sleep(0.05)

        if self._stop_event.is_set() or not rclpy.ok():
            raise MotionError('placement verification interrupted')
        if last_pose is None:
            detail = 'no world-to-object TF was received'
        else:
            last_translation, last_rotation, _ = last_pose
            xy_error, z_error, tilt_error, yaw_error = policy.errors(
                last_translation,
                target_translation,
                last_rotation,
                target_yaw,
            )
            detail = (
                f'last xy_error={xy_error:.4f} m, '
                f'z_error={z_error:.4f} m, '
                f'tilt={tilt_error:.2f} deg, '
                f'yaw_error={yaw_error:.2f} deg, '
                f'stable_duration={stability_window.duration_sec:.3f} s/'
                f'{policy.stable_duration_sec:.3f} s'
            )
        raise MotionError(
            f'placement verification timed out by {timeout_reason}: {detail}'
        )

    def _publish_event(
        self,
        episode_id: int,
        seed: int,
        phase: str,
        status: str,
        detail: str = '',
        extra: Optional[dict] = None,
    ) -> bool:
        """Publish one event, then commit its lifecycle transition."""
        with self._episode_state_lock:
            run_id = self._current_run_id
            if run_id is None:
                self.get_logger().error(
                    f'not publishing {status} for episode {episode_id}: '
                    'no collection run is active'
                )
                return False
            identity = (self._session_id, run_id, episode_id, seed)
            if status == EpisodeStatus.STARTED:
                if self._active_episode is not None:
                    self.get_logger().error(
                        'refusing to open a second episode before the first '
                        'one reaches a terminal event'
                    )
                    return False
            elif self._active_episode != identity:
                # Setup and dry-run failures occur outside the recordable
                # window. Do not emit orphan RUNNING/terminal events that a
                # recorder could mistake for an episode boundary.
                self.get_logger().error(
                    f'not publishing {status} for inactive episode '
                    f'{episode_id}'
                )
                return False
            elif status in EpisodeStatus.TERMINAL:
                if self._terminal_event_sent:
                    self.get_logger().error(
                        f'refusing duplicate terminal {status} for episode '
                        f'{episode_id}'
                    )
                    return False

            try:
                stamp = self.get_clock().now()
                event = EpisodeEvent(
                    session_id=self._session_id,
                    run_id=run_id,
                    episode_id=episode_id,
                    seed=seed,
                    phase=phase,
                    status=status,
                    instruction=self._task.instruction,
                    stamp_sec=stamp.nanoseconds / 1e9,
                    detail=detail,
                    extra=dict(extra or {}),
                )
                payload = event.to_json()
                self._event_publisher.publish(String(data=payload))
            except Exception as error:  # pragma: no cover - ROS runtime
                self.get_logger().error(
                    f'could not publish {status} for episode {episode_id}: '
                    f'{type(error).__name__}: {error}'
                )
                return False

            # Commit only after construction, strict serialization and DDS
            # publication all succeeded. A failed STARTED or terminal remains
            # retryable instead of poisoning the in-memory state machine.
            self._current_episode_phase = phase
            if status == EpisodeStatus.STARTED:
                self._active_episode = identity
                self._terminal_event_sent = False
            if status in EpisodeStatus.TERMINAL:
                self._terminal_event_sent = True
                self._active_episode = None
            return True

    def _publish_event_or_stop(
        self,
        episode_id: int,
        seed: int,
        phase: str,
        status: str,
        detail: str = '',
        extra: Optional[dict] = None,
    ) -> None:
        """Publish a required boundary event or stop before further motion."""
        if self._stop_event.is_set() or not rclpy.ok():
            self._stop_event.set()
            raise MotionError(
                f'collection stop requested before publishing {status} for '
                f'episode {episode_id}'
            )
        if self._publish_event(
            episode_id,
            seed,
            phase,
            status,
            detail=detail,
            extra=extra,
        ):
            return
        self._stop_event.set()
        raise MotionError(
            f'could not publish required {status} episode event for '
            f'episode {episode_id}'
        )

    def _abort_active_episode(self, detail: str) -> bool:
        """Close an open recording window exactly once as ABORTED."""
        with self._episode_state_lock:
            if self._active_episode is None or self._terminal_event_sent:
                return False
            _, _, episode_id, seed = self._active_episode
            return self._publish_event(
                episode_id,
                seed,
                self._current_episode_phase,
                EpisodeStatus.ABORTED,
                detail=detail,
            )

    def _episode_is_active(self, episode_id: int, seed: int) -> bool:
        """Return whether this run currently owns the requested identity."""
        with self._episode_state_lock:
            run_id = self._current_run_id
            return (
                run_id is not None
                and self._active_episode == (
                    self._session_id,
                    run_id,
                    episode_id,
                    seed,
                )
            )

    # -------------------------------------------------------------- motion --

    def _to_base_frame(
        self,
        translation: Sequence[float],
        rotation: Sequence[float],
    ):
        return pose_world_to_base(
            tuple(translation),
            tuple(rotation),
            self._scene.expected_world_to_base_translation,
            self._scene.expected_world_to_base_rotation,
        )

    def _to_base_goalset(
        self,
        translation: Sequence[float],
        rotations: Sequence[Sequence[float]],
    ):
        """Express one world position and candidate rotations in base."""
        base_translation = None
        base_rotations = []
        for rotation in rotations:
            converted_translation, converted_rotation = self._to_base_frame(
                translation,
                rotation,
            )
            if base_translation is None:
                base_translation = converted_translation
            base_rotations.append(converted_rotation)
        if base_translation is None:
            raise ValueError('pose goalset must contain at least one rotation')
        return base_translation, tuple(base_rotations)

    def _go_home(self) -> None:
        trajectory = self._plan_after_cumotion_reload(
            lambda: self._motion.plan_to_joint_state(
                ARM_JOINTS,
                self._home_positions,
            ),
            'home planning',
        )
        self._motion.execute(trajectory)

    def _reset_scene(self, seed: int) -> None:
        with self._scene_reset_ack_lock:
            token = self._next_scene_request_token
            self._next_scene_request_token = (
                1 if token == SCENE_MAX_REQUEST_TOKEN else token + 1
            )
            baseline_generation = self._scene_reset_ack_generation
        command = scene_command_for_request(seed, token)
        wall_started_at = time.monotonic()
        start_sim_ns = self.get_clock().now().nanoseconds
        self._scene_command_publisher.publish(
            Int32(data=command)
        )
        settle = self._task.episode.reset_settle_sec
        self._wait_for_scene_reset(
            command,
            baseline_generation,
            start_sim_ns,
            settle,
            wall_started_at,
        )

    def _dry_run_episode(self, episode_id: int, seed: int) -> None:
        """
        Plan every waypoint without executing, to prove the layout solves.

        A grasp that only fails at the fourth waypoint wastes a real episode
        and leaves the arm mid-trajectory; planning the whole sequence first
        surfaces an unreachable pose in seconds and moves nothing.
        """
        self._reset_scene(seed)
        failures = []
        source_rotation = None
        place_rotation = None
        for phase, translation, rotation in build_episode_waypoints(
            self._task,
            seed,
        ):
            world_rotations = phase_rotation_candidates(
                self._task,
                phase,
                rotation,
                source_rotation,
                place_rotation,
            )
            base_translation, base_rotations = self._to_base_goalset(
                translation,
                world_rotations,
            )
            try:
                _, goal_index = self._plan_after_cumotion_reload(
                    lambda: self._motion.plan_to_pose_goalset(
                        base_translation,
                        base_rotations,
                    ),
                    f'{phase} dry-run planning',
                )
            except MotionError as error:
                failures.append(f'{phase}: {error}')
                self.get_logger().error(f'  {phase}: FAILED -- {error}')
            else:
                selected_rotation = world_rotations[goal_index]
                if phase in (
                    EpisodePhase.APPROACH,
                    EpisodePhase.GRASP,
                    EpisodePhase.LIFT,
                ):
                    source_rotation = selected_rotation
                elif phase in (
                    EpisodePhase.PLACE,
                    EpisodePhase.RETREAT,
                ):
                    place_rotation = selected_rotation
                self.get_logger().info(
                    f'  {phase}: planned to base_link '
                    f'({base_translation[0]:.4f}, {base_translation[1]:.4f}, '
                    f'{base_translation[2]:.4f}), selected symmetry '
                    f'{goal_index + 1}/{len(world_rotations)}'
                )
        if failures:
            self.get_logger().error(
                f'dry-run episode {episode_id}, seed {seed}: '
                f'{len(failures)} of 6 waypoints unreachable'
            )
        else:
            self.get_logger().info(
                f'dry-run episode {episode_id}, seed {seed}: '
                'all 6 waypoints reachable'
            )

    def _run_episode(self, episode_id: int, seed: int) -> None:
        if self._dry_run:
            self._ensure_cumotion_detached()
            self._dry_run_episode(episode_id, seed)
            return

        # Setup, deliberately outside the recorded window.
        #
        # Open BEFORE homing so a ROS restart cannot carry a block that the
        # still-running simulator retained from an interrupted episode. Home
        # BEFORE the reset: going home with the new layout already in place
        # drives the arm through wherever the block was just teleported to.
        #
        # The STARTED event is published only once setup is complete, so a
        # recorded episode begins with the arm already at home and the block
        # already placed. The homing motion is not part of the demonstration
        # and would otherwise be learned as one.
        self._current_episode_phase = EpisodePhase.OPEN_GRIPPER
        self._ensure_cumotion_detached()
        self._open_gripper()
        self._current_episode_phase = EpisodePhase.HOME
        self._go_home()
        self._current_episode_phase = EpisodePhase.RESET
        self._reset_scene(seed)

        self._publish_event_or_stop(
            episode_id,
            seed,
            EpisodePhase.RESET,
            EpisodeStatus.STARTED,
        )
        self._publish_event_or_stop(
            episode_id,
            seed,
            EpisodePhase.OBSERVE,
            EpisodeStatus.RUNNING,
        )

        _, (place_center, place_yaw) = sample_scene_layout(self._task, seed)
        self._placement_target_yaw = place_yaw

        source_rotation = None
        place_rotation = None
        for phase, translation, rotation in build_episode_waypoints(
            self._task,
            seed,
        ):
            # Planning can fail before the phase event below is published.
            # Mark the attempted phase first so the terminal event is honest.
            self._current_episode_phase = phase
            carries_object = phase in (
                EpisodePhase.LIFT,
                EpisodePhase.TRANSFER,
                EpisodePhase.PLACE,
            )
            if carries_object:
                self._require_attachment(phase, 'before planning')
            if (
                self._object_attachment is not None
                and phase in (EpisodePhase.TRANSFER, EpisodePhase.PLACE)
                and self._official_attached is not True
            ):
                raise MotionError(
                    f'cuMotion attached-object model is not active before '
                    f'{phase} planning'
                )
            if (
                self._object_attachment is not None
                and phase == EpisodePhase.RETREAT
                and self._official_attached is not False
            ):
                raise MotionError(
                    'cuMotion attached-object model is still active before '
                    'retreat planning'
                )
            world_rotations = phase_rotation_candidates(
                self._task,
                phase,
                rotation,
                source_rotation,
                place_rotation,
            )
            base_translation, base_rotations = self._to_base_goalset(
                translation,
                world_rotations,
            )
            trajectory, goal_index = self._plan_after_cumotion_reload(
                lambda: self._motion.plan_to_pose_goalset(
                    base_translation,
                    base_rotations,
                ),
                f'{phase} planning',
            )
            if carries_object:
                self._require_attachment(phase, 'after planning')
            selected_rotation = world_rotations[goal_index]
            if phase in (
                EpisodePhase.APPROACH,
                EpisodePhase.GRASP,
                EpisodePhase.LIFT,
            ):
                source_rotation = selected_rotation
            elif phase in (
                EpisodePhase.PLACE,
                EpisodePhase.RETREAT,
            ):
                place_rotation = selected_rotation
            self._publish_event_or_stop(
                episode_id,
                seed,
                phase,
                EpisodeStatus.RUNNING,
                extra={
                    'world_translation': list(translation),
                    'world_rotation': list(selected_rotation),
                    'goalset_index': goal_index,
                    'goalset_size': len(world_rotations),
                },
            )
            if len(world_rotations) > 1:
                self.get_logger().info(
                    f'{phase}: cuMotion selected square-grasp symmetry '
                    f'{goal_index + 1}/{len(world_rotations)}'
                )
            self._motion.execute(trajectory)
            if carries_object:
                self._require_attachment(phase, 'after execution')

            if phase == EpisodePhase.GRASP:
                self._publish_event_or_stop(
                    episode_id,
                    seed,
                    EpisodePhase.CLOSE_GRIPPER,
                    EpisodeStatus.RUNNING,
                )
                self._close_gripper()
            elif phase == EpisodePhase.LIFT:
                # Match NVIDIA's official workflow: lift away from the support
                # surface first, then update the XRDF before transfer/place.
                self._attach_object_to_cumotion()
            elif phase == EpisodePhase.PLACE:
                self._publish_event_or_stop(
                    episode_id,
                    seed,
                    EpisodePhase.OPEN_GRIPPER,
                    EpisodeStatus.RUNNING,
                )
                self._open_gripper()
                self._detach_object_from_cumotion()

        # The arm has reached the retreat pose, but the episode remains open
        # while simulator ground truth proves that the released block has
        # settled at the target. Homing happens during the next episode's
        # setup, outside the recorded window.
        self._publish_event_or_stop(
            episode_id,
            seed,
            EpisodePhase.VERIFY_PLACE,
            EpisodeStatus.RUNNING,
        )
        placement_extra = self._wait_for_placement(place_center, place_yaw)
        self._publish_event_or_stop(
            episode_id,
            seed,
            EpisodePhase.VERIFY_PLACE,
            EpisodeStatus.SUCCEEDED,
            extra=placement_extra,
        )

    # ----------------------------------------------------------------- run --

    def _run(self, run_id: str, start_seed: int) -> None:
        """
        Run one identity-scoped collection batch.

        The broad exception guard is intentional: a programming/runtime error
        must still close an open recorder window and must not leave a daemon
        worker silently dying while the recorder waits forever for a terminal
        event.
        """
        try:
            try:
                self._motion.wait_for_servers(
                    timeout_sec=self._server_timeout_sec
                )
            except MotionError as error:
                self.get_logger().error(f'motion stack unavailable: {error}')
                return
            if self._object_attachment is not None:
                try:
                    self._object_attachment.wait_for_server(
                        timeout_sec=self._server_timeout_sec,
                    )
                except ObjectAttachmentError as error:
                    self.get_logger().error(
                        f'cuMotion object attachment unavailable: {error}'
                    )
                    return

            if self._stop_event.is_set() or not rclpy.ok():
                return

            # Planning without the static scene would treat the platform, the
            # column and the floor as empty space, so refuse to start rather
            # than generate demonstrations that drive through them.
            if not self._motion.request_static_planning_scene():
                self.get_logger().error(
                    'no static collision scene was published; refusing to '
                    'plan into an empty world'
                )
                return
            self.get_logger().info('static collision scene received')

            for index in range(self._episode_count):
                if self._stop_event.is_set() or not rclpy.ok():
                    return
                seed = start_seed + index
                episode_id = self._next_episode_id
                self._next_episode_id += 1
                self.get_logger().info(
                    f'episode {index + 1}/{self._episode_count} '
                    f'(id {episode_id}, seed {seed}, run {run_id[:8]})'
                )
                self._current_episode_phase = EpisodePhase.RESET
                try:
                    self._run_episode(episode_id, seed)
                except MotionError as error:
                    if self._stop_event.is_set() or not rclpy.ok():
                        self._abort_active_episode(
                            f'collection interrupted: {error}'
                        )
                        return
                    self.get_logger().error(
                        f'episode {episode_id} failed: {error}'
                    )
                    if self._episode_is_active(episode_id, seed):
                        if not self._publish_event(
                            episode_id,
                            seed,
                            self._current_episode_phase,
                            EpisodeStatus.FAILED,
                            detail=str(error),
                        ):
                            self._stop_event.set()
                            return
                    # The gripper/planner may still be holding the block;
                    # restore both states before the next reset.
                    if not self._cleanup_failed_episode():
                        self.get_logger().error(
                            'failed-episode cleanup left simulator/planner '
                            'state indeterminate; stopping this batch'
                        )
                        return
                except Exception as error:  # pragma: no cover - ROS runtime
                    detail = f'{type(error).__name__}: {error}'
                    self.get_logger().error(
                        f'episode {episode_id} crashed: {detail}\n'
                        f'{traceback.format_exc()}'
                    )
                    if self._episode_is_active(episode_id, seed):
                        if not self._publish_event(
                            episode_id,
                            seed,
                            self._current_episode_phase,
                            EpisodeStatus.FAILED,
                            detail=detail,
                        ):
                            self._stop_event.set()
                            return
                    if not self._cleanup_failed_episode():
                        self.get_logger().error(
                            'crash cleanup left simulator/planner state '
                            'indeterminate; stopping this batch'
                        )
                    return
            self.get_logger().info('collection run finished')
        except Exception as error:  # pragma: no cover - ROS runtime
            self.get_logger().error(
                f'collection run crashed: {type(error).__name__}: {error}\n'
                f'{traceback.format_exc()}'
            )
            self._abort_active_episode(
                f'collection worker crashed: {type(error).__name__}: {error}'
            )
        finally:
            # A shutdown can interrupt a blocking action between two phase
            # events. Never leave the recorder's current episode open.
            try:
                if self._stop_event.is_set() or not rclpy.ok():
                    self._abort_active_episode('collection worker stopped')
            finally:
                # Event construction, serialization or DDS publication must
                # never strand the run state and make every future collect
                # request look permanently busy.
                with self._run_lock:
                    if self._worker is threading.current_thread():
                        self._worker = None
                    if self._current_run_id == run_id:
                        self._current_run_id = None
                        self._current_run_start_seed = None

    def start_run(self) -> bool:
        """Start a collection run unless one is already in flight."""
        with self._run_lock:
            if self._recorder_faulted:
                return False
            if self._worker is not None and self._worker.is_alive():
                return False
            run_id = uuid.uuid4().hex
            start_seed = self._next_seed
            # Reserve this batch up front. Repeated calls to the collect
            # service must not silently regenerate the same seed sequence.
            self._next_seed += self._episode_count
            self._current_run_id = run_id
            self._current_run_start_seed = start_seed
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._run,
                args=(run_id, start_seed),
                name='vla_episode_run',
                daemon=False,
            )
            self._worker.start()
            return True

    def stop_run(
        self,
        join_timeout_sec: float = 0.0,
        detail: str = 'collection stopped by shutdown',
    ) -> bool:
        """Request shutdown and optionally wait for the worker to exit."""
        self._stop_event.set()
        if rclpy.ok():
            self._abort_active_episode(detail)
        worker = self._worker
        if (
            join_timeout_sec > 0.0
            and worker is not None
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=join_timeout_sec)
        return worker is None or not worker.is_alive()

    def _on_collect_request(self, _request, response):
        if self.start_run():
            response.success = True
            response.message = (
                f'started {self._episode_count} episodes from seed '
                f'{self._current_run_start_seed}'
            )
        else:
            response.success = False
            if self._recorder_faulted:
                response.message = (
                    'recorder is unhealthy and must be restarted: '
                    f'{self._recorder_fault_detail}'
                )
            else:
                response.message = 'a collection run is already in progress'
        return response


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Spin the episode driver on a multi-threaded executor."""
    parser = argparse.ArgumentParser(
        description=(
            'Drive scripted cuMotion episodes for VLA data collection.'
        ),
    )
    parsed, remaining = parser.parse_known_args(argv)
    del parsed

    rclpy.init(
        args=remaining,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    executor = MultiThreadedExecutor()
    spin_thread = None
    spin_errors = []
    shutdown_requested = threading.Event()
    shutdown_signal = []
    previous_signal_handlers = {}

    def request_shutdown(signum, _frame) -> None:
        shutdown_signal[:] = [int(signum)]
        shutdown_requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signum] = signal.signal(
            signum,
            request_shutdown,
        )
    worker_stopped = True
    try:
        node = EpisodeDriver()
        executor.add_node(node)

        def spin_executor() -> None:
            try:
                executor.spin()
            except Exception as error:  # pragma: no cover - ROS runtime
                spin_errors.append(error)
            finally:
                shutdown_requested.set()

        spin_thread = threading.Thread(
            target=spin_executor,
            name='vla_episode_executor',
            daemon=False,
        )
        spin_thread.start()
        while rclpy.ok() and not shutdown_requested.wait(timeout=0.1):
            pass
    except KeyboardInterrupt:
        shutdown_requested.set()
    finally:
        if node is not None:
            reason = (
                f'collection stopped by signal {shutdown_signal[0]}'
                if shutdown_signal else 'collection process stopping'
            )
            # Keep the executor and ROS context alive while the motion client
            # remotely cancels and drains the active action.
            worker_stopped = node.stop_run(
                join_timeout_sec=5.0,
                detail=reason,
            )
            if not worker_stopped:
                node.get_logger().error(
                    'collection worker did not stop within 5 seconds; '
                    'shutting down the executor to unblock remaining waits'
                )
        executor.shutdown(timeout_sec=5.0)
        if spin_thread is not None:
            spin_thread.join(timeout=5.0)
        if not worker_stopped and rclpy.ok():
            rclpy.shutdown()
        if node is not None and not worker_stopped:
            node.stop_run(join_timeout_sec=5.0, detail=reason)
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
    if spin_errors:
        raise spin_errors[0]
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
