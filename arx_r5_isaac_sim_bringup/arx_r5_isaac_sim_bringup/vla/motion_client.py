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
Minimal cuMotion plan/execute/grip client for scripted data collection.

The official behavior tree is not used for data collection: it discovers
objects, retries, and owns the episode structure, none of which a scripted
expert-demonstration generator wants. This client keeps only the three action
calls the tree ultimately makes.

Goal construction mirrors the official ``PoseToPoseNode``
(``isaac_ros_manipulation_pose_to_pose/pose_to_pose.py``) so that upstream
behaviour is reproduced without importing or modifying it.
"""

import math
import threading
import time
from typing import Callable, Optional, Sequence

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal

from arx_r5_isaac_sim_bringup.contracts import ARM_JOINTS

from control_msgs.action import GripperCommand

from geometry_msgs.msg import Pose, PoseArray

from isaac_ros_cumotion_interfaces.action import MotionPlan
from isaac_ros_cumotion_interfaces.srv import PublishStaticPlanningScene

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, PlanningScene

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from sensor_msgs.msg import JointState

from std_msgs.msg import Header


DEFAULT_MOTION_PLAN_ACTION = 'cumotion/motion_plan'
DEFAULT_EXECUTE_ACTION = 'execute_trajectory'
DEFAULT_GRIPPER_ACTION = '/gripper_controller/gripper_cmd'
PLANNING_SCENE_TOPIC = '/planning_scene'
PUBLISH_STATIC_SCENE_SERVICE = 'publish_static_planning_scene'
JOINT_STATES_TOPIC = '/joint_states'


class MotionError(RuntimeError):
    """Raised when a plan, execution or gripper command does not succeed."""


def _spin_until_complete(
    node: Node,
    future,
    timeout_sec: float,
    stop_predicate: Optional[Callable[[], bool]] = None,
):
    """
    Block the calling thread on a future while the node keeps spinning.

    The driver runs on a multi-threaded executor, so this only parks the
    episode thread; camera, TF and planning-scene callbacks keep running.
    """
    del node
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        if stop_predicate is not None and stop_predicate():
            raise MotionError('interrupted by a collection stop request')
        if time.monotonic() > deadline:
            raise MotionError(f'timed out after {timeout_sec:.1f} s')
        time.sleep(0.02)
    if not future.done():
        raise MotionError('interrupted before the goal completed')
    return future.result()


class CumotionMotionClient:
    """Plan to a Cartesian goal, execute it, and drive the gripper."""

    def __init__(
        self,
        node: Node,
        link_name: str = 'base_link',
        time_dilation_factor: float = 0.2,
        motion_plan_action: str = DEFAULT_MOTION_PLAN_ACTION,
        execute_action: str = DEFAULT_EXECUTE_ACTION,
        gripper_action: str = DEFAULT_GRIPPER_ACTION,
        gripper_max_effort: float = 10.0,
        stop_predicate: Optional[Callable[[], bool]] = None,
        execute_timeout_min_sec: float = 60.0,
        execute_timeout_scale: float = 1.5,
        execute_timeout_margin_sec: float = 10.0,
        execute_wall_timeout_sec: float = 10.0,
    ) -> None:
        """Create the three action clients and subscribe to the scene."""
        self._node = node
        self._link_name = link_name
        self._time_dilation_factor = float(time_dilation_factor)
        self._gripper_max_effort = float(gripper_max_effort)
        self._stop_predicate = stop_predicate
        self._execute_timeout_min_sec = float(execute_timeout_min_sec)
        self._execute_timeout_scale = float(execute_timeout_scale)
        self._execute_timeout_margin_sec = float(execute_timeout_margin_sec)
        self._execute_wall_timeout_sec = float(execute_wall_timeout_sec)
        if self._execute_timeout_min_sec <= 0.0:
            raise ValueError('execute_timeout_min_sec must be positive')
        if self._execute_timeout_scale <= 0.0:
            raise ValueError('execute_timeout_scale must be positive')
        if self._execute_timeout_margin_sec < 0.0:
            raise ValueError(
                'execute_timeout_margin_sec must not be negative'
            )
        if self._execute_wall_timeout_sec <= 0.0:
            raise ValueError('execute_wall_timeout_sec must be positive')

        self._plan_client = ActionClient(
            node,
            MotionPlan,
            motion_plan_action,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._execute_client = ActionClient(
            node,
            ExecuteTrajectory,
            execute_action,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._gripper_client = ActionClient(
            node,
            GripperCommand,
            gripper_action,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self._planning_scene_world = None
        node.create_subscription(
            PlanningScene,
            PLANNING_SCENE_TOPIC,
            self._on_planning_scene,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # The official object-attachment component hot-reloads cuMotion's
        # RobotManager when it adds or removes collision spheres.  That new
        # manager starts without a current robot state, even though the action
        # result already says that the description update has completed.  A
        # local generation counter lets the episode driver wait for messages
        # published *after* the hot reload before issuing the next plan.
        self._joint_state_condition = threading.Condition()
        self._joint_state_generation = 0
        node.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self._on_joint_state,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # The static scene server republishes when it sees a new subscriber, so
        # the subscription above is usually enough. This service is the
        # explicit fallback for the case where the subscription is created
        # before the server exists and therefore never triggers that hook.
        self._static_scene_client = node.create_client(
            PublishStaticPlanningScene,
            PUBLISH_STATIC_SCENE_SERVICE,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _on_planning_scene(self, message: PlanningScene) -> None:
        self._planning_scene_world = message.world

    def _on_joint_state(self, message: JointState) -> None:
        """Count structurally valid joint-state updates from the simulator."""
        names = tuple(message.name)
        if (
            not names
            or len(message.position) != len(names)
            or not set(ARM_JOINTS).issubset(names)
            or any(not math.isfinite(value) for value in message.position)
        ):
            return
        with self._joint_state_condition:
            self._joint_state_generation += 1
            self._joint_state_condition.notify_all()

    @property
    def joint_state_generation(self) -> int:
        """Return the number of valid joint-state messages seen so far."""
        with self._joint_state_condition:
            return self._joint_state_generation

    def wait_for_joint_state_updates(
        self,
        after_generation: int,
        minimum_updates: int = 1,
        timeout_sec: float = 2.0,
    ) -> int:
        """
        Wait for fresh state messages after a RobotManager hot reload.

        The counter is captured only after the attachment action has returned,
        so reaching the target proves that these messages were published
        after the replacement RobotManager became active.  Two updates are
        used by the caller to avoid racing the planner subscriber callback.
        """
        baseline = int(after_generation)
        updates = int(minimum_updates)
        timeout = float(timeout_sec)
        if baseline < 0:
            raise ValueError('after_generation must not be negative')
        if updates < 1:
            raise ValueError('minimum_updates must be at least one')
        if timeout <= 0.0:
            raise ValueError('timeout_sec must be positive')

        target = baseline + updates
        deadline = time.monotonic() + timeout
        with self._joint_state_condition:
            while rclpy.ok() and self._joint_state_generation < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._joint_state_condition.wait(timeout=remaining)
            generation = self._joint_state_generation
        if generation < target:
            raise MotionError(
                f'only {max(0, generation - baseline)} of {updates} fresh '
                f'joint-state updates arrived within {timeout:.1f} s'
            )
        return generation

    @property
    def has_planning_scene(self) -> bool:
        """Return whether a static collision world has been received."""
        return self._planning_scene_world is not None

    def request_static_planning_scene(self, timeout_sec: float = 20.0) -> bool:
        """
        Ask the cuMotion scene server to publish the static collision world.

        Returns True once a scene has arrived. Planning without one would treat
        the platform, the column and the floor as empty space.
        """
        if self.has_planning_scene:
            return True
        if self._static_scene_client.wait_for_service(timeout_sec=timeout_sec):
            request = PublishStaticPlanningScene.Request()
            # An empty path keeps the server's own configured scene file.
            request.scene_file_path = ''
            try:
                response = _spin_until_complete(
                    self._node,
                    self._static_scene_client.call_async(request),
                    timeout_sec,
                )
            except MotionError as error:
                self._node.get_logger().warning(
                    f'static planning scene request failed: {error}'
                )
            else:
                if response is not None and response.success:
                    # The response carries the scene directly, so this does not
                    # depend on the topic round trip completing.
                    self._planning_scene_world = response.planning_scene.world
                    return True
                self._node.get_logger().warning(
                    'static planning scene server reported failure: '
                    f'{getattr(response, "message", "no response")}'
                )
        else:
            self._node.get_logger().warning(
                f'service {PUBLISH_STATIC_SCENE_SERVICE} did not appear'
            )

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not self.has_planning_scene:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return self.has_planning_scene

    def wait_for_servers(self, timeout_sec: float = 60.0) -> None:
        """Block until all three action servers are reachable."""
        for name, client in (
            ('cuMotion motion plan', self._plan_client),
            ('execute trajectory', self._execute_client),
            ('gripper command', self._gripper_client),
        ):
            self._node.get_logger().info(f'waiting for the {name} server')
            if not client.wait_for_server(timeout_sec=timeout_sec):
                raise MotionError(
                    f'{name} action server did not appear within '
                    f'{timeout_sec:.0f} s'
                )

    @staticmethod
    def _consume_late_result(future) -> None:
        """Consume a late action result without canceling its client future."""
        try:
            future.result()
        except Exception:
            pass

    @staticmethod
    def _validate_terminal_result(response, label: str):
        """Return a response only when the goal is in a terminal state."""
        if response is None:
            raise MotionError(f'{label} returned no result')
        if response.status not in (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_ABORTED,
        ):
            raise MotionError(
                f'{label} result did not reach a terminal goal status: '
                f'{response.status}'
            )
        return response

    @staticmethod
    def _same_goal_id(left, right) -> bool:
        """Compare ROS GoalInfo UUIDs without depending on generated types."""
        left_uuid = getattr(left, 'uuid', left)
        right_uuid = getattr(right, 'uuid', right)
        try:
            return bytes(left_uuid) == bytes(right_uuid)
        except (TypeError, ValueError):
            return left_uuid == right_uuid

    @classmethod
    def _cancel_response_issue(cls, response, goal_handle) -> Optional[str]:
        """Describe why a CancelGoal response did not acknowledge this goal."""
        if response is None:
            return 'cancel service returned no response'
        if int(response.return_code) != int(CancelGoal.Response.ERROR_NONE):
            return (
                'cancel service rejected the goal with return code '
                f'{response.return_code}'
            )
        goals_canceling = tuple(getattr(response, 'goals_canceling', ()))
        if not goals_canceling:
            return 'cancel service acknowledged no goal for cancellation'
        if goal_handle is not None and not any(
            cls._same_goal_id(goal_info.goal_id, goal_handle.goal_id)
            for goal_info in goals_canceling
        ):
            return 'cancel service did not acknowledge this goal UUID'
        return None

    def _handle_late_cancel_response(
        self,
        future,
        goal_handle,
        label: str,
    ) -> None:
        """Log (and consume) a cancel response for a late accepted goal."""
        try:
            response = future.result()
        except Exception as error:
            self._node.get_logger().warning(
                f'late {label} cancel request failed: {error}'
            )
            return
        issue = self._cancel_response_issue(response, goal_handle)
        if issue:
            self._node.get_logger().warning(f'late {label}: {issue}')

    def _cancel_late_goal_response(self, future, label: str) -> None:
        """
        Cancel and drain a goal whose send response arrived after timeout.

        A local ``Future.cancel()`` only removes the pending request from
        rclpy; it does not cancel a goal that the server may already have
        accepted. This callback therefore keeps the request alive, obtains the
        result future, and sends a real remote cancel without blocking an
        executor callback while waiting for either response.
        """
        try:
            goal_handle = future.result()
        except Exception as error:
            self._node.get_logger().warning(
                f'late {label} goal response failed: {error}'
            )
            return
        if goal_handle is None or not goal_handle.accepted:
            return

        try:
            result_future = goal_handle.get_result_async()
        except Exception as error:
            result_future = None
            self._node.get_logger().warning(
                f'could not obtain late {label} result future: {error}'
            )
        if result_future is not None:
            result_future.add_done_callback(self._consume_late_result)

        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception as error:
            self._node.get_logger().warning(
                f'could not cancel late {label} goal: {error}'
            )
        else:
            cancel_future.add_done_callback(
                lambda done_future: self._handle_late_cancel_response(
                    done_future,
                    goal_handle,
                    label,
                )
            )

    def _cancel_goal(
        self,
        goal_handle,
        result_future,
        label: str,
        timeout_sec: float = 2.0,
    ):
        """
        Cancel one accepted action goal and drain its terminal result.

        Canceling only ``result_future`` leaves the server-side goal running.
        A later command then preempts it and rclpy reports the old result as an
        unexpected response. Always cancel through the goal handle instead.
        """
        if result_future.done():
            return self._validate_terminal_result(
                result_future.result(),
                label,
            )
        cancel_issue = None
        try:
            cancel_future = goal_handle.cancel_goal_async()
            cancel_response = _spin_until_complete(
                self._node,
                cancel_future,
                timeout_sec,
            )
            cancel_issue = self._cancel_response_issue(
                cancel_response,
                goal_handle,
            )
        except Exception as error:
            cancel_issue = f'cancel request failed: {error}'

        try:
            result_response = _spin_until_complete(
                self._node,
                result_future,
                timeout_sec,
            )
            result_response = self._validate_terminal_result(
                result_response,
                label,
            )
        except MotionError as error:
            # Do not call Future.cancel(): the action server still owns this
            # goal and its eventual result must retain a matching pending
            # request in rclpy. The callback consumes it when it arrives.
            result_future.add_done_callback(self._consume_late_result)
            detail = f'{label} cancellation did not reach a terminal result'
            if cancel_issue:
                detail += f' ({cancel_issue})'
            raise MotionError(detail) from error
        if cancel_issue:
            # The goal may have reached a terminal state between the cancel
            # request and its response. The terminal result is the decisive
            # proof that no remote goal remains, but retain the diagnostic.
            self._node.get_logger().warning(
                f'{label}: {cancel_issue}; terminal result received'
            )
        return result_response

    def _send(
        self,
        client,
        goal,
        timeout_sec: float,
        label: str,
        completion_predicate: Optional[Callable[[], bool]] = None,
        result_clock=None,
        result_wall_timeout_sec: Optional[float] = None,
    ):
        """
        Send one action goal and wait for its terminal result.

        Goal acceptance and cancellation always use wall time.  By default the
        result uses the same wall-time budget.  Passing ``result_clock`` makes
        the result budget run in that clock instead. The wall timeout then
        guards a clock that stops advancing without imposing a total wall
        deadline on a slow simulation.
        """
        timeout_sec = float(timeout_sec)
        if timeout_sec <= 0.0:
            raise ValueError('action timeout must be positive')
        if result_clock is None:
            if result_wall_timeout_sec is not None:
                raise ValueError(
                    'result_wall_timeout_sec requires result_clock'
                )
        else:
            if result_wall_timeout_sec is None:
                raise ValueError(
                    'result_clock requires result_wall_timeout_sec'
                )
            result_wall_timeout_sec = float(result_wall_timeout_sec)
            if result_wall_timeout_sec <= 0.0:
                raise ValueError(
                    'result_wall_timeout_sec must be positive'
                )

        send_future = client.send_goal_async(goal)
        stop_predicate = getattr(self, '_stop_predicate', None)
        try:
            if stop_predicate is None:
                goal_handle = _spin_until_complete(
                    self._node,
                    send_future,
                    timeout_sec,
                )
            else:
                goal_handle = _spin_until_complete(
                    self._node,
                    send_future,
                    timeout_sec,
                    stop_predicate=stop_predicate,
                )
        except MotionError:
            # Keep the pending request in rclpy. The server can accept a goal
            # just after this timeout; the callback then cancels that remote
            # goal and drains its result instead of creating an unexpected
            # response on the next action request.
            send_future.add_done_callback(
                lambda done_future: self._cancel_late_goal_response(
                    done_future,
                    label,
                )
            )
            raise
        if goal_handle is None or not goal_handle.accepted:
            raise MotionError(f'{label} goal was rejected')
        result_future = goal_handle.get_result_async()
        result_response = None
        wall_now = time.monotonic()
        wall_deadline = (
            wall_now + timeout_sec if result_clock is None else None
        )
        sim_started_ns = None
        last_sim_now_ns = None
        last_sim_progress_wall = wall_now
        if result_clock is not None:
            try:
                sim_started_ns = int(result_clock.now().nanoseconds)
            except Exception as error:
                self._cancel_goal(goal_handle, result_future, label)
                raise MotionError(
                    f'{label} could not read the simulation clock: {error}'
                ) from error
            last_sim_now_ns = sim_started_ns
        while rclpy.ok():
            if stop_predicate is not None and stop_predicate():
                if result_response is None and not result_future.done():
                    self._cancel_goal(goal_handle, result_future, label)
                raise MotionError(
                    f'{label} interrupted by a collection stop request'
                )
            try:
                externally_complete = (
                    completion_predicate is not None
                    and completion_predicate()
                )
            except Exception as error:
                if not result_future.done():
                    self._cancel_goal(goal_handle, result_future, label)
                raise MotionError(
                    f'{label} completion check failed: {error}'
                ) from error
            if externally_complete:
                if result_response is None:
                    if result_future.done():
                        result_response = self._validate_terminal_result(
                            result_future.result(),
                            label,
                        )
                    else:
                        try:
                            result_response = self._cancel_goal(
                                goal_handle,
                                result_future,
                                label,
                            )
                        except MotionError as error:
                            # For the simulated gripper, bilateral PhysX
                            # contact and the FixedJoint ACK are stronger proof
                            # of completion than ros2_control's velocity/stall
                            # heuristic. Some controller versions keep the
                            # canceled goal pending while they hold the
                            # measured finger position. The result future was
                            # retained by _cancel_goal and will be consumed by
                            # its done callback, so it is safe to continue the
                            # arm motion without turning a real grasp into a
                            # failed episode.
                            self._node.get_logger().warning(
                                f'{label} is physically complete but its ROS '
                                f'action did not reach terminal promptly: '
                                f'{error}; continuing and draining it in the '
                                'background'
                            )
                            return None
                return result_response.result

            if result_response is None and result_future.done():
                try:
                    result_response = result_future.result()
                except Exception as error:
                    raise MotionError(
                        f'{label} result request failed: {error}'
                    ) from error
                if result_response is None:
                    raise MotionError(f'{label} returned no result')
                if result_response.status != GoalStatus.STATUS_SUCCEEDED:
                    action_result = result_response.result
                    detail = str(getattr(action_result, 'message', '')).strip()
                    detail_suffix = f': {detail}' if detail else ''
                    raise MotionError(
                        f'{label} finished with goal status '
                        f'{result_response.status}{detail_suffix}'
                    )
                if completion_predicate is None:
                    return result_response.result

            wall_now = time.monotonic()
            timeout_detail = None
            if result_clock is None:
                if wall_now >= wall_deadline:
                    timeout_detail = (
                        f'wall-clock timeout after {timeout_sec:.1f} s'
                    )
            else:
                try:
                    sim_now_ns = int(result_clock.now().nanoseconds)
                except Exception as error:
                    if result_response is None:
                        self._cancel_goal(goal_handle, result_future, label)
                    raise MotionError(
                        f'{label} could not read the simulation clock: {error}'
                    ) from error
                if sim_now_ns < last_sim_now_ns:
                    self._node.get_logger().warning(
                        f'{label}: simulation clock moved backward; '
                        'restarting the execution timeout window'
                    )
                    sim_started_ns = sim_now_ns
                    last_sim_progress_wall = wall_now
                elif sim_now_ns > last_sim_now_ns:
                    last_sim_progress_wall = wall_now
                last_sim_now_ns = sim_now_ns
                sim_elapsed_sec = max(
                    0,
                    sim_now_ns - sim_started_ns,
                ) / 1e9
                wall_stalled_sec = wall_now - last_sim_progress_wall
                if sim_elapsed_sec >= timeout_sec:
                    timeout_detail = (
                        'simulation-time timeout '
                        f'({sim_elapsed_sec:.3f}/{timeout_sec:.3f} s)'
                    )
                elif wall_stalled_sec >= result_wall_timeout_sec:
                    timeout_detail = (
                        'wall-clock watchdog expired after '
                        f'{wall_stalled_sec:.3f} s without simulation-clock '
                        f'progress (limit {result_wall_timeout_sec:.3f} s, '
                        f'simulation elapsed {sim_elapsed_sec:.3f} s)'
                    )

            if timeout_detail is not None:
                if result_response is None:
                    self._cancel_goal(goal_handle, result_future, label)
                raise MotionError(f'{label} {timeout_detail}')
            time.sleep(0.02)

        if result_response is None:
            self._cancel_goal(goal_handle, result_future, label)
        raise MotionError(f'{label} was interrupted before completion')

    def plan_to_pose(
        self,
        translation: Sequence[float],
        rotation: Sequence[float],
        timeout_sec: float = 30.0,
    ):
        """Plan a collision-free motion to one link6 pose in ``link_name``."""
        trajectory, _ = self.plan_to_pose_goalset(
            translation,
            (rotation,),
            timeout_sec=timeout_sec,
        )
        return trajectory

    def plan_to_pose_goalset(
        self,
        translation: Sequence[float],
        rotations: Sequence[Sequence[float]],
        timeout_sec: float = 30.0,
    ):
        """
        Plan to one position with one or more candidate orientations.

        ``MotionPlan`` calls this a goalset: cuMotion solves the IK for every
        pose and returns the selected index in ``result.goal_index``.  A
        square parallel-jaw grasp has several physically equivalent wrist
        orientations; passing them together lets cuMotion avoid a joint limit
        instead of making the driver guess which wrist branch is reachable.

        Planning always starts from the live measured robot state.
        """
        rotations = tuple(tuple(float(value) for value in rotation)
                          for rotation in rotations)
        if not rotations:
            raise ValueError('at least one pose orientation is required')
        if any(len(rotation) != 4 for rotation in rotations):
            raise ValueError('each pose orientation must contain four values')
        if self._planning_scene_world is None:
            raise MotionError(
                f'no collision world received on {PLANNING_SCENE_TOPIC}; '
                'planning would ignore the platform and the floor'
            )
        goal = MotionPlan.Goal()
        goal.goal_pose = PoseArray()
        goal.goal_pose.header.frame_id = self._link_name
        goal.goal_pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.goal_pose.poses.extend(
            _pose_msg(translation, rotation) for rotation in rotations
        )
        goal.plan_pose = True
        goal.plan_cspace = False
        goal.plan_grasp = False
        goal.use_current_state = True
        goal.hold_partial_pose = False
        goal.use_planning_scene = True
        # nvblox is not running during collection, so there is no ESDF to
        # refresh; the static scene file is the whole collision world.
        goal.update_esdf = False
        goal.time_dilation_factor = self._time_dilation_factor
        goal.world = self._planning_scene_world

        result = self._send(
            self._plan_client,
            goal,
            timeout_sec,
            'cuMotion motion plan',
        )
        if not result.success or not result.planned_trajectory:
            raise MotionError(
                f'cuMotion planning failed: '
                f'{result.message or "no trajectory"} '
                f'(error code {result.error_code.val})'
            )
        goal_index = int(result.goal_index)
        if goal_index < 0 or goal_index >= len(rotations):
            raise MotionError(
                f'cuMotion returned invalid goal index {goal_index} for '
                f'{len(rotations)} pose candidates'
            )
        return result.planned_trajectory[0], goal_index

    def plan_to_joint_state(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        timeout_sec: float = 30.0,
    ):
        """
        Plan a configuration-space motion to an explicit joint goal.

        Used for the between-episode home pose, which is defined in joint space
        so every episode starts from the same arm configuration rather than the
        same tool pose reached through an arbitrary elbow solution.
        """
        if len(joint_names) != len(positions):
            raise ValueError(
                'joint names and positions must be the same length'
            )
        if self._planning_scene_world is None:
            raise MotionError(
                f'no collision world received on {PLANNING_SCENE_TOPIC}; '
                'planning would ignore the platform and the floor'
            )
        goal = MotionPlan.Goal()
        goal.goal_state = JointState()
        goal.goal_state.header.stamp = self._node.get_clock().now().to_msg()
        goal.goal_state.name = list(joint_names)
        goal.goal_state.position = [float(value) for value in positions]
        goal.plan_pose = False
        goal.plan_cspace = True
        goal.plan_grasp = False
        goal.use_current_state = True
        goal.hold_partial_pose = False
        goal.use_planning_scene = True
        goal.update_esdf = False
        goal.time_dilation_factor = self._time_dilation_factor
        goal.world = self._planning_scene_world

        result = self._send(
            self._plan_client,
            goal,
            timeout_sec,
            'cuMotion cspace plan',
        )
        if not result.success or not result.planned_trajectory:
            raise MotionError(
                f'cuMotion cspace planning failed: '
                f'{result.message or "no trajectory"} '
                f'(error code {result.error_code.val})'
            )
        return result.planned_trajectory[0]

    @staticmethod
    def _duration_seconds(duration) -> float:
        """Return seconds from a builtin or ROS duration-shaped value."""
        if duration is None:
            return 0.0
        if hasattr(duration, 'nanoseconds'):
            return max(0.0, float(duration.nanoseconds) / 1e9)
        return max(
            0.0,
            float(getattr(duration, 'sec', 0))
            + float(getattr(duration, 'nanosec', 0)) / 1e9,
        )

    @classmethod
    def trajectory_duration_sec(cls, trajectory) -> float:
        """Return the latest time_from_start carried by a trajectory."""
        durations = [0.0]
        for field_name in (
            'joint_trajectory',
            'multi_dof_joint_trajectory',
        ):
            sub_trajectory = getattr(trajectory, field_name, None)
            points = tuple(getattr(sub_trajectory, 'points', ()) or ())
            if points:
                durations.append(cls._duration_seconds(
                    getattr(points[-1], 'time_from_start', None)
                ))
        return max(durations)

    def execute(self, trajectory, timeout_sec: Optional[float] = None) -> None:
        """
        Run a trajectory with a simulation-time timeout.

        ``timeout_sec`` is measured in the node's ROS clock.  A separate wall
        watchdog cancels the action only when that clock stops advancing, so a
        low real-time factor does not consume the trajectory's timeout budget.
        """
        if timeout_sec is None:
            duration_sec = self.trajectory_duration_sec(trajectory)
            timeout_sec = max(
                self._execute_timeout_min_sec,
                duration_sec * self._execute_timeout_scale
                + self._execute_timeout_margin_sec,
            )
        timeout_sec = float(timeout_sec)
        if timeout_sec <= 0.0:
            raise ValueError('execute timeout must be positive')
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        # The controller rejects trajectories that carry a stale stamp.
        goal.trajectory.joint_trajectory.header = Header()
        goal.trajectory.multi_dof_joint_trajectory.header = Header()
        result = self._send(
            self._execute_client,
            goal,
            timeout_sec,
            'execute trajectory',
            result_clock=self._node.get_clock(),
            result_wall_timeout_sec=self._execute_wall_timeout_sec,
        )
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise MotionError(
                f'trajectory execution failed with MoveIt error code '
                f'{result.error_code.val}'
            )

    def move_to_pose(
        self,
        translation: Sequence[float],
        rotation: Sequence[float],
        plan_timeout_sec: float = 30.0,
        execute_timeout_sec: Optional[float] = None,
    ) -> None:
        """Plan and execute one Cartesian goal."""
        trajectory, _ = self.plan_to_pose_goalset(
            translation,
            (rotation,),
            timeout_sec=plan_timeout_sec,
        )
        self.execute(trajectory, timeout_sec=execute_timeout_sec)

    def move_to_pose_goalset(
        self,
        translation: Sequence[float],
        rotations: Sequence[Sequence[float]],
        plan_timeout_sec: float = 30.0,
        execute_timeout_sec: Optional[float] = None,
    ) -> int:
        """Plan and execute a pose goalset, returning the selected index."""
        trajectory, goal_index = self.plan_to_pose_goalset(
            translation,
            rotations,
            timeout_sec=plan_timeout_sec,
        )
        self.execute(trajectory, timeout_sec=execute_timeout_sec)
        return goal_index

    def command_gripper(
        self,
        position: float,
        timeout_sec: float = 15.0,
        completion_predicate: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Command the gripper and wait for its ROS or physical completion.

        A physical completion predicate is used for closing in Isaac Sim. The
        ros2_control action may never satisfy its stall detector while PhysX
        contact jitters, even though the simulator has already attached the
        object. When the predicate becomes true, the still-running action is
        canceled cleanly and its controller holds the current finger position.
        """
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = self._gripper_max_effort
        self._send(
            self._gripper_client,
            goal,
            timeout_sec,
            f'gripper command ({position:.4f} m)',
            completion_predicate=completion_predicate,
        )


def _pose_msg(translation: Sequence[float], rotation: Sequence[float]) -> Pose:
    """Build a geometry_msgs Pose from xyz and an xyzw quaternion."""
    pose = Pose()
    pose.position.x = float(translation[0])
    pose.position.y = float(translation[1])
    pose.position.z = float(translation[2])
    pose.orientation.x = float(rotation[0])
    pose.orientation.y = float(rotation[1])
    pose.orientation.z = float(rotation[2])
    pose.orientation.w = float(rotation[3])
    return pose


def lookup_optional_transform(
    tf_buffer,
    target_frame: str,
    source_frame: str,
) -> Optional[object]:
    """Return the latest transform, or None when it is unavailable."""
    from tf2_ros import TransformException

    try:
        return tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
        )
    except TransformException:
        return None
