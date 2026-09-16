# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab SkillGen adapter for the isolated native cuMotion worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation
import torch

try:
    from isaaclab_mimic.motion_planners.motion_planner_base import (
        MotionPlannerBase,
    )
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith('omni'):
        raise

    class MotionPlannerBase:  # type: ignore[no-redef]
        """Offline-only shape of the Isaac Lab interface for unit tests."""

        def __init__(self, env, robot, env_id=0, debug=False, **unused):
            self.env = env
            self.robot = robot
            self.env_id = env_id
            self.debug = debug

        def get_planner_info(self):
            return {
                'name': self.__class__.__name__,
                'env_id': self.env_id,
                'debug': self.debug,
            }


class Kinematics(Protocol):
    """Forward kinematics used to expose cuMotion joint paths to SkillGen."""

    def fk(self, joints: np.ndarray) -> np.ndarray:
        """Return base-to-link6 as a homogeneous matrix."""

    def ik(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Solve the target on a branch seeded by the current arm state."""


class PlannerRunner(Protocol):
    """Boundary between Isaac Lab Python and the native cuMotion runtime."""

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        """Plan one request and return the worker's complete JSON result."""


@dataclass(frozen=True)
class DoorCuMotionPlannerConfig:
    """Planner options consumed by SkillGen and the native worker."""

    sample_dt: float = 1.0 / 30.0
    time_dilation_factor: float = 0.5
    motion_noise_scale: float = 0.0


class SubprocessCuMotionRunner:
    """Persist planner requests/results and invoke the isolated worker."""

    def __init__(
        self,
        run_script: str | Path,
        artifact_dir: str | Path,
        *,
        timeout_s: float = 300.0,
    ) -> None:
        self.run_script = Path(run_script).expanduser().resolve()
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.timeout_s = float(timeout_s)
        if not self.run_script.is_file():
            raise FileNotFoundError(self.run_script)
        if self.timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute one request without hiding a native planning failure."""
        self._sequence += 1
        stem = f'plan_{self._sequence:04d}'
        request_path = self.artifact_dir / f'{stem}.request.json'
        result_path = self.artifact_dir / f'{stem}.result.json'
        log_path = self.artifact_dir / f'{stem}.log'
        request_path.write_text(
            json.dumps(request, indent=2, allow_nan=False) + '\n',
            encoding='utf-8',
        )
        try:
            process = subprocess.run(
                [
                    str(self.run_script),
                    '--request',
                    str(request_path),
                    '--output',
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            log_path.write_text(
                process.stdout + process.stderr,
                encoding='utf-8',
            )
        except subprocess.TimeoutExpired as error:
            result = {
                'success': False,
                'status': 'TIMEOUT',
                'reason': f'cuMotion exceeded {self.timeout_s:.3f} seconds',
                'positions': [],
                'timestamps': [],
            }
            result_path.write_text(
                json.dumps(result, indent=2) + '\n',
                encoding='utf-8',
            )
            log_path.write_text(str(error), encoding='utf-8')
            return result
        if not result_path.is_file():
            return {
                'success': False,
                'status': 'WORKER_ERROR',
                'reason': (
                    'cuMotion worker did not create a result file; '
                    f'exit={process.returncode}'
                ),
                'positions': [],
                'timestamps': [],
            }
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if process.returncode != 0 and result.get('success'):
            return {
                **result,
                'success': False,
                'status': 'WORKER_PROTOCOL_ERROR',
                'reason': 'worker returned non-zero with success=true',
            }
        return result


class DoorCuMotionPlanner(MotionPlannerBase):
    """Convert native cuMotion joint plans to SkillGen end-effector poses."""

    def __init__(
        self,
        env: Any,
        robot: Any,
        *,
        runner: PlannerRunner,
        kinematics: Kinematics,
        config: DoorCuMotionPlannerConfig | None = None,
        env_id: int = 0,
        debug: bool = False,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=debug)
        self.runner = runner
        self.kinematics = kinematics
        self.config = config or DoorCuMotionPlannerConfig()
        if self.config.sample_dt <= 0:
            raise ValueError('sample_dt must be positive')
        if not 0 < self.config.time_dilation_factor <= 1:
            raise ValueError('time_dilation_factor must be in (0, 1]')
        self.last_request: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self._planned_poses: list[torch.Tensor] = []
        self._plan_index = 0
        self.visualize_spheres = False
        self.approach_standoff_m = 0.0
        self.transition_goal_override: np.ndarray | None = None

    def reset_plan(self) -> None:
        """Discard every waypoint and reset the iteration cursor."""
        self._planned_poses = []
        self._plan_index = 0

    def _base_world(self, env_id: int) -> np.ndarray:
        value = self.env.get_robot_base_pose(env_id)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('robot base pose must be a finite 4x4 matrix')
        return matrix

    def _current_arm(self, env_id: int) -> np.ndarray:
        value = self.env.get_arm_joint_positions(env_id)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        joints = np.asarray(value, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('current arm state must contain six finite joints')
        return joints

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        *,
        expected_attached_object: str | None = None,
        env_id: int | None = None,
        **unused: Any,
    ) -> bool:
        """Refresh door collision cuboids and plan in the ARX base frame."""
        self.reset_plan()
        selected_env = self.env_id if env_id is None else int(env_id)
        if expected_attached_object is not None:
            self.last_result = {
                'success': False,
                'status': 'UNSUPPORTED_ATTACHMENT',
                'reason': (
                    'the door is articulated and must never be represented '
                    'as a rigid gripper attachment'
                ),
            }
            return False
        if isinstance(target_pose, torch.Tensor):
            target_world = target_pose.detach().cpu().numpy()
        else:
            target_world = np.asarray(target_pose)
        if self.transition_goal_override is not None:
            target_world = self.transition_goal_override
        target_world = np.array(target_world, dtype=np.float64, copy=True)
        if target_world.shape != (4, 4) or not np.isfinite(target_world).all():
            raise ValueError('target_pose must be a finite 4x4 matrix')
        if not np.isfinite(self.approach_standoff_m) or self.approach_standoff_m < 0:
            raise ValueError('approach_standoff_m must be finite and nonnegative')
        target_world[:3, 3] -= self.approach_standoff_m * target_world[:3, 0]
        base_world = self._base_world(selected_env)
        target_base = np.linalg.inv(base_world) @ target_world
        quaternion = Rotation.from_matrix(target_base[:3, :3]).as_quat()
        start = self._current_arm(selected_env)
        # Match the approved pilot: seeded ARX IK followed by cuMotion's
        # collision-checked joint-space planner. The generic task-space IK
        # rejects the demonstrated high side-grasp although ARX IK solves it.
        target_joints = self.kinematics.ik(target_base, start)
        request = {
            'start_positions': start.tolist(),
            'target_joints': target_joints.tolist(),
            'target_pose_base': {
                'position': target_base[:3, 3].tolist(),
                'quaternion_xyzw': quaternion.tolist(),
            },
            'cuboids': self.env.export_cumotion_cuboids(selected_env),
            'time_dilation_factor': self.config.time_dilation_factor,
            'sample_dt': self.config.sample_dt,
            'source': 'isaac_lab_skillgen',
            'approach_standoff_m': self.approach_standoff_m,
            'env_id': selected_env,
        }
        self.last_request = request
        result = self.runner.plan(request)
        self.last_result = result
        if not result.get('success'):
            return False
        positions = np.asarray(result.get('positions'), dtype=np.float64)
        timestamps = np.asarray(result.get('timestamps'), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 6:
            raise ValueError('cuMotion result positions must have shape (N, 6)')
        if timestamps.shape != (len(positions),):
            raise ValueError('cuMotion timestamps must match positions')
        if len(positions) < 1 or not np.isfinite(positions).all():
            raise ValueError('cuMotion returned no finite waypoints')
        if not np.isfinite(timestamps).all() or (
            len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0)
        ):
            raise ValueError('cuMotion timestamps must be finite and increasing')
        device = torch.device(self.env.device)
        self._planned_poses = [
            torch.as_tensor(
                base_world @ self.kinematics.fk(joints),
                dtype=torch.float32,
                device=device,
            )
            for joints in positions
        ]
        return True

    def has_next_waypoint(self) -> bool:
        """Return whether the current plan has an unconsumed pose."""
        return self._plan_index < len(self._planned_poses)

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        """Return and consume the next world-frame link6 target."""
        if not self.has_next_waypoint():
            raise IndexError('No more waypoints in the plan.')
        pose = self._planned_poses[self._plan_index]
        self._plan_index += 1
        return pose

    def get_planned_poses(self) -> list[torch.Tensor]:
        """Return a non-consuming copy of the complete current path."""
        return [pose.clone() for pose in self._planned_poses]

    def get_planner_info(self) -> dict[str, Any]:
        """Describe the actual backend used for provenance."""
        return {
            **super().get_planner_info(),
            'backend': 'nvidia_cumotion_native',
            'sample_dt': self.config.sample_dt,
            'time_dilation_factor': self.config.time_dilation_factor,
        }
