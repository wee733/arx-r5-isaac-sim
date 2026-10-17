# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Official AutoData planner interface over the isolated native cuMotion worker.

The file request/result runner follows the protocol implemented by
scripts/door/planner/door_cumotion_worker.py. Its subprocess boundary is derived
from arx_door/skillgen/cumotion_planner.py, with an explicit argv command rather
than an import of the historical Isaac Lab Mimic planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from autodata_interfaces.motion_planners.motion_planner_base import MotionPlannerBase

from arx_door.autodata.adapter import checked_pose, numpy_value, require_zero_noise, validate_joints, validate_plan_result
from arx_door.kinematics import ArmKinematics


def failure(status: str, reason: str) -> dict[str, Any]:
    return {'success': False, 'status': status, 'reason': reason, 'positions': [], 'timestamps': []}


@dataclass(frozen=True)
class DoorPlannerConfig:
    sample_dt: float = 1. / 30.
    time_dilation_factor: float = 0.5
    motion_noise_scale: float = 0.


class FileCuMotionRunner:
    """Invoke the real worker's --request/--output protocol with no shell."""

    def __init__(self, worker_command: list[str], artifact_dir: str | Path, timeout_s: float = 300.) -> None:
        if not isinstance(worker_command, list) or not worker_command or any(
            not isinstance(part, str) or not part for part in worker_command
        ):
            raise ValueError('worker_command must be a nonempty list of argv strings')
        if not math.isfinite(timeout_s) or timeout_s <= 0.:
            raise ValueError('worker timeout must be positive and finite')
        self.worker_command = list(worker_command)
        self.timeout_s = timeout_s
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        prior = [int(path.name.split('.')[0].removeprefix('plan_'))
                 for path in self.artifact_dir.glob('plan_*.request.json')
                 if path.name.split('.')[0].removeprefix('plan_').isdigit()]
        self.sequence = max(prior, default=0)

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        stem = f'plan_{self.sequence:06d}'
        request_path = self.artifact_dir / f'{stem}.request.json'
        result_path = self.artifact_dir / f'{stem}.result.json'
        log_path = self.artifact_dir / f'{stem}.log'
        request_path.write_text(json.dumps(request, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        try:
            process = subprocess.run(
                [*self.worker_command, '--request', str(request_path), '--output', str(result_path)],
                capture_output=True, text=True, timeout=self.timeout_s, check=False,
            )
            log_path.write_text(process.stdout + process.stderr, encoding='utf-8')
        except (subprocess.TimeoutExpired, OSError) as error:
            status = 'TIMEOUT' if isinstance(error, subprocess.TimeoutExpired) else 'WORKER_ERROR'
            result = failure(status, str(error))
            result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            log_path.write_text(str(error), encoding='utf-8')
            return result
        try:
            result = json.loads(result_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            return failure('WORKER_PROTOCOL_ERROR', f'worker result unavailable (exit={process.returncode}): {error}')
        if not isinstance(result, dict) or not isinstance(result.get('success'), bool):
            return failure('WORKER_PROTOCOL_ERROR', 'worker must return an object with boolean success')
        if process.returncode != 0 and result['success']:
            return {**result, **failure('WORKER_PROTOCOL_ERROR', 'nonzero worker exit with success=true')}
        return result


class DoorPlanner(MotionPlannerBase):
    """Explicit ARX/link6 planner with one independent state per environment."""

    def __init__(
        self,
        datastream,
        worker_command: list[str],
        env_id: int = 0,
        *,
        cuboids_path: str | Path | None = None,
        urdf_path: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        timeout_s: float = 300.,
        sample_dt: float = 1. / 30.,
        time_dilation_factor: float = 0.5,
        motion_noise_scale: float = 0.,
        debug: bool = False,
        robot_asset_name: str = 'robot',
        **kwargs: Any,
    ) -> None:
        super().__init__(datastream, env_id, debug, robot_asset_name, **kwargs)
        if robot_asset_name != 'robot':
            raise ValueError('ARX Arena scene key must be robot')
        if not isinstance(env_id, int) or not 0 <= env_id < self.env.num_envs:
            raise ValueError('env_id is outside the live environment batch')
        if not math.isfinite(sample_dt) or not math.isclose(sample_dt, 1. / 30., abs_tol=1e-10, rel_tol=0.):
            raise ValueError('ARX AutoData planner must sample at the 30 Hz action period')
        if not math.isfinite(time_dilation_factor) or not 0. < time_dilation_factor <= 1.:
            raise ValueError('time_dilation_factor must be in (0, 1]')
        require_zero_noise(motion_noise_scale)
        self.config = DoorPlannerConfig(sample_dt, time_dilation_factor)
        self.step_size = None  # Worker already samples its time-parameterized trajectory.
        selected_urdf = Path(urdf_path or datastream.embodiment_adapter.urdf_path).expanduser().resolve()
        if selected_urdf != datastream.embodiment_adapter.urdf_path:
            raise ValueError('planner and embodiment adapter must use the same URDF')
        self.urdf_path = selected_urdf
        self.kinematics = ArmKinematics(self.urdf_path)
        if cuboids_path is not None:
            datastream.configure_collision_world(cuboids_path)
        if datastream.cuboids_path is None:
            raise ValueError('cuboids_path must be configured before constructing the planner')
        if artifact_dir is None:
            self._temporary_artifacts = tempfile.TemporaryDirectory(prefix=f'arx_autodata_env_{env_id}_')
            artifact_dir = self._temporary_artifacts.name
        # env scopes also isolate requests if callers pass a shared run directory.
        self.runner = FileCuMotionRunner(worker_command, Path(artifact_dir) / f'env_{env_id}', timeout_s)
        self.last_request: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self.reset_plan()

    def reset_plan(self) -> None:
        self._planned_poses: list[torch.Tensor] = []
        self._planned_joint_positions: list[torch.Tensor] = []
        self._planned_timestamps: list[float] = []
        self._planned_gripper_joint_positions: torch.Tensor | None = None
        self._plan_index = 0

    def _reject(self, status: str, reason: str) -> bool:
        self.last_result = failure(status, reason)
        return False

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int | None = None,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        **kwargs: Any,
    ) -> bool:
        """Plan env-relative target poses through the base-frame native worker.

        The native backend owns time parametrization. cuRobo's additional
        joint-distance retiming is explicitly unsupported, never silently used.
        """
        self.reset_plan()
        self.last_request = None
        self.last_result = None
        selected_env = self.env_id if env_id is None else env_id
        if selected_env != self.env_id:
            return self._reject('ENV_SCOPE_MISMATCH', 'a planner instance may only plan for its own env_id')
        if expected_attached_object is not None:
            return self._reject('UNSUPPORTED_ATTACHMENT', 'articulated door bodies cannot be rigid gripper attachments')
        if step_size is not None or enable_retiming:
            return self._reject('UNSUPPORTED_RETIMING', 'native worker provides 30 Hz time-parameterized sampling')
        if kwargs:
            return self._reject('UNSUPPORTED_OPTIONS', f'unsupported native planner options: {sorted(kwargs)}')
        try:
            target_env = checked_pose(target_pose)
            base_env = checked_pose(self.datastream.get_robot_root_pose([selected_env])[0])
            target_base = np.linalg.inv(base_env) @ target_env
            start = validate_joints(self.datastream.get_arm_joint_positions(selected_env), self.kinematics)
            gripper = numpy_value(self.datastream.get_gripper_joint_positions(selected_env))
            if gripper.shape != (2,) or not np.isfinite(gripper).all():
                raise ValueError('measured gripper must contain joint7 and joint8 [m]')
            goal = validate_joints(self.kinematics.ik(target_base, start), self.kinematics)
            request = {
                'start_positions': start.tolist(), 'target_joints': goal.tolist(),
                'gripper_joint_positions': gripper.tolist(), 'gripper_state_source': 'measured_joint_state',
                'target_pose_base': {'position': target_base[:3, 3].tolist(),
                                     'quaternion_xyzw': Rotation.from_matrix(target_base[:3, :3]).as_quat().tolist()},
                'cuboids': self.datastream.export_cumotion_cuboids(selected_env),
                'time_dilation_factor': self.config.time_dilation_factor, 'sample_dt': self.config.sample_dt,
                'source': 'autodata_arena_arx', 'env_id': selected_env,
            }
            self.last_request = request
            result = self.runner.plan(request)
            self.last_result = result
            if not result.get('success'):
                return False
            if result.get('backend') != 'nvidia_cumotion_native':
                raise ValueError('worker must identify the explicit nvidia_cumotion_native backend')
            model = result.get('model', {})
            if not isinstance(model, dict):
                raise ValueError('worker model metadata must be an object')
            result_model = model.get('urdf')
            if result_model is not None and Path(result_model).resolve() != self.urdf_path:
                raise ValueError('worker loaded a different URDF from the embodiment adapter')
            positions, timestamps = validate_plan_result(result, start, target_base, self.kinematics)
            device = self.datastream.device
            self._planned_joint_positions = [torch.as_tensor(q.copy(), dtype=torch.float32, device=device) for q in positions]
            self._planned_poses = [torch.as_tensor(base_env @ self.kinematics.fk(q), dtype=torch.float32, device=device)
                                   for q in positions]
            self._planned_timestamps = timestamps.tolist()
            self._planned_gripper_joint_positions = torch.as_tensor(gripper.copy(), dtype=torch.float64, device=device)
            return True
        except (ValueError, TypeError, KeyError, OSError) as error:
            # Preserve the native response and its runtime/model provenance when
            # rejecting invalid success, rather than exposing executable poses.
            self.last_result = {**(self.last_result or {}), 'success': False,
                                'status': 'INVALID_PLAN', 'reason': str(error)}
            self.reset_plan()
            return False

    def has_next_waypoint(self) -> bool:
        return self._plan_index < len(self._planned_poses)

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        if not self.has_next_waypoint():
            raise IndexError('no remaining native planner waypoints')
        pose = self._planned_poses[self._plan_index].clone()
        self._plan_index += 1
        return pose

    def get_planned_poses(self) -> list[torch.Tensor]:
        return [pose.clone() for pose in self._planned_poses]

    def get_planned_joint_positions(self) -> list[torch.Tensor]:
        """Return targets paired one-for-one with get_planned_poses()."""
        return [joints.clone() for joints in self._planned_joint_positions]

    def get_planned_gripper_joint_positions(self) -> torch.Tensor:
        """Return frozen request joint7/joint8 [m] for the current successful plan.

        This is a snapshot of the worker's measured-state input, never a fresh
        read after planning or source-demo gripper passthrough.
        """
        if self._planned_gripper_joint_positions is None or not self._planned_poses:
            raise ValueError('no successful plan with measured gripper metadata is available')
        return self._planned_gripper_joint_positions.clone()

    def get_planner_info(self) -> dict[str, Any]:
        return {**super().get_planner_info(), 'backend': 'nvidia_cumotion_native',
                'sample_dt': self.config.sample_dt, 'worker_command': self.runner.worker_command,
                'artifact_dir': str(self.runner.artifact_dir), 'urdf': str(self.urdf_path),
                'runtime': (self.last_result or {}).get('runtime')}
