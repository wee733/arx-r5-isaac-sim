#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Generate one physically replayed ARX door trial with Isaac Lab SkillGen."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    ROOT.parent
    / 'arx-r5-moveit'
    / 'isaac_ros_manipulation_arx_r5a_robot_description'
    / 'urdf/r5a_cumotion.urdf'
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene', type=Path, required=True)
parser.add_argument('--scene-manifest', type=Path, required=True)
parser.add_argument('--cuboids', type=Path, required=True)
parser.add_argument('--input-file', type=Path, required=True)
parser.add_argument('--output-file', type=Path, required=True)
parser.add_argument('--artifact-dir', type=Path, required=True)
parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
parser.add_argument('--angle-deg', type=float, required=True)
parser.add_argument('--timeout-s', type=float, default=900.0)
parser.add_argument(
    '--smoke-only',
    action='store_true',
    help='Create/reset/step the environment without running SkillGen.',
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import asyncio  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402

from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.sim import activate_contact_sensors  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool  # noqa: E402
from isaaclab_mimic.datagen.generation import setup_env_config  # noqa: E402

from arx_r5_isaac_sim_bringup.door_skillgen.cumotion_planner import (  # noqa: E402
    DoorCuMotionPlanner,
    SubprocessCuMotionRunner,
)
from arx_r5_isaac_sim_bringup.door_skillgen.registry import register  # noqa: E402
from arx_r5_isaac_sim_bringup.door_skillgen.seed import ENV_ID  # noqa: E402
from arx_r5_isaac_sim_bringup.door_skillgen import mdp  # noqa: E402
from arx_r5_isaac_sim_bringup.door_skillgen.generator import DoorDataGenerator  # noqa: E402
from arx_r5_isaac_sim_bringup.door_skillgen.contract import evaluate_door_success  # noqa: E402
from arx_r5_isaac_sim_bringup.door_teaching_kinematics import (  # noqa: E402
    ArmKinematics,
)


class RawVideoWriter:
    """Stream RGB frames directly to ffmpeg without temporary image files."""

    def __init__(self, path: Path, width: int, height: int, fps: int = 30) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel',
                'error',
                '-y',
                '-f',
                'rawvideo',
                '-pixel_format',
                'rgb24',
                '-video_size',
                f'{width}x{height}',
                '-framerate',
                str(fps),
                '-i',
                'pipe:0',
                '-an',
                '-c:v',
                'libx264',
                '-preset',
                'veryfast',
                '-crf',
                '20',
                '-pix_fmt',
                'yuv420p',
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.frames = 0

    def append(self, image: torch.Tensor) -> None:
        frame = image.detach().cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if self.process.stdin is None:
            raise RuntimeError('ffmpeg input pipe is closed')
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode() if self.process.stderr else ''
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f'ffmpeg failed for {self.path}: {stderr}')


def _open_stage(scene: Path) -> None:
    if not scene.is_file():
        raise FileNotFoundError(scene)
    result = omni.usd.get_context().open_stage(
        str(scene.resolve()),
        load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE,
    )
    if result is False:
        raise RuntimeError(f'failed to open stage {scene}')
    for _ in range(5):
        simulation_app.update()
    activate_contact_sensors('/R5a', threshold=0.0)


def _configure_environment() -> tuple[object, object]:
    os.environ['ARX_DOOR_SKILLGEN_SCENE'] = str(args_cli.scene.resolve())
    os.environ['ARX_DOOR_SKILLGEN_MANIFEST'] = str(
        args_cli.scene_manifest.resolve()
    )
    os.environ['ARX_DOOR_SKILLGEN_CUBOIDS'] = str(args_cli.cuboids.resolve())
    os.environ['ARX_DOOR_SKILLGEN_URDF'] = str(args_cli.urdf.resolve())
    os.environ['ARX_DOOR_SKILLGEN_SEED'] = str(args_cli.input_file.resolve())
    register()
    output = args_cli.output_file.resolve()
    cfg, success_term = setup_env_config(
        env_name=ENV_ID,
        output_dir=str(output.parent),
        output_file_name=output.stem,
        num_envs=1,
        device=args_cli.device,
        generation_num_trials=1,
    )
    cfg.recorders.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    )
    cfg.datagen_config.use_skillgen = True
    cfg.datagen_config.generation_guarantee = False
    cfg.datagen_config.generation_keep_failed = True
    cfg.recorders.export_in_record_pre_reset = False
    return cfg, success_term


def _camera_frame(env, sensor_name: str) -> torch.Tensor:
    return env.scene.sensors[sensor_name].data.output['rgb']


def _capture(env, writers: dict[str, RawVideoWriter]) -> None:
    for sensor_name, writer in writers.items():
        writer.append(_camera_frame(env, sensor_name))


def _hdf5_episode_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with h5py.File(path, 'r') as handle:
        return len(handle.get('data', {}))


def _hdf5_frame_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with h5py.File(path, 'r') as handle:
        data = handle.get('data')
        if data is None:
            return 0
        names = sorted(data.keys())
        if not names:
            return 0
        return int(data[names[0]]['actions'].shape[0])


def _run_attempt(env, success_term, planner, writers, audit) -> dict:
    """Run one DataGenerator attempt while owning the synchronous sim loop."""
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    reset_queue = asyncio.Queue()
    action_queue = asyncio.Queue()
    lock = asyncio.Lock()
    pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=lock)
    pool.load_from_dataset_file(str(args_cli.input_file.resolve()))
    if pool.num_datagen_infos != 1:
        raise ValueError(f'expected one seed episode, found {pool.num_datagen_infos}')
    generator = DoorDataGenerator(env=env, src_demo_datagen_info_pool=pool, motion_planner=planner)

    async def generate_once():
        return await generator.generate(
            env_id=0,
            success_term=success_term,
            env_reset_queue=reset_queue,
            env_action_queue=action_queue,
            pause_subtask=False,
            export_demo=False,
            motion_planner=planner,
        )

    task = event_loop.create_task(generate_once())
    started = time.monotonic()
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    try:
        while not task.done():
            if time.monotonic() - started > args_cli.timeout_s:
                raise TimeoutError(
                    f'SkillGen attempt exceeded {args_cli.timeout_s:.1f} seconds'
                )
            event_loop.run_until_complete(asyncio.sleep(0))
            while not reset_queue.empty():
                env_id_tensor[0] = reset_queue.get_nowait()
                env.reset(env_ids=env_id_tensor)
                reset_queue.task_done()
            if action_queue.empty():
                continue
            env_id, action = action_queue.get_nowait()
            actions = torch.zeros(
                env.action_space.shape,
                dtype=torch.float32,
                device=env.device,
            )
            actions[env_id] = action.to(env.device)
            # Record the same pre-action observation as the native recorder.
            _capture(env, writers)
            env.recorder_manager.add_to_episodes('obs/diagnostics', {
                'door_joint_pos': mdp.door_joint_state(env),
                'camera_clearance': mdp.camera_panel_clearance(env),
                'finger_contacts': mdp.finger_contacts(env),
                'arm_contacts': mdp.arm_contact_forces(env),
                'timestamp': torch.full((1, 1), audit['physics_steps'] / 30., device=env.device),
                'transition_index': torch.full((1, 1), planner.runner._sequence, device=env.device),
            })
            env.recorder_manager.add_to_episodes('obs/datagen_info', {
                'eef_pose': {'link6': env.get_robot_eef_pose('link6')},
                'target_eef_pose': env.action_to_target_eef_pose(actions),
                'object_pose': env.get_object_poses(),
            })
            env.step(actions)
            audit['physics_steps'] += 1
            _audit_step(env, audit)
            if audit['physics_steps'] % 150 == 0:
                print(json.dumps({'progress': audit}), flush=True)
            action_queue.task_done()
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                event_loop.run_until_complete(task)
        event_loop.close()


def _audit_step(env, audit):
    """Accumulate physical evidence; no success from door angle alone."""
    door = mdp.door_joint_state(env)[0].detach().cpu().numpy()
    contacts = mdp.finger_contacts(env)[0].detach().cpu().numpy()
    clearance = float(mdp.camera_panel_clearance(env)[0, 0])
    arm_force = float(torch.linalg.vector_norm(mdp.arm_contact_forces(env), dim=-1).max())
    audit['final_door_angle_deg'] = float(np.rad2deg(door[0]))
    audit['max_handle_angle_deg'] = max(audit['max_handle_angle_deg'], float(np.rad2deg(door[1])))
    audit['max_latch_m'] = max(audit['max_latch_m'], float(door[2]))
    audit['min_camera_clearance_m'] = min(audit['min_camera_clearance_m'], clearance)
    audit['max_arm_contact_n'] = max(audit['max_arm_contact_n'], arm_force)
    knob_contact = bool(np.linalg.norm(contacts[:, 0], axis=-1).min() >= 3.)
    audit['knob_grasp_observed'] |= knob_contact
    audit['unlock_observed'] |= bool(np.rad2deg(door[1]) >= 43.5 and door[2] >= .0154)
    audit['gap_with_knob_grasp_observed'] |= bool(np.rad2deg(door[0]) >= 10. and knob_contact)
    if clearance < .01:
        raise RuntimeError(f'camera clearance {clearance:.6f} m below 0.01 m')
    if arm_force > 40.:
        raise RuntimeError(f'non-finger contact {arm_force:.3f} N exceeds 40 N')


def main() -> int:
    """Create the environment, execute one attempt, and publish its result."""
    output = args_cli.output_file.resolve()
    if output.exists() or (args_cli.artifact_dir / 'result.json').exists():
        raise FileExistsError('use a fresh output directory to preserve previous attempts')
    _open_stage(args_cli.scene.resolve())
    cfg, success_term = _configure_environment()
    env = gym.make(ENV_ID, cfg=cfg).unwrapped
    env.reset()
    if args_cli.smoke_only:
        robot = env.scene['robot']
        door = env.scene['door']
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos + 0.01,
            torch.zeros_like(robot.data.joint_vel),
        )
        door.write_joint_state_to_sim(
            torch.full_like(door.data.joint_pos, 0.01),
            torch.zeros_like(door.data.joint_vel),
        )
        env.reset()
        torch.testing.assert_close(robot.data.joint_pos, robot.data.default_joint_pos)
        torch.testing.assert_close(door.data.joint_pos, door.data.default_joint_pos)
        with h5py.File(args_cli.input_file, 'r') as source:
            expected_knob = source['data/demo_0/obs/datagen_info/object_pose/knob'][0]
        np.testing.assert_allclose(
            env.get_object_poses()['knob'][0].cpu().numpy(), expected_knob, atol=1e-4,
        )
        action = torch.tensor(
            [[*env.get_arm_joint_positions(0).tolist(), 0.028]],
            dtype=torch.float32,
            device=env.device,
        )
        for _ in range(30):
            env.step(action)
        tracking_error = float(torch.linalg.vector_norm(
            env.get_robot_eef_pose('link6')[:, :3, 3]
            - env.action_to_target_eef_pose(action)['link6'][:, :3, 3], dim=-1,
        ).max())
        print(json.dumps({
            'hold_tracking_error_m': tracking_error,
            'joint_stiffness': robot.data.joint_stiffness.tolist(),
            'joint_damping': robot.data.joint_damping.tolist(),
        }), flush=True)
        if tracking_error > 0.005:
            raise RuntimeError(f'static arm tracking error {tracking_error:.6f} m exceeds 5 mm')
        payload = {
            'env': ENV_ID,
            'action_shape': list(env.action_space.shape),
            'robot_joints': env.scene['robot'].joint_names,
            'door_joints': env.scene['door'].joint_names,
            'robot_bodies': env.scene['robot'].body_names,
            'door_bodies': env.scene['door'].body_names,
            'wrist_shape': list(_camera_frame(env, 'wrist_camera').shape),
            'base_world': env.get_robot_base_pose(0).tolist(),
            'reset_verified': True,
            'object_frames_verified': True,
            'hold_tracking_error_m': tracking_error,
        }
        print(json.dumps(payload, indent=2))
        env.close()
        return 0

    output = args_cli.output_file.resolve()
    args_cli.artifact_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        'wrist_camera': RawVideoWriter(
            args_cli.artifact_dir / 'wrist.mp4', 640, 480
        ),
        'overview_camera': RawVideoWriter(
            args_cli.artifact_dir / 'overview.mp4', 960, 720
        ),
    }
    runner = SubprocessCuMotionRunner(
        ROOT / 'scripts/run_door_cumotion.sh',
        args_cli.artifact_dir / 'cumotion',
    )
    planner = DoorCuMotionPlanner(
        env,
        env.scene['robot'],
        runner=runner,
        kinematics=ArmKinematics(args_cli.urdf),
    )
    results = None
    error = None
    audit = {
        'physics_steps': 0,
        'final_door_angle_deg': 0.0,
        'max_handle_angle_deg': 0.0,
        'max_latch_m': 0.0,
        'min_camera_clearance_m': 1e6,
        'max_arm_contact_n': 0.0,
        'knob_grasp_observed': False,
        'unlock_observed': False,
        'gap_with_knob_grasp_observed': False,
    }
    success = False
    acceptance_reasons = []
    try:
        results = _run_attempt(env, success_term, planner, writers, audit)
        acceptance = evaluate_door_success(
            door_angle_rad=float(mdp.door_joint_state(env)[0, 0]),
            finger_forces=mdp.finger_contacts(env)[0, :, 1].cpu().numpy(),
            panel_normal=env.get_object_poses()['panel'][0, :3, 1].cpu().numpy(),
            camera_clearance_m=float(mdp.camera_panel_clearance(env)[0, 0]),
            arm_contact_forces=mdp.arm_contact_forces(env)[0].cpu().numpy(),
        )
        acceptance_reasons = list(acceptance.reasons)
        acceptance_reasons.extend(key + '_not_observed' for key in (
            'knob_grasp_observed', 'unlock_observed', 'gap_with_knob_grasp_observed',
        ) if not audit[key])
        success = (
            bool(results and results.get('success'))
            and bool(mdp.door_success(env)[0])
            and all(audit[key] for key in (
                'knob_grasp_observed', 'unlock_observed', 'gap_with_knob_grasp_observed',
            ))
        )
    except Exception as exception:  # retain the failed attempt and diagnostics
        import traceback
        traceback.print_exc()
        error = f'{type(exception).__name__}: {exception}'
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exception:
                success = False
                error = f'{error or ""}; video: {exception}'
        if audit['physics_steps']:
            env.recorder_manager.set_success_to_episodes(
                torch.tensor([0], device=env.device),
                torch.tensor([[success]], device=env.device),
            )
            env.recorder_manager.export_episodes(
                torch.tensor([0], device=env.device)
            )
        env.close()

    selected_hdf5 = output if success else output.with_name(
        f'{output.stem}_failed{output.suffix}'
    )
    if not selected_hdf5.is_file():
        selected_hdf5 = output
    failure_stage = failure_reason = None
    if not success:
        if error:
            failure_stage, failure_reason = 'generation_exception', error
        else:
            failure_stage = 'physical_acceptance'
            failure_reason = ', '.join(acceptance_reasons) or 'success_gate_not_met'
        if planner.last_result and not planner.last_result.get('success'):
            failure_stage = f'cumotion_transition_{runner._sequence:02d}'
            failure_reason = str(
                planner.last_result.get('reason')
                or planner.last_result.get('status')
            )
    result = {
        'schema': 'arx-door-skillgen-result-v1',
        'environment': ENV_ID,
        'angle_deg': float(args_cli.angle_deg),
        'physics_executed': audit['physics_steps'] > 0,
        'success': success,
        'failure_stage': failure_stage,
        'failure_reason': failure_reason,
        'frames': max(writer.frames for writer in writers.values()),
        'hdf5_frames': _hdf5_frame_count(selected_hdf5),
        'hdf5_episodes': _hdf5_episode_count(selected_hdf5),
        'final_door_angle_deg': audit['final_door_angle_deg'],
        'physical_audit': audit,
        'source_seed': str(args_cli.input_file.resolve()),
        'video_alignment': 'pre_action_observation_at_30_hz',
        'generation_config': {
            'refresh_panel_once_after_transition': True,
            'panel_transition_standoff_m': 0.04,
            'physics_dt_s': 1.0 / 120.0,
            'control_dt_s': 1.0 / 30.0,
            'audit_sampling_hz': 30,
            'transform_actual_skill_entrance': True,
        },
        'planner': planner.get_planner_info(),
        'artifacts': {
            'scene': str(args_cli.scene.resolve()),
            'scene_manifest': str(args_cli.scene_manifest.resolve()),
            'cuboids': str(args_cli.cuboids.resolve()),
            'hdf5': str(selected_hdf5.resolve()),
            'wrist_video': str((args_cli.artifact_dir / 'wrist.mp4').resolve()),
            'overview_video': str((args_cli.artifact_dir / 'overview.mp4').resolve()),
            'cumotion': str((args_cli.artifact_dir / 'cumotion').resolve()),
        },
    }
    result_path = args_cli.artifact_dir / 'result.json'
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        result_path = args_cli.artifact_dir / 'result.json'
        if not result_path.exists():
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps({
                'schema': 'arx-door-skillgen-result-v1',
                'angle_deg': args_cli.angle_deg,
                'success': False,
                'physics_executed': False,
                'failure_stage': 'initialization_or_export',
                'failure_reason': traceback.format_exc(),
                'frames': 0,
            }, indent=2) + '\n')
    finally:
        # A partially constructed env has not removed the timeline STOP callback.
        # Remove it before Replicator stops; otherwise shutdown loops in render().
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        simulation_app.close()
    raise SystemExit(exit_code)
