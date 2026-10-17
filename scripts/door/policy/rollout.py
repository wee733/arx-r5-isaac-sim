#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run one N1.7 closed-loop episode in the original ARX door physics environment.

``--replay-hdf5`` executes a generated episode's recorded absolute actions
instead of the policy, open loop, through exactly the same scene, cameras,
gate-free action path and acceptance. A model can only be blamed for a
failure once its own training episodes pass this replay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.policy.config import DEFAULT_CONFIG, load_config, resource
from arx_door.runtime import close_app, planning_urdf, runtime_provenance, validate_runtime

RUNTIME = runtime_provenance()
PROFILE = validate_runtime(RUNTIME)
if PROFILE != 'isaac61':
    raise RuntimeError('this first policy rollout is validated against the existing Isaac 6.1 profile')

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
parser.add_argument('--scene', type=Path, required=True)
parser.add_argument('--cuboids', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--max-steps', type=int, default=1800)
parser.add_argument('--success-hold-steps', type=int, default=30)
parser.add_argument('--replay-hdf5', type=Path,
                    help='replay this generated.hdf5 demo_0 actions instead of querying the policy')
AppLauncher.add_app_launcher_args(parser)
for option in ('--headless', '--enable_cameras'):
    if option not in parser._option_string_actions:
        parser.add_argument(option, action='store_true')
args = parser.parse_args()
config = load_config(args.config)
if not 1 <= args.max_steps <= 2700 or args.success_hold_steps < 1:
    parser.error('max-steps must be within the original 90s cap; hold steps must be positive')
args.device = 'cpu'  # Training data used the original CPU PhysX / MBP / CCD profile.
args.enable_cameras = True
app = AppLauncher(args, width=1280, height=720).app

import numpy as np
import omni.usd
import torch
from isaaclab.sim import activate_contact_sensors
from PIL import Image

from arx_door.autodata.acceptance import RuntimeAcceptance, as_numpy
from arx_door.files import write_json
from arx_door.policy.guards import GripperGate, RuleStagePrompter
from arx_door.policy.simulation import MountedCamera, RemotePolicy, VideoWriter
from arx_door.scene.ego_camera import ego_camera_cfg, ego_mount
from arx_door.skillgen.compat import quaternion_from_xyzw
from arx_door.skillgen import mdp
from arx_door.skillgen.env import DoorSkillGenEnv
from arx_door.skillgen.env_cfg import DoorSkillGenEnvCfg
from arx_door.skillgen.physics_profile import (
    configure_physics_profile,
    physics_material_bindings,
    physics_runtime_evidence,
    source_cpu_scene_settings,
    validate_physics_runtime,
)


def main() -> int:
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    video_keys = config.get('video_keys', ['wrist'])
    stage_language = config.get('language_mode', 'fixed') == 'stage'
    gate_config = config.get('gripper_gate')
    result = {'status': 'initializing', 'success': False, 'scope': 'simulation_closed_loop',
              'checkpoint': str(resource(config, 'checkpoint')), 'language': config['language'],
              'video_keys': video_keys, 'language_mode': 'stage' if stage_language else 'fixed',
              'gripper_gate': gate_config,
              'execution_horizon': config['execution_horizon'], 'runtime': RUNTIME,
              'clock_mode': 'synchronous simulation; physics pauses during inference',
              'mode': 'expert_replay' if args.replay_hdf5 else 'policy',
              'real_time_hardware_validated': False, 'steps': 0}
    write_json(output / 'result.json', result)
    env = remote = acceptance = None
    writers = []
    history, chunks, inference = [], [], []
    exit_code = 1
    try:
        manifest_path = args.scene.with_suffix('.json')
        manifest = json.loads(manifest_path.read_text())
        for key, value in {
            'SCENE': args.scene.absolute(), 'MANIFEST': manifest_path.absolute(),
            'CUBOIDS': args.cuboids.absolute(), 'URDF': planning_urdf(),
            'SEED': resource(config, 'seed_dataset'),
        }.items():
            os.environ['ARX_DOOR_SKILLGEN_' + key] = str(value)
        context = omni.usd.get_context()
        if context.open_stage(str(args.scene.absolute()),
                              load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE) is False:
            raise RuntimeError('could not open the prepared evaluation scene')
        for _ in range(5):
            app.update()
        stage = context.get_stage()
        activate_contact_sensors('/R5a', threshold=0.0, stage=stage)
        source_settings = source_cpu_scene_settings(stage)
        bindings = physics_material_bindings(stage)
        cfg = DoorSkillGenEnvCfg()
        cfg.seed = config['seed']
        cfg.sim.device = 'cpu'
        cfg.observations.rgb_camera = None
        cfg.scene.overview_camera.update_period = 0.0
        if 'ego' in video_keys:
            cfg.scene.ego_camera = ego_camera_cfg(
                manifest['placement'], ego_mount(config.get('ego_camera')),
                quaternion_from_xyzw=quaternion_from_xyzw)
        cfg.terminations.success = None
        cfg.terminations.time_out = None
        cfg.recorders = None
        configure_physics_profile(cfg.sim, config['physics_profile'], source_settings)
        env = DoorSkillGenEnv(cfg=cfg)
        env.cfg.sim.render_interval = 1000000
        env.reset()
        evidence = physics_runtime_evidence(
            stage, env.cfg.sim, str(env.sim.device), config['physics_profile'],
            source_settings=source_settings, original_bindings=bindings)
        validate_physics_runtime(evidence)
        result['physics_runtime'] = evidence
        result['scene'] = str(args.scene.absolute())
        result['placement'] = manifest['placement']
        if not np.isclose(env.sim.get_physics_dt(), 1/120):
            raise RuntimeError('physics clock differs from training data')
        camera = MountedCamera(env)
        result['initial_camera_check'] = camera.render()
        check = result['initial_camera_check']
        Image.fromarray(camera.rgb('wrist_camera')).save(output / 'initial_wrist.png')
        Image.fromarray(camera.rgb('overview_camera')).save(output / 'initial_overview.png')
        writers = [VideoWriter(output / 'wrist.mp4', 640, 480),
                   VideoWriter(output / 'overview.mp4', 960, 720)]
        replay = None
        if args.replay_hdf5:
            import h5py
            with h5py.File(args.replay_hdf5, 'r') as file:
                demo = file['data/demo_0']
                replay = np.asarray(demo['processed_actions'], dtype=np.float32)
                recorded_start = np.asarray(demo['obs/joint_pos'][0], dtype=np.float32)
            start = as_numpy(mdp.robot_state(env))[0]
            result['replay'] = {'hdf5': str(args.replay_hdf5.absolute()), 'frames': len(replay),
                                'initial_state_max_abs_error': float(np.abs(start - recorded_start).max())}
            gate_config, stage_language = None, False
        else:
            remote = RemotePolicy(config, args.config, output)
            remote.reset()
        gate = GripperGate(**gate_config) if gate_config else None
        prompter = RuleStagePrompter() if stage_language else None
        sensors = {'wrist': 'wrist_camera', 'ego': 'ego_camera'}
        if 'ego' in video_keys:
            writers.append(VideoWriter(output / 'ego.mp4', 640, 480))
        stages, raw_gripper = [], []
        acceptance = RuntimeAcceptance(1, behavior_family='knob_hold_open')
        robot = env.scene['robot']
        ids, names = robot.find_joints([f'joint{i}' for i in range(1, 8)], preserve_order=True)
        if names != [f'joint{i}' for i in range(1, 8)]:
            raise RuntimeError('simulator joint order differs from policy state')
        limits = as_numpy(robot.data.soft_joint_pos_limits)[0, ids[:6]]
        held = 0
        chunk = None
        chunk_offset = 0
        result['status'] = 'running'
        result['stop_reason'] = 'timeout'
        write_json(output / 'result.json', result)
        for step in range(args.max_steps):
            if not app.is_running():
                result['stop_reason'] = 'window_closed'
                break
            state = as_numpy(mdp.robot_state(env))[0].copy()
            pre_arms = as_numpy(mdp.arm_contact_forces(env))
            pre_clearance = as_numpy(mdp.camera_panel_clearance(env))
            acceptance.observe_safety([0], pre_arms, pre_clearance)
            if replay is not None:
                if step >= len(replay):
                    result['stop_reason'] = 'replay_exhausted'
                    break
                chunk, chunk_offset = replay[step:step + 1], 0
                inference.append([step, 0.0])
            elif chunk is None or chunk_offset == config['execution_horizon']:
                views = {key: camera.rgb(sensors[key]) for key in video_keys}
                language = prompter.instruction if prompter else None
                chunk, elapsed = remote.infer(views, state, language)
                chunks.append(chunk)
                inference.append([step, elapsed])
                chunk_offset = 0
            command = chunk[chunk_offset].copy()
            raw_gripper.append(float(command[6]))
            if gate is not None:
                command[6] = gate(float(command[6]))
            if np.any(command[:6] < limits[:, 0] - 1e-3) or np.any(command[:6] > limits[:, 1] + 1e-3):
                result.update(stop_reason='predicted_joint_limit',
                              rejected_command=command.tolist(),
                              arm_joint_limits_rad=limits.tolist(), rejected_step=step)
                break
            env.step(torch.as_tensor(command[None], dtype=torch.float32, device=env.device))
            check = camera.render()
            door = as_numpy(mdp.door_joint_state(env))
            fingers = as_numpy(mdp.finger_contacts(env))
            arms = as_numpy(mdp.arm_contact_forces(env))
            clearance = as_numpy(mdp.camera_panel_clearance(env))
            acceptance.update([0], door, fingers, arms, clearance)
            actual = as_numpy(mdp.robot_state(env))[0].copy()
            history.append((state, command, actual, door[0].copy(), fingers[0].copy(),
                            arms[0].copy(), float(clearance.reshape(-1)[0])))
            writers[0].append(camera.rgb('wrist_camera'))
            writers[1].append(camera.rgb('overview_camera'))
            if 'ego' in video_keys:
                writers[2].append(camera.rgb('ego_camera'))
            if prompter is not None:
                previous = prompter.stage
                stage, missed = prompter.update(actual, raw_gripper[-1],
                                                gate.closed if gate else command[6] < 0.014)
                if missed and gate is not None:
                    gate.force_open()
                if stage != previous:
                    chunk = None  # the instruction changed: re-plan from this observation
                    result.setdefault('stage_transitions', []).append(
                        {'step': step, 'stage': prompter.name, 'missed_grasp': missed})
                stages.append(stage)
            held = held + 1 if acceptance.accepted()[0] else 0
            result['steps'] = step + 1
            chunk_offset += 1
            if step % 120 == 0:
                result['physical_metrics'] = acceptance.summary(0)
                write_json(output / 'result.json', result)
                print(f'ROLLOUT step={step} door_deg={np.rad2deg(door[0,0]):.2f} '
                      f'inference_s={inference[-1][1]:.3f}', flush=True)
            if held >= args.success_hold_steps:
                result.update(success=True, stop_reason='physical_success_held')
                break
            if not acceptance.finite[0] or acceptance.max_arm[0] > 2 or acceptance.min_clearance[0] < .01:
                result['stop_reason'] = 'physical_safety_limit'
                break
        result.update(status=('interrupted' if result['stop_reason'] == 'window_closed' else 'completed'),
                      physical_metrics=acceptance.summary(0),
                      final_camera_check=check, inference_calls=len(inference))
        exit_code = 130 if result['status'] == 'interrupted' else 0
    except BaseException as exc:  # noqa: BLE001 - the process records failures before exiting 1.
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        import traceback
        traceback.print_exc()
    finally:
        if acceptance is not None:
            result['physical_metrics'] = acceptance.summary(0)
        if history:
            columns = list(zip(*history))
            np.savez_compressed(output / 'trajectory.npz',
                                state=np.asarray(columns[0]), command=np.asarray(columns[1]),
                                actual=np.asarray(columns[2]), door_joint_pos=np.asarray(columns[3]),
                                finger_contacts=np.asarray(columns[4]), arm_contacts=np.asarray(columns[5]),
                                camera_clearance=np.asarray(columns[6]),
                                inference=np.asarray(inference), predicted_chunks=np.asarray(chunks),
                                raw_gripper_command=np.asarray(raw_gripper), stage=np.asarray(stages))
        for item in [*writers, remote, env]:
            if item is not None:
                try:
                    item.close()
                except Exception as exc:  # noqa: BLE001 - all owned resources must be closed.
                    result.setdefault('cleanup_errors', []).append(str(exc))
                    result['status'] = 'error'
                    exit_code = 1
        result.update(elapsed_seconds=time.monotonic() - started)
        write_json(output / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return exit_code


code = 1
try:
    code = main()
finally:
    close_app(app, code)
raise SystemExit(code)
