#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run one N1.7 closed-loop episode in the ARX door environment (arx_door.envs.door, joint_pos).

``--replay-hdf5`` executes a generated episode's recorded actions instead of
the policy, open loop, through the same environment and acceptance. AutoData
episodes (8-D absolute link6 pose + gripper) replay in the ``ik_abs`` action mode
they were generated with; 7-D joint episodes replay in ``joint_pos``. A model can only be blamed for a
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
from arx_door.runtime import close_app
from arx_door.autodata.runtime import runtime_provenance
from arx_door.envs.physics import PHYSICS_PROFILES

RUNTIME = runtime_provenance()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
parser.add_argument('--assets', type=Path, required=True,
                    help='asset bundle manifest.json (scripts/door/autodata/prepare.py)')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--physics', choices=PHYSICS_PROFILES, help='override policy config; replay uses run.json')
parser.add_argument('--max-steps', type=int, default=1800)
parser.add_argument('--success-hold-steps', type=int, default=30)
parser.add_argument('--replay-hdf5', type=Path,
                    help='replay recorded actions of one episode instead of querying the policy')
parser.add_argument('--replay-demo', default='demo_0', help='episode name inside --replay-hdf5')
AppLauncher.add_app_launcher_args(parser)
for option in ('--headless', '--enable_cameras'):
    if option not in parser._option_string_actions:
        parser.add_argument(option, action='store_true')
args = parser.parse_args()
config = load_config(args.config)
if args.max_steps < 1 or (not args.replay_hdf5 and args.max_steps > 2700) or args.success_hold_steps < 1:
    parser.error('max-steps and hold steps must be positive; policy runs have a 2700-step cap')
physics = args.physics or config['physics_profile']
if physics not in PHYSICS_PROFILES:
    parser.error(f'unknown physics profile: {physics}')
args.device = 'cpu' if physics == 'source_cpu' else 'cuda:0'
args.enable_cameras = True
app = AppLauncher(args, width=1280, height=720).app

import gymnasium as gym
import numpy as np
import omni.usd
import torch
from PIL import Image

from arx_door.envs import mdp
from arx_door.envs.acceptance import RuntimeAcceptance, as_numpy
from arx_door.envs.assets import load_asset_manifest
from arx_door.envs.door import IK_ABS_ACTION_DIM, JOINT_POS_ACTION_DIM, build
from arx_door.envs.physics import physics_material_bindings, physics_runtime_evidence, validate_physics_runtime
from arx_door.files import write_json
from arx_door.policy.guards import GripperGate, RuleStagePrompter
from arx_door.policy.simulation import MountedCamera, RemotePolicy, VideoWriter


def main() -> int:
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    video_keys = config.get('video_keys', ['wrist'])
    stage_language = config.get('language_mode', 'fixed') == 'stage'
    gate_config = config.get('gripper_gate')
    result = {'status': 'initializing', 'success': False,
              'scope': 'simulation_expert_replay' if args.replay_hdf5 else 'simulation_closed_loop',
              'checkpoint': str(resource(config, 'checkpoint')), 'language': config['language'],
              'video_keys': video_keys, 'language_mode': 'stage' if stage_language else 'fixed',
              'gripper_gate': gate_config,
              'execution_horizon': config['execution_horizon'], 'runtime': RUNTIME,
              'success_hold_steps': args.success_hold_steps,
              'clock_mode': 'synchronous simulation; physics pauses during inference',
              'mode': 'expert_replay' if args.replay_hdf5 else 'policy',
              'real_time_hardware_validated': False, 'steps': 0}
    write_json(output / 'result.json', result)
    env = remote = acceptance = None
    writers = {}
    history, chunks, inference = [], [], []
    exit_code = 1
    try:
        replay = recorded_start = None
        action_mode = 'joint_pos'
        if args.replay_hdf5:
            import h5py
            with h5py.File(args.replay_hdf5, 'r') as file:
                demo = file['data'][args.replay_demo]
                replay = np.asarray(demo['actions'], dtype=np.float32)
                # Generated episodes use the policy observation; converted
                # expert sources retain the original diagnostics observation.
                state_key = ('obs/joint_pos' if 'obs/joint_pos' in demo
                             else 'obs/diagnostics/joint_state')
                recorded_start = np.asarray(demo[state_key][0], dtype=np.float32)
            widths = {JOINT_POS_ACTION_DIM: 'joint_pos', IK_ABS_ACTION_DIM: 'ik_abs'}
            if replay.ndim != 2 or replay.shape[1] not in widths:
                raise ValueError(f'replay actions must be [T, 7] joint or [T, 8] pose commands: {replay.shape}')
            action_mode = widths[replay.shape[1]]
        gripper_index = IK_ABS_ACTION_DIM - 1 if action_mode == 'ik_abs' else JOINT_POS_ACTION_DIM - 1
        manifest = load_asset_manifest(args.assets)
        # One door environment serves AutoData (ik_abs) and policy evaluation (joint_pos).
        env_id, cfg, env_kwargs = build(
            manifest, 1, action=action_mode, physics=physics, name='Arx-Door-Eval-v0',
            ego_camera=(config.get('ego_camera') or {}) if 'ego' in video_keys else None)
        cfg.seed = config['seed']
        cfg.terminations.success = None
        cfg.recorders = None
        env = gym.make(env_id, cfg=cfg, **env_kwargs).unwrapped
        stage = omni.usd.get_context().get_stage()
        bindings = physics_material_bindings(stage)
        env.cfg.sim.render_interval = 1000000
        env.reset()
        evidence = physics_runtime_evidence(
            stage, env.cfg.sim, str(env.sim.device), physics,
            source_settings=manifest['source_settings'], original_bindings=bindings)
        evidence['material_comparison_origin'] = 'after_arena_construction_before_first_reset'
        validate_physics_runtime(evidence)
        result['physics_runtime'] = evidence
        result['assets'] = str(args.assets.absolute())
        result['source_scene'] = manifest['source_scene']
        if not np.isclose(env.sim.get_physics_dt(), 1/120):
            raise RuntimeError('physics clock differs from training data')
        camera = MountedCamera(env)
        result['initial_camera_check'] = camera.render()
        check = result['initial_camera_check']
        video_sensors = {'wrist': 'wrist_camera'}
        if 'overview_camera' in env.scene.sensors:
            video_sensors['overview'] = 'overview_camera'
        if 'ego' in video_keys:
            video_sensors['ego'] = 'ego_camera'
        for key, sensor in video_sensors.items():
            Image.fromarray(camera.rgb(sensor)).save(output / f'initial_{key}.png')
        writers = {key: VideoWriter(output / f'{key}.mp4', *camera.size(sensor)) for key, sensor in video_sensors.items()}
        if replay is not None:
            start = as_numpy(mdp.robot_state(env))[0]
            result['replay'] = {'hdf5': str(args.replay_hdf5.absolute()), 'demo': args.replay_demo,
                                'action_mode': action_mode, 'frames': len(replay),
                                'initial_state_max_abs_error': float(np.abs(start - recorded_start).max())}
            gate_config, stage_language = None, False
        else:
            remote = RemotePolicy(config, args.config, output)
            remote.reset()
        gate = GripperGate(**gate_config) if gate_config else None
        prompter = RuleStagePrompter() if stage_language else None
        sensors = {'wrist': 'wrist_camera', 'ego': 'ego_camera'}
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
            raw_gripper.append(float(command[gripper_index]))
            if gate is not None:
                command[gripper_index] = gate(float(command[gripper_index]))
            if action_mode == 'joint_pos' and (np.any(command[:6] < limits[:, 0] - 1e-3)
                                               or np.any(command[:6] > limits[:, 1] + 1e-3)):
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
            for key, writer in writers.items():
                writer.append(camera.rgb(video_sensors[key]))
            if prompter is not None:
                previous = prompter.stage
                stage, missed = prompter.update(actual, raw_gripper[-1],
                                                gate.closed if gate else command[gripper_index] < 0.014)
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
            replay_complete = replay is None or step + 1 == len(replay)
            if held >= args.success_hold_steps and replay_complete:
                result.update(success=True, stop_reason=(
                    'physical_replay_accepted' if replay is not None else 'physical_success_held'))
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
        for item in [*writers.values(), remote, env]:
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
