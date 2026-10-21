#!/usr/bin/env python3
"""Fresh LLM-authored geometric/feedback experts in the pinned Arena runtime.

No demonstration file is read. This is project expert acquisition, followed by
official AutoData annotation/generation; it is not an upstream LLM policy API.
Every pose command and actual IK actuator substep is recorded with RGB and
strict physical diagnostics. Failed attempts remain in their own directories.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.autodata.runtime import runtime_provenance
from arx_door.runtime import close_app
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--strategy', choices=['measured_arc', 'staged_arc'], required=True)
parser.add_argument('--seed', type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
for option in ('--headless', '--enable_cameras'):
    if option not in parser._option_string_actions:
        parser.add_argument(option, action='store_true')
args = parser.parse_args()
args.device, args.enable_cameras = 'cpu', True
START = time.monotonic()
RUNTIME = runtime_provenance()
args.output.mkdir(parents=True, exist_ok=False)
(args.output/'collector_source.py').write_text(Path(__file__).read_text())
app = AppLauncher(args).app

import gymnasium as gym
import h5py
import numpy as np
import omni.usd
import torch
from scipy.spatial.transform import Rotation, Slerp
from pxr import UsdGeom
from arx_door.envs import mdp
from arx_door.envs.assets import load_asset_manifest
from arx_door.envs.door import build, ENV_NAME
from arx_door.envs.acceptance import acceptance_state, as_numpy
from arx_door.envs.recorders import (_body_pose, capture_pre_step, capture_post_step,
                                     capture_joint_target_substep, camera_evidence, read_physical_state)
from arx_door.envs.physics import physics_material_bindings, physics_runtime_evidence, validate_physics_runtime
from arx_door.autodata.source import pose_matrix, absolute_pose_actions
from arx_door.kinematics import ArmKinematics
from arx_door.policy.simulation import VideoWriter


def main():
    cfgs = {
        'measured_arc': dict(roll_deg=0., stand_off_m=.060, staging_height_m=.0,
                             arc_increment_deg=3., arc_seconds=1.8, final_deg=32.),
        'staged_arc': dict(roll_deg=-3., stand_off_m=.075, staging_height_m=.025,
                           arc_increment_deg=4., arc_seconds=2.2, final_deg=34.),
    }
    spec = cfgs[args.strategy]
    result = dict(schema='arx-fresh-expert-v1', strategy=args.strategy, parameters=spec,
                  seed=args.seed, runtime=RUNTIME, success=False, source_demonstrations=[],
                  method='LLM-authored geometry and measured-feedback controller; no model training',
                  provenance='fresh physics execution; no K001 action input', phases=[],
                  rationale=['pregrasp keeps fingers outside knob before axial insertion',
                             'roll changes the round-knob contact configuration between experts',
                             'unlock requires measured handle and latch progress',
                             'hinge arc preserves grasp geometry; each arc starts from measured link6',
                             'physical safety and continuous grip gates are unchanged'])
    def save():
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    manifest = load_asset_manifest(args.assets)
    (args.output/'run.json').write_text(json.dumps(dict(
        schema='arx-autodata-run-v1', assets=os.path.relpath(args.assets.resolve(), args.output),
        input='generated.hdf5', physics='source_cpu', num_envs=1, trials=1, seed=args.seed,
        scene_parameters=manifest.get('trial_parameters', {}), expert_strategy=args.strategy), indent=2))
    env = file = None
    writers = {}
    frames, phase_ids, languages = 0, [], []
    phase_names = ['transit_pregrasp', 'approach_knob', 'grasp_knob', 'unlock_round_knob',
                   'pull_knob_for_gap', 'pull_door_by_knob', 'hold_door_open']
    instructions = ['Move the open gripper in front of the door knob.',
                    'Approach the door knob with the open gripper.', 'Grasp the door knob.',
                    'Rotate the knob until the latch is fully retracted.',
                    'Pull the unlocked door while holding the knob.',
                    'Continue opening the door while holding the knob.',
                    'Hold the door open.']
    events = (args.output/'controller_events.jsonl').open('w')
    try:
        env_id, cfg, kwargs = build(manifest, 1, action='ik_abs', physics='source_cpu',
                                    name='Arx-Fresh-Expert-v0', ego_camera={})
        cfg.seed, cfg.recorders, cfg.terminations.success = args.seed, None, None
        env = gym.make(env_id, cfg=cfg, **kwargs).unwrapped
        stage = omni.usd.get_context().get_stage()
        bindings = physics_material_bindings(stage)
        env.cfg.sim.render_interval = 1000000
        env.reset()
        evidence = physics_runtime_evidence(stage, env.cfg.sim, str(env.sim.device), 'source_cpu',
                     source_settings=manifest['source_settings'], original_bindings=bindings)
        validate_physics_runtime(evidence)
        result['physics_runtime'] = evidence
        result['startup_to_ready_seconds'] = time.monotonic()-START
        save()
        base = manifest['assets']['robot']['initial_pose']
        world_base = pose_matrix(base['position'], base['quaternion_xyzw'])
        to_base = np.linalg.inv(world_base)
        door = manifest['assets']['door']['initial_pose']
        door_base = to_base @ pose_matrix(door['position'], door['quaternion_xyzw'])
        kin = ArmKinematics(ROOT/'assets/arx_r5a/urdf/r5a_cumotion.urdf')
        physical = read_physical_state(env)
        if np.max(np.abs(physical['door_joint_pos'])) > .0035:
            raise RuntimeError('expert requires a fresh closed, locked door')
        panel = _body_pose(env, 'door', 'door_panel')[0]
        grasp_prim_path = '/World/envs/env_0/Door/door_handle/grasp_target'
        grasp_prim = stage.GetPrimAtPath(grasp_prim_path)
        if not grasp_prim.IsValid():
            raise RuntimeError('asset lacks the authored knob grasp_target')
        grasp_local = np.asarray(UsdGeom.Xformable(grasp_prim).GetLocalTransformation()).T
        knob_pose = _body_pose(env, 'door', 'door_handle')[0] @ grasp_local
        knob = knob_pose[:3, 3]
        grasp = np.eye(4)
        grasp[:3, :3] = panel[:3, :3] @ Rotation.from_euler('z', 90, degrees=True).as_matrix()
        grasp[:3, :3] = grasp[:3, :3] @ Rotation.from_euler('x', spec['roll_deg'], degrees=True).as_matrix()
        grasp[:3, 3] = knob - .145*grasp[:3, 0]
        result['geometry'] = dict(panel_world=panel.tolist(), knob_world=knob.tolist(),
                                   grasp_world=grasp.tolist(), robot_world=world_base.tolist(),
                                   grasp_target_path=grasp_prim_path, grasp_local=grasp_local.tolist())
        file = h5py.File(args.output/'generated.hdf5', 'x')
        data = file.create_group('data')
        data.attrs['env_args'] = json.dumps(dict(env_name=ENV_NAME, type=2, env_kwargs={}))
        episode = data.create_group('demo_0')
        episode.attrs['phase_names'] = json.dumps(phase_names)
        episode.attrs['behavior_family'] = 'knob_hold_open'
        episode.attrs['expert_strategy'] = args.strategy
        episode.attrs['seed'] = args.seed
        episode.require_group('obs/diagnostics').attrs['phase_names'] = json.dumps(phase_names)
        def initial_state(group, values):
            for key, value in values.items():
                if isinstance(value, dict):
                    initial_state(group.create_group(key), value)
                else:
                    group.create_dataset(key, data=as_numpy(value)[0:1])
        initial_state(episode.create_group('initial_state'), env.scene.get_state(is_relative=True))
        apply = env.action_manager.apply_action
        def apply_and_record(*a, **kw):
            ret = apply(*a, **kw)
            capture_joint_target_substep(env)
            return ret
        env.action_manager.apply_action = apply_and_record
        def append(path, value):
            value = np.asarray(value)
            if path not in episode:
                episode.create_dataset(path, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                       dtype=value.dtype, chunks=(1, *value.shape), compression='lzf')
            ds = episode[path]
            ds.resize(frames+1, axis=0)
            ds[frames] = value
        def step(target, grip, phase):
            nonlocal frames
            target_base = to_base @ target
            action = absolute_pose_actions(target_base[None], np.array([grip]))[0]
            _, obs = capture_pre_step(env)
            measured = to_base @ _body_pose(env, 'robot', 'link6')[0]
            for key, value in obs.items():
                append('obs/'+key, as_numpy(value)[0])
            for camera in ('wrist', 'ego', 'overview'):
                sensor = env.scene.sensors[camera+'_camera']
                if camera == 'overview':
                    sensor.update(0., force_recompute=True)
                rgb = as_numpy(sensor.data.output['rgb'])[0, ..., :3]
                if camera not in writers:
                    writers[camera] = VideoWriter(args.output/f'{camera}.mp4', rgb.shape[1], rgb.shape[0])
                writers[camera].append(rgb)
            env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
            _, diag = capture_post_step(env)
            append('actions', action)
            append('obs/datagen_info/eef_pose/link6', measured.astype(np.float32))
            append('obs/datagen_info/target_eef_pose/link6', target_base.astype(np.float32))
            append('obs/datagen_info/object_pose/door', door_base.astype(np.float32))
            append('obs/datagen_info/subtask_start_signals/knob_interaction', np.array([phase != 'transit_pregrasp'], dtype=np.int64))
            append('obs/datagen_info/subtask_term_signals/knob_interaction', np.array([phase == 'hold_door_open'], dtype=np.int64))
            for key, value in diag.items():
                append('obs/diagnostics/'+key, as_numpy(value)[0])
            phase_id = phase_names.index(phase)
            append('obs/diagnostics/phase_id', np.array(phase_id, dtype=np.int64))
            phase_ids.append(phase_id)
            languages.append(instructions[phase_id])
            frames += 1
            state = acceptance_state(env)
            if state.max_arm[0] > 2 or state.min_clearance[0] < .01 or state.first_loss[0] >= 0:
                raise RuntimeError('irreversible physical violation: '+json.dumps(state.summary(0)))
        def feedback():
            physical = read_physical_state(env)
            forces = np.linalg.norm(physical['finger_contacts'][0, :, 0], axis=-1)
            return physical['door_joint_pos'][0], forces
        def event(phase, begin, tick):
            door, forces = feedback()
            row = dict(phase=phase, start_frame=begin, end_frame_exclusive=frames,
                       simulated_seconds=(frames-begin)/30, wall_seconds=time.monotonic()-tick,
                       elapsed_seconds=time.monotonic()-START, door_deg=float(np.rad2deg(door[0])),
                       handle_deg=float(np.rad2deg(door[1])), latch_m=float(door[2]),
                       finger_knob_force_n=forces.tolist())
            result['phases'].append(row)
            events.write(json.dumps(row)+'\n'); events.flush()
            print('EXPERT_PHASE '+json.dumps(row), flush=True)
            file.flush(); save()
        def move(target, phase, grip, seconds, *, joint_path=False):
            begin, tick = frames, time.monotonic()
            current = _body_pose(env, 'robot', 'link6')[0]
            if joint_path:
                q0 = as_numpy(mdp.robot_state(env))[0, :6]
                q1 = kin.ik(to_base @ target, q0)
                seconds = max(seconds, float(np.max(np.abs(q1-q0)))*2/.45)
            slerp = Slerp([0., 1.], Rotation.from_matrix(np.stack([current[:3,:3], target[:3,:3]])))
            for i in range(1, round(seconds*30)+1):
                t = i/round(seconds*30)
                blend = t*t*(3-2*t)
                if joint_path:
                    command = world_base @ kin.fk(q0 + blend*(q1-q0))
                else:
                    command = np.eye(4)
                    command[:3,:3] = slerp(blend).as_matrix()
                    command[:3,3] = (1-blend)*current[:3,3] + blend*target[:3,3]
                step(command, grip, phase)
            event(phase, begin, tick)
        pregrasp = grasp.copy()
        pregrasp[:3,3] -= spec['stand_off_m']*grasp[:3,0]
        pregrasp[2,3] += spec['staging_height_m']
        move(pregrasp, 'transit_pregrasp', .028, 5, joint_path=True)
        if spec['staging_height_m']:
            pregrasp[2,3] -= spec['staging_height_m']
            move(pregrasp, 'transit_pregrasp', .028, 1.5)
        move(grasp, 'approach_knob', .028, 4)
        # Keep the commanded grasp throughout closure; measured tracking error
        # is evidence, not an excuse to teleport or relax acceptance.
        begin, tick = frames, time.monotonic()
        for _ in range(60):
            step(grasp, 0., 'grasp_knob')
        event('grasp_knob', begin, tick)
        if min(feedback()[1]) < 3:
            raise RuntimeError('failed to establish both knob contacts')
        unlocked = grasp.copy()
        previous = None
        for angle in (45., 47., 49., 51.):
            unlocked[:3,:3] = grasp[:3,:3] @ Rotation.from_euler('x', -angle, degrees=True).as_matrix()
            move(unlocked, 'unlock_round_knob', 0., 3 if angle == 45 else .8)
            door, forces = feedback()
            if min(forces) < 3:
                raise RuntimeError('lost knob grip during unlocking')
            if np.rad2deg(door[1]) >= 43.5 and door[2] >= .0154:
                break
            if previous is not None and not (door[1] > previous[1] and door[2] > previous[2]):
                raise RuntimeError('unlock recovery made no measured progress')
            previous = door.copy()
        else:
            raise RuntimeError('bounded unlock exhausted without handle/latch threshold')
        for _ in range(20):
            door, forces = feedback()
            angle = float(np.rad2deg(door[0]))
            delta = min(spec['arc_increment_deg'], spec['final_deg']-angle)
            if delta < .15:
                break
            target = _body_pose(env, 'robot', 'link6')[0]
            hinge = _body_pose(env, 'door', 'door_panel')[0, :3,3]
            turn = Rotation.from_euler('z', -delta, degrees=True).as_matrix()
            target[:3,3] = hinge + turn@(target[:3,3]-hinge)
            target[:3,:3] = turn@target[:3,:3]
            move(target, 'pull_knob_for_gap' if angle < 10 else 'pull_door_by_knob', 0., spec['arc_seconds'])
            now, forces = feedback()
            if min(forces) < 3 or abs(float(np.rad2deg(now[0]))-angle-delta) > 2:
                raise RuntimeError('grip or door-following gate failed')
        # Dwell at the last commanded target; no rebase of the contact preload.
        begin, tick = frames, time.monotonic()
        for _ in range(30):
            step(target, 0., 'hold_door_open')
        event('hold_door_open', begin, tick)
        result['success'] = bool(acceptance_state(env).accepted()[0])
        if not result['success']:
            raise RuntimeError('final strict acceptance failed')
    except Exception as error:
        result['error'] = str(error)
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    finally:
        result.update(frames=frames, simulated_seconds=frames/30,
                      acquisition_wall_seconds=time.monotonic()-START)
        if env is not None:
            result['physical_metrics'] = acceptance_state(env).summary(0)
            result['camera_evidence'] = camera_evidence(env, 0)
        if file is not None:
            file['data'].attrs['total'] = frames
            file['data/demo_0'].attrs['num_samples'] = frames
            file['data/demo_0'].attrs['success'] = result['success']
            file.close()
        events.close()
        for writer in writers.values():
            writer.close()
        (args.output/'language.json').write_text(json.dumps(dict(phase_names=phase_names,
                    instructions=instructions, phase_ids=phase_ids, alignment='pre_action'), indent=2))
        save()
        if env is not None:
            env.close()
    return 0 if result['success'] else 1


code = 1
try:
    code = main()
finally:
    close_app(app, code)
