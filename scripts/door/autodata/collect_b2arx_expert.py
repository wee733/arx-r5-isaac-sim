#!/usr/bin/env python3
"""B2ARX expert collection, official AutoData generation and independent replay.

The default geometric/feedback expert does not read demonstrations. Explicit
generation/replay modes consume the selected source in the pinned Arena runtime;
this is a project adapter, not an upstream LLM policy API. Pose commands and
actual IK actuator substeps retain strict physical diagnostics. Generation also
records RGB; numeric replay omits duplicate images. Failed attempts are retained.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import shutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.autodata.runtime import runtime_provenance
from arx_door.runtime import close_app
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--autodata-source', type=Path)
parser.add_argument('--source-demo', default='demo_0')
parser.add_argument('--variation', type=Path)
parser.add_argument('--replay-pose', action='store_true')
parser.add_argument('--pose-codec', action='store_true', help='execute the decoded GR00T XYZ_ROTVEC action representation')
parser.add_argument('--replay-joint-held', action='store_true')
parser.add_argument('--joint-sampling', choices=['first','last','mean'], default='first')
parser.add_argument('--numeric-replay', action='store_true', help='replay physics and numeric commands without duplicating training RGB/video')
parser.add_argument('--visual-preview', action='store_true', help='capture settled camera previews only; never accept as a demonstration')
parser.add_argument('--strategy', choices=['measured_arc', 'staged_arc'], default='measured_arc')
parser.add_argument('--pose', type=float, nargs=4)
parser.add_argument('--replication-of', default=None)
parser.add_argument('--replay-source',type=Path,help='independently execute recorded full-body substep commands in a fresh reset; never count as a new seed')
parser.add_argument('--batch-plan', type=Path)
parser.add_argument('--stop-on-failure', action='store_true')
parser.add_argument('--planner-contract', action='store_true',help='check live cuRobo FK and linkwise world sync without controlling the episode')
parser.add_argument('--continuous-targets',action='store_true',help='preserve command continuity at loaded Cartesian phase boundaries')
parser.add_argument('--unlock-seconds',type=float,default=3.,help='physical feedback-controlled first knob rotation duration')
parser.add_argument('--seed', type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
for option in ('--headless', '--enable_cameras'):
    if option not in parser._option_string_actions:
        parser.add_argument(option, action='store_true')
args = parser.parse_args()
if args.pose is None and args.batch_plan is None:parser.error('--pose or --batch-plan is required')
args.device, args.enable_cameras = 'cpu', True
START = time.monotonic()
COLLECTOR_SOURCE = Path(__file__).read_text()
RUNTIME = runtime_provenance()
args.output.mkdir(parents=True, exist_ok=False)
(args.output/'collector_source.py').write_text(COLLECTOR_SOURCE)
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
from arx_door.envs.b2arx import build, arm_state, ARM, LEGS
from arx_door.envs.b2arx_stability import Stability
ENV_NAME = "B2Arx-Door-Expert-v0"
from arx_door.envs.acceptance import acceptance_state, as_numpy
from arx_door.envs.recorders import (_body_pose, _pose, capture_pre_step, capture_post_step,
                                     capture_joint_target_substep, camera_evidence, read_physical_state)
from arx_door.envs.physics import physics_material_bindings, physics_runtime_evidence, validate_physics_runtime
from arx_door.autodata.source import pose_matrix, absolute_pose_actions
from arx_door.kinematics import ArmKinematics
from arx_door.policy.simulation import VideoWriter
SHARED_ENV=None


def main():
    global SHARED_ENV
    (args.output/'collector_source.py').write_text(COLLECTOR_SOURCE)
    cfgs = {
        'measured_arc': dict(roll_deg=0., stand_off_m=.060, staging_height_m=.0,
                             arc_increment_deg=3., arc_seconds=1.8, final_deg=32.),
        'staged_arc': dict(roll_deg=-3., stand_off_m=.075, staging_height_m=.025,
                           arc_increment_deg=4., arc_seconds=2.2, final_deg=34.),
    }
    spec = cfgs[args.strategy]
    spec['continuous_targets']=args.continuous_targets
    spec['unlock_seconds']=args.unlock_seconds
    result = dict(schema='b2arx-fresh-expert-v1', strategy=args.strategy, parameters=spec,
                  seed=args.seed, pose=args.pose, replication_of=args.replication_of, runtime=RUNTIME, success=False, source_demonstrations=[],
                  method='LLM-authored geometry and measured-feedback controller; no model training',
                  provenance='fresh physics execution; no K001 action input', phases=[],
                  rationale=['pregrasp keeps fingers outside knob before axial insertion',
                             'body placement changes the arm workspace while door geometry stays fixed',
                             'unlock requires measured handle and latch progress',
                             'hinge arc preserves the nominal unlocked grasp without accumulating compliance error',
                             'physical safety and continuous grip gates are unchanged'])
    def save():
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    replay_actions=replay_commands=replay_phases=None
    if args.numeric_replay and args.replay_source is None:
        raise ValueError('numeric-only capture is restricted to independent replay')
    if args.replay_source is not None:
        source_result=json.loads((args.replay_source/'result.json').read_text())
        if not source_result['success'] or source_result['pose']!=args.pose:
            raise ValueError('physical replay requires an accepted source at the same initialization pose')
        with h5py.File(args.replay_source/'generated.hdf5') as source:
            replay_actions=source['data/demo_0/actions'][:]
            replay_commands=None if args.replay_pose else source['data/demo_0/obs/whole_body_substeps/joint_target'][:]
            replay_phases=source['data/demo_0/obs/diagnostics/phase_id'][:]
        result['source_demonstrations']=[str(args.replay_source)]
        result['provenance']='fresh-reset independent full-body actuator-command replay, not a new expert source'
        if args.replay_pose:result['provenance']='fresh-reset independent live-mount pose-action replay'
        if args.pose_codec:
            if not args.replay_pose:raise ValueError('pose codec requires pose replay')
            from arx_door.autodata.b2arx_pose_codec import encode,decode
            replay_actions=decode(encode(replay_actions))
            result['pose_codec']='XYZ_ROTVEC float32 -> quaternion_xyzw float32'
        if args.replay_joint_held:
            target=(replay_commands[:,-1:,:] if args.joint_sampling=='last' else
                    replay_commands.mean(axis=1,keepdims=True) if args.joint_sampling=='mean' else replay_commands[:,0:1,:])
            replay_commands=np.repeat(target,4,axis=1)
            result['provenance']=f'fresh-reset independent {args.joint_sampling}-substep joint-target held at 30 Hz replay'
            result['joint_sampling']=args.joint_sampling
        result['replay_capture']='numeric physics only; no RGB validation' if args.numeric_replay else 'full RGB and physics'
    snapshot=args.output/'implementation';snapshot.mkdir(exist_ok=True)
    for path in ('arx_door/envs/b2arx.py','arx_door/envs/b2arx_stability.py','arx_door/envs/recorders.py',
                 'arx_door/envs/acceptance.py','configs/autodata/embodiments/b2arx.yaml',
                 'arx_door/autodata/b2arx_generation.py','arx_door/autodata/b2arx_variations.py',
                 'arx_door/autodata/b2arx_replay.py','arx_door/autodata/b2arx_pose_codec.py'):
        shutil.copy2(ROOT/path,snapshot/Path(path).name)
    shutil.copy2(args.assets,snapshot/'manifest.json')
    manifest = load_asset_manifest(args.assets)
    (args.output/'run.json').write_text(json.dumps(dict(
        schema='arx-autodata-run-v1', assets=os.path.relpath(args.assets.resolve(), args.output),
        input='generated.hdf5', physics='source_cpu', num_envs=1, trials=1, seed=args.seed,
        robot_initial_pose_xyz_yaw=args.pose,embodiment='b2arx',action_reference_link='R5a',
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
        env_id, cfg, kwargs = build(manifest, pose=args.pose, action='ik_abs')
        from isaaclab.utils.dict import class_to_dict
        (args.output/'environment_config.json').write_text(json.dumps(class_to_dict(cfg),default=str,indent=2))
        cfg.seed, cfg.recorders, cfg.terminations.success = args.seed, None, None
        if SHARED_ENV is None:
            env = gym.make(env_id, cfg=cfg, **kwargs).unwrapped
            if args.batch_plan is not None:SHARED_ENV=env
        else:
            env=SHARED_ENV
            env.seed(args.seed)
        stage = omni.usd.get_context().get_stage()
        if args.variation is not None:
            from arx_door.autodata.b2arx_variations import apply_variation
            result['variation'] = apply_variation(stage, json.loads(args.variation.read_text()))
        if args.autodata_source is not None:
            result['schema']='b2arx-autodata-generation-v1'
            result['method']='official AutoData DataGenerator and SkillGen, custom floating-mount embodiment and live collision-world adapter'
            result['provenance']='official AutoData SkillGen physical generation'
            result['source_demonstrations']=[str(args.autodata_source)]
        bindings = physics_material_bindings(stage)
        env.cfg.sim.render_interval = 1000000
        env.reset()
        if args.batch_plan is not None:
            # Episode initialization only: clear all articulation and controller
            # state through reset, then set this trial's free-root starting pose.
            import math
            root_pose=torch.tensor([[*args.pose[:3],0.,0.,math.sin(args.pose[3]/2),math.cos(args.pose[3]/2)]],device=env.device)
            env.scene['robot'].write_root_pose_to_sim(root_pose)
            env.scene['robot'].write_root_velocity_to_sim(torch.zeros((1,6),device=env.device))
            env.sim.forward();env.scene.update(env.physics_dt)
        from arx_door.envs.recorders import reset_recording
        reset_recording(env)
        evidence = physics_runtime_evidence(stage, env.cfg.sim, str(env.sim.device), 'source_cpu',
                     source_settings=manifest['source_settings'], original_bindings=bindings)
        validate_physics_runtime(evidence)
        result['physics_runtime'] = evidence
        result['startup_to_ready_seconds'] = time.monotonic()-START
        save()
        robot = env.scene['robot']
        if robot.is_fixed_base:
            raise RuntimeError('fixed-base debug cannot be an expert seed')
        result['controller'] = cfg.b2arx_control
        result['joint_names'] = list(robot.joint_names)
        if replay_commands is not None and source_result['joint_names']!=result['joint_names']:
            raise RuntimeError('recorded full-body command joint ordering differs')
        result['contact_sensor_bodies']={name:list(env.scene.sensors[name].body_names)
            for name in ('feet_contact','body_contact','leg_contact','arm_contacts')}
        monitor = Stability(env)
        settled=[]
        mount0 = _body_pose(env,'robot','R5a')[0]
        initial_target = np.linalg.inv(mount0) @ _body_pose(env,'robot','link6')[0]
        hold = absolute_pose_actions(initial_target[None], np.array([.028]))
        for i in range(120):
            env.step(torch.as_tensor(hold,dtype=torch.float32,device=env.device))
            settled.append(monitor.snapshot())
            if i==60: monitor.start()
            if i>60: monitor.observe(settled[-1])
        np.savez_compressed(args.output/'settling.npz', **{k:np.stack([v[k] for v in settled]) for k in settled[0]})
        result['standing_baseline'] = monitor.summary()
        result['standing_baseline']['sample_hz'] = 30
        if not result['standing_baseline']['accepted']:
            raise RuntimeError('standing baseline failed: '+json.dumps(result['standing_baseline']))
        # Flush rendering after visual scene edits without advancing physics.
        # Reject missing images before planner work or data collection.
        from arx_door.envs.recorders import _expected_camera_pose
        expected_camera=_expected_camera_pose(env)
        env.scene.sensors['wrist_camera'].set_world_poses(
            torch.as_tensor(expected_camera[:,:3,3],device=env.device,dtype=torch.float32),
            torch.as_tensor(Rotation.from_matrix(expected_camera[:,:3,:3]).as_quat(),device=env.device,dtype=torch.float32),
            convention='opengl')
        minimum_warmup=int(result.get('variation',{}).get('render_warmup_frames',4))
        for warmup in range(max(30,minimum_warmup)):
            env.sim.render()
            valid=True
            for camera in ('wrist','ego'):
                sensor=env.scene.sensors[camera+'_camera'];sensor.update(0.,force_recompute=True)
                if as_numpy(sensor.data.output['rgb'])[0,...,:3].std()<1.:valid=False
            if valid and warmup>=minimum_warmup-1:break
        if not valid:
            from PIL import Image
            diagnostic={}
            for mode in ('scene_visible','scene_hidden'):
                if mode=='scene_hidden':
                    background=stage.GetPrimAtPath('/World/AugmentationBackground')
                    if background:UsdGeom.Imageable(background).MakeInvisible()
                for _ in range(10):env.sim.render()
                diagnostic[mode]={}
                for name in ('wrist','ego','overview'):
                    sensor=env.scene.sensors[name+'_camera'];sensor.update(0.,force_recompute=True)
                    pixels=as_numpy(sensor.data.output['rgb'])[0,...,:3]
                    diagnostic[mode][name]={'mean':float(pixels.mean()),'std':float(pixels.std()),'max':int(pixels.max())}
                    Image.fromarray(pixels).save(args.output/f'diagnostic_{mode}_{name}.png')
            result['render_diagnostic']=diagnostic
            raise RuntimeError('camera_initialization: empty RGB after bounded render warmup')
        if args.visual_preview:
            from PIL import Image
            previews={}
            for name in ('wrist','ego','overview'):
                sensor=env.scene.sensors[name+'_camera'];sensor.update(0.,force_recompute=True)
                pixels=as_numpy(sensor.data.output['rgb'])[0,...,:3]
                Image.fromarray(pixels).save(args.output/f'{name}_preview.png')
                previews[name]={'mean':float(pixels.mean()),'std':float(pixels.std()),'max':int(pixels.max())}
            result.update(schema='b2arx-visual-preview-v1',visual_preview=previews,eligible_for_training=False)
            return 0
        monitor = Stability(env); monitor.start()
        result['measured_initial_root'] = monitor.reference['root_state'].tolist()
        substeps=[]
        update_scene=env.scene.update
        def update_and_record(*a,**kw):
            ret=update_scene(*a,**kw)
            sample=monitor.snapshot(); monitor.observe(sample);substeps.append(sample)
            return ret
        env.scene.update=update_and_record
        world_base = _body_pose(env,'robot','R5a')[0]
        to_base = np.linalg.inv(world_base)
        door = manifest['assets']['door']['initial_pose']
        door_base = to_base @ pose_matrix(door['position'], door['quaternion_xyzw'])
        kin = ArmKinematics(ROOT/'assets/b2arx_robot/my_robot.urdf',joint_prefix='R5a_')
        fk = kin.fk(as_numpy(arm_state(env))[0,:6])
        measured_fk = to_base @ _body_pose(env,'robot','link6')[0]
        result['kinematic_contract'] = dict(position_error_m=float(np.linalg.norm(fk[:3,3]-measured_fk[:3,3])),
            rotation_error_rad=float(Rotation.from_matrix(fk[:3,:3].T@measured_fk[:3,:3]).magnitude()),
            action_frame='R5a',quaternion_order='xyzw',control_hz=30,physics_hz=120,
            episode_start_after_settling_s=4.,torque_evidence='implicit actuator applied_torque estimate and enforced USD effort limits')
        if result['kinematic_contract']['position_error_m']>1e-4 or result['kinematic_contract']['rotation_error_rad']>1e-3:
            raise RuntimeError('full URDF FK and measured USD mount frame disagree')
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
        actual_root=monitor.reference['root_state']
        result['placement_metrics']=dict(body_position_m=actual_root[:3].tolist(),
            signed_distance_closed_plane_m=float(-.3082-actual_root[1]),
            mount_to_handle_distance_m=float(np.linalg.norm(world_base[:3,3]-knob)),
            yaw_deg=float(Rotation.from_quat(actual_root[3:7]).as_euler('xyz',degrees=True)[2]),
            standing_height_m=float(actual_root[2]),door_plane_y_m=-.3082,door_normal=[0,-1,0],
            opening_direction='clockwise world Z toward negative Y')
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
        replay_substep=0
        def apply_and_record(*a, **kw):
            nonlocal replay_substep
            ret = apply(*a, **kw)
            if replay_commands is not None:
                if replay_substep>=4:raise RuntimeError('too many replay physics substeps')
                robot.set_joint_position_target(torch.as_tensor(replay_commands[frames,replay_substep][None],device=env.device))
                replay_substep+=1
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
        planner_probe=None
        if args.planner_contract:
            from arx_door.autodata.b2arx_planner import PlannerContractProbe
            planner_probe=PlannerContractProbe(env,args.output/'planner_contract')
        last_command=None
        def step(target, grip, phase):
            nonlocal frames,last_command,replay_substep
            if planner_probe is not None and frames in (0,600,1000):planner_probe.record(frames)
            world_base = _body_pose(env,'robot','R5a')[0]
            to_base = np.linalg.inv(world_base)
            target_base = to_base @ target
            door_root = as_numpy(env.scene['door'].data.root_state_w)[0]
            door_base = to_base @ pose_matrix(door_root[:3],door_root[3:7])
            before = monitor.snapshot()
            for key,value in before.items(): append('obs/whole_body/'+key,value)
            substeps.clear()
            replay_substep=0
            action = absolute_pose_actions(target_base[None], np.array([grip]))[0]
            if args.numeric_replay:
                from arx_door.autodata.b2arx_replay import capture_numeric_pre_step
                obs=capture_numeric_pre_step(env)
            else:
                _, obs = capture_pre_step(env)
                ego_sensor=env.scene.sensors['ego_camera']
                append('obs/ego_camera_pose',_pose(ego_sensor.data.pos_w,ego_sensor.data.quat_w_opengl)[0])
            jacobian=env.action_manager.get_term('arm_action')._compute_frame_jacobian()
            append('obs/whole_body/arm_min_singular_value',np.asarray(np.linalg.svd(as_numpy(jacobian)[0],compute_uv=False)[-1]))
            measured = to_base @ _body_pose(env, 'robot', 'link6')[0]
            for key, value in obs.items():
                append('obs/'+key, as_numpy(value)[0])
            for camera in (() if args.numeric_replay else ('wrist', 'ego', 'overview')):
                sensor = env.scene.sensors[camera+'_camera']
                if camera == 'overview':
                    sensor.update(0., force_recompute=True)
                rgb = as_numpy(sensor.data.output['rgb'])[0, ..., :3]
                if camera in ('wrist','ego') and rgb.std()<1.:
                    raise RuntimeError(f'empty RGB: {camera} frame {frames}')
                if camera not in writers:
                    writers[camera] = VideoWriter(args.output/f'{camera}.mp4', rgb.shape[1], rgb.shape[0])
                writers[camera].append(rgb)
                if frames==0:
                    from PIL import Image
                    Image.fromarray(rgb).save(args.output/f'{camera}_start.png')
            env.step(torch.as_tensor(action[None], dtype=torch.float32, device=env.device))
            _, diag = capture_post_step(env)
            if len(substeps)!=4: raise RuntimeError(f'expected four full-body samples, got {len(substeps)}')
            for key in substeps[0]: append('obs/whole_body_substeps/'+key,np.stack([v[key] for v in substeps]))
            append('obs/whole_body/timestamp_s',np.asarray(frames/30.))
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
            last_command=target.copy()
            if monitor.failures: raise RuntimeError('full-body stability violation: '+json.dumps(monitor.summary()))
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
            if args.continuous_targets and last_command is not None and not joint_path:
                current=last_command.copy()
            world_base = _body_pose(env,'robot','R5a')[0]
            to_base = np.linalg.inv(world_base)
            if joint_path:
                q0 = as_numpy(arm_state(env))[0, :6]
                q1 = kin.ik(to_base @ target, q0)
                seconds = max(seconds, float(np.max(np.abs(q1-q0)))*2/.45)
            slerp = Slerp([0., 1.], Rotation.from_matrix(np.stack([current[:3,:3], target[:3,:3]])))
            for i in range(1, round(seconds*30)+1):
                t = i/round(seconds*30)
                blend = t*t*(3-2*t)
                if joint_path:
                    command = _body_pose(env,'robot','R5a')[0] @ kin.fk(q0 + blend*(q1-q0))
                else:
                    command = np.eye(4)
                    command[:3,:3] = slerp(blend).as_matrix()
                    command[:3,3] = (1-blend)*current[:3,3] + blend*target[:3,3]
                step(command, grip, phase)
            event(phase, begin, tick)
        if args.autodata_source is not None:
            from arx_door.autodata.b2arx_generation import generate
            result['generation'] = generate(env,args.autodata_source,args.source_demo,args.output,step,phase_names,args.seed)
            result['success']=bool(acceptance_state(env).accepted()[0]) and monitor.summary()['accepted']
            if not result['success']:raise RuntimeError('generated episode failed strict acceptance')
            return 0
        if replay_actions is not None:
            begin,tick,last_phase=0,time.monotonic(),int(replay_phases[0])
            for index,recorded_action in enumerate(replay_actions):
                phase=int(replay_phases[index])
                if phase!=last_phase:
                    event(phase_names[last_phase],begin,tick);begin,tick,last_phase=frames,time.monotonic(),phase
                target=_body_pose(env,'robot','R5a')[0]@pose_matrix(recorded_action[:3],recorded_action[3:7])
                step(target,float(recorded_action[7]),phase_names[phase])
            event(phase_names[last_phase],begin,tick)
            result['success']=bool(acceptance_state(env).accepted()[0]) and monitor.summary()['accepted']
            if not result['success']:raise RuntimeError('independent command replay failed strict acceptance')
            return 0
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
            move(unlocked, 'unlock_round_knob', 0., args.unlock_seconds if angle == 45 else .8)
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
            hinge = _body_pose(env, 'door', 'door_panel')[0, :3,3]
            # Preserve the commanded unlocked grasp about the measured hinge.
            # Rebasing from the loaded measured wrist at every arc step accumulates
            # compliance error and gradually releases the knob's torsional preload.
            target = unlocked.copy()
            turn = Rotation.from_euler('z', -(angle+delta), degrees=True).as_matrix()
            target[:3,3] = hinge + turn@(unlocked[:3,3]-hinge)
            target[:3,:3] = turn@unlocked[:3,:3]
            move(target, 'pull_knob_for_gap' if angle < 10 else 'pull_door_by_knob', 0., spec['arc_seconds'])
            now, forces = feedback()
            if min(forces) < 3 or abs(float(np.rad2deg(now[0]))-angle-delta) > 2:
                raise RuntimeError('grip or door-following gate failed')
        # Dwell at the last commanded target; no rebase of the contact preload.
        begin, tick = frames, time.monotonic()
        for _ in range(30):
            step(target, 0., 'hold_door_open')
        event('hold_door_open', begin, tick)
        result['success'] = bool(acceptance_state(env).accepted()[0]) and monitor.summary()['accepted']
        if not result['success']:
            raise RuntimeError('final strict acceptance failed')
    except Exception as error:
        result['error'] = str(error)
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    finally:
        result.update(frames=frames, simulated_seconds=frames/30,
                      acquisition_wall_seconds=time.monotonic()-START)
        if 'monitor' in locals(): result['whole_body_acceptance'] = monitor.summary()
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
            if 'update_scene' in locals():env.scene.update=update_scene
            if 'apply' in locals():env.action_manager.apply_action=apply
            if args.batch_plan is None:env.close()
    return 0 if result['success'] else 1


code = 1
try:
    if args.batch_plan is None:
        code = main()
    else:
        batch_root=args.output
        plan=json.loads(args.batch_plan.read_text())
        completed=[]
        for row in plan:
            if row.get('requires_success_of'):
                prerequisite=batch_root/row['requires_success_of']/'result.json'
                if not prerequisite.is_file() or not json.loads(prerequisite.read_text())['success']:
                    completed.append(dict(name=row['name'],skipped='source did not pass'))
                    continue
            args.output=batch_root/row['name'];args.output.mkdir(exist_ok=False)
            args.autodata_source=Path(row['autodata_source']) if row.get('autodata_source') else None
            args.source_demo=row.get('source_demo','demo_0')
            args.variation=Path(row['variation']) if row.get('variation') else None
            args.replay_pose=row.get('replay_pose',False)
            args.pose_codec=row.get('pose_codec',False)
            args.replay_joint_held=row.get('replay_joint_held',False)
            args.joint_sampling=row.get('joint_sampling','first')
            args.numeric_replay=row.get('numeric_replay',False)
            args.pose=row['pose'];args.seed=row['seed'];args.replication_of=row.get('replication_of')
            args.replay_source=batch_root/row['replay_source'] if row.get('replay_source') else None
            args.continuous_targets=row.get('continuous_targets',True)
            args.unlock_seconds=row.get('unlock_seconds',3.)
            args.planner_contract=row.get('planner_contract',False)
            print('B2ARX_BATCH_BEGIN '+json.dumps(row),flush=True)
            trial_code=main()
            completed.append(dict(name=row['name'],exit_code=trial_code))
            (batch_root/'batch_results.json').write_text(json.dumps(completed,indent=2))
            print('B2ARX_BATCH_END '+json.dumps(completed[-1]),flush=True)
            if trial_code and args.stop_on_failure:break
        code=0 if len(completed)==len(plan) and all(v.get('exit_code')==0 for v in completed) else 1
finally:
    if SHARED_ENV is not None:SHARED_ENV.close()
    close_app(app, code)
