"""Persistent 24-env SkillGen collection with frozen-state camera multiplexing.

Only the environment/capture scheduler is project code. Upstream DataGenerator,
SkillGen, source transforms and cuRobo planning are used without vendor edits.
"""
import asyncio
import json
import time
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation

from .b2arx_generation import FixedEpisodeEmbodiment, FloatingPlanner
from .b2arx_planner import LiveMountCollisionWorld, robot_config_file, configure_warp_interop
from .b2arx_shared_scene import reset_source_poses, randomize_panel_appearance
from .source import absolute_pose_actions, pose_matrix
from arx_door.envs.recorders import (_body_pose, _expected_camera_pose, _camera_clearance,
    read_physical_state, capture_joint_target_substep, capture_post_step, reset_recording)
from arx_door.envs.acceptance import acceptance_state, as_numpy
from arx_door.envs.b2arx import arm_state
from arx_door.envs.b2arx_stability import Stability
from arx_door.policy.simulation import VideoWriter

PHASES=['transit_pregrasp','approach_knob','grasp_knob','unlock_round_knob',
        'pull_knob_for_gap','pull_door_by_knob','hold_door_open']
TEXTS=['Move the open gripper in front of the door knob.',
       'Approach the door knob with the open gripper.','Grasp the door knob.',
       'Rotate the knob until the latch is fully retracted.',
       'Pull the unlocked door while holding the knob.',
       'Continue opening the door while holding the knob.','Hold the door open.']
ROOT=Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n')


class PlannerSlot:
    """Serialize cuRobo workspace use; retain an independent plan for each env."""
    def __init__(self, owner, stream, env_id):
        self.owner,self.stream,self.env_id=owner,stream,env_id
        self.config=owner.config;self.step_size=owner.step_size
        self.poses=[]

    def update_world_and_plan_motion(self, **kwargs):
        engine=self.owner
        engine.env_id=self.env_id;engine.datastream=self.stream
        engine.live_world=LiveMountCollisionWorld(engine.env,engine.usd_helper,self.env_id)
        engine.motion_gen.update_world(engine.live_world.world)
        ok=engine.update_world_and_plan_motion(**kwargs)
        self.poses=[p.clone() for p in engine.get_planned_poses()] if ok else []
        return ok

    def get_planned_poses(self):return self.poses


class CameraBank:
    """Four pairs of real RTX cameras; physics is frozen across the six groups.

    Every env keeps its own camera extrinsics. Reusing render products limits
    memory, not env count. No step() occurs between cameras for a control row.
    """
    def __init__(self,env,stage,manifest,pairs=4):
        from pxr import UsdGeom,Gf
        import omni.replicator.core as rep
        import carb
        self.env,self.stage,self.pairs=env,stage,pairs
        self.cameras=[];self.products=[];self.annotators=[];self.frame=0
        # A multiplexed camera must not blend a previous env's temporal history.
        carb.settings.get_settings().set('/rtx/post/aa/op',0)
        _,cfg,_=__import__('arx_door.envs.b2arx',fromlist=['build']).build(manifest)
        self.ego_local=pose_matrix(cfg.scene.ego_camera.offset.pos,cfg.scene.ego_camera.offset.rot)
        # Lab's world convention is +X forward,+Z up; USD is -Z forward,+Y up.
        self.ego_local[:3,:3] = self.ego_local[:3,:3] @ np.array([[0,0,-1],[-1,0,0],[0,1,0]])
        for slot in range(pairs):
            for name,cfgcam in [('wrist',cfg.scene.wrist_camera),('ego',cfg.scene.ego_camera)]:
                cam=UsdGeom.Camera.Define(stage,f'/World/TrainingCapture/{name}_{slot}')
                spawn=cfgcam.spawn
                cam.CreateFocalLengthAttr(spawn.focal_length)
                cam.CreateHorizontalApertureAttr(spawn.horizontal_aperture)
                cam.CreateVerticalApertureAttr(spawn.vertical_aperture or spawn.horizontal_aperture*480/640)
                cam.CreateClippingRangeAttr(Gf.Vec2f(spawn.clipping_range[0],max(100.,spawn.clipping_range[1])))
                cam.CreateHorizontalApertureOffsetAttr(spawn.horizontal_aperture_offset or 0.)
                cam.CreateVerticalApertureOffsetAttr(spawn.vertical_aperture_offset or 0.)
                cam.AddTransformOp()
                product=rep.create.render_product(str(cam.GetPath()),(640,480))
                annot=rep.AnnotatorRegistry.get_annotator('rgb');annot.attach([product])
                self.cameras.append(cam);self.products.append(product);self.annotators.append(annot)
        self.intrinsics={name:dict(focal_length=c.spawn.focal_length,horizontal_aperture=c.spawn.horizontal_aperture,
                                  vertical_aperture=c.spawn.vertical_aperture or c.spawn.horizontal_aperture*.75,
                                  resolution=[640,480]) for name,c in [('wrist',cfg.scene.wrist_camera),('ego',cfg.scene.ego_camera)]}

    def capture(self,ids):
        from pxr import Gf,UsdGeom
        wrist=_expected_camera_pose(self.env)
        ego=_body_pose(self.env,'robot','b2_description')@self.ego_local
        images={};self.frame+=1
        # Check that frozen render calls never advance the physical state.
        before=as_numpy(self.env.scene['robot'].data.root_state_w).copy()
        for begin in range(0,len(ids),self.pairs):
            group=ids[begin:begin+self.pairs]
            for slot,i in enumerate(group):
                for view,poses in enumerate((wrist,ego)):
                    UsdGeom.Xformable(self.cameras[slot*2+view]).GetOrderedXformOps()[0].Set(Gf.Matrix4d(poses[i].T.tolist()))
            # Two render updates flush the pose and render-product readback.
            self.env.sim.render();self.env.sim.render()
            for slot,i in enumerate(group):
                images[i]={}
                for view,name in enumerate(('wrist','ego')):
                    rgb=self.annotators[slot*2+view].get_data()[...,:3].copy()
                    if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or rgb.std()<1:
                        raise RuntimeError(f'invalid training image env={i} view={name}')
                    images[i][name]=rgb
        if not np.array_equal(before,as_numpy(self.env.scene['robot'].data.root_state_w)):
            raise RuntimeError('physics advanced during multiplexed capture')
        return images,wrist,ego


class EpisodeWriter:
    def __init__(self,path,meta,env,index):
        self.path=path;path.mkdir(parents=True,exist_ok=False)
        self.meta=meta;self.n=0;self.labels=[];self.videos={}
        self.file=h5py.File(path/'generated.hdf5','x')
        data=self.file.create_group('data');data.attrs['env_args']=json.dumps(dict(env_name='B2Arx-Wurenyuan-Shared',type=2,env_kwargs={}))
        self.ep=data.create_group('demo_0');self.ep.attrs['phase_names']=json.dumps(PHASES)
        self.ep.attrs['behavior_family']='knob_hold_open'
        def copy_state(group,values):
            for k,v in values.items():
                if isinstance(v,dict):copy_state(group.create_group(k),v)
                else:group.create_dataset(k,data=as_numpy(v)[index:index+1])
        copy_state(self.ep.create_group('initial_state'),env.scene.get_state(is_relative=True))
        write_json(path/'result.json',dict(**meta,success=False,state='collecting'))

    def add(self,path,value):
        value=np.asarray(value)
        if path not in self.ep:self.ep.create_dataset(path,shape=(0,*value.shape),maxshape=(None,*value.shape),dtype=value.dtype,chunks=(1,*value.shape),compression='lzf')
        ds=self.ep[path];ds.resize(self.n+1,axis=0);ds[self.n]=value

    def image(self,name,rgb):
        self.add('obs/'+name+'_rgb',rgb)
        if name not in self.videos:self.videos[name]=VideoWriter(self.path/f'{name}.mp4',640,480)
        self.videos[name].append(rgb)
        if self.n==0:
            from PIL import Image
            Image.fromarray(rgb).save(self.path/f'{name}_start.png')

    def close(self,result):
        self.ep.attrs['num_samples']=self.n;self.file['data'].attrs['total']=self.n
        self.file.close()
        for v in self.videos.values():v.close()
        write_json(self.path/'language.json',dict(instructions=TEXTS,phase_ids=self.labels))
        write_json(self.path/'result.json',dict(**self.meta,**result,frames=self.n))


def create_jobs(env,source,rows,out,seed,owner=None):
    import yaml
    from autodata_interfaces.datastream import Datastream
    from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
    from autodata_core.algorithms import SkillGen
    from autodata_core.data_generator import DataGenerator
    from isaaclab.managers import TerminationTermCfg
    from curobo.util import usd_helper
    configure_warp_interop();usd_helper._WARP_AVAILABLE=False
    spec=yaml.safe_load((ROOT/'configs/autodata/embodiments/b2arx.yaml').read_text());spec.pop('type')
    slots=[];loop=asyncio.get_event_loop()
    for i,row in enumerate(rows):
        emb=FixedEpisodeEmbodiment(**spec);emb.latch(env)
        task=TaskDescriptor.from_yaml(str(ROOT/'configs/autodata/tasks/arx_door_knob_hold_open.yaml'))
        task.get_generation_policy().seed=seed+i
        demo='demo_'+str(row['source_index'])
        stream=Datastream(env=env,task_descriptor=task,embodiment_adapter=emb,source_dataset_path=str(source),select_demo_keys=[demo],uses_start_signals=True)
        if owner is None:
            cfg=CuroboPlannerCfg(robot_config_file=str(robot_config_file(out/'planner')),robot_name='b2arx',ee_link_name='R5a_link6',
                gripper_joint_names=['R5a_joint7','R5a_joint8'],gripper_open_positions={'R5a_joint7':.044,'R5a_joint8':.044},
                gripper_closed_positions={'R5a_joint7':0.,'R5a_joint8':0.},hand_link_names=['R5a_link6','R5a_link7','R5a_link8'],
                grasp_gripper_open_val=.044,approach_distance=0.,retreat_distance=0.,collision_activation_distance=.01,
                max_planning_attempts=3,enable_finetune_trajopt=True,motion_noise_scale=0.,motion_step_size=.006,
                time_dilation_factor=.2,visualize_spheres=False,visualize_plan=False,debug_planner=False)
            owner=FloatingPlanner(datastream=stream,config=cfg,env_id=i)
        algorithm=SkillGen(motion_planners={i:PlannerSlot(owner,stream,i)})
        generator=DataGenerator(datastream=stream,algorithm=algorithm)
        with h5py.File(source) as f:
            ep=f['data/'+demo];start=int(stream.source_pool.subtask_boundaries['link6'][0][0][0])
            obj=ep['obs/datagen_info/object_pose/door'][start]
            delta=stream.get_object_poses([i])['door'][0].cpu().numpy()@np.linalg.inv(obj)
            if np.linalg.norm(delta[:3,3])>.15:
                raise ValueError(f'fixed-source/env-origin frame contract failed env={i}: {delta[:3,3]}')
            expected=delta@np.concatenate([ep['obs/datagen_info/eef_pose/link6'][start:start+1],ep['obs/datagen_info/target_eef_pose/link6'][start:]])
            raw=Path(str(ep.attrs['raw_dataset']))
            if not raw.exists():raw=Path('/outputs')/str(raw).split('generated/door_autodata/',1)[1]
            with h5py.File(raw) as rf:ph=rf['data/demo_0/obs/diagnostics/phase_id'][:]
            phase_ids=np.concatenate([ph[start:start+1],ph[start:]])
        state=dict(motion=True,skill_index=0,matched=0,expected=expected,phase_ids=phase_ids,fixed=emb.fixed[i].cpu().numpy(),delta=delta,source_start=start)
        original=algorithm.plan_subtask_trajectory
        def track(_original=original,_state=state,**kwargs):
            result=_original(**kwargs)
            if result is not None:_state['motion']=result[1]
            return result
        algorithm.plan_subtask_trajectory=track
        resets=asyncio.Queue();actions=asyncio.Queue()
        term=TerminationTermCfg(func=lambda e:torch.as_tensor(acceptance_state(e).accepted(),device=e.device))
        job=loop.create_task(generator.generate(i,term,env_reset_queue=resets,env_action_queue=actions,export_demo=False))
        slots.append(dict(job=job,resets=resets,actions=actions,state=state,generator=generator))
    return slots,owner


def run_campaign(env,stage,config,out,sources,layout,manifest,overview,status):
    import carb,warp as wp
    from PIL import Image
    from pxr import UsdGeom,Gf
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.envs.mdp import randomize_rigid_body_material
    from .shared_preview import apply
    n=env.num_envs
    if n!=24 or len(config['rounds'])!=3:raise ValueError('campaign requires exactly 3 x 24 attempts')
    write_json(out/'campaign.json',config)
    import shutil
    snapshot=out/'implementation';snapshot.mkdir(exist_ok=True)
    for name in ('shared_generation.py','b2arx_generation.py','b2arx_planner.py','b2arx_shared_scene.py','shared_preview.py'):
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    camera_bank=CameraBank(env,stage,manifest,pairs=config.get('camera_pairs',4))
    settings=carb.settings.get_settings();base_iso=settings.get('/rtx/post/tonemap/filmIso') or 100.
    owner=None;all_results=[];loop=asyncio.get_event_loop()
    source=Path(config['source_pool'])
    base_control=json.loads(Path(config['render_control']).read_text())
    scene_update=env.scene.update;action_apply=env.action_manager.apply_action
    for round_index,lighting in enumerate(config['rounds']):
        seed=73020+round_index;env.seed(seed)
        rd=out/f'round_{round_index+1:02d}_{lighting["name"]}';rd.mkdir(exist_ok=False)
        rng=np.random.default_rng(seed);rows=[]
        for i in range(n):
            si=(i+round_index*3)%10
            pose=np.array(sources[si]['meta']['pose'],dtype=float)
            jitter=np.array([rng.uniform(-.006,.006),rng.uniform(-.006,.006),0,rng.uniform(-.006,.006)])
            rows.append(dict(env_id=i,source_index=si,source_id=sources[si]['id'].split('/')[-1],pose=(pose+jitter).tolist(),pose_jitter=jitter.tolist()))
        env.reset();reset_source_poses(env,None,[r['pose'] for r in rows])
        randomize_panel_appearance(env,None,seed)
        foot_event=env.event_manager.get_term_cfg('foot_material')
        foot_event.func(env,None,**foot_event.params)
        env.sim.forward();env.scene.update(env.physics_dt)
        control=dict(base_control);control['settings']=dict(base_control['settings'])
        control['settings']['/rtx/post/aa/op']=0
        control['settings']['/rtx/post/tonemap/filmIso']=float(base_iso*lighting['exposure_ratio'])
        control['light_intensity']={p:float(v*lighting['light_ratio']) for p,v in base_control['light_intensity'].items()}
        render_report=apply(stage,control)
        cam=UsdGeom.Camera.Get(stage,'/World/ArenaOverview')
        eye=np.array(control['eye']);target=np.array(control['target']);z=(eye-target)/np.linalg.norm(eye-target);x=np.cross([0,0,1],z);x=x/np.linalg.norm(x);y=np.cross(z,x)
        matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=eye
        UsdGeom.Xformable(cam).GetOrderedXformOps()[0].Set(Gf.Matrix4d(matrix.T.tolist()));cam.GetFocalLengthAttr().Set(control['focal_length'])
        mats=wp.to_torch(env.scene['robot'].root_view.get_material_properties()).cpu().numpy()
        variation=dict(round_index=round_index,seed=seed,lighting=lighting,render_control=control,render_settings=render_report,
            appearance=env._shared_scene_appearance,rows=rows,actual_physics_materials=mats.tolist(),intrinsics=camera_bank.intrinsics,
            gaussian_lighting='captured radiance retained; exposure variation is separate from inserted-object illumination')
        write_json(rd/'variation.json',variation)
        hold=absolute_pose_actions(np.linalg.inv(_body_pose(env,'robot','R5a'))@_body_pose(env,'robot','link6'),np.full(n,.028))
        monitors=[Stability(env,i) for i in range(n)]
        status(dict(state='generation_settling',round=round_index+1,attempts=24))
        for frame in range(120):
            env.step(torch.tensor(hold,dtype=torch.float32))
            if frame==60:
                for m in monitors:m.start()
            if frame>60:
                for m in monitors:m.observe(m.snapshot())
        standing=[m.summary() for m in monitors]
        if not all(m['accepted'] for m in standing):raise RuntimeError('campaign standing failed; no trials started')
        for _ in range(60):env.sim.render()
        Image.fromarray(overview.get_data()[...,:3].copy()).save(rd/'overview_start.png')
        reset_recording(env)
        monitors=[Stability(env,i) for i in range(n)]
        for m in monitors:m.start()
        status(dict(state='planning',round=round_index+1,attempts=24))
        slots,owner=create_jobs(env,source,rows,rd,seed,owner)
        episodes=[];results=[None]*n;substeps=[];recording=False;min_ego_clearance=np.full(n,np.inf)
        for i,row in enumerate(rows):
            meta=dict(schema='b2arx-shared-autodata-v1',seed=seed,pose=row['pose'],source_id=row['source_id'],
                source_demonstrations=[str(source)+'::demo_'+str(row['source_index'])],
                provenance='official AutoData SkillGen physical generation',joint_names=list(env.scene['robot'].joint_names),
                variation=dict(**row,lighting=lighting,appearance=variation['appearance'][i],round_seed=seed),
                standing_baseline=standing[i],training_data_generation=True,
                contact_sensor_bodies={name:list(env.scene.sensors[name].body_names) for name in ('feet_contact','body_contact','leg_contact','arm_contacts')})
            episodes.append(EpisodeWriter(rd/f'env_{i:02d}',meta,env,i))
        def update(*args,**kwargs):
            ret=scene_update(*args,**kwargs)
            if recording:
                sample=[m.snapshot() for m in monitors];substeps.append(sample)
                for i in np.flatnonzero(env.active_episode_mask):monitors[i].observe(sample[i])
            return ret
        def apply_action(*args,**kwargs):
            ret=action_apply(*args,**kwargs)
            if recording:capture_joint_target_substep(env)
            return ret
        env.scene.update=update;env.action_manager.apply_action=apply_action
        commands=hold.copy();tick=time.monotonic();step_index=0;video=VideoWriter(rd/'overview.mp4',1920,1080)
        try:
            while any(not s['job'].done() for s in slots):
                pending={}
                # Pump generators until each live env has exactly one action.
                for pump in range(1000):
                    loop.run_until_complete(asyncio.sleep(0))
                    for i,s in enumerate(slots):
                        if not s['resets'].empty():s['resets'].get_nowait();s['resets'].task_done()
                        if i not in pending and not s['actions'].empty():
                            _,a=s['actions'].get_nowait();pending[i]=a.cpu().numpy()
                    if all(s['job'].done() or i in pending for i,s in enumerate(slots)):break
                else:raise RuntimeError('generator scheduler failed to reach batch barrier')
                if not pending:break
                ids=sorted(pending);env.active_episode_mask.copy_(torch.tensor([i in pending for i in range(n)],device=env.device))
                phases={}
                for i,a in pending.items():
                    st=slots[i]['state'];commands[i]=a
                    if st['motion']:phases[i]=0
                    else:
                        k=st['skill_index'];world=_body_pose(env,'robot','R5a')[i]@pose_matrix(a[:3],a[3:7]);fixed=np.linalg.inv(st['fixed'])@world
                        if k>=len(st['expected']) or not np.allclose(fixed,st['expected'][k],atol=2e-5):raise RuntimeError(f'skill coordinate mismatch env={i} index={k}')
                        phases[i]=int(st['phase_ids'][k]);st['skill_index']+=1;st['matched']+=1
                before=[m.snapshot() for m in monitors]
                images,wrist,ego=camera_bank.capture(ids)
                frame=acceptance_state(env).frames.copy();physical=read_physical_state(env)
                ego_clearance=_camera_clearance(ego,physical['panel_pose']).reshape(-1)
                min_ego_clearance[ids]=np.minimum(min_ego_clearance[ids],ego_clearance[ids])
                acceptance_state(env).observe_safety(ids,physical['arm_contacts'][ids],physical['camera_clearance'][ids])
                env._arx_pre_action_camera=dict(wrist_camera_pose=wrist,wrist_camera_expected_pose=wrist,timestamp=frame[:,None]/30.,action_index=frame[:,None],camera_frame_id=np.full((n,1),camera_bank.frame))
                env._arx_joint_target_substeps=[]
                mount=_body_pose(env,'robot','R5a');eef=np.linalg.inv(mount)@_body_pose(env,'robot','link6')
                door=as_numpy(env.scene['door'].data.root_state_w);objects=np.linalg.inv(mount)@np.stack([pose_matrix(x[:3],x[3:7]) for x in door])
                joints=as_numpy(arm_state(env));jac=as_numpy(env.action_manager.get_term('arm_action')._compute_frame_jacobian())
                for i in ids:
                    w=episodes[i]
                    for key,value in before[i].items():w.add('obs/whole_body/'+key,value)
                    w.add('obs/whole_body/arm_min_singular_value',np.linalg.svd(jac[i],compute_uv=False)[-1])
                    w.add('obs/whole_body/timestamp_s',np.asarray(w.n/30.));w.add('obs/ego_camera_pose',ego[i]);w.add('obs/joint_pos',joints[i])
                    w.add('obs/ego_camera_clearance',ego_clearance[i])
                    for name in ('wrist','ego'):w.image(name,images[i][name])
                    w.add('actions',commands[i]);w.add('obs/datagen_info/eef_pose/link6',eef[i].astype('float32'))
                    w.add('obs/datagen_info/target_eef_pose/link6',pose_matrix(commands[i,:3],commands[i,3:7]).astype('float32'))
                    w.add('obs/datagen_info/object_pose/door',objects[i].astype('float32'))
                    w.add('obs/datagen_info/subtask_start_signals/knob_interaction',np.array([phases[i]!=0],dtype=np.int64))
                    w.add('obs/datagen_info/subtask_term_signals/knob_interaction',np.array([phases[i]==6],dtype=np.int64))
                video.append(overview.get_data()[...,:3].copy())
                substeps.clear();recording=True;env.step(torch.tensor(commands,dtype=torch.float32));recording=False
                _,diag=capture_post_step(env)
                if len(substeps)!=4:raise RuntimeError('missing 120 Hz physics evidence')
                for i in ids:
                    w=episodes[i]
                    for key in substeps[0][i]:w.add('obs/whole_body_substeps/'+key,np.stack([s[i][key] for s in substeps]))
                    for key,value in diag.items():w.add('obs/diagnostics/'+key,as_numpy(value)[i])
                    w.add('obs/diagnostics/phase_id',np.array(phases[i],dtype=np.int64));w.labels.append(phases[i]);w.n+=1
                    slots[i]['actions'].task_done()
                step_index+=1
                if step_index%30==0:
                    for w in episodes:w.file.flush()
                    status(dict(state='generating',round=round_index+1,frames=step_index,active_envs=len(ids),wall_seconds=time.monotonic()-tick))
                if step_index>2500:raise RuntimeError('generation exceeded bounded 2500 control frames')
            loop.run_until_complete(asyncio.sleep(0))
            for i,s in enumerate(slots):
                error=None
                try:official=bool(s['job'].result().success)
                except Exception as exc:official=False;error=repr(exc)
                physical=acceptance_state(env).summary(i);whole=monitors[i].summary();st=s['state'];w=episodes[i]
                complete=st['matched']==len(st['expected']);success=official and complete and physical['accepted'] and whole['accepted'] and min_ego_clearance[i]>=.01
                result=dict(success=bool(success),physical_metrics=physical,whole_body_acceptance=whole,error=error,
                    camera_evidence=dict(capture_protocol_verified=w.n>0,frames_verified=w.n,alignment='pre-action RGB/state; physics frozen across all multiplexed views',render_updates_per_group=2,view_env_mapping='explicit world pose from each live body'),
                    generation=dict(algorithm='official AutoData SkillGen',source=str(source),demo='demo_'+str(rows[i]['source_index']),
                        generated_frames=w.n,matched_skill_frames=st['matched'],expected_skill_frames=len(st['expected']),
                        source_start=st['source_start'],skill_transform=st['delta'].tolist(),official_success=official,
                        planner='serialized shared cuRobo workspace with per-env live collision world; 24 batched physical actions'),
                    camera_intrinsics=camera_bank.intrinsics,min_ego_camera_clearance_m=float(min_ego_clearance[i]))
                w.close(result);results[i]=dict(run=str(w.path),env_id=i,source_id=rows[i]['source_id'],success=bool(success),frames=w.n)
            all_results.extend(results)
            write_json(rd/'result.json',dict(attempts=n,passed=sum(r['success'] for r in results),episodes=results,wall_seconds=time.monotonic()-tick))
            write_json(out/'attempts.json',all_results)
        finally:
            recording=False;env.scene.update=scene_update;env.action_manager.apply_action=action_apply
            for s in slots:
                if not s['job'].done():s['job'].cancel()
            loop.run_until_complete(asyncio.sleep(0))
            for w in episodes:
                if w.file.id.valid:
                    w.file.close()
                    for v in w.videos.values():v.close()
            video.close()
        replay_round(env,rd,rows,results,status)
    status(dict(state='generation_complete',attempts=len(all_results),passed=sum(r['success'] for r in all_results),source_count=10))


def replay_round(env,rd,rows,results,status):
    """Fresh reset, decode the exported pose representation and re-execute it."""
    from .b2arx_pose_codec import encode,decode
    selected=[r['env_id'] for r in results if r['success']]
    if not selected:
        write_json(rd/'replay_report.json',dict(episodes=[],passed=0));return
    env.active_episode_mask.zero_()
    env.reset();reset_source_poses(env,None,[r['pose'] for r in rows])
    env.sim.forward();env.scene.update(env.physics_dt)
    commands=absolute_pose_actions(np.linalg.inv(_body_pose(env,'robot','R5a'))@_body_pose(env,'robot','link6'),np.full(env.num_envs,.028))
    for _ in range(120):env.step(torch.tensor(commands,dtype=torch.float32))
    reset_recording(env)
    monitors=[Stability(env,i) for i in range(env.num_envs)]
    for m in monitors:m.start()
    writers={};actions={};labels={};sizes={}
    for i in selected:
        run=rd/f'env_{i:02d}';original=json.loads((run/'result.json').read_text())
        with h5py.File(run/'generated.hdf5') as f:
            actions[i]=decode(encode(f['data/demo_0/actions'][:]));labels[i]=f['data/demo_0/obs/diagnostics/phase_id'][:]
        sizes[i]=len(actions[i])
        writers[i]=EpisodeWriter(rd/f'replay_{i:02d}',dict(schema='b2arx-shared-pose-replay-v1',pose=rows[i]['pose'],
            seed=original['seed'],source_demonstrations=[str(run)],joint_names=original['joint_names'],
            provenance='fresh-reset independent live-mount pose-action replay',pose_codec='XYZ_ROTVEC float32 -> quaternion_xyzw float32',
            replay_capture='numeric physics only; no duplicate training RGB'),env,i)
    scene_update=env.scene.update;action_apply=env.action_manager.apply_action;substeps=[]
    def update(*args,**kwargs):
        ret=scene_update(*args,**kwargs);sample=[m.snapshot() for m in monitors];substeps.append(sample)
        for i in np.flatnonzero(env.active_episode_mask):monitors[i].observe(sample[i])
        return ret
    def apply_action(*args,**kwargs):
        ret=action_apply(*args,**kwargs);capture_joint_target_substep(env);return ret
    env.scene.update=update;env.action_manager.apply_action=apply_action
    report=[]
    try:
        for frame in range(max(sizes.values())):
            ids=[i for i in selected if frame<sizes[i]]
            env.active_episode_mask.copy_(torch.tensor([i in ids for i in range(env.num_envs)],device=env.device))
            for i in ids:commands[i]=actions[i][frame]
            physical=read_physical_state(env);state=acceptance_state(env)
            state.observe_safety(ids,physical['arm_contacts'][ids],physical['camera_clearance'][ids])
            env._arx_pre_action_camera=dict(wrist_camera_expected_pose=_expected_camera_pose(env),timestamp=state.frames[:,None]/30.,action_index=state.frames[:,None])
            env._arx_joint_target_substeps=[]
            joints=as_numpy(arm_state(env));eef=np.linalg.inv(_body_pose(env,'robot','R5a'))@_body_pose(env,'robot','link6')
            for i in ids:
                w=writers[i];w.add('actions',commands[i]);w.add('obs/joint_pos',joints[i]);w.add('obs/datagen_info/eef_pose/link6',eef[i])
                for key,value in monitors[i].snapshot().items():w.add('obs/whole_body/'+key,value)
            substeps.clear();env.step(torch.tensor(commands,dtype=torch.float32));_,diag=capture_post_step(env)
            for i in ids:
                w=writers[i]
                for key,value in diag.items():w.add('obs/diagnostics/'+key,as_numpy(value)[i])
                for key in substeps[0][i]:w.add('obs/whole_body_substeps/'+key,np.stack([s[i][key] for s in substeps]))
                w.add('obs/diagnostics/phase_id',labels[i][frame]);w.labels.append(int(labels[i][frame]));w.n+=1
            if frame%60==0:status(dict(state='independent_pose_replay',round=rd.name,frame=frame,active_envs=len(ids)))
        for i,w in writers.items():
            physical=acceptance_state(env).summary(i);whole=monitors[i].summary()
            success=physical['accepted'] and whole['accepted']
            w.close(dict(success=bool(success),physical_metrics=physical,whole_body_acceptance=whole))
            report.append(dict(env_id=i,run=str(w.path),success=bool(success)))
    finally:
        env.scene.update=scene_update;env.action_manager.apply_action=action_apply
        for w in writers.values():
            if w.file.id.valid:w.file.close()
    write_json(rd/'replay_report.json',dict(episodes=report,passed=sum(r['success'] for r in report)))
