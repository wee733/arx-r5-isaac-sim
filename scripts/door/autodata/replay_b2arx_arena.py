#!/usr/bin/env python3
"""Replay ten accepted sources concurrently under Arena physics; no pose playback."""
import argparse,json,math,time,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--table',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
for flag in ('--headless','--enable_cameras'):
 if flag not in p._option_string_actions:p.add_argument(flag,action='store_true')
a=p.parse_args();a.device='cpu';a.enable_cameras=True
a.kit_args=(getattr(a,'kit_args','') or '')+' --/rtx/scenePartitioning/showAllPartitionsByDefault=true'
out=a.output;out.mkdir(parents=True,exist_ok=True)
def status(value):
 q=out/'status.tmp';q.write_text(json.dumps(value));q.replace(out/'status.json')
status(dict(state='starting',message='Starting pinned Isaac Sim / Arena'))
app=AppLauncher(a).app
import gymnasium as gym
import h5py,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from arx_door.envs.b2arx import build
from arx_door.envs.b2arx_stability import Stability
from arx_door.envs.assets import load_asset_manifest
from arx_door.envs.recorders import _body_pose,read_physical_state
from arx_door.envs.acceptance import as_numpy,RuntimeAcceptance
from arx_door.autodata.source import absolute_pose_actions
from arx_door.policy.simulation import VideoWriter
from arx_door.envs.physics import physics_material_bindings,physics_runtime_evidence,validate_physics_runtime
import omni.usd

def local(path):
 s=str(path)
 if 'generated/door_autodata/' in s:return Path('/outputs')/s.split('generated/door_autodata/',1)[1]
 return Path(path)
rows=json.loads(a.table.read_text());assert len(rows)==10
sources=[]
for row in rows:
 path=local(row['source']);meta=json.loads((path/'result.json').read_text());assert meta['success']
 with h5py.File(path/'generated.hdf5') as f:
  ep=f['data/demo_0'];sources.append(dict(id=row['id'],path=str(path),meta=meta,actions=ep['actions'][:],
    commands=ep['obs/whole_body_substeps/joint_target'][:],phases=ep['obs/diagnostics/phase_id'][:]))
lengths=np.array([len(s['actions']) for s in sources]);total=int(lengths.max());n=len(sources)
manifest=load_asset_manifest(a.assets)
env_id,cfg,kwargs=build(manifest,pose=sources[0]['meta']['pose'],num_envs=n,name='B2Arx-Ten-Arena-Replay-v0')
cfg.seed=73019;cfg.scene.wrist_camera=None;cfg.scene.ego_camera=None
cfg.observations.camera_obs=None
cfg.scene.overview_camera.width=480;cfg.scene.overview_camera.height=360
cfg.terminations.success=None;cfg.recorders=None
from isaaclab.utils.dict import class_to_dict
(out/'environment_config.json').write_text(json.dumps(class_to_dict(cfg),default=str,indent=2))
shutil.copy2(__file__,out/'replay_source.py')
env=None;writers={};file=None
try:
 env=gym.make(env_id,cfg=cfg,**kwargs).unwrapped
 robot=env.scene['robot'];assert not robot.is_fixed_base
 assert all(s['meta']['joint_names']==list(robot.joint_names) for s in sources)
 env.cfg.sim.render_interval=1000000
 env.reset()
 origins=as_numpy(env.scene.env_origins).copy()
 poses=np.array([[*s['meta']['pose'][:3],0,0,math.sin(s['meta']['pose'][3]/2),math.cos(s['meta']['pose'][3]/2)] for s in sources])
 poses[:,:3]+=origins
 robot.write_root_pose_to_sim(torch.tensor(poses,dtype=torch.float32,device=env.device))
 robot.write_root_velocity_to_sim(torch.zeros((n,6),device=env.device))
 env.sim.forward();env.scene.update(env.physics_dt)
 stage=omni.usd.get_context().get_stage()
 # Dome/distant lights are global: cloned copies otherwise sum ten times.
 lighting=[]
 for prim in stage.Traverse():
  path=str(prim.GetPath())
  if prim.GetTypeName() in ('DomeLight','DistantLight') and path.startswith('/World/envs/env_'):
   env_number=int(path.split('/')[3].split('_')[-1]);attr=prim.GetAttribute('inputs:intensity')
   lighting.append(dict(path=path,intensity_before=attr.Get(),retained=env_number==0))
   if env_number!=0:attr.Set(0.)
 (out/'lighting.json').write_text(json.dumps(lighting,indent=2))
 evidence=physics_runtime_evidence(stage,env.cfg.sim,str(env.sim.device),'source_cpu',source_settings=manifest['source_settings'],original_bindings=physics_material_bindings(stage))
 validate_physics_runtime(evidence)
 center=origins.mean(axis=0)+np.array([0,-.5,.8])
 detail=env.scene.sensors['overview_camera']
 eye=center+np.array([11.,-15.,13.])
 from pxr import UsdGeom,Gf
 import omni.replicator.core as rep
 camera=UsdGeom.Camera.Define(stage,'/World/ArenaOverview')
 camera.CreateFocalLengthAttr(18.);camera.CreateHorizontalApertureAttr(20.955)
 camera.CreateClippingRangeAttr(Gf.Vec2f(.1,100.))
 z=(eye-center)/np.linalg.norm(eye-center);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
 matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=eye
 UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
 product=rep.create.render_product('/World/ArenaOverview',(1920,1080))
 arena=rep.AnnotatorRegistry.get_annotator('rgb');arena.attach(product)
 # Each close view uses its own translated copy of the same physical scene.
 eyes=origins+np.array([2.6,-3.6,2.1]);targets=origins+np.array([0,-.8,.65])
 detail.set_world_poses_from_view(torch.tensor(eyes,dtype=torch.float32),torch.tensor(targets,dtype=torch.float32))
 for _ in range(4):env.sim.render()
 mount=_body_pose(env,'robot','R5a');eef=_body_pose(env,'robot','link6')
 hold=absolute_pose_actions(np.linalg.inv(mount)@eef,np.full(n,.028))
 monitors=[Stability(env,i) for i in range(n)]
 status(dict(state='settling',message='Ten floating robots standing under gravity'))
 for frame in range(120):
  env.step(torch.tensor(hold,dtype=torch.float32))
  if frame==60:
   for m in monitors:m.start()
  if frame>60:
   for m in monitors:m.observe(m.snapshot())
 standing=[m.summary() for m in monitors]
 for item in standing:item['sample_hz']=30
 if not all(s['accepted'] for s in standing):
  (out/'standing_failure.json').write_text(json.dumps(standing,indent=2));raise RuntimeError('parallel standing gate failed')
 monitors=[Stability(env,i) for i in range(n)]
 for m in monitors:m.start()
 acceptance=RuntimeAcceptance(n);finished={};samples=[];current_frame=0;substep=0
 original_apply=env.action_manager.apply_action;original_update=env.scene.update
 def apply_commands(*args,**kw):
  global substep
  ret=original_apply(*args,**kw)
  targets=np.stack([s['commands'][min(current_frame,len(s['commands'])-1),substep] for s in sources])
  robot.set_joint_position_target(torch.tensor(targets,dtype=torch.float32));substep+=1
  return ret
 def observe_substep(*args,**kw):
  ret=original_update(*args,**kw);batch=[]
  for i,m in enumerate(monitors):
   snap=m.snapshot();batch.append(snap)
   if current_frame<lengths[i]:m.observe(snap)
  samples.append(batch);return ret
 env.action_manager.apply_action=apply_commands;env.scene.update=observe_substep
 file=h5py.File(out/'parallel_replay.hdf5','x');file.attrs['sources']=json.dumps([s['path'] for s in sources]);file.attrs['joint_names']=json.dumps(list(robot.joint_names));file.create_dataset('env_origins',data=origins)
 try:font=ImageFont.truetype('DejaVuSans.ttf',20)
 except OSError:font=ImageFont.load_default()
 phase_names=['pregrasp','approach','grasp','unlock','gap','pull','hold']
 start=time.monotonic()
 for current_frame in range(total+60):
  actions=np.stack([s['actions'][min(current_frame,len(s['actions'])-1)] for s in sources])
  samples.clear();substep=0
  env.step(torch.tensor(actions,dtype=torch.float32))
  assert substep==4 and len(samples)==4
  physical=read_physical_state(env);ids=np.flatnonzero(current_frame<lengths)
  if len(ids):acceptance.update(ids,physical['door_joint_pos'][ids],physical['finger_contacts'][ids],physical['arm_contacts'][ids],physical['camera_clearance'][ids],panel_normal=physical['panel_pose'][ids,:3,1])
  for i in ids:
   if current_frame+1==lengths[i]:
    finished[int(i)]=dict(id=sources[i]['id'],source=sources[i]['path'],frames=int(lengths[i]),physical=acceptance.summary(int(i)),whole_body=monitors[i].summary())
  for key in ('root_state','joint_pos','joint_vel','joint_target','applied_torque','feet_position','foot_ground_force','body_force','leg_force'):
   data=np.stack([[row[key] for row in step] for step in samples])
   if key not in file:file.create_dataset(key,shape=(0,*data.shape),maxshape=(None,*data.shape),dtype=data.dtype,chunks=(1,*data.shape),compression='lzf')
   ds=file[key];ds.resize(current_frame+1,axis=0);ds[current_frame]=data
  env.sim.render();detail.update(0.,force_recompute=True)
  overview=arena.get_data()[...,:3].copy()
  tiles=as_numpy(detail.data.output['rgb'])[...,:3]
  mosaic=Image.new('RGB',(2400,800),(16,22,32));draw=ImageDraw.Draw(mosaic)
  angles=np.rad2deg(physical['door_joint_pos'][:,0])
  states=[]
  for i,s in enumerate(sources):
   x=(i%5)*480;y=(i//5)*400;mosaic.paste(Image.fromarray(tiles[i]),(x,y+40))
   phase='completed / holding' if current_frame>=lengths[i] else phase_names[int(np.asarray(s['phases'][current_frame]).item())]
   draw.text((x+10,y+8),f"{s['id']}  {angles[i]:.1f} deg  {phase}",font=font,fill=(230,240,255))
   states.append(dict(id=s['id'],frame=min(current_frame+1,int(lengths[i])),frames=int(lengths[i]),phase=phase,door_deg=float(angles[i])))
  for name,rgb in [('arena',overview),('grid',np.asarray(mosaic))]:
   if name not in writers:writers[name]=VideoWriter(out/(name+'.mp4'),rgb.shape[1],rgb.shape[0])
   writers[name].append(rgb)
   if current_frame%10==0:
    temp=out/(name+'.tmp.jpg');Image.fromarray(rgb).save(temp,quality=85);temp.replace(out/(name+'.jpg'))
  if current_frame%30==0:
   status(dict(state='replaying',sim_seconds=(current_frame+1)/30,wall_seconds=time.monotonic()-start,episodes=states))
   print('ARENA_PROGRESS',current_frame+1,total,angles.round(2).tolist(),flush=True)
 result=dict(schema='b2arx-ten-concurrent-physics-replay-v1',num_envs=n,new_expert_sources=0,
  replayed_source_count=n,passed=sum(r['physical']['accepted'] and r['whole_body']['accepted'] for r in finished.values()),
  environment_origins=origins.tolist(),standing=standing,physics=evidence,
  replay_protocol='same-time start, original 30Hz/120Hz commands; shorter episodes hold final commands; root writes only at initialization',
  episodes=[finished[i] for i in range(n)],extra_final_hold_frames=60)
 (out/'result.json').write_text(json.dumps(result,indent=2));status(dict(state='complete',passed=result['passed'],episodes=states,sim_seconds=(total+60)/30))
 print('ARENA_COMPLETE',result['passed'],flush=True)
except Exception as error:
 status(dict(state='error',message=str(error)));raise
finally:
 if file is not None:file.close()
 for writer in writers.values():writer.close()
 if env is not None:env.close()
 app.close()
