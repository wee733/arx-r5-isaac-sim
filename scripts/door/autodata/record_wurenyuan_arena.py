#!/usr/bin/env python3
"""Record shared 3DGS+mesh previews or an explicitly configured SkillGen campaign."""
import argparse,json,math,time,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--table',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--layout',type=Path,required=True)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--visual-assets',type=Path,help='derived USD instances of immutable link visuals')
p.add_argument('--preview-only',action='store_true')
p.add_argument('--control-file',type=Path,help='wait in the loaded scene for JSON preview/record commands')
p.add_argument('--generation-campaign',type=Path,help='official SkillGen campaign; three persistent 24-env rounds')
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
import carb
carb.settings.get_settings().set('/rtx/rendermode','RayTracedLighting')
from pxr import Usd
source_stage=Usd.Stage.Open(str(a.scene),load=Usd.Stage.LoadNone)
for key,value in source_stage.GetRootLayer().customLayerData.get('renderSettings',{}).items():
 if key!='rtx:rendermode':carb.settings.get_settings().set('/'+key.replace(':','/'),value)
del source_stage
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
layout=json.loads(a.layout.read_text())
base_rows=json.loads(a.table.read_text());assert len(base_rows)==10
rows=[{**base_rows[i%10],'id':f'E{i:02d}/'+base_rows[i%10]['id']} for i in range(layout['num_envs'])]
sources=[]
for row in rows:
 path=local(row['source']);meta=json.loads((path/'result.json').read_text());assert meta['success']
 with h5py.File(path/'generated.hdf5') as f:
  ep=f['data/demo_0'];sources.append(dict(id=row['id'],path=str(path),meta=meta,actions=ep['actions'][:],
    commands=ep['obs/whole_body_substeps/joint_target'][:],phases=ep['obs/diagnostics/phase_id'][:]))
lengths=np.array([len(s['actions']) for s in sources]);total=int(lengths.max());n=len(sources)
manifest=load_asset_manifest(a.assets)
env_id,cfg,kwargs=build(manifest,pose=sources[0]['meta']['pose'],num_envs=n,name='B2Arx-Wurenyuan-Shared-Round-v0')
cfg.seed=73019;cfg.scene.wrist_camera=None;cfg.scene.ego_camera=None
if a.visual_assets is not None:
 cfg.scene.robot.spawn.usd_path=str(a.visual_assets/'robot.usda')
 cfg.scene.door.spawn.usd_path=str(a.visual_assets/'door.usda')
 if (a.visual_assets/'workcell.usda').exists():cfg.scene.workcell.spawn.usd_path=str(a.visual_assets/'workcell.usda')
cfg.scene.ground.spawn.visible=False
cfg.observations.camera_obs=None
cfg.scene.overview_camera=None
detail_indices=[7,16]
cfg.terminations.success=None;cfg.recorders=None
cfg.scene.env_spacing=layout['spacing_m']
# The upstream ground expansion assumes a square grid. Size its XY support to
# this rectangle while preserving the original thickness and height.
xy=np.asarray(layout['env_origins'])[:,:2]
ground_scale=cfg.scene.ground.spawn.scale
cfg.scene.ground.spawn.scale=(max(ground_scale[0],float(np.max(np.abs(xy[:,0]))*2+6)),
                              max(ground_scale[1],float(np.max(np.abs(xy[:,1]))*2+6)),ground_scale[2])
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.managers import EventTermCfg,SceneEntityCfg
from isaaclab.envs.mdp import randomize_rigid_body_material
from arx_door.autodata.b2arx_shared_scene import reset_source_poses,randomize_panel_appearance,configure_shared_lighting,rectangular_clone_layout
cfg.scene.shared_reconstruction=AssetBaseCfg(prim_path='/World/SharedReconstruction',
 spawn=UsdFileCfg(usd_path=str(a.scene)),collision_group=-1,
 init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(layout['background_translation'])))
cfg.events.reset_source_poses=EventTermCfg(func=reset_source_poses,mode='reset',params={'poses':[s['meta']['pose'] for s in sources]})
cfg.events.panel_appearance=EventTermCfg(func=randomize_panel_appearance,mode='startup',params={'seed':73019})
cfg.events.foot_material=EventTermCfg(func=randomize_rigid_body_material,mode='startup',params={
 'asset_cfg':SceneEntityCfg('robot',body_names=['b2_description_.*_foot']),
 'static_friction_range':(.75,1.05),'dynamic_friction_range':(.65,.85),
 'restitution_range':(0.,0.),'num_buckets':64,'make_consistent':True})
from isaaclab.utils.dict import class_to_dict
(out/'environment_config.json').write_text(json.dumps(class_to_dict(cfg),default=str,indent=2))
shutil.copy2(__file__,out/'replay_source.py')
env=None;writers={};file=None
try:
 status(dict(state='building_physics',num_envs=n))
 with rectangular_clone_layout(layout['env_origins']):
  env=gym.make(env_id,cfg=cfg,**kwargs).unwrapped
 robot=env.scene['robot'];assert not robot.is_fixed_base
 assert all(s['meta']['joint_names']==list(robot.joint_names) for s in sources)
 env.cfg.sim.render_interval=1000000
 env.reset()
 origins=as_numpy(env.scene.env_origins).copy()
 assert np.allclose(origins,np.asarray(layout['env_origins']),atol=1e-5)
 env.sim.forward();env.scene.update(env.physics_dt)
 env.sim.set_setting('/isaaclab/video/enabled',True)
 stage=omni.usd.get_context().get_stage()
 from pxr import UsdGeom,Sdf
 UsdGeom.Imageable(stage.GetPrimAtPath('/World/ground')).MakeInvisible()
 # Meshes serve collision/placement only; do not double-shadow the captured field.
 for prim in stage.Traverse():
  if str(prim.GetPath()).startswith('/World/SharedReconstruction') and prim.IsA(UsdGeom.Mesh):
   UsdGeom.Imageable(prim).MakeInvisible()
   for key in ('doNotCastShadows','invisibleToSecondaryRays'):
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(key,Sdf.ValueTypeNames.Bool).Set(True)
 lighting=configure_shared_lighting(stage,origins)
 carb.settings.get_settings().set('/rtx/rendermode','RayTracedLighting')
 from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering
 ok,nurec,spg,problems=setup_for_rendering(stage)
 if not ok or not nurec:raise RuntimeError('Gaussian setup failed: '+str(problems))
 fields=[str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() in ('ParticleField3DGaussianSplat','OmniNuRecFieldAsset')]
 assert len(fields) in (1,2),fields
 (out/'lighting.json').write_text(json.dumps(lighting,indent=2))
 import warp as wp
 actual_materials=wp.to_torch(robot.root_view.get_material_properties()).cpu().numpy()
 (out/'variation_manifest.json').write_text(json.dumps(dict(
  schema='shared-scene-round-v1',round_seed=73019,num_envs=n,layout=layout,
  gaussian_fields=fields,background_instances=1,appearance=env._shared_scene_appearance,
  initial_robot_poses=[s['meta']['pose'] for s in sources],
  physics_materials_per_env=actual_materials.tolist(),
  material_columns=['static_friction','dynamic_friction','restitution'],
  material_randomization='official Isaac Lab randomize_rigid_body_material startup event on feet',
  source_ids=[s['id'] for s in sources],training_data_generation=bool(a.generation_campaign),
  record_scope='startup scene; generation rounds record their actual variations separately',
  detail_camera_env_ids=detail_indices),indent=2))
 evidence=physics_runtime_evidence(stage,env.cfg.sim,str(env.sim.device),'source_cpu',source_settings=manifest['source_settings'],original_bindings=physics_material_bindings(stage))
 validate_physics_runtime(evidence)
 center=origins.mean(axis=0)+np.array([0,-.5,.8])
 eye=np.array([0.,-10.,7.5])
 from pxr import UsdGeom,Gf
 import omni.replicator.core as rep
 camera=UsdGeom.Camera.Define(stage,'/World/ArenaOverview')
 camera.CreateFocalLengthAttr(10.);camera.CreateHorizontalApertureAttr(20.955)
 camera.CreateClippingRangeAttr(Gf.Vec2f(.1,100.))
 z=(eye-center)/np.linalg.norm(eye-center);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
 matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=eye
 UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
 product=rep.create.render_product('/World/ArenaOverview',(1920,1080))
 arena=rep.AnnotatorRegistry.get_annotator('rgb');arena.attach(product)
 # Four diagnostic views; all 32 environments still execute every physics step.
 from types import SimpleNamespace
 class DetailViews:
  def __init__(self):
   self.cameras=[];self.annotators=[];self.products=[]
   for index in detail_indices:
    cam=UsdGeom.Camera.Define(stage,f'/World/Detail_{index:02d}')
    cam.CreateFocalLengthAttr(15.);cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.1,100.))
    cam.AddTransformOp();self.cameras.append(cam)
    prod=rep.create.render_product(str(cam.GetPath()),(640,480))
    self.products.append(prod)
    annot=rep.AnnotatorRegistry.get_annotator('rgb');annot.attach(prod);self.annotators.append(annot)
  def set_world_poses_from_view(self,eyes,targets):
   for cam,index in zip(self.cameras,detail_indices):
    eye=np.asarray(eyes[index]);target=np.asarray(targets[index])
    z=(eye-target)/np.linalg.norm(eye-target);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=eye
    UsdGeom.Xformable(cam).GetOrderedXformOps()[0].Set(Gf.Matrix4d(matrix.T.tolist()))
  def update(self,*args,**kwargs):
   self.data=SimpleNamespace(output={'rgb':np.stack([a.get_data()[...,:3].copy() for a in self.annotators])})
 detail=DetailViews()

 eyes=origins+np.array([1.2,-2.1,2.2]);targets=origins+np.array([0,-.6,.8])
 detail.set_world_poses_from_view(torch.tensor(eyes,dtype=torch.float32),torch.tensor(targets,dtype=torch.float32))
 for _ in range(4):env.sim.render()
 mount=_body_pose(env,'robot','R5a');eef=_body_pose(env,'robot','link6')
 hold=absolute_pose_actions(np.linalg.inv(mount)@eef,np.full(n,.028))
 monitors=[Stability(env,i) for i in range(n)]
 status(dict(state='settling',message=f'{n} floating robots in one shared Gaussian/mesh scene'))
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
 if a.generation_campaign is not None:
  from arx_door.autodata.shared_generation import run_campaign
  run_campaign(env, stage, json.loads(a.generation_campaign.read_text()), out, sources,
               layout, manifest, arena, status)
  sys.exit(0)
 if a.control_file is not None:
  status(dict(state='awaiting_camera_command',num_envs=n,standing_passed=True))
  last_command=None
  while app.is_running():
   if not a.control_file.exists():
    env.sim.render();time.sleep(.1);continue
   content=a.control_file.read_text()
   if content==last_command:
    env.sim.render();time.sleep(.1);continue
   try:command=json.loads(content)
   except json.JSONDecodeError:continue
   last_command=content
   import importlib
   from arx_door.autodata import shared_preview
   render_settings=importlib.reload(shared_preview).apply(stage,command)
   (out/'render_settings.json').write_text(json.dumps(render_settings,indent=2))
   if 'eye' in command:
    eye=np.asarray(command['eye']);target=np.asarray(command.get('target',center.tolist()))
    z=(eye-target)/np.linalg.norm(eye-target);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=eye
    UsdGeom.Xformable(camera).GetOrderedXformOps()[0].Set(Gf.Matrix4d(matrix.T.tolist()))
   camera.GetFocalLengthAttr().Set(float(command.get('focal_length',10.)))
   if 'detail_eye' in command:
    eyes=origins+np.asarray(command['detail_eye'])
    detail.set_world_poses_from_view(torch.tensor(eyes,dtype=torch.float32),torch.tensor(targets,dtype=torch.float32))
   for _ in range(120):env.sim.render()
   detail.update(0.,force_recompute=True)
   tag=str(command.get('id','preview'))
   Image.fromarray(arena.get_data()[...,:3].copy()).save(out/f'camera_{tag}.png')
   for i,pixels in zip(detail_indices,as_numpy(detail.data.output['rgb'])[...,:3]):Image.fromarray(pixels).save(out/f'camera_{tag}_env{i:02d}.png')
   status(dict(state='camera_preview_ready',command=command,num_envs=n))
   if command.get('action')=='record':break
 for _ in range(120):env.sim.render()
 detail.update(0.,force_recompute=True)
 preview=arena.get_data()[...,:3].copy()
 Image.fromarray(preview).save(out/'overview_start.png')
 for i,pixels in zip(detail_indices,as_numpy(detail.data.output['rgb'])[...,:3]):Image.fromarray(pixels).save(out/f'env_{i:02d}_start.png')
 if a.preview_only:
  (out/'result.json').write_text(json.dumps(dict(num_envs=n,preview_only=True,standing=standing,gaussian_fields=fields,training_data_generation=False),indent=2))
  status(dict(state='preview_complete',num_envs=n));sys.exit(0)
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
  env.sim.render();detail.update(0.,force_recompute=True)
  overview=arena.get_data()[...,:3].copy()
  tiles=as_numpy(detail.data.output['rgb'])[...,:3]
  mosaic=Image.new('RGB',(1280,520),(16,22,32));draw=ImageDraw.Draw(mosaic)
  angles=np.rad2deg(physical['door_joint_pos'][:,0])
  states=[]
  for i,s in enumerate(sources):
   phase='completed / holding' if current_frame>=lengths[i] else phase_names[int(np.asarray(s['phases'][current_frame]).item())]
   if i in detail_indices:
    tile_index=detail_indices.index(i);x=(tile_index%2)*640;y=(tile_index//2)*520
    mosaic.paste(Image.fromarray(tiles[tile_index]),(x,y+40))
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
 result=dict(schema='b2arx-shared-scene-visual-round-v1',num_envs=n,new_expert_sources=0,training_data_generation=False,gaussian_fields=fields,shared_background_count=1,
  wall_seconds=time.monotonic()-start,sim_seconds=(total+60)/30,
  replayed_source_count=n,passed=sum(r['physical']['accepted'] and r['whole_body']['accepted'] for r in finished.values()),
  environment_origins=origins.tolist(),standing=standing,physics=evidence,
  replay_protocol='same-time start, original 30Hz/120Hz commands; shorter episodes hold final commands; root writes only at initialization',
  episodes=[finished[i] for i in range(n)],extra_final_hold_frames=60)
 (out/'result.json').write_text(json.dumps(result,indent=2));status(dict(state='complete',num_envs=n,passed=result['passed'],episodes=states,sim_seconds=(total+60)/30))
 print('ARENA_COMPLETE',result['passed'],flush=True)
except Exception as error:
 status(dict(state='error',message=str(error)));raise
finally:
 if file is not None:file.close()
 for writer in writers.values():writer.close()
 if env is not None:env.close()
 app.close()
