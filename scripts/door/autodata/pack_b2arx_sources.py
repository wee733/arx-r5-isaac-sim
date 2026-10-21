#!/usr/bin/env python3
"""Pack accepted, independently reproduced full-body experts and test the official pool."""
import argparse,json
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from autodata_core.pool import DataGenInfoPool
from autodata_interfaces.embodiments import embodiment_adapter_from_yaml
from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
from arx_door.autodata.b2arx_frames import to_fixed_frame,to_live_mount
from arx_door.autodata.source import absolute_pose_actions
from arx_door.envs.recorders import _camera_clearance

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--replays',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[3]
if len(a.runs)!=len(a.replays):raise ValueError('one independent replay per source is required')
report=dict(schema='b2arx-autodata-sources-v2',source_reference_frame='world-fixed initial R5a pose per episode',
            live_action_reference_frame='current robot/R5a',requires_live_generation_frame_bridge=True,
            annotation_actions='actual commanded poses reexpressed in the fixed source frame; raw actions retained separately',
            episodes=[],passed=False)
urdf=ET.parse(root/'assets/b2arx_robot/my_robot.urdf').getroot()
arm_limits=np.array([[float(j.find('limit').get(k)) for k in ('lower','upper')]
    for name in [f'R5a_joint{i}' for i in range(1,7)] for j in urdf.findall('joint') if j.get('name')==name])
def audit_raw(old,result):
 n=len(old['actions']);obs=old['obs'];whole=obs['whole_body'];sub=obs['whole_body_substeps']
 bodies=result.get('contact_sensor_bodies',{})
 if len(bodies.get('leg_contact',[]))!=24 or len(bodies.get('arm_contacts',[]))!=7:
  raise ValueError('requires complete leg/rotor and nonfinger arm/base collision coverage')
 if 'foot_ground_force' not in whole:raise ValueError('requires separate ground contact evidence')
 required=('wrist_rgb','ego_rgb','ego_camera_pose','joint_pos','diagnostics/joint_target_substeps')
 for name in required:
  if len(obs[name])!=n:raise ValueError(f'unsynchronized {name}')
 if sub['joint_target'].shape!=(n,4,20):raise ValueError('missing full-body physical command history')
 arm=[result['joint_names'].index(f'R5a_joint{i}') for i in range(1,7)]
 q=sub['joint_pos'][:][:,:,arm]
 margin=float(np.minimum(q-arm_limits[:,0],arm_limits[:,1]-q).min())
 singular=float(whole['arm_min_singular_value'][:].min())
 if margin<.04 or singular<.005:raise ValueError('full trajectory joint margin or singularity screening failed')
 command_error=float(np.abs(sub['joint_target'][:][:,:,arm]-obs['diagnostics/joint_target_substeps'][:,:,:6]).max())
 if command_error>1e-6:raise ValueError('recorded arm command differs from actuator target')
 if not np.allclose(whole['timestamp_s'][:],np.arange(n)/30):raise ValueError('control timestamps disagree')
 roots=whole['root_state'][:]
 if not np.allclose(roots[0,:3],result['placement_metrics']['body_position_m'],atol=1e-6):
  raise ValueError('reported placement differs from the physical raw state')
 measured_plane=-.3082-float(roots[0,1])
 if abs(measured_plane-result['placement_metrics']['signed_distance_closed_plane_m'])>1e-6:
  raise ValueError('reported door distance differs from physical state')
 world_body=np.repeat(np.eye(4)[None],n,axis=0)
 world_body[:,:3,:3]=Rotation.from_quat(roots[:,3:7]).as_matrix();world_body[:,:3,3]=roots[:,:3]
 ego_mount=np.linalg.inv(world_body)@obs['ego_camera_pose'][:]
 ego_error=float(np.abs(ego_mount-ego_mount[0]).max())
 if ego_error>5e-5:raise ValueError('body camera extrinsics changed relative to the robot body')
 ego_clearance=float(_camera_clearance(obs['ego_camera_pose'][:],whole['world_door'][:]).min())
 if ego_clearance<.01:raise ValueError('body camera violates the door-panel clearance')
 rpy=Rotation.from_quat(sub['root_state'][:].reshape(-1,13)[:,3:7]).as_euler('xyz',degrees=True)
 for name in ('wrist_rgb','ego_rgb'):
  if obs[name].shape[1:]!=(480,640,3):raise ValueError('unexpected camera dimensions')
  for index in (0,n//2,n-1):
   if obs[name][index].std()<1:raise ValueError('empty camera frame')
 return dict(min_joint_margin_rad=margin,min_jacobian_singular_value=singular,
             max_command_record_error=command_error,whole_body_physics_samples=4*n,
             ego_camera_mount=ego_mount[0].tolist(),max_ego_mount_matrix_error=ego_error,
             min_body_camera_clearance_m=ego_clearance,
             max_abs_roll_deg=float(np.abs(rpy[:,0]).max()),max_abs_pitch_deg=float(np.abs(rpy[:,1]).max()))
with h5py.File(a.output,'x') as dst:
 data=dst.create_group('data');data.attrs['env_args']=json.dumps(dict(env_name='B2Arx-Door-Expert-v0',type=2,env_kwargs={}))
 for i,(run,replay) in enumerate(zip(a.runs,a.replays)):
  result=json.loads((run/'result.json').read_text());rep=json.loads((replay/'result.json').read_text())
  for label,check in [('source',result),('replay',rep)]:
   if not (check['success'] and check['physical_metrics']['accepted'] and check['whole_body_acceptance']['accepted']):
    raise ValueError(f'{label} {run} has not passed all gates')
  if run.resolve()==replay.resolve():raise ValueError('independent replay must be a separate physical execution')
  if result['pose']!=rep['pose']:raise ValueError('replay placement differs')
  if 'actuator-command replay' not in rep.get('provenance',''):raise ValueError('requires independent recorded-command replay')
  with h5py.File(run/'generated.hdf5') as src:
   old=src['data/demo_0'];ep=data.create_group(f'demo_{i}')
   raw_audit=audit_raw(old,result)
   with h5py.File(replay/'generated.hdf5') as rf:
    replay_audit=audit_raw(rf['data/demo_0'],rep)
    if not np.allclose(old['obs/whole_body_substeps/joint_target'][:],rf['data/demo_0/obs/whole_body_substeps/joint_target'][:],atol=1e-6):
     raise ValueError('independent replay did not execute the recorded full-body commands')
   for k,v in old.attrs.items():ep.attrs[k]=v
   for key in ('actions','initial_state','obs/datagen_info','obs/joint_pos','obs/whole_body','obs/whole_body_substeps'):
    parent,_,leaf=key.rpartition('/');src.copy(old[key],ep.require_group(parent) if parent else ep,name=leaf)
   src.copy(old['obs/datagen_info'],ep['obs'],name='raw_datagen_info_live_mount')
   actions=old['actions'][:];mount=old['obs/whole_body/world_mount'][:];eef=old['obs/whole_body/world_eef'][:]
   recorded=old['obs/datagen_info/eef_pose/link6'][:]
   if not np.allclose(np.linalg.inv(mount)@eef,recorded,atol=2e-5):raise ValueError('moving mount-frame eef mismatch')
   if actions.shape[1]!=8:raise ValueError('pose action must retain 8 dimensions')
   ep.create_dataset('raw_mount_frame_actions',data=actions)
   ep.create_dataset('source_world_frame',data=mount[0])
   ep.attrs['pose_frame']='world-fixed initial R5a pose; not directly executable in a drifting mount frame'
   ep.attrs['raw_dataset']=str(run/'generated.hdf5')
   original_target=old['obs/datagen_info/target_eef_pose/link6'][:]
   original_object=old['obs/datagen_info/object_pose/door'][:]
   canonical_target=to_fixed_frame(mount[0],mount,original_target)
   canonical_object=to_fixed_frame(mount[0],mount,original_object)
   canonical_eef=to_fixed_frame(mount[0],mount,recorded)
   roundtrip_error=float(np.abs(to_live_mount(mount[0],mount,canonical_target)-original_target).max())
   object_relative_error=float(np.abs(np.linalg.inv(canonical_object)@canonical_target-np.linalg.inv(original_object)@original_target).max())
   object_drift=float(np.abs(canonical_object-canonical_object[0]).max())
   if max(roundtrip_error,object_relative_error,object_drift)>2e-5:raise ValueError('fixed-source/live-mount pose contract failed')
   ep['actions'][:]=absolute_pose_actions(canonical_target,actions[:,7])
   ep['obs/datagen_info/eef_pose/link6'][:]=canonical_eef
   ep['obs/datagen_info/target_eef_pose/link6'][:]=canonical_target
   ep['obs/datagen_info/object_pose/door'][:]=canonical_object
   report['episodes'].append(dict(source=str(run),replay=str(replay),frames=len(actions),pose=result['pose'],
                                  placement_metrics=result['placement_metrics'],
                                  frame_bridge_audit=dict(roundtrip_max_error=roundtrip_error,object_relative_max_error=object_relative_error,
                                                          fixed_source_object_max_drift=object_drift),
                                  raw_audit=raw_audit,replay_audit=replay_audit))
 data.attrs['total']=sum(x['frames'] for x in report['episodes'])
if len(a.runs)>1:
 positions=np.array([r['placement_metrics']['body_position_m'][:2] for r in report['episodes']])
 distance=np.array([r['placement_metrics']['signed_distance_closed_plane_m'] for r in report['episodes']])
 pairs=np.triu_indices(len(positions),1)
 planar=float(np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)[pairs].min())
 gap=float(np.abs(distance[:,None]-distance[None,:])[pairs].min())
 report['variation']=dict(min_pair_planar_distance_m=planar,min_plane_distance_gap_m=gap,
                          plane_distance_span_m=float(np.ptp(distance)))
 if planar<.025 or gap<.02:raise ValueError('placements are too similar for distinct seed variation')
task=TaskDescriptor.from_yaml(str(root/'configs/autodata/tasks/arx_door_knob_hold_open.yaml'))
adapter=embodiment_adapter_from_yaml(str(root/'configs/autodata/embodiments/b2arx.yaml'));task.validate_embodiment(adapter)
pool=DataGenInfoPool.from_hdf5(str(a.output),task,adapter,'cpu',True)
if pool.num_datagen_infos!=len(a.runs):raise ValueError('official pool dropped sources')
with h5py.File(a.output) as f:
 for i,info in enumerate(pool.datagen_infos):
  decoded=adapter.action_to_target_eef_pose(torch.tensor(f[f'data/demo_{i}/actions'][:]))['link6']
  if not torch.allclose(decoded,info.target_eef_pose['link6'].to(decoded.dtype),atol=1e-6):raise ValueError('official action codec mismatch')
report.update(passed=True,official_pool_count=pool.num_datagen_infos,subtask_boundaries=pool.subtask_boundaries)
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
