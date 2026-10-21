#!/usr/bin/env python3
"""Measure a floating full robot stand in the pinned door scene."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--pose',type=float,nargs=4,default=[.4,-1.2,.65,1.57079632679]);p.add_argument('--steps',type=int,default=240)
AppLauncher.add_app_launcher_args(p)
for option in ('--headless','--enable_cameras'):
    if option not in p._option_string_actions:p.add_argument(option,action='store_true')
a=p.parse_args();a.device='cpu';a.enable_cameras=True
a.output.mkdir(parents=True,exist_ok=False)
app=AppLauncher(a).app
import gymnasium as gym
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
from arx_door.envs.b2arx import build,ARM,LEGS,STAND
from arx_door.envs.acceptance import as_numpy
from arx_door.envs.recorders import _field,_body_pose
from arx_door.runtime import close_app
from arx_door.policy.simulation import VideoWriter
thresholds=dict(max_tilt_deg=8.,max_translation_m=.04,max_yaw_deg=5.,max_foot_slip_m=.025,
                min_foot_force_n=5.,max_body_contact_n=2.,max_leg_contact_n=2.)
result=dict(success=False,thresholds=thresholds,scope='standing_only_not_door_success',pose=a.pose)
rows=[];env=None;video=None
try:
    eid,cfg,kw=build(a.assets,pose=a.pose,action='joint_pos')
    cfg.sim.render_interval=1000000
    env=gym.make(eid,cfg=cfg,**kw).unwrapped;env.reset()
    import omni.usd
    omni.usd.get_context().get_stage().Export(str(a.output/'runtime_scene.usda'))
    r=env.scene['robot']; result['fixed_base']=r.is_fixed_base
    if r.is_fixed_base: raise RuntimeError('robot is fixed base')
    result['joint_names']=list(r.joint_names);result['body_names']=list(r.body_names)
    result['controller']=cfg.b2arx_control
    result['joint_limits']=as_numpy(r.data.joint_pos_limits).tolist()
    result['effort_limits']=as_numpy(r.data.joint_effort_limits).tolist()
    result['total_mass_kg']=float(as_numpy(r.data.default_mass).sum())
    ids,_=r.find_joints(ARM+['R5a_joint7'],preserve_order=True)
    action=as_numpy(r.data.joint_pos)[:,ids].copy()
    feet=[r.body_names.index('b2_description_'+leg+'_foot') for leg in ('FL','FR','RL','RR')]
    original_step=env.sim.step
    def measure(*args,**kwargs):
        value=original_step(*args,**kwargs)
        return value
    for i in range(a.steps):
        env.step(torch.tensor(action,dtype=torch.float32,device=env.device))
        body=as_numpy(r.data.root_state_w)[0].copy()
        row=dict(root=body,joints=as_numpy(r.data.joint_pos)[0].copy(),velocity=as_numpy(r.data.joint_vel)[0].copy(),
            feet=as_numpy(r.data.body_link_state_w)[0,feet,:3].copy(),
            foot_force=_field(env.scene.sensors['feet_contact'].data,'net_normal_forces_w','net_forces_w')[0].copy(),
            body_force=_field(env.scene.sensors['body_contact'].data,'net_normal_forces_w','net_forces_w')[0].copy(),
            leg_force=_field(env.scene.sensors['leg_contact'].data,'net_normal_forces_w','net_forces_w')[0].copy(),
            effort=as_numpy(r.data.applied_torque)[0].copy(),target=as_numpy(r.actuators.target_command.position)[0].copy())
        rows.append(row)
        if i%30==0:
            print('STAND',i,body[:7].tolist(),'feet',row['foot_force'].tolist(),flush=True)
            env.sim.render();cam=env.scene.sensors['overview_camera'];cam.update(0.,force_recompute=True)
            rgb=as_numpy(cam.data.output['rgb'])[0,:,:,:3]
            if video is None: video=VideoWriter(a.output/'overview.mp4',rgb.shape[1],rgb.shape[0])
            video.append(rgb)
        if body[2]<.3:raise RuntimeError('fell below 0.3m')
    stable=rows[60:]
    root=np.stack([v['root'] for v in stable]);fp=np.stack([v['feet'] for v in stable]);ff=np.stack([v['foot_force'] for v in stable])
    rotations=Rotation.from_quat(root[:,3:7]); tilt=np.rad2deg(np.arccos(np.clip(rotations.as_matrix()[:,2,2],-1,1)))
    metrics=dict(max_tilt_deg=float(tilt.max()),max_translation_m=float(np.linalg.norm(root[:,:3]-root[0,:3],axis=1).max()),
        max_yaw_deg=float(np.abs(np.rad2deg(rotations.as_euler('xyz')[:,2]-rotations.as_euler('xyz')[0,2])).max()),
        max_foot_slip_m=float(np.linalg.norm(fp[:,:,:2]-fp[0,:,:2],axis=-1).max()),
        min_foot_force_n=float(np.linalg.norm(ff,axis=-1).min()),
        max_body_contact_n=float(max(np.linalg.norm(v['body_force'],axis=-1).max() for v in stable)),
        max_leg_contact_n=float(max(np.linalg.norm(v['leg_force'],axis=-1).max() for v in stable)))
    result['metrics']=metrics;result['settled_root']=root[0].tolist()
    result['checks']={k:(metrics[k]>=v if k.startswith('min_') else metrics[k]<=v) for k,v in thresholds.items()}
    result['success']=all(result['checks'].values())
except Exception as e:
    result['error']=str(e);result['traceback']=traceback.format_exc();print(result['traceback'],flush=True)
finally:
    if rows:
        with h5py.File(a.output/'standing.hdf5','w') as f:
            for key in rows[0]:f.create_dataset(key,data=np.stack([r[key] for r in rows]))
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    if video:video.close()
    if env:env.close()
    close_app(app,0 if result['success'] else 1)
