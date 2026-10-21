#!/usr/bin/env python3
"""Continuous arm IK screening only; survivors still require full physics validation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from arx_door.kinematics import ArmKinematics
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--height',type=float,default=.58)
a=p.parse_args();kin=ArmKinematics(ROOT/'assets/b2arx_robot/my_robot.urdf',joint_prefix='R5a_')
hinge=np.array([-.378,-.331,0]);knob=np.array([.313,-.3752,.928])
grasp=np.eye(4);grasp[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();grasp[:3,3]=knob-.145*grasp[:3,0]
path=[]
for stand in np.linspace(.06,0,4):
 t=grasp.copy();t[:3,3]-=stand*t[:3,0];path.append(t)
for ang in np.linspace(0,51,8)[1:]:
 t=grasp.copy();t[:3,:3]=grasp[:3,:3]@Rotation.from_euler('x',-ang,degrees=True).as_matrix();path.append(t)
unlocked=path[-1]
for ang in np.linspace(0,34,13)[1:]:
 t=unlocked.copy();rot=Rotation.from_euler('z',-ang,degrees=True).as_matrix()
 t[:3,3]=hinge+rot@(unlocked[:3,3]-hinge);t[:3,:3]=rot@unlocked[:3,:3];path.append(t)
rows=[]
for y in np.linspace(-1.25,-.85,9):
 for x in [0.,.15,.30,.45,.60]:
  for yaw in [0.,30.,60.,90.,120.,150.]:
   world=np.eye(4);world[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix();world[:3,3]=[x,y,a.height]
   mount=world.copy();mount[:3,3]+=world[:3,:3]@np.array([.22,0,.1]);inv=np.linalg.inv(mount)
   row=dict(position=[x,float(y),a.height],yaw_deg=yaw,distance_plane_m=-.3082-y,
       mount_to_handle_m=float(np.linalg.norm(mount[:3,3]-knob)),geometric_pass=False)
   q=np.array([0.,1.,1.5,0.,0.,0.]);qs=[];margin=100.;sv=100.
   try:
    for t in path:
     q1=kin.ik(inv@t,q,attempts=2)
     if qs and np.max(np.abs(q1-q))>.3:raise ValueError('IK branch discontinuity')
     q=q1;qs.append(q.tolist());margin=min(margin,float(np.min(np.r_[q-kin.lower,kin.upper-q])))
     fk=kin.fk(q);jac=np.zeros((6,6))
     for j in range(6):
      dq=np.zeros(6);dq[j]=1e-5;f=kin.fk(q+dq)
      jac[:,j]=np.r_[(f[:3,3]-fk[:3,3])/1e-5,Rotation.from_matrix(f[:3,:3]@fk[:3,:3].T).as_rotvec()/1e-5]
     sv=min(sv,float(np.linalg.svd(jac,compute_uv=False)[-1]))
    row.update(geometric_pass=margin>=.04 and sv>=.005,min_limit_margin_rad=margin,min_jacobian_singular_value=sv,path_q=qs)
   except ValueError as e:row['rejection']=str(e)
   rows.append(row)
   if row['geometric_pass']:print('IK_SURVIVOR',row['position'],yaw,margin,sv,flush=True)
 a.output.write_text(json.dumps(dict(scope='geometric_screen_only',door_normal=[0,-1,0],door_lateral=[1,0,0],
    closed_plane_y=-.3082,opening='positive hinge = clockwise world Z, toward negative Y',candidates=rows),indent=2))
print('CANDIDATES',len(rows),'IK_SURVIVORS',sum(r['geometric_pass'] for r in rows))
