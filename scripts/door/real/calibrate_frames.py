#!/usr/bin/env python3
"""Resolve SDK's zero-position origin from its URDF, without fitting away geometry differences."""
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]


def chain(path,prefix):
    tree=ET.parse(path);result=[]
    for i in range(1,7):
        j=tree.find(f"joint[@name='{prefix}{i}']");origin=j.find('origin');t=np.eye(4)
        t[:3,3]=np.fromstring(origin.get('xyz'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy'),sep=' ')).as_matrix()
        result.append((t,np.fromstring(j.find('axis').get('xyz'),sep=' ')))
    return result


def fk(chain,q):
    t=np.eye(4)
    for (origin,axis),angle in zip(chain,q):
        r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix();t=t@origin@r
    return t


def main():
    sdk_path=ROOT/'drivers/arx_r5/ROS2/R5_ws/src/ARX_R5_ros2_V7/arx_r5_controller/X5liteaa0.urdf'
    model_path=ROOT/'assets/b2arx_robot/my_robot.urdf'
    sdk=chain(sdk_path,'joint');model=chain(model_path,'R5a_joint')
    home=fk(sdk,[0]*6)[:3,3]
    rng=np.random.default_rng(733)
    q=rng.uniform([-1.,0.,0.,-.7,-.7,-.7],[1.,2.,2.,.7,.7,.7],(100,6));q[0]=0
    probe=ROOT/'install/real/arx_vla_control/lib/arx_vla_control/kinematics_probe'
    result=subprocess.run([str(probe)],input='\n'.join(json.dumps(x.tolist()) for x in q)+'\n',text=True,capture_output=True,check=True)
    native=[json.loads(x)['eef'] for x in result.stdout.splitlines() if x.startswith('{')]
    if len(native)!=len(q):raise RuntimeError(result.stderr+result.stdout)
    position_error=[];rotation_error=[];geometry_error=[]
    for joints,p in zip(q,native):
        t=fk(sdk,joints);m=fk(model,joints)
        position_error.append(float(np.linalg.norm(np.array(p[:3])+home-t[:3,3])))
        rotation_error.append(float((Rotation.from_rotvec(p[3:]).inv()*Rotation.from_matrix(t[:3,:3])).magnitude()))
        geometry_error.append(float(np.linalg.norm(m[:3,3]-t[:3,3])))
    verified=max(position_error)<1e-7 and max(rotation_error)<1e-7
    report={'schema':'arx_frame_calibration_v1','method':'SDK URDF FK minus zero-position translation; independently checked against 100 native FK samples',
            'training_asset_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'sources':[str(p.relative_to(ROOT)) for p in (sdk_path,model_path)],
            'sdk_base_from_model_mount':(-home).tolist()+[0.,0.,0.], 'model_link6_from_sdk_tool':[0.]*6,
            'software_frame_convention_verified':verified,'external_metrology_performed':False,
            'native_fk_position_max_error_m':max(position_error),'native_fk_rotation_max_error_rad':max(rotation_error),
            'model_geometry_difference_max_m':max(geometry_error),
            'geometry_differences':[{'joint':i+1,'sdk_origin':s[0][:3,3].tolist(),'training_origin':m[0][:3,3].tolist()} for i,(s,m) in enumerate(zip(sdk,model)) if not np.allclose(s[0][:3,3],m[0][:3,3],atol=1e-7)],
            'interpretation':'Both APIs describe the installation-base axes and link6, not fingertip/TCP. SDK subtracts only zero-position XYZ, not zero rotation. Wrist geometry differs; do not invent a fitted base/tool offset to absorb that q-dependent difference.'}
    path=ROOT/'configs/real/frame_calibration.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not verified:raise RuntimeError('SDK convention verification failed')


if __name__=='__main__':main()
