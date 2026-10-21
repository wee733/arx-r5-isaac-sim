"""Predeclared full-body support gates, sampled at the physics rate."""
import numpy as np
from scipy.spatial.transform import Rotation
from .acceptance import as_numpy
from .recorders import _field,_body_pose

THRESHOLDS = dict(max_tilt_deg=8.,max_translation_m=.04,max_yaw_deg=5.,max_foot_slip_m=.025,
                  min_foot_force_n=5.,max_body_contact_n=2.,max_leg_contact_n=2.,
                  max_foot_non_ground_contact_n=2.,
                  max_arm_velocity_rad_s=1.,joint_limit_tolerance_rad=.002,effort_tolerance_nm=.1)

class Stability:
    def __init__(self,env,env_index=0):
        from .b2arx import ARM,LEGS
        self.env=env;self.robot=env.scene['robot'];self.reference=None;self.count=0
        self.env_index=env_index
        self.feet=[self.robot.body_names.index('b2_description_'+l+'_foot') for l in ('FL','FR','RL','RR')]
        self.foot_sensor_order=[env.scene.sensors['feet_contact'].body_names.index('b2_description_'+l+'_foot')
                                for l in ('FL','FR','RL','RR')]
        self.arm=self.robot.find_joints(ARM,preserve_order=True)[0]
        self.legs=self.robot.find_joints(LEGS,preserve_order=True)[0]
        self.limits=as_numpy(self.robot.data.joint_pos_limits)[env_index].copy()
        self.effort_limits=as_numpy(self.robot.data.joint_effort_limits)[env_index].copy()
        self.extrema=dict(max_tilt_deg=0.,max_translation_m=0.,max_yaw_deg=0.,max_foot_slip_m=0.,
            min_foot_force_n=float('inf'),max_body_contact_n=0.,max_leg_contact_n=0.,max_arm_velocity_rad_s=0.,
            max_foot_non_ground_contact_n=0.,
            joint_limit_violation_rad=0.,effort_limit_violation_nm=0.)
        self.failures=set()

    def snapshot(self):
        r=self.robot;i=self.env_index
        return dict(root_state=as_numpy(r.data.root_state_w)[i].copy(),
            joint_pos=as_numpy(r.data.joint_pos)[i].copy(),joint_vel=as_numpy(r.data.joint_vel)[i].copy(),
            joint_target=as_numpy(r.actuators.target_command.position)[i].copy(),
            applied_torque=as_numpy(r.data.applied_torque)[i].copy(),
            finger_self_force=_field(self.env.scene.sensors['finger_self_contact'].data,'normal_force_matrix_w','force_matrix_w')[i].copy(),
            feet_position=as_numpy(r.data.body_link_state_w)[i,self.feet,:3].copy(),
            foot_force=_field(self.env.scene.sensors['feet_contact'].data,'net_normal_forces_w','net_forces_w')[i,self.foot_sensor_order].copy(),
            foot_ground_force=np.stack([_field(self.env.scene.sensors['foot_ground_'+leg].data,
                'normal_force_matrix_w','force_matrix_w')[i,0,0].copy() for leg in ('FL','FR','RL','RR')]),
            body_force=_field(self.env.scene.sensors['body_contact'].data,'net_normal_forces_w','net_forces_w')[i].copy(),
            leg_force=_field(self.env.scene.sensors['leg_contact'].data,'net_normal_forces_w','net_forces_w')[i].copy(),
            world_mount=_body_pose(self.env,'robot','R5a')[i],
            world_eef=_body_pose(self.env,'robot','link6')[i],
            world_door=_body_pose(self.env,'door','door_panel')[i])

    def start(self): self.reference=self.snapshot()

    def observe(self,row):
        if self.reference is None:return
        self.count+=1
        if not all(np.isfinite(v).all() for v in row.values()):self.failures.add('nonfinite');return
        root=row['root_state'];ref=self.reference['root_state'];rot=Rotation.from_quat(root[3:7]);rr=Rotation.from_quat(ref[3:7])
        values=dict(max_tilt_deg=float(np.rad2deg(np.arccos(np.clip(rot.as_matrix()[2,2],-1,1)))),
            max_translation_m=float(np.linalg.norm(root[:3]-ref[:3])),
            max_yaw_deg=float(abs(Rotation.from_matrix(rr.as_matrix().T@rot.as_matrix()).as_euler('xyz',degrees=True)[2])),
            max_foot_slip_m=float(np.linalg.norm(row['feet_position'][:,:2]-self.reference['feet_position'][:,:2],axis=-1).max()),
            min_foot_force_n=float(np.linalg.norm(row['foot_ground_force'],axis=-1).min()),
            max_foot_non_ground_contact_n=float(np.linalg.norm(row['foot_force']-row['foot_ground_force'],axis=-1).max()),
            max_body_contact_n=float(np.linalg.norm(row['body_force'],axis=-1).max()),
            max_leg_contact_n=float(np.linalg.norm(row['leg_force'],axis=-1).max()),
            max_arm_velocity_rad_s=float(np.abs(row['joint_vel'][self.arm]).max()),
            joint_limit_violation_rad=float(max(0.,np.max(self.limits[:,0]-row['joint_pos']),np.max(row['joint_pos']-self.limits[:,1]))),
            effort_limit_violation_nm=float(max(0.,np.max(np.abs(row['applied_torque'])-self.effort_limits))))
        for key,val in values.items():
            self.extrema[key]=(min if key.startswith('min_') else max)(self.extrema[key],val)
            threshold=THRESHOLDS.get(key,THRESHOLDS['joint_limit_tolerance_rad'] if key=='joint_limit_violation_rad' else THRESHOLDS['effort_tolerance_nm'])
            if (val<threshold if key.startswith('min_') else val>threshold):self.failures.add(key)

    def summary(self):
        return dict(accepted=self.count>0 and not self.failures,samples=self.count,thresholds=THRESHOLDS,
                    metrics=self.extrema,failures=sorted(self.failures),sample_hz=120)
