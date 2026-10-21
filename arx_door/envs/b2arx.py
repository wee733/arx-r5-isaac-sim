"""Floating B2 + ARX variant. Leg hold is independent of the eight-dimensional arm action."""
from __future__ import annotations
import math
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.sim import ArticulationRootPropertiesCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as mu
from . import door, mdp
from .compat import as_torch

ARM = [f'R5a_joint{i}' for i in range(1,7)]
LEGS = [f'b2_description_{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
STAND = [v for side in (1,-1,1,-1) for v in (.15*side,.67,-1.32)]

def arm_state(env):
    r=env.scene['robot']; ids,_=r.find_joints(ARM+['R5a_joint7'],preserve_order=True)
    return as_torch(r.data.joint_pos)[:,ids]

def arm_velocity(env):
    r=env.scene['robot']; ids,_=r.find_joints(ARM+['R5a_joint7'],preserve_order=True)
    return as_torch(r.data.joint_vel)[:,ids]

def eef_pose(env):
    r=env.scene['robot']; bp,bq=mdp._body_pose(r,'R5a'); ep,eq=mdp._body_pose(r,'R5a_link6')
    return mu.subtract_frame_transforms(bp,bq,ep,eq)

def eef_pos(env): return eef_pose(env)[0]
def eef_quat(env): return eef_pose(env)[1]

class MountFrameIK(DifferentialInverseKinematicsAction):
    """IK and AutoData pose frame is the measured R5a installation link."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.mount_id=self._asset.find_bodies('R5a')[0][0]

    def _compute_frame_pose(self):
        s=as_torch(self._asset.data.body_link_state_w)
        return mu.subtract_frame_transforms(s[:,self.mount_id,:3],s[:,self.mount_id,3:7],
                                            s[:,self._body_idx,:3],s[:,self._body_idx,3:7])

    @property
    def jacobian_b(self):
        j=self.jacobian_w.clone()
        q=as_torch(self._asset.data.body_link_state_w)[:,self.mount_id,3:7]
        rot=mu.matrix_from_quat(mu.quat_inv(q))
        j[:,:3,:]=torch.bmm(rot,j[:,:3,:]); j[:,3:,:]=torch.bmm(rot,j[:,3:,:])
        return j

def build(manifest, *, pose=None, action='ik_abs', name='B2Arx-Door-Expert-v0', num_envs=1):
    env_id,cfg,kwargs=door.build(manifest,num_envs,action=action,physics='source_cpu',name=name,ego_camera={})
    r=cfg.scene.robot
    if pose is not None:
        r.init_state.pos=tuple(pose[:3]); yaw=pose[3]
        r.init_state.rot=(0.,0.,math.sin(yaw/2),math.cos(yaw/2))
    r.init_state.joint_pos={**dict(zip(LEGS,STAND)),**dict(zip(ARM,door.DEFAULT_ARM_STATE)),
                            'R5a_joint7':.028,'R5a_joint8':.028}
    r.spawn.articulation_props=ArticulationRootPropertiesCfg(fix_root_link=False,enabled_self_collisions=True,
        solver_position_iteration_count=32,solver_velocity_iteration_count=4)
    r.actuators={
        'legs':ImplicitActuatorCfg(joint_names_expr=LEGS,stiffness=800.,damping=40.,
            effort_limit_sim={'.*_hip_joint':200.,'.*_thigh_joint':200.,'.*_calf_joint':320.}),
        'arm':ImplicitActuatorCfg(joint_names_expr=ARM,stiffness=2000.,damping=100.,effort_limit_sim=100.),
        'gripper':ImplicitActuatorCfg(joint_names_expr=['R5a_joint[78]'],stiffness=1200.,damping=60.,effort_limit_sim=60.)}
    cfg.actions.arm_action.joint_names=ARM
    if action=='ik_abs':
        cfg.actions.arm_action.body_name='R5a_link6'
        cfg.actions.arm_action.class_type=MountFrameIK
    cfg.actions.gripper_action.joint_names=('R5a_joint7','R5a_joint8')
    for term,func in [('joint_pos',arm_state),('joint_vel',arm_velocity),('eef_pos',eef_pos),('eef_quat',eef_quat)]:
        getattr(cfg.observations.policy,term).func=func
    cfg.robot_body_aliases={f'link{i}':f'R5a_link{i}' for i in range(1,9)}
    cfg.arm_state_names=ARM+['R5a_joint7']
    for attr,link in [('left_panel_contact','R5a_link7'),('right_panel_contact','R5a_link8'),('arm_contacts','(R5a|R5a_link[1-6])')]:
        getattr(cfg.scene,attr).prim_path='{ENV_REGEX_NS}/R5a/'+link
    cfg.scene.feet_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/R5a/b2_description_.*_foot',history_length=1)
    for foot in ('FL','FR','RL','RR'):
        setattr(cfg.scene,'foot_ground_'+foot,ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/R5a/b2_description_'+foot+'_foot',history_length=1,
            filter_prim_paths_expr=['/World/ground/geometry']))
    cfg.scene.finger_self_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/R5a/R5a_link7',history_length=1,
        filter_prim_paths_expr=['{ENV_REGEX_NS}/R5a/R5a_link8'])
    cfg.scene.body_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/R5a/b2_description(|_head_Link|_tail_link)',history_length=1)
    cfg.scene.leg_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/R5a/b2_description_.*_(hip|thigh|calf)(_rotor)?',history_length=1)
    # Real body-bound camera: local +X forward, +Z up, pitched up 20 degrees.
    from arx_door.scene.ego_camera import ego_camera_cfg,ego_mount
    cfg.scene.ego_camera=ego_camera_cfg(dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1]),
        ego_mount(dict(position_base_m=[.42,0,.14],yaw_deg=-60.,pitch_up_deg=30.,horizontal_fov_deg=110.,
                       prim_path='{ENV_REGEX_NS}/R5a/b2_description/EgoCamera')),
        quaternion_from_xyzw=lambda q:q)
    cfg.scene.wrist_camera.prim_path='{ENV_REGEX_NS}/R5a/R5a_link6/WristCamera'
    if cfg.scene.overview_camera is not None:
        import numpy as np
        from scipy.spatial.transform import Rotation
        position=np.array([2.6,-3.6,2.1]);lookat=np.array([0.,-.8,.65])
        z=(position-lookat)/np.linalg.norm(position-lookat)
        x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        cfg.scene.overview_camera.offset=CameraCfg.OffsetCfg(pos=tuple(position),
            rot=tuple(Rotation.from_matrix(np.stack([x,y,z],axis=1)).as_quat()),convention='opengl')
        cfg.scene.overview_camera.spawn.focal_length=24.
        cfg.scene.overview_camera.spawn.clipping_range=(.02,20.)
    cfg.terminations.success=None
    cfg.recorders=None
    cfg.b2arx_control=dict(leg_kp=800.,leg_kd=40.,leg_target=STAND,arm_kp=2000.,arm_kd=100.,
                          gripper_kp=1200.,gripper_kd=60.,pose_frame='R5a',floating=True,self_collision=True)
    return env_id,cfg,kwargs
