"""Floating-mount cuRobo frame and linkwise obstacle synchronization.

The fresh expert does not call cuRobo. This separate adapter is required before
using its sources for motion generation. It never uses the pedestal exemption.
"""
from pathlib import Path
import copy
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]

def configure_warp_interop():
    """cuRobo v1's one legacy namespace lookup, mapped to Warp 1.16's public API."""
    import warp,types
    if not hasattr(warp,'torch'):
        warp.torch=types.SimpleNamespace(device_from_torch=warp.device_from_torch)

def robot_config_file(output_dir):
    """Reuse arm sphere geometry in identical link frames, with full-model names."""
    import yaml
    from curobo.types.file_path import ContentPath
    from curobo.util.xrdf_utils import convert_xrdf_to_curobo
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    old=ROOT/'assets/arx_r5a/xrdf/r5a.xrdf'
    names={'base_link':'R5a',**{f'link{i}':f'R5a_link{i}' for i in range(1,9)},
           **{f'joint{i}':f'R5a_joint{i}' for i in range(1,9)}}
    def rename(v):
        if isinstance(v,dict):return {names.get(k,k):rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        return names.get(v,v) if isinstance(v,str) else v
    xrdf=rename(yaml.safe_load(old.read_text()))
    # No attached-object link is necessary for a door handle that stays articulated.
    xrdf['modifiers']=[{'set_base_frame':'R5a'}]
    for section in ('collision','self_collision'):
        xrdf[section].get('buffer_distance',{}).pop('attached_object',None)
    xrdf['self_collision']['ignore'].pop('attached_object',None)
    for key,items in xrdf['self_collision']['ignore'].items():
        xrdf['self_collision']['ignore'][key]=[x for x in items if x!='attached_object']
    xp=output/'b2arx.xrdf';xp.write_text(yaml.safe_dump(xrdf,sort_keys=False))
    urdf=ROOT/'assets/b2arx_robot/my_robot.urdf'
    cfg=convert_xrdf_to_curobo(ContentPath(robot_urdf_absolute_path=str(urdf),
        robot_xrdf_absolute_path=str(xp),robot_asset_absolute_path=str(urdf.parent)),xrdf)
    cfg=cfg.get('robot_cfg',cfg);kin=cfg['kinematics']
    kin.update(urdf_path=str(urdf),asset_root_path=str(urdf.parent))
    # Both fingers in this URDF are independent prismatic joints.
    kin['lock_joints']={'R5a_joint7':.044,'R5a_joint8':.044}
    if any(v<0 for v in kin['collision_sphere_buffer'].values()):
        raise ValueError('negative world collision exemptions are forbidden for the floating robot')
    path=output/'b2arx_curobo.yaml';path.write_text(yaml.safe_dump({'robot_cfg':cfg},sort_keys=False))
    return path


def matrix(pose):
    value=np.eye(4);value[:3,3]=pose[:3]
    value[:3,:3]=Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()
    return value


def pose_list(transform):
    q=Rotation.from_matrix(transform[:3,:3]).as_quat()
    return [*transform[:3,3],q[3],*q[:3]]


class LiveMountCollisionWorld:
    """Cache local geometry, never articulated world poses or mounting transforms.

    Physics/Fabric body poses drive every obstacle on the dog and door. Even
    fixed walls and the floor are reexpressed in the current measured R5a frame.
    Arm shapes are handled by cuRobo's self-collision model; the complete dog is
    retained as obstacles. No robot-root substring is blanket-ignored.
    """
    def __init__(self,env,usd_helper,env_id=0):
        from pxr import UsdGeom,Usd,UsdPhysics
        from arx_door.envs.acceptance import as_numpy
        self.env,self.env_id=env,env_id
        self.as_numpy=as_numpy
        self.stage=usd_helper.stage
        env_path=f'/World/envs/env_{env_id}'
        robot_path=env_path+'/R5a'
        ignore=[robot_path+'/'+name+'/' for name in ['R5a',*[f'R5a_link{i}' for i in range(1,9)],'R5a_ee']]
        ignore+=['WristCamera','EgoCamera','OverviewCamera','/curobo']
        roots=[self.stage.GetPrimAtPath(env_path)]
        if self.stage.GetPrimAtPath('/World/ground').IsValid():roots.append(self.stage.GetPrimAtPath('/World/ground'))
        collision_paths=[str(p.GetPath()) for root in roots for p in Usd.PrimRange(root)
                         if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        if not collision_paths:raise ValueError('no enabled collision geometry in the live environment')
        # cuRobo's Cube reader assumes a local xformOp:scale. The valid wrapped
        # ground instead inherits its scale from the parent. Normalize only the
        # planning reader, never the live USD or physical collider.
        from curobo.util import usd_helper as helper_module
        from curobo.geom.types import Cuboid
        original_cube_reader=helper_module.get_cube_attrs
        def cube_reader(prim,cache=None,transform=None):
            mat=np.asarray(cache.GetLocalToWorldTransform(prim)).T.copy()
            scale=np.linalg.norm(mat[:3,:3],axis=0)
            mat[:3,:3]/=scale[None,:]
            if transform is not None:mat=np.asarray(transform)@mat
            size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            return Cuboid(name=str(prim.GetPath()),pose=pose_list(mat),dims=(scale*size).tolist())
        helper_module.get_cube_attrs=cube_reader
        try:
            self.world=usd_helper.get_obstacles_from_stage(only_paths=collision_paths,ignore_substring=ignore,
                                                          reference_prim_path='/World').get_collision_check_world()
        finally:helper_module.get_cube_attrs=original_cube_reader
        self.bindings=[]
        cache=UsdGeom.XformCache()
        body_paths={}
        for asset,prefix in [('robot',robot_path),('door',env_path+'/Door')]:
            for index,name in enumerate(env.scene[asset].body_names):body_paths[prefix+'/'+name]=(asset,index)
        for obstacle in self.world.objects:
            path=obstacle.name
            match=next((b for b in sorted(body_paths,key=len,reverse=True) if path==b or path.startswith(b+'/')),None)
            t=matrix(obstacle.pose)
            if match:
                usd_body=np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(match))).T
                self.bindings.append((obstacle,body_paths[match],np.linalg.inv(usd_body)@t))
            else:self.bindings.append((obstacle,None,t))
        self.mount_index=env.scene['robot'].body_names.index('R5a')
        if not any('/b2_description' in o.name for o,_,_ in self.bindings):
            raise ValueError('collision world omitted the dog')
        if not any('/Door/' in o.name for o,_,_ in self.bindings):raise ValueError('collision world omitted the door')
        self.update()

    def update(self,checker=None,tensor_args=None):
        from curobo.types.math import Pose
        from arx_door.autodata.source import pose_matrix
        states={a:self.as_numpy(self.env.scene[a].data.body_link_state_w)[self.env_id] for a in ('robot','door')}
        mount=states['robot'][self.mount_index]
        inverse=np.linalg.inv(pose_matrix(mount[:3],mount[3:7]))
        for obstacle,binding,local in self.bindings:
            if binding:
                state=states[binding[0]][binding[1]]
                world=pose_matrix(state[:3],state[3:7])@local
            else:world=local
            obstacle.pose=pose_list(inverse@world)
            if checker is not None:
                checker.update_obstacle_pose(obstacle.name,Pose.from_list(obstacle.pose,tensor_args),update_cpu_reference=True)
        return self.world


def install_on_planner(planner):
    """Install only on a b2arx planner after its arm model is initialized.

    Caller/embodiment targets already use R5a coordinates, so planner->caller
    transforms are identity, while world obstacles follow the live mount.
    """
    import torch,types
    if planner.config.robot_name!='b2arx':raise ValueError('requires the prefixed full-model arm configuration')
    live=LiveMountCollisionWorld(planner.datastream.env,planner.usd_helper,planner.env_id)
    planner.motion_gen.update_world(live.world)
    def refresh(self):
        self._base_pose_in_env=torch.eye(4,device=self.datastream.device)
        self._env_to_base_transform=self._base_pose_in_env.clone();self._base_frame_is_identity=True
    def update(self):
        refresh(self)
        live.update(self.motion_gen.world_coll_checker,self.tensor_args)
    planner._refresh_base_frame_transform=types.MethodType(refresh,planner)
    planner.update_world=types.MethodType(update,planner)
    planner.b2arx_live_world=live
    planner.update_world()
    return live

class PlannerContractProbe:
    """Observe a real episode without changing controls; check cuRobo interfaces."""
    def __init__(self,env,output):
        import torch
        configure_warp_interop()
        from curobo.util import usd_helper
        from curobo.geom.sdf.world import WorldCollisionConfig,CollisionCheckerType
        from curobo.geom.sdf.world_mesh import WorldMeshCollision
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel,CudaRobotModelConfig
        from curobo.types.base import TensorDeviceType
        usd_helper._WARP_AVAILABLE=False
        helper=usd_helper.UsdHelper();helper.load_stage(env.scene.stage)
        self.live=LiveMountCollisionWorld(env,helper)
        self.tensor_args=TensorDeviceType();self.output=Path(output);self.rows=[];self.env=env
        config=WorldCollisionConfig.load_from_dict({'checker_type':CollisionCheckerType.MESH,'cache':{'mesh':256,'obb':256}},
                                                    self.live.world,self.tensor_args)
        self.checker=WorldMeshCollision(config)
        self.model=CudaRobotModel(CudaRobotModelConfig.from_robot_yaml_file(str(robot_config_file(output))))
        self.names=self.checker.get_obstacle_names()
        expected=[o.name for o,_,_ in self.live.bindings]
        if not set(expected).issubset(self.names):raise ValueError('cuRobo omitted live obstacles')
        self.ids=env.scene['robot'].find_joints(self.model.joint_names,preserve_order=True)[0]

    def record(self,frame):
        import json,torch
        from pxr import Usd,UsdPhysics,PhysxSchema
        from arx_door.envs.acceptance import as_numpy
        from arx_door.envs.recorders import _body_pose
        self.live.update(self.checker,self.tensor_args)
        robot=self.env.scene['robot']
        roots=[p for p in Usd.PrimRange(self.live.stage.GetPrimAtPath('/World/envs/env_0/R5a'))
               if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        self_collision={str(p.GetPath()):PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get() for p in roots}
        if robot.is_fixed_base or not self_collision or not all(self_collision.values()):
            raise ValueError('live robot must be floating with enabled self collision')
        q=as_numpy(robot.data.joint_pos)[:,self.ids]
        state=self.model.get_state(self.tensor_args.to_device(q))
        fk=matrix([*as_numpy(state.ee_position)[0],*as_numpy(state.ee_quaternion)[0]])
        measured=np.linalg.inv(_body_pose(self.env,'robot','R5a')[0])@_body_pose(self.env,'robot','link6')[0]
        error=float(np.abs(fk-measured).max())
        if error>2e-4:raise ValueError(f'cuRobo/R5a measured tool-frame mismatch: {error}')
        row=dict(frame=frame,fk_max_matrix_error=error,obstacle_count=len(self.live.bindings),
                 obstacles={o.name:o.pose for o,_,_ in self.live.bindings})
        self.rows.append(row)
        report=dict(scope='live cuRobo FK and collision-world synchronization; no planning/generation claim',
                    passed=True,kinematic_base='R5a',robot_config=str(self.output/'b2arx_curobo.yaml'),
                    no_pedestal_exemption=True,self_collision=self_collision,
                    floating=not robot.is_fixed_base,total_mass_kg=float(as_numpy(robot.data.default_mass).sum()),frames=self.rows)
        (self.output/'live_contract.json').write_text(json.dumps(report,indent=2))
