"""Official SkillGen with a fixed episode frame and live floating-mount commands.

The upstream algorithm, source pool, object retargeting and cuRobo solver remain
unchanged. This adapter owns coordinates and connects queued actions to the
project's existing fully instrumented physics collector.
"""
import asyncio
from pathlib import Path
import numpy as np
import torch
from autodata_interfaces.embodiments.absolute_pose_embodiment_adapter import AbsolutePoseEmbodimentAdapter
from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
from arx_door.envs.recorders import _body_pose
from arx_door.autodata.source import pose_matrix
from arx_door.autodata.b2arx_planner import LiveMountCollisionWorld, robot_config_file, configure_warp_interop


class FixedEpisodeEmbodiment(AbsolutePoseEmbodimentAdapter):
    def latch(self, env):
        self.fixed = torch.as_tensor(_body_pose(env, 'robot', 'R5a'), dtype=torch.float32, device=env.device)

    def get_reference_frame_pose(self, env_ids=None):
        # Datastream first removes env_origins from object poses. Its reference
        # frame must therefore also be env-local; fixed remains world-space for
        # live-mount action conversion and the motion planner.
        idx = slice(None) if env_ids is None else env_ids
        reference = self.fixed[idx].clone()
        reference[:, :3, 3] -= self.env.scene.env_origins[idx].to(reference)
        return reference

    def get_eef_poses(self, env_ids=None):
        idx = slice(None) if env_ids is None else env_ids
        world = torch.as_tensor(_body_pose(self.env, 'robot', 'link6'), dtype=self.fixed.dtype, device=self.fixed.device)
        return {'link6': torch.linalg.inv(self.fixed[idx]) @ world[idx]}

    def target_eef_pose_to_action(self, target_eef_pose_dict, passthrough_action_dict, action_noise_dict=None, env_id=0):
        live = torch.as_tensor(_body_pose(self.env, 'robot', 'R5a')[env_id], dtype=self.fixed.dtype, device=self.fixed.device)
        targets = {k: torch.linalg.inv(live) @ self.fixed[env_id] @ v for k, v in target_eef_pose_dict.items()}
        return super().target_eef_pose_to_action(targets, passthrough_action_dict, action_noise_dict, env_id)


class FloatingPlanner(CuroboPlanner):
    def _initialize_static_world(self):
        self.live_world = LiveMountCollisionWorld(self.env, self.usd_helper, self.env_id)
        self._static_world_config = self.live_world.world
        self.motion_gen.update_world(self._static_world_config)

    def update_world(self):
        live = torch.as_tensor(_body_pose(self.env, 'robot', 'R5a')[self.env_id], dtype=torch.float32, device=self.datastream.device)
        fixed = self.datastream.embodiment_adapter.fixed[self.env_id]
        self._base_pose_in_env = torch.linalg.inv(fixed) @ live
        self._env_to_base_transform = torch.linalg.inv(self._base_pose_in_env)
        self._base_frame_is_identity = False
        self.live_world.update(self.motion_gen.world_coll_checker, self.tensor_args)


def generate(env, source, demo, output, step, phase_names, seed):
    import h5py, yaml, json
    from isaaclab.managers import TerminationTermCfg
    from autodata_interfaces.datastream import Datastream
    from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
    from autodata_core.algorithms import SkillGen
    from autodata_core.data_generator import DataGenerator
    from arx_door.envs.acceptance import acceptance_state
    from curobo.util import usd_helper
    root = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((root/'configs/autodata/embodiments/b2arx.yaml').read_text()); spec.pop('type')
    embodiment = FixedEpisodeEmbodiment(**spec); embodiment.latch(env)
    task = TaskDescriptor.from_yaml(str(root/'configs/autodata/tasks/arx_door_knob_hold_open.yaml'))
    task.get_generation_policy().seed = seed
    stream = Datastream(env=env, task_descriptor=task, embodiment_adapter=embodiment,
                        source_dataset_path=str(source), select_demo_keys=[demo], uses_start_signals=True)
    configure_warp_interop(); usd_helper._WARP_AVAILABLE = False
    cfg = CuroboPlannerCfg(robot_config_file=str(robot_config_file(output)), robot_name='b2arx',
        ee_link_name='R5a_link6', gripper_joint_names=['R5a_joint7','R5a_joint8'],
        gripper_open_positions={'R5a_joint7':.044,'R5a_joint8':.044},
        gripper_closed_positions={'R5a_joint7':0.,'R5a_joint8':0.},
        hand_link_names=['R5a_link6','R5a_link7','R5a_link8'], grasp_gripper_open_val=.044,
        approach_distance=0.,retreat_distance=0.,collision_activation_distance=.01,
        max_planning_attempts=3,enable_finetune_trajopt=True,motion_noise_scale=0.,
        motion_step_size=.006,time_dilation_factor=.2,visualize_spheres=False,visualize_plan=False,debug_planner=True)
    planner = getattr(env, '_b2arx_generation_planner', None)
    if planner is None:
        planner = FloatingPlanner(datastream=stream, config=cfg, env_id=0)
        env._b2arx_generation_planner = planner
    else:
        # Same physical scene and robot; refresh live obstacle poses below.
        # Reuse CUDA allocations across serial episodes, not stale transforms.
        planner.datastream = stream
    algorithm = SkillGen(motion_planners={0:planner})
    generator = DataGenerator(datastream=stream, algorithm=algorithm)
    with h5py.File(source) as f:
        ep = f['data/'+demo]
        source_targets = ep['obs/datagen_info/target_eef_pose/link6'][:]
        source_eef = ep['obs/datagen_info/eef_pose/link6'][:]
        source_object = ep['obs/datagen_info/object_pose/door'][:]
        phases = ep['obs/diagnostics/phase_id'][:] if 'obs/diagnostics/phase_id' in ep else None
        source_names = json.loads(ep.attrs['phase_names'])
        raw_dataset = str(ep.attrs['raw_dataset'])
    # Packed source may keep phase ids in a separate original source run.
    if phases is None:
        with h5py.File(raw_dataset) as f:
            phases=f['data/demo_0/obs/diagnostics/phase_id'][:]
    start = int(stream.source_pool.subtask_boundaries['link6'][0][0][0])
    delta = stream.get_object_poses([0])['door'][0].cpu().numpy() @ np.linalg.inv(source_object[start])
    # Upstream prepends the measured EEF at the first skill boundary.
    transformed = delta @ np.concatenate([source_eef[start:start+1],source_targets[start:]])
    skill_phases=np.concatenate([phases[start:start+1],phases[start:]])
    index = 0; skill_index = 0; matched = 0
    original_plan = algorithm.plan_subtask_trajectory
    state = {'motion': True}
    def track(**kwargs):
        result=original_plan(**kwargs)
        if result is not None:state['motion']=result[1]
        return result
    algorithm.plan_subtask_trajectory=track
    loop=asyncio.get_event_loop()
    resets=asyncio.Queue();actions=asyncio.Queue()
    term=TerminationTermCfg(func=lambda e:torch.as_tensor(acceptance_state(e).accepted(),device=e.device))
    job=loop.create_task(generator.generate(0,term,env_reset_queue=resets,env_action_queue=actions,export_demo=False))
    try:
        while not job.done():
            # Match the upstream env_loop contract: pump the async generator,
            # then step Kit/PhysX outside the running asyncio event loop.
            loop.run_until_complete(asyncio.sleep(0))
            if not resets.empty():
                resets.get_nowait()
                if index:raise RuntimeError('unexpected mid-episode reset')
                resets.task_done()
            if not actions.empty():
                _, action=actions.get_nowait(); a=action.cpu().numpy()
                world=_body_pose(env,'robot','R5a')[0]@pose_matrix(a[:3],a[3:7])
                if state['motion']:
                    phase='transit_pregrasp'
                else:
                    fixed=np.linalg.inv(embodiment.fixed[0].cpu().numpy())@world
                    if skill_index>=len(transformed) or not np.allclose(fixed,transformed[skill_index],atol=2e-5):
                        raise ValueError(f'official skill/source alignment mismatch at {skill_index}')
                    phase=source_names[int(skill_phases[skill_index])];skill_index+=1;matched+=1
                step(world,float(a[7]),phase)
                index+=1
                if index%100==0:print('AUTODATA_PROGRESS',index,phase,flush=True)
                actions.task_done()
        result=job.result()
        if not result.success or matched!=len(transformed):raise RuntimeError(f'official generation failed or incomplete skill: {matched}/{len(transformed)}')
    finally:
        if not job.done():
            job.cancel()
            try:loop.run_until_complete(job)
            except asyncio.CancelledError:pass
    return dict(algorithm='official AutoData SkillGen',source=str(source),demo=demo,
        fixed_world_mount=embodiment.fixed[0].cpu().numpy().tolist(),skill_transform=delta.tolist(),
        matched_skill_frames=matched,source_start=start,generated_frames=index,
        planner_obstacles=len(planner.live_world.bindings),no_pedestal_exemption=True)
