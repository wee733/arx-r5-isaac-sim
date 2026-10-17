# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Build the ARX door task through Arena's ordinary environment builder.

Import this module before AppLauncher safely; call ``build_arena_environment``
only after AppLauncher inside the pinned AutoData container.
"""
from __future__ import annotations

import math
from pathlib import Path

from arx_door.skillgen.physics_profile import CFG_FIELDS, SOURCE_CPU_SCENE, configure_physics_profile
from .assets import load_asset_manifest

DEFAULT_ARM_STATE = (-7.822022212167212e-7, 1.0002118349075317, 1.49851655960083,
                     -0.0019079649355262518, -4.5365965206656256e-7, -1.1146544238727074e-6)


def reset_domain(env, env_ids) -> None:
    """Reset only the selected domain state; physical reset is a separate term."""
    import torch
    from arx_door.autodata.acceptance import acceptance_state
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    acceptance_state(env).reset(ids)
    if not hasattr(env, '_arx_terminal'):
        env._arx_terminal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._arx_terminal[ids] = False


def _display_gap(env):
    from arx_door.skillgen import mdp
    return mdp.door_gap_reached(env).squeeze(-1)


def _display_open(env):
    from arx_door.skillgen import mdp
    return mdp.door_joint_state(env)[:, 0] > math.radians(29.0)


def _domain_success(env):
    from arx_door.autodata.acceptance import domain_success
    return domain_success(env)


def _assert_environment_contract(cfg, profile: str, source_settings: dict) -> None:
    if not math.isclose(cfg.sim.dt, 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError('Arena physics dt must be 1/120 second')
    if cfg.decimation != 4 or cfg.sim.render_interval != 4:
        raise RuntimeError('Arena action and render rates must be 30 Hz')
    if cfg.scene.replicate_physics or not cfg.scene.filter_collisions:
        raise RuntimeError('ARX environments require separate physics and collision filtering')
    if cfg.scene.env_spacing < 3.0:
        raise RuntimeError('ARX environment spacing must be at least 3 metres')
    if profile == 'source_cpu':
        if cfg.sim.device != 'cpu' or not cfg.sim.use_fabric:
            raise RuntimeError('source_cpu requires CPU device and enabled fabric')
        for key in ('broadphaseType', 'enableGPUDynamics', 'enableCCD', 'solverType', 'timeStepsPerSecond'):
            if source_settings['values'][key] != SOURCE_CPU_SCENE[key]:
                raise RuntimeError(f'source_cpu manifest has incompatible {key}')
        backend = cfg.sim.physics
        for field, source_name in CFG_FIELDS.items():
            if getattr(backend, field) != source_settings['values'][source_name]:
                raise RuntimeError(f'Arena overwrote source_cpu {field}')
        if backend.solver_type != 1:
            raise RuntimeError('source_cpu requires the TGS solver')


def build_arena_environment(asset_manifest: str | Path | dict, num_envs: int,
                            profile: str = 'source_cpu'):
    """Return ``(env_id, cfg, env_kwargs)`` from ``ArenaEnvBuilder.build_registered``.

    This does not create or step a simulator. The generation entrypoint calls
    AutoData ``setup_env_config`` after merging its recorders, then gym.make.
    TaskProgress criteria only display progress; the registered success term
    is the terminal-waypoint domain predicate.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    if profile not in ('source_cpu', 'lab_default'):
        raise ValueError(f'unknown physics profile: {profile}')
    manifest = load_asset_manifest(asset_manifest)
    for name in ('robot', 'door', 'workcell'):
        if name not in manifest['assets']:
            raise ValueError(f'asset manifest is missing {name}')
    # Runtime imports intentionally follow AppLauncher, never import the old
    # DoorSkillGenEnvCfg/Mimic environment or task subtask configuration.
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs.mdp import JointPositionActionCfg, reset_scene_to_default
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.sim import PinholeCameraCfg, UsdFileCfg
    from isaaclab.utils import configclass
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import TaskBase
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
    from isaaclab_arena.utils.cameras import ArenaCameraCfg
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from arx_door.skillgen import mdp

    def spawn(name, contact=False, scale_multiplier=1.0):
        entry = manifest['assets'][name]
        return UsdFileCfg(usd_path=entry['usd_path'], activate_contact_sensors=contact,
                          scale=tuple(value * scale_multiplier if index < 2 else value
                                      for index, value in enumerate(entry['initial_pose']['scale'])))

    def init_state(name, cfg_type, **kwargs):
        pose = manifest['assets'][name]['initial_pose']
        return cfg_type.InitialStateCfg(pos=tuple(pose['position']),
                                       rot=tuple(pose['quaternion_xyzw']), **kwargs)

    robot_cfg = ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/R5a', spawn=spawn('robot', contact=True),
        init_state=init_state('robot', ArticulationCfg,
            joint_pos={**{f'joint{i + 1}': value for i, value in enumerate(DEFAULT_ARM_STATE)},
                       'joint7': 0.028, 'joint8': 0.028}, joint_vel={'.*': 0.0}),
        actuators={
            'arm': ImplicitActuatorCfg(joint_names_expr=['joint[1-6]'], effort_limit_sim=100.0,
                                      stiffness=None, damping=None),
            # USD preserves the joint7 drive and joint8 passive zero gains.
            'gripper': ImplicitActuatorCfg(joint_names_expr=['joint[78]'], effort_limit_sim=60.0,
                                          stiffness=None, damping=None),
        })
    door_cfg = ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/Door', spawn=spawn('door', contact=True),
        init_state=init_state('door', ArticulationCfg,
            joint_pos={'hinge_joint': 0.0, 'handle_joint': 0.0, 'latch_joint': 0.0},
            joint_vel={'.*': 0.0}), actuators={})
    background_cfg = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Workcell', spawn=spawn('workcell'),
                                  init_state=init_state('workcell', AssetBaseCfg))

    @configclass
    class WorkcellCfg:
        door = door_cfg
        workcell = background_cfg
        ground = None

    workcell_config = WorkcellCfg()
    if 'ground' in manifest['assets']:
        # One shared collider with its original material; expand only its X/Y
        # coverage for larger grids. A -1 collision group is global in Lab.
        coverage = max(1, math.ceil(math.sqrt(num_envs)))
        workcell_config.ground = AssetBaseCfg(
            prim_path='/World/ground', spawn=spawn('ground', scale_multiplier=coverage),
            init_state=init_state('ground', AssetBaseCfg), collision_group=-1)

    @configclass
    class RobotSceneCfg:
        robot = robot_cfg
        left_panel_contact = ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/R5a/link7', update_period=0.0, history_length=1,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Door/door_panel', '{ENV_REGEX_NS}/Door/door_handle'])
        right_panel_contact = ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/R5a/link8', update_period=0.0, history_length=1,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Door/door_panel', '{ENV_REGEX_NS}/Door/door_handle'])
        arm_contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/R5a/link[1-6]',
                                        update_period=0.0, history_length=1)

    camera = manifest['camera']
    camera_attrs = camera['attributes']

    @configclass
    class WristCameraRig(ArenaCameraCfg):
        wrist_camera = CameraCfg(
            prim_path=camera['prim_path'], width=camera['width'], height=camera['height'],
            data_types=['rgb'], update_period=0.0, update_latest_camera_pose=True,
            spawn=PinholeCameraCfg(focal_length=camera_attrs['focalLength'],
                horizontal_aperture=camera_attrs['horizontalAperture'],
                vertical_aperture=camera_attrs['verticalAperture'],
                horizontal_aperture_offset=camera_attrs['horizontalApertureOffset'],
                vertical_aperture_offset=camera_attrs['verticalApertureOffset'],
                clipping_range=tuple(camera_attrs['clippingRange']),
                focus_distance=camera_attrs['focusDistance']))

    @configclass
    class ActionsCfg:
        arm_action = JointPositionActionCfg(asset_name='robot',
            joint_names=[f'joint{i}' for i in range(1, 7)], scale=1.0,
            use_default_offset=False, preserve_order=True)
        gripper_action = mdp.MirroredGripperActionCfg(asset_name='robot')

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.robot_state)
        joint_vel = ObservationTermCfg(func=mdp.robot_velocity)
        actions = ObservationTermCfg(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiagnosticsCfg(ObservationGroupCfg):
        door_joint_pos = ObservationTermCfg(func=mdp.door_joint_state)
        finger_contacts = ObservationTermCfg(func=mdp.finger_contacts)
        arm_contacts = ObservationTermCfg(func=mdp.arm_contact_forces)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ObservationsCfg:
        policy = PolicyCfg()
        diagnostics = DiagnosticsCfg()

    @configclass
    class ResetEventsCfg:
        reset_scene = EventTermCfg(func=reset_scene_to_default, mode='reset',
                                   params={'reset_joint_targets': True})
        reset_arx_domain = EventTermCfg(func=reset_domain, mode='reset')

    class ArxDoorScene(Scene):
        def __init__(self):
            super().__init__()
        def get_scene_cfg(self):
            return workcell_config

    class ArxEmbodiment(EmbodimentBase):
        name = 'arx_r5a'
        def __init__(self):
            super().__init__(enable_cameras=True)
            self.scene_config = RobotSceneCfg()
            self.action_config = ActionsCfg()
            self.observation_config = ObservationsCfg()
            self.camera_config = WristCameraRig()
            self.camera_config.set_use_tiled_camera(False)

    class DoorTask(TaskBase):
        name = 'arx_door_knob_hold_open'
        def __init__(self):
            super().__init__(episode_length_s=240.0, task_description='Unlock and open the door while holding the knob.')
        def get_scene_cfg(self):
            return None
        def get_events_cfg(self):
            return ResetEventsCfg()
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=None,
                success=[CompletionCriteria(name='door_progress', predicate_sequence=[_display_gap, _display_open])])
        def get_mimic_env_cfg(self, arm_mode):
            raise NotImplementedError('AutoData task YAML owns subtasks; Arena mimic mode is disabled')
        def get_metrics(self):
            return []

    def configure(cfg):
        cfg.sim.dt = 1.0 / 120.0
        cfg.decimation = 4
        cfg.sim.render_interval = 4
        cfg.scene.replicate_physics = False
        cfg.scene.filter_collisions = True
        cfg.arx_wrist_mount = manifest['camera']['mount_transform']
        cfg.arx_asset_manifest = manifest
        cfg.behavior_family = 'knob_hold_open'
        cfg.physics_profile = profile
        cfg.arx_source_settings = manifest['source_settings']
        configure_physics_profile(cfg.sim, profile, manifest['source_settings'])
        # Arena's TaskProgress remains display-only. AutoData extracts this
        # atomic terminal predicate before clearing manager terminations.
        cfg.terminations.success = TerminationTermCfg(func=_domain_success)
        return cfg

    descriptor = IsaacLabArenaEnvironment(
        name=f'Arx-Door-AutoData-{profile}-v0', scene=ArxDoorScene(),
        embodiment=ArxEmbodiment(), task=DoorTask(), env_cfg_callback=configure,
        default_physics_backend=PhysicsBackend.PHYSX)
    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=num_envs, env_spacing=max(3.0, float(manifest['env_spacing_m'])),
        mimic=False, solve_relations=False, device='cpu' if profile == 'source_cpu' else 'cuda:0',
        disable_fabric=False, record_trajectories=False)
    env_id, cfg, kwargs = ArenaEnvBuilder(descriptor, builder_cfg).build_registered()
    _assert_environment_contract(cfg, profile, manifest['source_settings'])
    if type(cfg.scene.wrist_camera) is not CameraCfg:
        raise RuntimeError('ARX requires an ordinary independent CameraCfg per environment')
    for name in ('robot', 'door', 'workcell', 'wrist_camera', 'left_panel_contact', 'right_panel_contact', 'arm_contacts'):
        if not getattr(cfg.scene, name).prim_path.startswith('{ENV_REGEX_NS}/'):
            raise RuntimeError(f'ARX scene entity {name} is outside the per-environment namespace')
    return env_id, cfg, kwargs


__all__ = ['build_arena_environment', 'reset_domain']
