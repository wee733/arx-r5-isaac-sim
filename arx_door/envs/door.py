# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""The ARX door environment: one Arena definition for AutoData and policy evaluation.

AutoData reaches it through ``arx_door.autodata.register`` (the official
``generate_dataset.py --external_callback``) with ``action='ik_abs'``: an absolute
link6 pose in the robot root frame plus one gripper value, matching the official
``absolute_pose`` embodiment. Policy evaluation builds it with ``action='joint_pos'``.

Importing this module before AppLauncher is safe; call :func:`build` afterwards.
"""
from __future__ import annotations

import math
from pathlib import Path

from .assets import load_asset_manifest
from .physics import CFG_FIELDS, SOURCE_CPU_SCENE, configure_physics_profile

ENV_NAME = 'Arx-Door-KnobHoldOpen-v0'
ACTION_MODES = ('ik_abs', 'joint_pos')
# Action layout of ik_abs: [x, y, z, qx, qy, qz, qw] of link6 in the robot root frame,
# then the mirrored gripper command. configs/autodata/embodiments/arx_r5a.yaml mirrors it.
IK_ABS_ACTION_DIM = 8
JOINT_POS_ACTION_DIM = 7
DEFAULT_ARM_STATE = (-7.822022212167212e-7, 1.0002118349075317, 1.49851655960083,
                     -0.0019079649355262518, -4.5365965206656256e-7, -1.1146544238727074e-6)


def reset_domain(env, env_ids) -> None:
    """Reset only the per-episode acceptance state; physical reset is a separate term."""
    from .acceptance import reset_acceptance
    reset_acceptance(env, None if env_ids is None else env_ids)


def _display_gap(env):
    from . import mdp
    return mdp.door_gap_reached(env).squeeze(-1)


def _display_open(env):
    from . import mdp
    return mdp.door_joint_state(env)[:, 0] > math.radians(29.0)


def _domain_success(env):
    from .acceptance import domain_success
    return domain_success(env)


def _assert_environment_contract(cfg, physics: str, source_settings: dict) -> None:
    if not math.isclose(cfg.sim.dt, 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError('door physics dt must be 1/120 second')
    if cfg.decimation != 4 or cfg.sim.render_interval != 4:
        raise RuntimeError('door action and render rates must be 30 Hz')
    if cfg.scene.replicate_physics or not cfg.scene.filter_collisions:
        raise RuntimeError('ARX environments require separate physics and collision filtering')
    if cfg.scene.env_spacing < 3.0:
        raise RuntimeError('ARX environment spacing must be at least 3 metres')
    if physics == 'source_cpu':
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


def build(asset_manifest: str | Path | dict, num_envs: int, *, action: str = 'ik_abs',
          physics: str = 'source_cpu', name: str = ENV_NAME, ego_camera: dict | None = None):
    """Register the door env with gym and return ``(env_id, cfg, env_kwargs)``.

    Nothing is created or stepped. ``ego_camera`` is an optional
    :func:`arx_door.scene.ego_camera.ego_mount` spec placed on the robot base.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    if action not in ACTION_MODES:
        raise ValueError(f'unknown action mode {action!r}; expected one of {ACTION_MODES}')
    if physics not in ('source_cpu', 'lab_default'):
        raise ValueError(f'unknown physics profile: {physics}')
    manifest = load_asset_manifest(asset_manifest)
    initial_arm = manifest.get('initial_arm_state_rad', DEFAULT_ARM_STATE)
    if len(initial_arm) != 6 or not all(math.isfinite(float(value)) for value in initial_arm):
        raise ValueError('initial_arm_state_rad must contain six finite joint angles')
    for asset in ('robot', 'door', 'workcell'):
        if asset not in manifest['assets']:
            raise ValueError(f'asset manifest is missing {asset}')
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.controllers import DifferentialIKControllerCfg
    from isaaclab.envs.mdp import (DifferentialInverseKinematicsActionCfg, JointPositionActionCfg,
                                   reset_scene_to_default)
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
    from . import mdp

    def spawn(asset, contact=False, scale_multiplier=1.0):
        entry = manifest['assets'][asset]
        return UsdFileCfg(usd_path=entry['usd_path'], activate_contact_sensors=contact,
                          scale=tuple(value * scale_multiplier if index < 2 else value
                                      for index, value in enumerate(entry['initial_pose']['scale'])))

    def init_state(asset, cfg_type, **kwargs):
        pose = manifest['assets'][asset]['initial_pose']
        return cfg_type.InitialStateCfg(pos=tuple(pose['position']),
                                       rot=tuple(pose['quaternion_xyzw']), **kwargs)

    def pinhole(attrs):
        return PinholeCameraCfg(
            focal_length=attrs['focalLength'], horizontal_aperture=attrs['horizontalAperture'],
            vertical_aperture=attrs['verticalAperture'],
            horizontal_aperture_offset=attrs['horizontalApertureOffset'],
            vertical_aperture_offset=attrs['verticalApertureOffset'],
            clipping_range=tuple(attrs['clippingRange']), focus_distance=attrs['focusDistance'])

    robot_cfg = ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/R5a', spawn=spawn('robot', contact=True),
        init_state=init_state('robot', ArticulationCfg,
            joint_pos={**{f'joint{i + 1}': value for i, value in enumerate(initial_arm)},
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

    wrist = manifest['camera']
    overview = manifest.get('overview_camera')

    @configclass
    class CameraRig(ArenaCameraCfg):
        wrist_camera = CameraCfg(
            prim_path=wrist['prim_path'], width=wrist['width'], height=wrist['height'],
            data_types=['rgb'], update_period=0.0, update_latest_camera_pose=True,
            spawn=pinhole(wrist['attributes']))
        overview_camera = None if overview is None else CameraCfg(
            prim_path=overview['prim_path'], width=overview['width'], height=overview['height'],
            data_types=['rgb'], update_period=0.0, update_latest_camera_pose=True,
            spawn=pinhole(overview['attributes']),
            offset=CameraCfg.OffsetCfg(pos=tuple(overview['position']),
                                       rot=tuple(overview['quaternion_xyzw']), convention='opengl'))
        ego_camera = None

    camera_rig = CameraRig()
    if ego_camera is not None:
        from arx_door.scene.ego_camera import ego_camera_cfg, ego_mount
        from .compat import quaternion_from_xyzw
        base = manifest['assets']['robot']['initial_pose']
        camera_rig.ego_camera = ego_camera_cfg(
            base, {**ego_mount(ego_camera), 'prim_path': '{ENV_REGEX_NS}/EgoCamera'},
            quaternion_from_xyzw=quaternion_from_xyzw)

    gripper = mdp.MirroredGripperActionCfg(asset_name='robot')
    if action == 'ik_abs':
        @configclass
        class ActionsCfg:
            arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name='robot', joint_names=[f'joint{i}' for i in range(1, 7)],
                body_name='link6', scale=1.0,
                controller=DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False,
                                                       ik_method='dls'))
            gripper_action = gripper
    else:
        @configclass
        class ActionsCfg:
            arm_action = JointPositionActionCfg(asset_name='robot',
                joint_names=[f'joint{i}' for i in range(1, 7)], scale=1.0,
                use_default_offset=False, preserve_order=True)
            gripper_action = gripper

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.robot_state)
        joint_vel = ObservationTermCfg(func=mdp.robot_velocity)
        actions = ObservationTermCfg(func=mdp.last_action)
        # Read by the official absolute_pose embodiment (pose_obs_keys).
        eef_pos = ObservationTermCfg(func=mdp.eef_pos_base)
        eef_quat = ObservationTermCfg(func=mdp.eef_quat_base)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskTermsCfg(ObservationGroupCfg):
        # Read by the official Datastream; names match configs/autodata/tasks.
        knob_interaction = ObservationTermCfg(func=mdp.knob_macro_completed)
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
        subtask_terms = SubtaskTermsCfg()
        diagnostics = DiagnosticsCfg()

    @configclass
    class ResetEventsCfg:
        reset_scene = EventTermCfg(func=reset_scene_to_default, mode='reset',
                                   params={'reset_joint_targets': True})
        reset_arx_domain = EventTermCfg(func=reset_domain, mode='reset')

    class ArxDoorScene(Scene):
        def get_scene_cfg(self):
            return workcell_config

    class ArxEmbodiment(EmbodimentBase):
        name = 'arx_r5a'
        def __init__(self):
            super().__init__(enable_cameras=True)
            self.scene_config = RobotSceneCfg()
            self.action_config = ActionsCfg()
            self.observation_config = ObservationsCfg()
            self.camera_config = camera_rig
            self.camera_config.set_use_tiled_camera(False)

    class DoorTask(TaskBase):
        name = 'arx_door_knob_hold_open'
        def __init__(self):
            super().__init__(episode_length_s=240.0,
                             task_description='Unlock and open the door while holding the knob.')
        def get_scene_cfg(self):
            return None
        def get_events_cfg(self):
            return ResetEventsCfg()
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=None,
                success=[CompletionCriteria(name='door_progress', predicate_sequence=[_display_gap, _display_open])])
        def get_mimic_env_cfg(self, arm_mode):
            raise NotImplementedError('the AutoData task YAML owns subtasks; Arena mimic mode is disabled')
        def get_metrics(self):
            return []

    def configure(cfg):
        cfg.sim.dt = 1.0 / 120.0
        cfg.decimation = 4
        cfg.sim.render_interval = 4
        cfg.scene.replicate_physics = False
        cfg.scene.filter_collisions = True
        cfg.arx_wrist_mount = wrist['mount_transform']
        cfg.arx_asset_manifest = manifest
        cfg.behavior_family = 'knob_hold_open'
        cfg.physics_profile = physics
        cfg.arx_action_mode = action
        cfg.arx_source_settings = manifest['source_settings']
        configure_physics_profile(cfg.sim, physics, manifest['source_settings'])
        # Arena's TaskProgress is display-only; AutoData reads this predicate.
        cfg.terminations.success = TerminationTermCfg(func=_domain_success)
        return cfg

    descriptor = IsaacLabArenaEnvironment(
        name=name, scene=ArxDoorScene(), embodiment=ArxEmbodiment(), task=DoorTask(),
        env_cfg_callback=configure, default_physics_backend=PhysicsBackend.PHYSX)
    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=num_envs, env_spacing=max(3.0, float(manifest['env_spacing_m'])),
        mimic=False, solve_relations=False, device='cpu' if physics == 'source_cpu' else 'cuda:0',
        disable_fabric=False, record_trajectories=False)
    env_id, cfg, kwargs = ArenaEnvBuilder(descriptor, builder_cfg).build_registered()
    _assert_environment_contract(cfg, physics, manifest['source_settings'])
    if type(cfg.scene.wrist_camera) is not CameraCfg:
        raise RuntimeError('ARX requires an ordinary independent CameraCfg per environment')
    for entity in ('robot', 'door', 'workcell', 'wrist_camera', 'left_panel_contact',
                   'right_panel_contact', 'arm_contacts'):
        if not getattr(cfg.scene, entity).prim_path.startswith('{ENV_REGEX_NS}/'):
            raise RuntimeError(f'ARX scene entity {entity} is outside the per-environment namespace')
    return env_id, cfg, kwargs


__all__ = ['ACTION_MODES', 'ENV_NAME', 'IK_ABS_ACTION_DIM', 'JOINT_POS_ACTION_DIM', 'build', 'reset_domain']
