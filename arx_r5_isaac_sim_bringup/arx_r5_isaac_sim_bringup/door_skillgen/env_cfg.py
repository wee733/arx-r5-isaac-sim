# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the single-environment ARX door SkillGen task."""

from __future__ import annotations

from dataclasses import MISSING
import json
import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp


DEFAULT_ARM_STATE = (
    -7.822022212167212e-07,
    1.0002118349075317,
    1.49851655960083,
    -0.0019079649355262518,
    -4.5365965206656256e-07,
    -1.1146544238727074e-06,
)


@configclass
class DoorSceneCfg(InteractiveSceneCfg):
    """Bind Isaac Lab entities to a prepared, already-opened USD stage."""

    robot = ArticulationCfg(
        prim_path='/R5a',
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                **{f'joint{i + 1}': value for i, value in enumerate(DEFAULT_ARM_STATE)},
                'joint7': 0.028,
                'joint8': 0.028,
            },
            joint_vel={'.*': 0.0},
        ),
        actuators={
            'arm': ImplicitActuatorCfg(
                joint_names_expr=['joint[1-6]'],
                effort_limit_sim=100.0,
                # Keep the approved USD gains. PhysX converts angular USD
                # degree units to radians; copying the literal 2000/100 into
                # this tensor API would weaken both gains by 180/pi.
                stiffness=None,
                damping=None,
            ),
            'gripper': ImplicitActuatorCfg(
                joint_names_expr=['joint[78]'],
                effort_limit_sim=60.0,
                # joint8 is a passive PhysX mimic, not a second motor.
                # Preserve its zero gains and joint7's reviewed force drive.
                stiffness=None,
                damping=None,
            ),
        },
    )
    door = ArticulationCfg(
        prim_path='/World/Door',
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                'hinge_joint': 0.0,
                'handle_joint': 0.0,
                'latch_joint': 0.0,
            },
            joint_vel={'.*': 0.0},
        ),
        actuators={},
    )
    wrist_camera = CameraCfg(
        prim_path='/World/SkillGenWristCamera',
        spawn=None,
        width=640,
        height=480,
        data_types=['rgb'],
        # Recording explicitly renders and refreshes without advancing physics.
        # A positive period would retain the pre-synchronization RGB/pose cache.
        update_period=0.0,
        update_latest_camera_pose=True,
    )
    overview_camera = CameraCfg(
        prim_path='/World/OverviewCamera',
        spawn=None,
        width=960,
        height=720,
        data_types=['rgb'],
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
    )
    left_panel_contact = ContactSensorCfg(
        prim_path='/R5a/link7',
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=['/World/Door/door_panel', '/World/Door/door_handle'],
    )
    right_panel_contact = ContactSensorCfg(
        prim_path='/R5a/link8',
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=['/World/Door/door_panel', '/World/Door/door_handle'],
    )
    arm_contacts = ContactSensorCfg(
        prim_path='/R5a/link[1-6]',
        update_period=0.0,
        history_length=1,
    )


@configclass
class ActionsCfg:
    """Six absolute arm joints and one mirrored gripper command."""

    arm_action = JointPositionActionCfg(
        asset_name='robot',
        joint_names=[f'joint{index}' for index in range(1, 7)],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    gripper_action = mdp.MirroredGripperActionCfg(asset_name='robot')


@configclass
class ObservationsCfg:
    """Training-facing and diagnostic observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.robot_state)
        joint_vel = ObsTerm(func=mdp.robot_velocity)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraCfgGroup(ObsGroup):
        wrist = ObsTerm(
            func=mdp.image,
            params={
                'sensor_cfg': SceneEntityCfg('wrist_camera'),
                'data_type': 'rgb',
                'normalize': False,
            },
        )
        overview = ObsTerm(
            func=mdp.image,
            params={
                'sensor_cfg': SceneEntityCfg('overview_camera'),
                'data_type': 'rgb',
                'normalize': False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiagnosticsCfg(ObsGroup):
        door_joint_pos = ObsTerm(func=mdp.door_joint_state)
        camera_panel_clearance = ObsTerm(func=mdp.camera_panel_clearance)
        finger_contacts = ObsTerm(func=mdp.finger_contacts)
        arm_contacts = ObsTerm(func=mdp.arm_contact_forces)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        knob_interaction = ObsTerm(func=mdp.door_gap_reached)
        edge_interaction = ObsTerm(func=mdp.door_opened)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: CameraCfgGroup = CameraCfgGroup()
    diagnostics: DiagnosticsCfg = DiagnosticsCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Generation stops only through the externally evaluated success term."""

    success = DoneTerm(func=mdp.door_success)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset both physical articulations, not just manager bookkeeping."""

    reset_scene = EventTermCfg(
        func=reset_scene_to_default,
        mode='reset',
        params={'reset_joint_targets': True},
    )


@configclass
class DoorSkillGenEnvCfg(ManagerBasedRLEnvCfg, MimicEnvCfg):
    """Native ManagerBasedRLMimicEnv configuration for ARX door opening."""

    scene: DoorSceneCfg = DoorSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=False,
        filter_collisions=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    events: EventsCfg = EventsCfg()
    curriculum = None

    scene_path: str = ''
    cuboids_path: str = ''
    urdf_path: str = ''
    source_seed_path: str = ''

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.seed = 0
        self.num_rerenders_on_reset = 2
        self.episode_length_s = 90.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        # GPU articulation visuals require Fabric. The recording camera is an
        # independent static-parent prim tracked from live link6, not stale USD.
        self.sim.use_fabric = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.num_envs = 1

        self.datagen_config.name = 'arx_door_skillgen'
        self.datagen_config.generation_guarantee = False
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 1
        self.datagen_config.generation_select_src_per_subtask = False
        # Plan to each skill's measured entrance pose before replaying its
        # commands. The first edge command already starts moving inward.
        self.datagen_config.generation_transform_first_robot_pose = True
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 1
        self.datagen_config.seed = 0
        self.datagen_config.use_skillgen = True
        self.subtask_configs['link6'] = [
            SubTaskConfig(
                object_ref='knob',
                subtask_term_signal='knob_interaction',
                selection_strategy='nearest_neighbor_object',
                selection_strategy_kwargs={'nn_k': 1},
                subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                action_noise=0.0,
                num_interpolation_steps=1,
                num_fixed_steps=0,
            ),
            SubTaskConfig(
                object_ref='panel',
                subtask_term_signal='edge_interaction',
                selection_strategy='nearest_neighbor_object',
                selection_strategy_kwargs={'nn_k': 1},
                subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                action_noise=0.0,
                num_interpolation_steps=1,
                num_fixed_steps=0,
            ),
        ]

        self.scene_path = os.environ.get('ARX_DOOR_SKILLGEN_SCENE', '')
        self.cuboids_path = os.environ.get('ARX_DOOR_SKILLGEN_CUBOIDS', '')
        self.urdf_path = os.environ.get('ARX_DOOR_SKILLGEN_URDF', '')
        self.source_seed_path = os.environ.get('ARX_DOOR_SKILLGEN_SEED', '')
        manifest_path = os.environ.get('ARX_DOOR_SKILLGEN_MANIFEST')
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            pose = manifest['placement']
            xyzw = pose['quaternion_xyzw']
            self.scene.robot.init_state.pos = tuple(pose['position'])
            self.scene.robot.init_state.rot = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
