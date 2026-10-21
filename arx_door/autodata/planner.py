# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""ARX R5A planner profile for AutoData's official cuRobo SkillGen planner.

``register_planner_profile`` adds ``arx_r5a_door`` to
``autodata_interfaces.motion_planners.curobo.curobo_planner_cfg.PLANNER_PROFILES``;
``configs/autodata/env_profiles/arx_door.yaml`` selects it. The cuRobo robot model
is generated from the project's own URDF + XRDF in ``assets/arx_r5a``.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

PROFILE_NAME = 'arx_r5a_door'
DESCRIPTION = Path(__file__).resolve().parents[2] / 'assets' / 'arx_r5a'
URDF = DESCRIPTION / 'urdf' / 'r5a_cumotion.urdf'
XRDF = DESCRIPTION / 'xrdf' / 'r5a.xrdf'
# cuRobo's URDF parser resolves joint8 as a mimic of joint7; it is not an
# independently lockable planning DOF. The simulator still drives both fingers.
GRIPPER_JOINTS = ('joint7',)
GRIPPER_OPEN_M = 0.044  # URDF upper limit of both mirrored fingers
# Prims that are not obstacles for the arm: shared ground, cameras, cuRobo debug.
# A custom list replaces cuRobo's defaults, including its robot-root exclusion.
# Self-collision is handled by the robot model, not by static world meshes.
WORLD_IGNORE = ['/R5a/', '/World/ground', '/curobo', 'WristCamera', 'OverviewCamera', 'EgoCamera']


def robot_config_file(output_dir: str | Path | None = None) -> Path:
    """Write a cuRobo robot YAML (``{'robot_cfg': ...}``) converted from the ARX XRDF."""
    import yaml
    from curobo.types.file_path import ContentPath
    from curobo.util.xrdf_utils import convert_xrdf_to_curobo

    for path in (URDF, XRDF):
        if not path.is_file():
            raise FileNotFoundError(f'ARX planning model missing from assets: {path}')
    content = ContentPath(robot_urdf_absolute_path=str(URDF),
                          robot_xrdf_absolute_path=str(XRDF),
                          robot_asset_absolute_path=str(DESCRIPTION))
    converted = convert_xrdf_to_curobo(content, yaml.safe_load(XRDF.read_text()))
    robot_cfg = converted.get('robot_cfg', converted)
    kinematics = robot_cfg['kinematics']
    kinematics['urdf_path'] = str(URDF)
    kinematics['asset_root_path'] = str(DESCRIPTION)
    # Fixed base is bolted to BaseSupport: its coarse spheres protrude 17 mm
    # through that mounting plane. It cannot move during this closed-door
    # transit, and the per-obstacle audit found no other base/world conflict.
    # Exempt the fixed base from world queries only. cuRobo compensates this
    # world buffer in self_collision_buffer, preserving base/arm self checks.
    # Moving links and the physical simulator's colliders are unchanged.
    kinematics['collision_sphere_buffer']['base_link'] = -1.1
    if kinematics.get('ee_link') != 'link6':
        raise ValueError(f"XRDF tool frame must be link6, got {kinematics.get('ee_link')!r}")
    output = Path(output_dir or tempfile.gettempdir()) / 'arx_r5a_curobo.yml'
    output.write_text(yaml.safe_dump({'robot_cfg': robot_cfg}, sort_keys=False))
    return output


def arx_r5a_door_config():
    """cuRobo planner settings for transit to the knob; the skill segment is replayed, not planned."""
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    return CuroboPlannerCfg(
        robot_config_file=str(robot_config_file()),
        robot_name='arx_r5a',
        ee_link_name='link6',
        gripper_joint_names=list(GRIPPER_JOINTS),
        gripper_open_positions={name: GRIPPER_OPEN_M for name in GRIPPER_JOINTS},
        gripper_closed_positions={name: 0.0 for name in GRIPPER_JOINTS},
        hand_link_names=['link6', 'link7', 'link8'],
        grasp_gripper_open_val=GRIPPER_OPEN_M,
        # This task plans only the initial transit with the door closed. The
        # official rigid-object sync maps one child collider to the articulation
        # root pose, losing its local offset (e.g. the door-frame post). Retain
        # the USD geometry here; planning after opening would need linkwise sync.
        static_objects=['workcell', 'door'],
        world_ignore_substrings=list(WORLD_IGNORE),
        # SkillGen replays the recorded approach/grasp; the planner only bridges to it.
        approach_distance=0.0,
        retreat_distance=0.0,
        collision_activation_distance=0.01,
        max_planning_attempts=3,
        enable_finetune_trajopt=True,
        # ARX seeds are deterministic: no motion noise on planned transits.
        motion_noise_scale=0.0,
        visualize_spheres=False,
        visualize_plan=False,
        debug_planner=True,
    )


def register_planner_profile() -> None:
    """Make ``arx_r5a_door`` selectable from an environment profile; idempotent."""
    from autodata_interfaces.motion_planners.curobo.curobo_planner_cfg import PLANNER_PROFILES
    from curobo.util import usd_helper

    # The pinned runtime crashes inside Warp's quad-triangulation launch while
    # importing the door mesh (pilot_20261018/generated_baseline_02 traceback).
    # Select cuRobo's existing CPU triangulation fallback only. Motion planning
    # stays on CUDA; simulator meshes, physics and the official files are unchanged.
    usd_helper._WARP_AVAILABLE = False

    # Upstream counts planning failures, but checks the trial limit only after
    # an env.step(). A failure before the first action otherwise resets forever.
    # Surface that failure without changing the planner or accepting any data.
    from autodata_interfaces.motion_planners.curobo.curobo_planner import CuroboPlanner
    if not getattr(CuroboPlanner.plan_motion, '_arx_fail_fast', False):
        plan_motion = CuroboPlanner.plan_motion

        def checked_plan(self, *args, **kwargs):
            success = plan_motion(self, *args, **kwargs)
            if not success:
                from .planner_diagnostics import report_start_collisions
                report_start_collisions(self)
                raise RuntimeError('ARX transit planning failed before execution; see cuRobo debug log')
            return success

        checked_plan._arx_fail_fast = True
        CuroboPlanner.plan_motion = checked_plan

    PLANNER_PROFILES.setdefault(PROFILE_NAME, arx_r5a_door_config)
