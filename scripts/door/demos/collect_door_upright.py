#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Try a horizontal door-edge grasp near knob height with viewpoint guards.

Use a fresh door_teach_session with camera clearance and arm-contact monitoring.
The virtual camera has no physical housing. Trials remain pending visual review.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door import paths  # noqa: E402

import numpy as np
from scipy.spatial.transform import Rotation
from arx_door.kinematics import ArmKinematics
from arx_door.runtime import (
    planning_urdf, runtime_provenance, usd_environment, validate_runtime,
)


def measured_planning_joints(state):
    """Read named arm and finger coordinates from the same teacher snapshot."""
    names = state.get('joint_names', [])
    positions = np.asarray(state.get('joints'), dtype=np.float64)
    required = [f'joint{index}' for index in range(1, 9)]
    if (positions.shape != (len(names),) or not np.isfinite(positions).all()
            or any(names.count(name) != 1 for name in required)):
        raise ValueError('snapshot must contain finite, uniquely named joint1..joint8 positions')
    ordered = positions[[names.index(name) for name in required]]
    return ordered[:6], ordered[6:8]


def native_plan_request(state, target_joints, cuboids):
    """Keep native auxiliary finger geometry at the observed articulation state."""
    arm, fingers = measured_planning_joints(state)
    return {'start_positions': arm.tolist(), 'target_joints': target_joints.tolist(),
            'gripper_joint_positions': fingers.tolist(),
            'gripper_state_source': 'measured_joint_state', 'cuboids': cuboids,
            'time_dilation_factor': .5, 'sample_dt': 1/30}


def unlock_feedback_policy(command_continuity=False):
    """Describe the bounded tool commands, independently of physical knob travel."""
    return {
        'schema': 'arx-door-measured-unlock-feedback-v3',
        'normal_tool_angles_deg': [45, 47],
        'normal_durations_s': [3, .8],
        'recovery_tool_angles_deg': [49, 51],
        'recovery_duration_s': .8,
        'tool_command_cap_deg': 51,
        'required_handle_angle_deg': 43.5,
        'required_latch_m': .0154,
        'minimum_finger_knob_contact_N': 3,
        'additional_confirmation_motion': False,
        'success_feedback': 'post_command_measured_handle_and_latch',
        'recovery_status': 'experimental_not_physically_validated',
        'required_recovery_progress': 'strict_increase_in_both_measured_handle_and_latch',
        'scope': 'fresh_expert_acquisition',
        'command_continuity': {
            'enabled': bool(command_continuity),
            'status': 'experimental_not_physically_validated' if command_continuity else 'disabled',
            'first_45_interpolation_start': 'measured',
            'subsequent_unlock_interpolation_start': 'previous_commanded' if command_continuity else 'measured',
        },
    }


def measured_knob_unlocked(state):
    """Use current measured feedback; a command or earlier peak is insufficient."""
    joints = np.asarray(state.get('door_joints'), dtype=float)
    if joints.shape != (3,) or not np.isfinite(joints).all():
        raise ValueError('unlock feedback requires three finite measured door joints')
    return bool(np.rad2deg(joints[1]) >= 43.5 and joints[2] >= .0154)


def unlock_feedback_progress(previous, current):
    """Require positive transmission before another experimental tool increment."""
    measured_knob_unlocked(previous)  # Validate the actual measurement buffer.
    measured_knob_unlocked(current)
    before = np.asarray(previous['door_joints'], dtype=float)
    after = np.asarray(current['door_joints'], dtype=float)
    return bool(after[1] > before[1] and after[2] > before[2])


def unlock_knob_with_feedback(grasp, pose, contact, *, audit=None, command_continuity=False):
    """Keep the original 45/47 path, then at most two measured-feedback retries.

    The teacher retains its strict IK, speed, camera and non-finger guards.
    Each returned state must still show both intended knob contacts. Positive
    angle and latch progress are also required: contact alone does not prove
    torque transmission. Extra tool rotation remains experimental and changes
    neither the physical handle limit nor the grasp position.
    """
    if audit is None:
        audit = {}
    audit.update({'attempted_tool_angles_deg': [], 'confirmation_tool_angles_deg': [],
                  'stopped_tool_angle_deg': None, 'success': False,
                  'stop_reason': None, 'steps': [],
                  'command_continuity_enabled': bool(command_continuity)})
    unlocked = grasp.copy()

    def move(angle, duration):
        unlocked[:3, :3] = grasp[:3, :3] @ Rotation.from_euler(
            'x', -angle, degrees=True).as_matrix()
        audit['attempted_tool_angles_deg'].append(angle)
        audit['stopped_tool_angle_deg'] = angle
        continuation = command_continuity and angle != 45
        try:
            if continuation:
                state = pose(unlocked, 'unlock_round_knob', 0, duration,
                             interpolation_start='previous_commanded')
            else:
                state = pose(unlocked, 'unlock_round_knob', 0, duration)
            measured = measured_knob_unlocked(state)
        except Exception:
            audit['stop_reason'] = 'motion_or_feedback_error'
            raise
        audit['steps'].append({'tool_angle_deg': angle, 'duration_s': duration,
                               'interpolation_start': 'previous_commanded' if continuation else 'measured',
                               'confirmation': False,
                               'measured_handle_angle_deg': float(np.rad2deg(state['door_joints'][1])),
                               'measured_latch_m': float(state['door_joints'][2]),
                               'measured_unlock_thresholds_passed': measured})
        try:
            contact(state, 0)
        except Exception:
            audit['stop_reason'] = 'contact_lost'
            raise
        return state

    initial = move(45, 3)
    state = move(47, .8)
    if measured_knob_unlocked(state):
        # A second same-target pose is a new Cartesian motion from measured
        # joints, not a dwell at the previous joint command; it can unload the
        # grasp. Preserve the proven post-command gate and subsequent pull.
        audit['success'] = True
        audit['stop_reason'] = 'unlocked_by_post_command_feedback'
        return state, unlocked
    if not unlock_feedback_progress(initial, state):
        audit['stop_reason'] = 'normal_sequence_no_measured_progress'
        raise RuntimeError('knob unlock recovery stopped: normal 45/47 sequence '
                           'did not improve both measured handle and latch')
    for angle in [49, 51]:
        previous = state
        state = move(angle, .8)
        delta = np.asarray(state['door_joints']) - np.asarray(previous['door_joints'])
        audit['steps'][-1].update({'measured_handle_delta_deg': float(np.rad2deg(delta[1])),
                                  'measured_latch_delta_m': float(delta[2])})
        if not unlock_feedback_progress(previous, state):
            audit['stop_reason'] = 'recovery_no_measured_progress'
            raise RuntimeError(f'knob unlock recovery stopped at {angle} degrees: '
                               'measured handle or latch stalled or regressed')
        if measured_knob_unlocked(state):
            audit['success'] = True
            audit['stop_reason'] = 'unlocked_by_post_command_feedback'
            return state, unlocked
    audit['stop_reason'] = 'command_cap_exhausted'
    raise RuntimeError('knob rotation/latch retraction did not reach unlock threshold '
                       'after bounded recovery to 51 degrees')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-output', type=Path, required=True)
    parser.add_argument('--name', default='candidate_000')
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--behavior-family', choices=['edge_regrasp', 'knob_hold_open'],
                        default='edge_regrasp')
    parser.add_argument('--final-angle', type=float, default=30.0)
    parser.add_argument('--edge-height', type=float, default=1.01)
    parser.add_argument('--edge-yaw', type=float, default=165.0)
    parser.add_argument('--edge-pregrasp-distance', type=float, default=.18,
                        help='Wrist distance along the approach axis from the door edge, metres.')
    parser.add_argument('--retreat-distance', type=float, default=.08,
                        help='Wrist retreat after knob release, metres (0.02 to 0.15).')
    parser.add_argument('--edge-x', type=float, default=.743,
                        help='Panel-local X coordinate of the grasp target (0.68 to 0.75 m).')
    parser.add_argument('--urdf', type=Path,
                        help='ARX URDF; defaults to ARX_DESCRIPTION_DIR or workspace discovery.')
    parser.add_argument('--unlock-command-continuity', action='store_true',
                        help='Experimental: continue 47/49/51 degree unlock commands from the '
                             'previous completed pose target; first45 remains measured.')
    args = parser.parse_args()
    if not 15 < args.final_angle <= 35:
        parser.error('pilot final angle must be above 15 and at most 35 degrees')
    if not np.isfinite([args.edge_height, args.edge_yaw,
                        args.edge_pregrasp_distance]).all() or args.edge_pregrasp_distance <= .125:
        parser.error('edge parameters must be finite and pregrasp distance above 0.125 m')
    if (not np.isfinite([args.retreat_distance, args.edge_x]).all()
            or not .02 <= args.retreat_distance <= .15 or not .68 <= args.edge_x <= .75):
        parser.error('retreat distance must be 0.02..0.15 m and edge X 0.68..0.75 m')
    urdf = planning_urdf(args.urdf)
    kin = ArmKinematics(urdf)
    collector_runtime = runtime_provenance()
    collector_profile = validate_runtime(collector_runtime)
    output = args.session_output.resolve()
    results = output / f'{args.name}_commands'
    results.mkdir(exist_ok=False)
    expert_parameters = {
        'behavior_family': args.behavior_family,
        'edge_height_m': args.edge_height,
        'edge_yaw_deg': args.edge_yaw,
        'edge_pregrasp_distance_m': args.edge_pregrasp_distance,
        'retreat_distance_m': args.retreat_distance,
        'edge_x_m': args.edge_x,
        'final_angle_deg': args.final_angle,
        'urdf_path': str(urdf),
        'unlock_tool_command_cap_deg': 51,
        'unlock_command_continuity': args.unlock_command_continuity,
        'unlock_feedback_policy': unlock_feedback_policy(args.unlock_command_continuity),
    }
    command_manifest = {
        'schema': 'arx-door-expert-commands-v1',
        'name': args.name,
        'session_output': str(output),
        'expert_parameters': expert_parameters,
        'collector_runtime_profile': collector_profile,
        'collector_runtime': collector_runtime,
        'review_status': 'pending',
        'unlock_tool_command_cap_deg': 51,
        'unlock_command_continuity': args.unlock_command_continuity,
        'unlock_feedback_policy': unlock_feedback_policy(args.unlock_command_continuity),
    }
    (results / 'command_manifest.json').write_text(json.dumps(command_manifest, indent=2))
    sequence = 0

    def call(command):
        nonlocal sequence
        sequence += 1
        prefix = results / f'{sequence:03d}'
        prefix.with_suffix('.request.json').write_text(json.dumps(command, indent=2))
        request = urllib.request.Request(
            f'http://127.0.0.1:{args.port}', data=json.dumps(command).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            state = json.load(response)
        prefix.with_suffix('.response.json').write_text(json.dumps(state, indent=2))
        if not state['ok']:
            raise RuntimeError(state.get('error', 'teaching command failed'))
        print(json.dumps({'phase': command.get('phase', command['op']),
                          'door_deg': np.rad2deg(state['door_joints'][0]),
                          'knob_deg': np.rad2deg(state['door_joints'][1])}), flush=True)
        return state

    def pose(matrix, phase, grip, duration=3, *, interpolation_start='measured'):
        command = {'op': 'pose', 'frame': 'world',
                     'position': matrix[:3, 3].tolist(),
                     'quaternion_xyzw': Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
                     'duration': duration, 'gripper': grip, 'phase': phase}
        if interpolation_start != 'measured':
            command['interpolation_start'] = interpolation_start
        return call(command)

    def contact(state, body_index):
        forces = np.asarray(state['finger_contact_forces'])[:, body_index, :]
        if np.min(np.linalg.norm(forces, axis=1)) < 3:
            raise RuntimeError('both fingers must maintain the intended object contact')
        if body_index == 1:
            normal = np.asarray(state['panel'])[:3, 1]
            components = forces @ normal
            if components[0]*components[1] >= 0 or np.min(np.abs(components)) < 3:
                raise RuntimeError('edge fingers are not gripping opposite panel faces')

    def rotate_about_hinge(matrix, hinge, degrees):
        target = matrix.copy()
        rotation = Rotation.from_euler('z', -degrees, degrees=True).as_matrix()
        target[:3, 3] = hinge + rotation @ (matrix[:3, 3] - hinge)
        target[:3, :3] = rotation @ matrix[:3, :3]
        return target

    def plan_and_move(matrix, phase, grip):
        state = call({'op': 'snapshot'})
        base = np.asarray(state['world_to_base'])
        if Path(state['images']['wrist']).parent != output:
            raise RuntimeError('session output does not match connected simulator')
        cuboids = results / f'{phase}_cuboids.json'
        subprocess.run([sys.executable, str(paths.EXPORT_CUBOIDS),
                        '--scene', str(output / 'live_scene.usd'), '--output', str(cuboids)],
                       env=usd_environment(), check=True)
        goal = np.linalg.inv(base) @ matrix
        arm, _ = measured_planning_joints(state)
        request = results / f'{phase}_plan_request.json'
        request.write_text(json.dumps(native_plan_request(
            state, kin.ik(goal, arm), json.loads(cuboids.read_text())['cuboids'],
        ), indent=2))
        planned = results / f'{phase}_plan.json'
        with (results / f'{phase}_planner.log').open('w') as log:
            subprocess.run([str(paths.CUMOTION_LAUNCHER), '--request',
                            str(request), '--output', str(planned)],
                           cwd=ROOT, stdout=log, stderr=log, check=True)
        trajectory = json.loads(planned.read_text())
        speed = np.max(np.abs(np.diff(trajectory['positions'], axis=0)) /
                       np.diff(trajectory['timestamps'])[:, None])
        return call({'op': 'trajectory', 'path': str(planned), 'phase': phase,
                     'gripper': grip, 'time_scale': max(1, float(speed)/.6)})

    state = call({'op': 'observe'})
    if 'camera_panel_clearance_m' not in state or 'arm_contact_forces' not in state:
        raise RuntimeError('updated viewpoint/contact-monitored teaching session required')
    if abs(np.rad2deg(state['door_joints'][0])) > .2:
        raise RuntimeError('start a fresh closed-door scene before each candidate')
    if state.get('urdf_path') != str(urdf):
        raise RuntimeError('collector URDF does not match the connected teaching session')
    if state.get('runtime_profile') != collector_profile:
        raise RuntimeError('collector Isaac runtime does not match the connected teaching session')
    command_manifest.update({
        'runtime_profile': state.get('runtime_profile'),
        'runtime': state.get('runtime'),
        'core_quaternion_order': state.get('core_quaternion_order'),
    })
    (results / 'command_manifest.json').write_text(json.dumps(command_manifest, indent=2))
    call({'op': 'begin', 'name': args.name, 'expert_parameters': expert_parameters})
    try:
        panel = np.asarray(state['panel'])
        hinge = panel[:3, 3]
        knob = np.asarray(state['knob'])[:3, 3]
        grasp = np.eye(4)
        grasp[:3, :3] = panel[:3, :3] @ Rotation.from_euler('z', 90, degrees=True).as_matrix()
        grasp[:3, 3] = knob - .145 * grasp[:3, 0]
        pregrasp = grasp.copy()
        pregrasp[:3, 3] -= .045 * grasp[:3, 0]
        plan_and_move(pregrasp, 'cumotion_approach_knob', .028)
        pose(grasp, 'approach_knob', .028, 4)
        state = call({'op': 'hold', 'gripper': 0, 'duration': 2, 'phase': 'grasp_knob'})
        contact(state, 0)
        unlock_audit = {}
        try:
            state, unlocked = unlock_knob_with_feedback(
                grasp, pose, contact, audit=unlock_audit,
                command_continuity=args.unlock_command_continuity,
            )
        finally:
            expert_parameters['unlock_feedback_outcome'] = unlock_audit
            command_manifest['unlock_feedback_outcome'] = unlock_audit
            (results / 'command_manifest.json').write_text(json.dumps(command_manifest, indent=2))
        for angle in [2, 8, 15]:
            state = pose(rotate_about_hinge(unlocked, hinge, angle), 'pull_knob_for_gap', 0)
            contact(state, 0)
            if abs(np.rad2deg(state['door_joints'][0])-angle) > 2:
                raise RuntimeError('door did not follow knob pull')
        if args.behavior_family == 'edge_regrasp':
            state = call({'op': 'hold', 'gripper': .044, 'duration': 1, 'phase': 'release_knob'})
            retreat = np.asarray(state['link6'])
            retreat[:3, 3] -= args.retreat_distance * retreat[:3, 0]
            state = pose(retreat, 'retreat_from_knob', .044, 2)
            if np.rad2deg(state['door_joints'][0]) < 10:
                raise RuntimeError('door gap closed before regrasp')
            # Bounds include rear rail: panel Y=.0038..0418, rail Y=.0457..0657.
            # These edge dimensions are specific to the current door asset.
            panel = np.asarray(state['panel'])
            edge = (panel @ np.array([args.edge_x, .03475, args.edge_height, 1]))[:3]
            edge_pose = np.eye(4)
            edge_pose[:3, :3] = panel[:3, :3] @ Rotation.from_euler(
                'z', args.edge_yaw, degrees=True).as_matrix()
            edge_pose[:3, 3] = edge - args.edge_pregrasp_distance * edge_pose[:3, 0]
            plan_and_move(edge_pose, 'cumotion_transfer_to_upright_edge', .044)
            edge_pose[:3, 3] = edge - .125 * edge_pose[:3, 0]
            state = pose(edge_pose, 'approach_door_edge', .044)
            state = call({'op': 'hold', 'gripper': 0, 'duration': 2, 'phase': 'grasp_door_edge'})
            contact(state, 1)
        grasp_body = 0 if args.behavior_family == 'knob_hold_open' else 1
        pull_phase = 'pull_door_by_knob' if grasp_body == 0 else 'pull_door_by_edge'
        for _ in range(8):
            angle = float(np.rad2deg(state['door_joints'][0]))
            delta = min(5, args.final_angle-angle)
            if delta <= .1:
                break
            target = rotate_about_hinge(np.asarray(state['link6']),
                                        np.asarray(state['panel'])[:3, 3], delta)
            state = pose(target, pull_phase, 0)
            contact(state, grasp_body)
            if abs(float(np.rad2deg(state['door_joints'][0]))-angle-delta) > 2:
                raise RuntimeError(f'door did not follow {pull_phase}')
        state = call({'op': 'hold', 'gripper': 0, 'duration': 1, 'phase': 'hold_door_open'})
        contact(state, grasp_body)
        if np.rad2deg(state['door_joints'][0]) < args.final_angle-1:
            raise RuntimeError('final door opening below required angle')
        call({'op': 'end', 'success': True,
              'expert_parameters': expert_parameters,
              'notes': f'Door expert behavior family {args.behavior_family}, '
                       'with native cuMotion free-space approach, '
                       'virtual viewpoint clearance and non-finger contact monitoring. '
                       'Single wrist, absolute 7D commands, uncalibrated camera and force model. '
                       'Human review pending; no claim of sim-to-real validity.'})
    except Exception as error:
        call({'op': 'end', 'success': False, 'expert_parameters': expert_parameters,
              'notes': f'Pilot stopped: {error}'})
        raise


if __name__ == '__main__':
    main()
