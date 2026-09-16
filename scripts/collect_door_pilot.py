#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Withdrawn legacy door-teaching recipe, retained for failure analysis.

The wrist camera crossed the panel and local IK bypassed collision validation.
The entry point is disabled until whole-robot collision coverage is corrected.
No door joint is commanded by this script.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-output', type=Path, required=True)
    parser.add_argument('--name', default='candidate_000')
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--final-angle', type=float, default=30.0)
    args = parser.parse_args()
    raise RuntimeError(
        'This pilot recipe is withdrawn after collision review: wrist-camera '
        'geometry is missing and local IK segments are not collision validated. '
        'See docs/door_collision_review.md. No simulator commands were sent.')
    if not 15 < args.final_angle <= 35:
        parser.error('pilot final angle must be above 15 and at most 35 degrees')
    root = Path(__file__).resolve().parents[1]
    output = args.session_output.resolve()
    results = output / f'{args.name}_commands'
    results.mkdir(exist_ok=False)
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

    def pose(matrix, phase, grip, duration=3):
        return call({'op': 'pose', 'frame': 'world',
                     'position': matrix[:3, 3].tolist(),
                     'quaternion_xyzw': Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
                     'duration': duration, 'gripper': grip, 'phase': phase})

    def contact(state, body_index):
        forces = np.asarray(state['finger_contact_forces'])[:, body_index, :]
        if np.min(np.linalg.norm(forces, axis=1)) < 3:
            raise RuntimeError('both fingers must maintain the intended object contact')

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
        spec = importlib.util.find_spec('isaacsim')
        sim_root = Path(next(iter(spec.submodule_search_locations)))
        usd = next((sim_root / 'extscache').glob('omni.usd.libs-*'))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(usd)
        env['LD_LIBRARY_PATH'] = f'{usd / "bin"}:{Path(sys.prefix) / "lib"}'
        cuboids = results / f'{phase}_cuboids.json'
        subprocess.run([sys.executable, str(root / 'scripts/export_door_cuboids.py'),
                        '--scene', str(output / 'live_scene.usd'), '--output', str(cuboids)],
                       env=env, check=True)
        goal = np.linalg.inv(base) @ matrix
        request = results / f'{phase}_plan_request.json'
        request.write_text(json.dumps({
            'start_positions': state['joints'][:6],
            'target_position': goal[:3, 3].tolist(),
            'target_quaternion_xyzw': Rotation.from_matrix(goal[:3, :3]).as_quat().tolist(),
            'cuboids': json.loads(cuboids.read_text())['cuboids'],
            'time_dilation_factor': .5, 'sample_dt': 1/30}, indent=2))
        planned = results / f'{phase}_plan.json'
        with (results / f'{phase}_planner.log').open('w') as log:
            subprocess.run([str(root / 'scripts/run_door_cumotion.sh'), '--request',
                            str(request), '--output', str(planned)],
                           cwd=root, stdout=log, stderr=log, check=True)
        trajectory = json.loads(planned.read_text())
        speed = np.max(np.abs(np.diff(trajectory['positions'], axis=0)) /
                       np.diff(trajectory['timestamps'])[:, None])
        return call({'op': 'trajectory', 'path': str(planned), 'phase': phase,
                     'gripper': grip, 'time_scale': max(1, float(speed)/.6)})

    state = call({'op': 'observe'})
    if abs(np.rad2deg(state['door_joints'][0])) > .2:
        raise RuntimeError('start a fresh closed-door scene before each candidate')
    call({'op': 'begin', 'name': args.name})
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
        unlocked = grasp.copy()
        for angle, duration in [(45, 3), (47, .8)]:
            unlocked[:3, :3] = grasp[:3, :3] @ Rotation.from_euler('x', -angle, degrees=True).as_matrix()
            state = pose(unlocked, 'unlock_round_knob', 0, duration)
        if np.rad2deg(state['door_joints'][1]) < 43.5 or state['door_joints'][2] < .0154:
            raise RuntimeError('knob rotation/latch retraction did not reach unlock threshold')
        for angle in [2, 8, 15]:
            state = pose(rotate_about_hinge(unlocked, hinge, angle), 'pull_knob_for_gap', 0)
            contact(state, 0)
            if abs(np.rad2deg(state['door_joints'][0])-angle) > 2:
                raise RuntimeError('door did not follow knob pull')
        state = call({'op': 'hold', 'gripper': .044, 'duration': 1, 'phase': 'release_knob'})
        retreat = np.asarray(state['link6'])
        retreat[:3, 3] -= .08 * retreat[:3, 0]
        state = pose(retreat, 'retreat_from_knob', .044, 2)
        if np.rad2deg(state['door_joints'][0]) < 10:
            raise RuntimeError('door gap closed before regrasp')
        # Bounds include the rear rail: panel Y=.0038..0418, rail Y=.0457..0657.
        # These dimensions and the low grasp height are specific to this asset.
        panel = np.asarray(state['panel'])
        transit = np.eye(4)
        transit[:3, :3] = panel[:3, :3] @ (
            Rotation.from_euler('z', 180, degrees=True) *
            Rotation.from_euler('y', 50, degrees=True)).as_matrix()
        transit[:3, 3] = (panel @ np.array([.759, .0228, .6, 1]))[:3] - .2 * transit[:3, 0]
        plan_and_move(transit, 'cumotion_transfer_to_edge', .044)
        edge = (panel @ np.array([.743, .03475, .56, 1]))[:3]
        edge_pose = np.eye(4)
        edge_pose[:3, :3] = panel[:3, :3] @ (
            Rotation.from_euler('z', 180, degrees=True) *
            Rotation.from_euler('y', 60, degrees=True)).as_matrix()
        edge_pose[:3, 3] = edge - .18 * edge_pose[:3, 0]
        pose(edge_pose, 'lower_toward_edge_contact_region', .044)
        edge_pose[:3, 3] = edge - .125 * edge_pose[:3, 0]
        state = pose(edge_pose, 'approach_door_edge', .044)
        state = call({'op': 'hold', 'gripper': 0, 'duration': 2, 'phase': 'grasp_door_edge'})
        contact(state, 1)
        for _ in range(8):
            angle = float(np.rad2deg(state['door_joints'][0]))
            delta = min(5, args.final_angle-angle)
            if delta <= .1:
                break
            target = rotate_about_hinge(np.asarray(state['link6']),
                                        np.asarray(state['panel'])[:3, 3], delta)
            state = pose(target, 'pull_door_by_edge', 0)
            contact(state, 1)
            if abs(float(np.rad2deg(state['door_joints'][0]))-angle-delta) > 2:
                raise RuntimeError('door did not follow edge pull')
        state = call({'op': 'hold', 'gripper': 0, 'duration': 1, 'phase': 'hold_door_open'})
        contact(state, 1)
        if np.rad2deg(state['door_joints'][0]) < args.final_angle-1:
            raise RuntimeError('final door opening below required angle')
        call({'op': 'end', 'success': True,
              'notes': 'Complete physical pilot with cuMotion approach/transfer. '
                       'Single wrist, absolute 7D commands, uncalibrated camera and force model. '
                       'Human review pending; no claim of sim-to-real validity.'})
    except Exception as error:
        call({'op': 'end', 'success': False, 'notes': f'Pilot stopped: {error}'})
        raise


if __name__ == '__main__':
    main()
