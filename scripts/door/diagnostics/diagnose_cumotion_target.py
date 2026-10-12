#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Inspect a saved native cuMotion request without optimizing or simulating.

The original request and model files remain unchanged. Compare the original
XRDF against explicitly fixed non-cspace gripper positions, retaining every
collision sphere, world obstacle and arm joint limit. This is a model diagnosis,
not a substitute for physical collision validation or a successful plan.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

# Runs in the isolated cuMotion venv, alongside the planner worker it inspects.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planner'))
from door_cumotion_worker import (  # noqa: E402
    resolve_description_dir, rotation, runtime_provenance, vector,
)


def with_fixed_gripper(xrdf, position):
    """Add explicit defaults only to the diagnostic in-memory XRDF."""
    if not np.isfinite(position) or not 0 <= position <= .044:
        raise ValueError('gripper position must be within original 0..0.044 m limits')
    if re.search(r'^\s+joint[78]:', xrdf, re.M):
        raise ValueError('diagnosis requires source XRDF without existing joint7/8 defaults')
    # joint8 is a URDF mimic. The native importer rejects an explicit default
    # for a mimic coordinate; setting joint7 propagates to both fingers.
    return xrdf.replace('default_joint_positions:\n',
                        f'default_joint_positions:\n  joint7: {position}\n', 1)


def inspect(request, description_dir, gripper_positions, candidates=None):
    import cumotion as cm

    urdf_path = description_dir / 'urdf/r5a_cumotion.urdf'
    xrdf_path = description_dir / 'xrdf/r5a.xrdf'
    urdf = urdf_path.read_text().replace(
        'package://isaac_ros_manipulation_arx_r5a_robot_description/', str(description_dir) + '/')
    xrdf = xrdf_path.read_text()
    source_limits = {}
    for joint in ET.fromstring(urdf).findall('joint'):
        limit = joint.find('limit')
        if limit is not None:
            source_limits[joint.attrib['name']] = {
                'lower': float(limit.attrib['lower']), 'upper': float(limit.attrib['upper'])}
    start = vector(request['start_positions'], 6, 'start_positions')
    target = vector(request['target_joints'], 6, 'target_joints')
    world = cm.create_world()
    obstacles = []
    for spec in request.get('cuboids', []):
        obstacle = cm.create_obstacle(cm.Obstacle.Type.CUBOID)
        obstacle.set_attribute(cm.Obstacle.Attribute.SIDE_LENGTHS,
                               cm.Obstacle.AttributeValue(vector(spec['size'], 3, 'size')))
        handle = world.add_obstacle(obstacle, cm.Pose3(
            rotation(cm, spec['quaternion_xyzw'], 'quaternion'),
            vector(spec['center'], 3, 'center')))
        obstacles.append((spec['name'], handle))
    view = world.add_world_view()
    view.update()
    report = {'schema': 'arx-cumotion-target-inspection-v1',
              'optimizer_executed': False, 'simulation_executed': False,
              'runtime': runtime_provenance(), 'model_dir': str(description_dir),
              'request_start': start.tolist(), 'request_target': target.tolist(),
              'urdf_joint_limits': source_limits, 'variants': []}
    for position in [None, *gripper_positions]:
        variant_xrdf = xrdf if position is None else with_fixed_gripper(xrdf, position)
        robot = cm.load_robot_from_memory(variant_xrdf, urdf)
        kin = robot.kinematics()
        inspector = cm.create_robot_world_inspector(robot, view)
        limits = {kin.cspace_coord_name(i): {
            'lower': kin.cspace_coord_limits(i).lower,
            'upper': kin.cspace_coord_limits(i).upper} for i in range(robot.num_cspace_coords())}
        item = {'fixed_gripper_m': position, 'native_cspace_limits': limits,
                'native_within_limits': kin.within_cspace_limits(target, False),
                'start_self_collision': inspector.frames_in_self_collision(start),
                'start_min_obstacle_distance_m': inspector.min_distance_to_obstacle(start),
                'target_self_collision': inspector.frames_in_self_collision(target),
                'target_min_obstacle_distance_m': inspector.min_distance_to_obstacle(target),
                'gripper_link_positions_in_link6': {
                    link: kin.position(target, link, 'link6').tolist() for link in ['link7', 'link8']},
                'target_link6_position': kin.position(target, 'link6').tolist(),
                'contacts': []}
        centers = inspector.world_collision_sphere_positions(target)
        radii = inspector.world_collision_sphere_radii()
        for name, handle in obstacles:
            for index in range(inspector.num_world_collision_spheres()):
                distance = inspector.distance_to_obstacle(handle, index, target)
                if distance <= 0:
                    item['contacts'].append({'obstacle': name, 'sphere_index': index,
                        'link': inspector.world_collision_sphere_frame_name(index),
                        'distance_m': distance, 'center_base': centers[index].tolist(),
                        'radius_with_buffer_m': radii[index]})
        item['standoff_candidates'] = []
        for candidate in candidates or []:
            joints = vector(candidate['target_joints'], 6, 'candidate target_joints')
            item['standoff_candidates'].append({
                'source_waypoint_index': candidate['source_waypoint_index'],
                'standoff_m': candidate.get('standoff_m', 0.0),
                'native_within_limits': kin.within_cspace_limits(joints, False),
                'self_collision_frames': inspector.frames_in_self_collision(joints),
                'min_obstacle_distance_m': inspector.min_distance_to_obstacle(joints),
                'world_collision': inspector.in_collision_with_obstacle(joints),
            })
        report['variants'].append(item)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--description-dir', type=Path)
    parser.add_argument('--gripper-positions', type=float, nargs='+', default=[0, .044])
    parser.add_argument('--candidate-report', type=Path,
                        help='Optional CPU IK shortlist with rows of target_joints')
    args = parser.parse_args()
    candidates = json.loads(args.candidate_report.read_text())['rows'] if args.candidate_report else None
    result = inspect(json.loads(args.request.read_text()),
                     resolve_description_dir(args.description_dir), args.gripper_positions, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{key: value for key, value in item.items() if key != 'contacts'}
                      for item in result['variants']], indent=2))


if __name__ == '__main__':
    main()
