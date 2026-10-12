#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Plan an ARX motion with official cuMotion in a separate Python process.

All Cartesian values are in base_link, distances in metres, joints in radians,
and request quaternions in xyzw order. Only successful, collision-checked plans
are returned. This worker neither drives a robot nor changes the simulation.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


JOINT_NAMES = [f'joint{index}' for index in range(1, 7)]


def resolve_description_dir(explicit=None):
    """Resolve the same ARX model on both the original and migrated machines."""
    requested = explicit or os.environ.get('ARX_DESCRIPTION_DIR')
    if requested:
        candidates = [Path(requested).expanduser().resolve()]
    else:
        workspace = Path(__file__).resolve().parents[4]
        package = 'isaac_ros_manipulation_arx_r5a_robot_description'
        candidates = [
            workspace / 'arx-r5-moveit' / package,
            workspace / 'B2ARX_Train_12DOF' / 'Isaac_Ros_CuMotion_ArxR5a' / package,
        ]
    for candidate in candidates:
        if (candidate / 'urdf/r5a_cumotion.urdf').is_file() and (candidate / 'xrdf/r5a.xrdf').is_file():
            return candidate
    raise FileNotFoundError(
        'ARX URDF/XRDF not found; set ARX_DESCRIPTION_DIR or pass --description-dir. '
        f'Searched: {[str(path) for path in candidates]}'
    )


def runtime_provenance():
    """Record the actual isolated interpreter and native dependency versions."""
    cm = sys.modules.get('cumotion')
    result = {
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'numpy': np.__version__,
        'cumotion': getattr(cm, '__version__', 'unavailable'),
        'cumotion_module': getattr(cm, '__file__', None),
        'platform': platform.system(),
        'machine': platform.machine(),
        'isolation': 'separate_process',
    }
    manifest = os.environ.get('CUMOTION_RUNTIME_MANIFEST')
    if manifest and Path(manifest).is_file():
        try:
            installation = json.loads(Path(manifest).read_text(encoding='utf-8'))
            if not isinstance(installation, dict):
                raise ValueError('runtime manifest must be a JSON object')
            result['installation'] = installation
            result['installation_matches_python'] = (
                installation.get('python_executable') == sys.executable
            )
        except (OSError, ValueError) as error:
            result['installation_manifest_error'] = str(error)
    return result


def vector(value, length, label):
    """Validate a finite vector before it reaches the native solver."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f'{label} must have {length} finite numbers')
    return array


def rotation(cm, value, label):
    """Convert an explicitly normalized xyzw request to cuMotion wxyz."""
    quaternion = vector(value, 4, label)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-10 or abs(norm - 1.0) > 1e-3:
        raise ValueError(f'{label} must be a unit quaternion')
    x, y, z, w = quaternion / norm
    return cm.Rotation3(float(w), float(x), float(y), float(z))


def configure_gripper_defaults(xrdf, urdf, request):
    """Fix auxiliary joint7 from measured state; joint8 remains its URDF mimic.

    All collision geometry stays unchanged. Native c-space contains only the
    arm, so the measured gripper is held constant during this arm-only plan.
    Older saved requests without gripper state explicitly retain XRDF defaults.
    """
    if 'gripper_joint_positions' not in request:
        return xrdf, {'source': 'legacy_xrdf_default',
                      'measured_joint_positions_m': None,
                      'state_supplied': False, 'held_constant_during_plan': True}
    if request.get('gripper_state_source') != 'measured_joint_state':
        raise ValueError('gripper state must identify measured_joint_state as its source')
    observed = vector(request['gripper_joint_positions'], 2, 'gripper_joint_positions')
    tree = ET.fromstring(urdf)
    configured = observed.copy()
    limits = {}
    for index, name in enumerate(('joint7', 'joint8')):
        joint = tree.find(f"joint[@name='{name}']")
        if joint is None or joint.attrib.get('type') != 'prismatic':
            raise ValueError(f'{name} must be the original prismatic gripper coordinate')
        limit = joint.find('limit')
        if limit is None:
            raise ValueError(f'{name} has no URDF position limit')
        lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
        limits[name] = {'lower': lower, 'upper': upper}
        # Only absorb floating-point boundary noise from the articulation.
        if not lower - 1e-5 <= observed[index] <= upper + 1e-5:
            raise ValueError(f'measured {name}={observed[index]} outside URDF limits')
        configured[index] = np.clip(observed[index], lower, upper)
    mimic = tree.find("joint[@name='joint8']/mimic")
    if mimic is None or mimic.attrib.get('joint') != 'joint7' or (
        float(mimic.attrib.get('multiplier', 1)) != 1
        or float(mimic.attrib.get('offset', 0)) != 0
    ):
        raise ValueError('joint8 must mimic joint7 with multiplier 1 and offset 0')
    if abs(observed[0] - observed[1]) > 1e-4:
        raise ValueError('measured gripper fingers differ by more than 0.1 mm; '
                         'the static native mimic model cannot represent this state')
    if not re.search(r'^default_joint_positions:\s*$', xrdf, re.M):
        raise ValueError('XRDF has no default_joint_positions block')
    # The native importer rejects explicit defaults for mimic joint8.
    if re.search(r'^  joint8:', xrdf, re.M):
        raise ValueError('XRDF must not specify an auxiliary default for mimic joint8')
    value = float(configured[0])
    if re.search(r'^  joint7:', xrdf, re.M):
        xrdf = re.sub(r'^  joint7:.*$', f'  joint7: {value}', xrdf, flags=re.M)
    else:
        xrdf = xrdf.replace('default_joint_positions:\n',
                            f'default_joint_positions:\n  joint7: {value}\n', 1)
    return xrdf, {'source': 'measured_joint_state', 'state_supplied': True,
                  'measured_joint_positions_m': observed.tolist(),
                  'native_joint_positions_m': [value, value],
                  'mimic_difference_m': float(abs(observed[0] - observed[1])),
                  'urdf_limits_m': limits,
                  'boundary_noise_clipped': bool(not np.array_equal(configured, observed)),
                  'held_constant_during_plan': True}


def plan(request, description_dir):
    """Return a sampled trajectory, or retain the native solver failure."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('the pinned cuMotion cp312 runtime requires Python 3.12')
    import cumotion as cm

    response = {
        'success': False,
        'status': 'NOT_PLANNED',
        'backend': 'nvidia_cumotion_native',
        'version': cm.__version__,
        'joint_names': JOINT_NAMES,
        'positions': [],
        'timestamps': [],
        'frame_id': 'base_link',
        'tool_frame': 'link6',
        'collision_checking': True,
        'runtime': runtime_provenance(),
        'model': {
            'description_dir': str(description_dir),
            'urdf': str(description_dir / 'urdf/r5a_cumotion.urdf'),
            'xrdf': str(description_dir / 'xrdf/r5a.xrdf'),
        },
    }
    start = vector(request['start_positions'], 6, 'start_positions')
    cartesian = 'target_position' in request
    joint_target = 'target_joints' in request
    if cartesian == joint_target:
        raise ValueError('specify exactly one of target_position or target_joints')
    sample_dt = float(request.get('sample_dt', 1.0 / 30.0))
    speed = float(request.get('time_dilation_factor', 0.2))
    if not math.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError('sample_dt must be positive and finite')
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError('time_dilation_factor must be in (0, 1]')

    urdf_path = description_dir / 'urdf' / 'r5a_cumotion.urdf'
    xrdf_path = description_dir / 'xrdf' / 'r5a.xrdf'
    # Resolve package URLs in memory: the source model remains unchanged and
    # this process does not require a sourced ROS environment.
    urdf = urdf_path.read_text().replace(
        'package://isaac_ros_manipulation_arx_r5a_robot_description/',
        str(description_dir) + '/',
    )
    xrdf, gripper_model = configure_gripper_defaults(xrdf_path.read_text(), urdf, request)
    response['model']['gripper'] = gripper_model
    robot = cm.load_robot_from_memory(xrdf, urdf)
    names = [robot.cspace_coord_name(i) for i in range(robot.num_cspace_coords())]
    if names != JOINT_NAMES:
        raise ValueError(f'unexpected XRDF joint order: {names}')

    world = cm.create_world()
    cuboids = request.get('cuboids', [])
    if not isinstance(cuboids, list):
        raise ValueError('cuboids must be a list')
    obstacle_names = []
    obstacle_handles = []
    for index, specification in enumerate(cuboids):
        label = specification.get('name', f'cuboid_{index}')
        center = vector(specification['center'], 3, f'{label}.center')
        size = vector(specification['size'], 3, f'{label}.size')
        if np.any(size <= 0):
            raise ValueError(f'{label}.size must be positive')
        orientation = rotation(
            cm, specification.get('quaternion_xyzw', [0, 0, 0, 1]),
            f'{label}.quaternion_xyzw',
        )
        obstacle = cm.create_obstacle(cm.Obstacle.Type.CUBOID)
        obstacle.set_attribute(
            cm.Obstacle.Attribute.SIDE_LENGTHS,
            cm.Obstacle.AttributeValue(size),
        )
        obstacle_handles.append(world.add_obstacle(obstacle, cm.Pose3(orientation, center)))
        obstacle_names.append(label)
    view = world.add_world_view()
    view.update()
    response['world'] = {
        'obstacle_names': obstacle_names,
        'cuboids': cuboids,
        'empty_world': not bool(cuboids),
        'label': 'EMPTY_WORLD_DIAGNOSTIC' if not cuboids else 'REQUEST_CUBOIDS',
    }
    config = cm.create_default_trajectory_optimizer_config(robot, 'link6', view)
    planner = cm.create_trajectory_optimizer(config)
    if joint_target:
        target = cm.TrajectoryOptimizer.CSpaceTarget(
            vector(request['target_joints'], 6, 'target_joints')
        )
        result = planner.plan_to_cspace_target(start, target)
    else:
        target = cm.TrajectoryOptimizer.TaskSpaceTarget(
            cm.TrajectoryOptimizer.TranslationConstraint.target(
                vector(request['target_position'], 3, 'target_position')
            ),
            cm.TrajectoryOptimizer.OrientationConstraint.terminal_target(
                rotation(cm, request['target_quaternion_xyzw'], 'target_quaternion_xyzw')
            ),
        )
        result = planner.plan_to_task_space_target(start, target)
    status = result.status()
    response['status'] = status.name
    response['status_code'] = int(status)
    if status != cm.TrajectoryOptimizer.Results.Status.SUCCESS:
        response['reason'] = f'cuMotion returned {status.name}; no trajectory exported'
        if joint_target:
            inspector = cm.create_robot_world_inspector(robot, view)
            goal = vector(request['target_joints'], 6, 'target_joints')
            response['target_diagnostics'] = {
                'within_native_cspace_limits': robot.kinematics().within_cspace_limits(goal, False),
                'native_cspace_limits': [{
                    'joint': name,
                    'lower': robot.kinematics().cspace_coord_limits(index).lower,
                    'upper': robot.kinematics().cspace_coord_limits(index).upper,
                } for index, name in enumerate(names)],
                'self_collision_frames': inspector.frames_in_self_collision(goal),
                'min_obstacle_distance_m': inspector.min_distance_to_obstacle(goal),
                'obstacle_contacts': [
                    {'obstacle': name,
                     'link': inspector.world_collision_sphere_frame_name(index),
                     'sphere_index': index,
                     'distance_m': inspector.distance_to_obstacle(handle, index, goal)}
                    for name, handle in zip(obstacle_names, obstacle_handles)
                    for index in range(inspector.num_world_collision_spheres())
                    if inspector.distance_to_obstacle(handle, index, goal) <= 0.0
                ],
            }
        return response

    trajectory = result.trajectory()
    domain = trajectory.domain()
    duration = (domain.upper - domain.lower) / speed
    if not math.isfinite(duration) or duration < 0 or duration / sample_dt > 1_000_000:
        raise RuntimeError('native trajectory has an invalid or excessive duration')
    timestamps = np.arange(0.0, duration, sample_dt).tolist()
    if not timestamps or duration - timestamps[-1] > 1e-10:
        timestamps.append(duration)
    positions = [trajectory.eval(domain.lower + t * speed).tolist() for t in timestamps]
    if np.asarray(positions).shape != (len(timestamps), 6):
        raise RuntimeError('native trajectory has an unexpected joint shape')
    if not np.isfinite(positions).all():
        raise RuntimeError('native trajectory contains non-finite joint positions')
    final_pose = robot.kinematics().pose(np.asarray(positions[-1]), 'link6')
    final_rotation = final_pose.rotation
    response.update({
        'success': True,
        'positions': positions,
        'timestamps': timestamps,
        'duration_sec': duration,
        'sample_dt': sample_dt,
        'time_dilation_factor': speed,
        'final_position': final_pose.translation.tolist(),
        'final_quaternion_xyzw': [
            final_rotation.x(), final_rotation.y(), final_rotation.z(), final_rotation.w(),
        ],
    })
    return response


def main():
    """Always write a machine-readable result, including validation errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--description-dir', type=Path,
        help='ARX description package (defaults to ARX_DESCRIPTION_DIR or workspace discovery)',
    )
    args = parser.parse_args()
    started = time.monotonic()
    try:
        request = json.loads(args.request.read_text())
        if not isinstance(request, dict):
            raise ValueError('request must be a JSON object')
        response = plan(request, resolve_description_dir(args.description_dir))
    except Exception as error:
        response = {
            'success': False, 'status': 'ERROR',
            'reason': f'{type(error).__name__}: {error}',
            'backend': 'nvidia_cumotion_native',
            'version': getattr(sys.modules.get('cumotion'), '__version__', 'unavailable'),
            'joint_names': JOINT_NAMES, 'positions': [], 'timestamps': [],
            'runtime': runtime_provenance(),
        }
        print(response['reason'], file=sys.stderr)
    response['planning_wall_sec'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(response, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    return 0 if response['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
