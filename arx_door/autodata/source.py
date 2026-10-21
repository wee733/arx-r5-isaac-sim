# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Turn an ARX expert seed into an AutoData annotated source dataset.

The expert pipeline (``scripts/door/demos``) records joint-space actions and
world-frame datagen poses. AutoData's official ``absolute_pose`` embodiment and
Datastream need instead, in the robot root frame:

* ``actions``: ``[x, y, z, qx, qy, qz, qw, gripper]`` of the target link6 pose;
* ``obs/datagen_info/object_pose/door``: the door articulation root pose, which is
  what the official Datastream reports for the live scene.

With the door closed at the subtask start, the knob is rigidly fixed to the door
root, so a door-relative transform equals the knob-relative one the seed was
annotated with. Two checks guard the frame change: the base pose recovered from
every frame (link6 pose times inverse FK) must be constant, and it must match the
robot placement of the asset bundle that the generation run will load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'arx-autodata-source-v1'
ACTION_LAYOUT = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
BASE_TOLERANCE_M = 1e-3
BASE_TOLERANCE_RAD = 2e-3
MANIFEST_TOLERANCE_M = 5e-3
MANIFEST_TOLERANCE_RAD = 1e-2


def pose_matrix(position, quaternion_xyzw) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = position
    return result


def _pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rotation = Rotation.from_matrix(a[..., :3, :3].swapaxes(-1, -2) @ b[..., :3, :3])
    return (float(np.max(np.linalg.norm(a[..., :3, 3] - b[..., :3, 3], axis=-1))),
            float(np.max(rotation.magnitude())))


def recover_base(eef_world: np.ndarray, joints: np.ndarray, kinematics) -> tuple[np.ndarray, dict]:
    """Return the world base pose implied by every frame, and its spread."""
    bases = np.stack([pose @ np.linalg.inv(kinematics.fk(q)) for pose, q in zip(eef_world, joints)])
    base = bases[0]
    position, rotation = _pose_error(bases, base)
    if position > BASE_TOLERANCE_M or rotation > BASE_TOLERANCE_RAD:
        raise ValueError(f'seed base pose is not constant: {position:.6f} m, {rotation:.6f} rad')
    return base, {'max_position_spread_m': position, 'max_rotation_spread_rad': rotation}


def absolute_pose_actions(target_base: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Encode link6 targets like ``AbsolutePoseEmbodimentAdapter`` (w >= 0)."""
    quaternions = Rotation.from_matrix(target_base[:, :3, :3]).as_quat()
    quaternions[quaternions[:, 3] < 0] *= -1
    return np.concatenate([target_base[:, :3, 3], quaternions, gripper.reshape(-1, 1)],
                          axis=1).astype(np.float32)


def convert_episode(episode, manifest: dict, kinematics) -> tuple[dict, dict]:
    """Return ``(datasets, provenance)`` for one seed episode group."""
    info = episode['obs/datagen_info']
    actions = np.asarray(episode['actions'], dtype=float)
    joints = np.asarray(episode['obs/diagnostics/joint_state'], dtype=float)[:, :6]
    eef_world = np.asarray(info['eef_pose/link6'], dtype=float)
    target_world = np.asarray(info['target_eef_pose/link6'], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError('seed actions must be [T, 7] absolute joint commands')
    base, spread = recover_base(eef_world, joints, kinematics)
    robot = manifest['assets']['robot']['initial_pose']
    door = manifest['assets']['door']['initial_pose']
    placement = pose_matrix(robot['position'], robot['quaternion_xyzw'])
    position, rotation = _pose_error(base, placement)
    if position > MANIFEST_TOLERANCE_M or rotation > MANIFEST_TOLERANCE_RAD:
        raise ValueError('seed and asset bundle disagree on the robot placement: '
                         f'{position:.4f} m, {rotation:.4f} rad; prepare the bundle from the seed scene')
    to_base = np.linalg.inv(base)
    target_base = to_base @ target_world
    planned = np.stack([kinematics.fk(q) for q in actions[:, :6]])
    fk_position, fk_rotation = _pose_error(target_base, planned)
    if fk_position > 1e-4 or fk_rotation > 1e-3:
        raise ValueError('target_eef_pose is not FK of the recorded joint actions')
    door_base = to_base @ pose_matrix(door['position'], door['quaternion_xyzw'])
    frames = len(actions)
    datasets = {
        'actions': absolute_pose_actions(target_base, actions[:, 6]),
        'obs/datagen_info/eef_pose/link6': (to_base @ eef_world).astype(np.float32),
        'obs/datagen_info/target_eef_pose/link6': target_base.astype(np.float32),
        'obs/datagen_info/object_pose/door': np.repeat(door_base[None], frames, 0).astype(np.float32),
    }
    for name in info['object_pose']:
        datasets[f'obs/datagen_info/object_pose/{name}'] = (
            to_base @ np.asarray(info[f'object_pose/{name}'], dtype=float)).astype(np.float32)
    knob_in_door = np.linalg.inv(door_base) @ datasets['obs/datagen_info/object_pose/knob'][0]
    provenance = {
        'base_world_from_seed': base.tolist(), 'base_spread': spread,
        'base_vs_bundle_placement': {'position_m': position, 'rotation_rad': rotation},
        'target_fk_residual': {'position_m': fk_position, 'rotation_rad': fk_rotation},
        'knob_in_door_root_at_start': knob_in_door.tolist(),
    }
    return datasets, provenance


def convert_seed(seed: str | Path, manifest: str | Path, output: str | Path, urdf: str | Path) -> dict:
    """Write the AutoData source HDF5 next to its JSON provenance; never overwrite."""
    import h5py
    from arx_door.envs.door import ENV_NAME
    from arx_door.kinematics import ArmKinematics

    seed, manifest_path, output = Path(seed), Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_data = json.loads(manifest_path.read_text())
    kinematics = ArmKinematics(urdf)
    report = {'schema': SCHEMA, 'seed': str(seed.resolve()),
              'seed_sha256': hashlib.sha256(seed.read_bytes()).hexdigest(),
              'assets': str(manifest_path.resolve()), 'frame': 'robot_root',
              'action_layout': ACTION_LAYOUT, 'object_ref': 'door', 'episodes': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    with h5py.File(seed, 'r') as source, h5py.File(partial, 'w') as target:
        data = target.create_group('data')
        for key, value in source['data'].attrs.items():
            data.attrs[key] = value
        data.attrs['env_args'] = json.dumps({'env_name': ENV_NAME, 'type': 2, 'env_kwargs': {}})
        for name, episode in source['data'].items():
            datasets, provenance = convert_episode(episode, manifest_data, kinematics)
            group = data.create_group(name)
            for key, value in episode.attrs.items():
                group.attrs[key] = value
            group.attrs['autodata_source'] = json.dumps(provenance)
            for path in ('obs/datagen_info/subtask_start_signals', 'obs/datagen_info/subtask_term_signals',
                         'obs/diagnostics'):
                parent, _, leaf = path.rpartition('/')
                source.copy(episode[path], group.require_group(parent), name=leaf)
            for path, value in datasets.items():
                group.create_dataset(path, data=value)
            report['episodes'][name] = provenance
    partial.rename(output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
