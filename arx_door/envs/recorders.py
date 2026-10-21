# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Ordinary Isaac Lab recorder terms for Arena's ARX scene.

Helpers stay importable on a NumPy-only host. Recorder classes are created lazily
after SimulationApp starts, and use RecorderTerm rather than Mimic environment
methods. Camera mount/render alignment still needs the real Sim 6.1 acceptance
run; sensor pose checks alone are not a rendered-image validation claim.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import numpy as np

from .acceptance import acceptance_state, as_numpy, reset_acceptance


ARM_STATE_NAMES = [*(f'joint{i}' for i in range(1, 7)), 'joint7']
DOOR_STATE_NAMES = ['hinge_joint', 'handle_joint', 'latch_joint']
PANEL_BOX_CENTER = np.array((.3795, .0228, .915))
PANEL_BOX_HALF_SIZE = np.array((.3625, .019, .395))


def _tensor(env, value):
    if not hasattr(env, 'device'):
        return np.asarray(value).copy()
    import torch
    return torch.as_tensor(np.asarray(value).copy(), device=env.device)


def _joint_state(env, asset_name, names):
    asset = env.scene[asset_name]
    ids, matched = asset.find_joints(names, preserve_order=True)
    if list(matched) != names:
        raise ValueError(f'{asset_name} joints do not match {names}: {matched}')
    return as_numpy(asset.data.joint_pos)[:, ids].copy()


def _pose(positions, quaternions):
    # The pinned Lab 91a0906 camera and articulation API uses XYZW.
    from scipy.spatial.transform import Rotation
    positions, quaternions = as_numpy(positions), as_numpy(quaternions)
    count = len(positions)
    if positions.shape != (count, 3) or quaternions.shape != (count, 4):
        raise ValueError('pose must have per-environment position and XYZW quaternion rows')
    result = np.repeat(np.eye(4)[None], count, axis=0)
    result[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    result[:, :3, 3] = positions
    return result


def _body_pose(env, asset_name, body_name):
    if asset_name == 'robot':
        body_name = getattr(env.cfg, 'robot_body_aliases', {}).get(body_name, body_name)
    asset = env.scene[asset_name]
    ids, matched = asset.find_bodies([body_name], preserve_order=True)
    if list(matched) != [body_name]:
        raise ValueError(f'{asset_name} body {body_name!r} is missing')
    state = as_numpy(asset.data.body_link_state_w)[:, ids[0]]
    return _pose(state[:, :3], state[:, 3:7])


def _mount(env):
    mount = getattr(env, '_approved_wrist_mount', None)
    if mount is None:
        mount = getattr(env.cfg, 'arx_wrist_mount', None)
    if mount is None:
        raise RuntimeError('Arena must supply the authored link6 OpenGL arx_wrist_mount')
    mount = np.asarray(mount, dtype=float)
    if (mount.shape != (4, 4) or not np.isfinite(mount).all()
            or not np.allclose(mount[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose(mount[:3, :3].T @ mount[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(mount[:3, :3]), 1, rtol=0, atol=1e-6)):
        raise ValueError('arx_wrist_mount must be a finite rigid 4x4 transform')
    env._approved_wrist_mount = mount.copy()
    return mount


def _expected_camera_pose(env):
    return _body_pose(env, 'robot', 'link6') @ _mount(env)


def _camera_clearance(camera_pose, panel_pose):
    center = panel_pose[:, :3, 3] + np.einsum('nij,j->ni', panel_pose[:, :3, :3], PANEL_BOX_CENTER)
    point = np.einsum('nji,nj->ni', panel_pose[:, :3, :3], camera_pose[:, :3, 3] - center)
    delta = np.abs(point) - PANEL_BOX_HALF_SIZE
    return (np.linalg.norm(np.maximum(delta, 0), axis=-1)
            + np.minimum(np.max(delta, axis=-1), 0))[:, None]


def _field(data, primary, legacy):
    value = getattr(data, primary, None)
    if value is None:
        value = getattr(data, legacy, None)
    if value is None:
        raise RuntimeError(f'contact sensor is missing {primary}')
    return as_numpy(value)


def read_physical_state(env) -> dict[str, np.ndarray]:
    """Read ordinary articulation/sensor data without a legacy environment API."""
    fingers = []
    for name in ('left_panel_contact', 'right_panel_contact'):
        data = env.scene.sensors[name].data
        forces = _field(data, 'normal_force_matrix_w', 'force_matrix_w')
        if forces.shape != (env.num_envs, 1, 2, 3):
            raise ValueError(f'{name} must filter panel then knob for every environment')
        fingers.append(forces[:, 0, [1, 0]])
    panel_pose = _body_pose(env, 'door', 'door_panel')
    expected = _expected_camera_pose(env)
    return {
        'door_joint_pos': _joint_state(env, 'door', DOOR_STATE_NAMES),
        'finger_contacts': np.stack(fingers, axis=1),
        'arm_contacts': _field(env.scene.sensors['arm_contacts'].data,
                               'net_normal_forces_w', 'net_forces_w').copy(),
        'camera_clearance': _camera_clearance(expected, panel_pose),
        'panel_pose': panel_pose,
    }


def _new_camera_evidence(env, env_id):
    sensor = env.scene.sensors['wrist_camera']
    return {
        'schema': 'arx-autodata-wrist-camera-v1', 'env_id': env_id,
        'mount_link': 'link6', 'mount_transform': _mount(env).tolist(),
        'camera_prim_pattern': sensor.cfg.prim_path,
        'camera_batch_index': env_id,
        'frames_verified': 0, 'max_position_error_m': 0., 'max_rotation_error_rad': 0.,
        'pose_verified': False, 'rgb_frames_captured': 0,
        'valid_rgb_frames': 0, 'capture_protocol_verified': False,
        'timestamp_action_index_mapping_complete': False, 'human_reviewed': False,
        'first_camera_frame_id': None, 'last_camera_frame_id': None,
        'camera_frame_ids_available': False, 'camera_frame_ids_strictly_increasing': True,
        'capture_order': ['set_world_poses', 'render', 'force_sensor_update', 'read_rgb'],
        'alignment': 'pre_action_observation_at_30_hz',
        'physics_diagnostics_alignment': 'post_action_at_30_hz',
        'rendered_image_alignment_verified': False,
    }


def camera_evidence(env, env_id: int) -> dict:
    """Return a detached JSON sidecar for finalization; no video claims are inferred."""
    acceptance_state(env)._ids([env_id])
    evidence = getattr(env, '_arx_camera_evidence', {})
    if env_id in evidence:
        return deepcopy(evidence[env_id])
    return _new_camera_evidence(env, env_id)


def _camera_frame_ids(sensor, num_envs):
    raw = getattr(sensor, 'frame', None)
    if raw is None:
        raw = getattr(sensor.data, 'frame', None)
    if raw is None:
        raise RuntimeError('wrist camera has no reliable per-environment frame counter')
    frame_ids = as_numpy(raw).reshape(-1)
    if frame_ids.shape != (num_envs,) or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError('camera frame IDs must be integer rows indexed by environment')
    return frame_ids.copy()


def capture_pre_step(env):
    """Synchronize each wrist camera, then capture its RGB and the pre-action state."""
    from scipy.spatial.transform import Rotation
    sensor = env.scene.sensors['wrist_camera']
    expected = _expected_camera_pose(env)
    if expected.shape != (env.num_envs, 4, 4):
        raise ValueError('link6 camera poses must have one row per environment')
    quaternions = Rotation.from_matrix(expected[:, :3, :3]).as_quat()
    before_frame_ids = _camera_frame_ids(sensor, env.num_envs)
    sensor.set_world_poses(_tensor(env, expected[:, :3, 3]), _tensor(env, quaternions),
                           convention='opengl')
    env.sim.render()
    sensor.update(0., force_recompute=True)
    actual = _pose(sensor.data.pos_w, sensor.data.quat_w_opengl)
    rgb = as_numpy(sensor.data.output['rgb'])
    if actual.shape != expected.shape or rgb.ndim != 4 or rgb.shape[0] != env.num_envs or rgb.shape[-1] < 3:
        raise ValueError('wrist camera batch must contain distinct rows for all environments')
    if rgb.dtype != np.uint8 or rgb.shape[1] < 1 or rgb.shape[2] < 1:
        raise ValueError('wrist RGB must be nonempty uint8 camera evidence')
    frame_ids = _camera_frame_ids(sensor, env.num_envs)
    # Reject an accidental TiledCamera setup or a namespace-to-row permutation.
    if 'TiledCamera' in type(sensor).__name__:
        raise RuntimeError('the first AutoData profile requires ordinary Camera, not TiledCamera')
    sensor_prims = getattr(sensor, '_sensor_prims', ())
    if sensor_prims:
        import re
        for env_id, camera_prim in enumerate(sensor_prims):
            path = str(camera_prim.GetPath())
            if re.search(rf'/env_{env_id}/', path) is None:
                raise RuntimeError(f'wrist camera batch row {env_id} belongs to {path}')
    pos_error = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1)
    relative = np.swapaxes(actual[:, :3, :3], 1, 2) @ expected[:, :3, :3]
    rot_error = Rotation.from_matrix(relative).magnitude()
    evidence = getattr(env, '_arx_camera_evidence', None)
    if evidence is None:
        evidence = env._arx_camera_evidence = {}
    active = as_numpy(getattr(env, 'active_episode_mask', np.ones(env.num_envs, bool))).astype(bool)
    state = acceptance_state(env)
    for env_id in np.flatnonzero(active):
        env_id = int(env_id)
        item = evidence.setdefault(env_id, _new_camera_evidence(env, env_id))
        if (not np.isfinite([pos_error[env_id], rot_error[env_id]]).all()
                or pos_error[env_id] > 1e-4 or rot_error[env_id] > 1e-3):
            raise RuntimeError(f'wrist camera mount mismatch in env {env_id}: '
                               f'{pos_error[env_id]:.9g} m, {rot_error[env_id]:.9g} rad')
        if item['frames_verified'] != int(state.frames[env_id]):
            item['capture_protocol_verified'] = False
            raise RuntimeError(f'camera pre-action/action_index mapping is incomplete in env {env_id}')
        if frame_ids[env_id] <= before_frame_ids[env_id]:
            item['camera_frame_ids_strictly_increasing'] = False
            item['capture_protocol_verified'] = False
            raise RuntimeError(f'wrist camera frame did not advance in env {env_id}')
        item['frames_verified'] += 1
        item['pose_verified'] = True
        item['rgb_frames_captured'] += 1
        item['valid_rgb_frames'] += 1
        frame_id = int(frame_ids[env_id])
        previous = item['last_camera_frame_id']
        item['camera_frame_ids_available'] = True
        if previous is not None and frame_id <= previous:
            item['camera_frame_ids_strictly_increasing'] = False
            item['capture_protocol_verified'] = False
            raise RuntimeError(f'wrist camera frame did not advance in env {env_id}')
        if item['first_camera_frame_id'] is None:
            item['first_camera_frame_id'] = frame_id
        item['last_camera_frame_id'] = frame_id
        item['capture_protocol_verified'] = True
        item['timestamp_action_index_mapping_complete'] = True
        item['max_position_error_m'] = max(item['max_position_error_m'], float(pos_error[env_id]))
        item['max_rotation_error_rad'] = max(item['max_rotation_error_rad'], float(rot_error[env_id]))
    panel_pose = _body_pose(env, 'door', 'door_panel')
    arms = _field(env.scene.sensors['arm_contacts'].data, 'net_normal_forces_w', 'net_forces_w')
    ids = np.flatnonzero(active)
    state.observe_safety(ids, arms[ids], _camera_clearance(expected, panel_pose)[ids])
    frame = state.frames.copy()
    env._arx_pre_action_camera = {
        'wrist_camera_pose': actual.copy(), 'wrist_camera_expected_pose': expected.copy(),
        'timestamp': frame[:, None] / 30., 'action_index': frame[:, None],
    }
    env._arx_pre_action_camera['camera_frame_id'] = frame_ids[:, None].copy()
    # Replaces the official flat policy recorder rather than double-appending
    # obs/joint_pos. Preserve additional Arena policy fields in the same record.
    obs = dict(getattr(env, 'obs_buf', {}).get('policy', {}))
    obs['joint_pos'] = _tensor(env, _joint_state(env, 'robot', getattr(env.cfg, 'arm_state_names', ARM_STATE_NAMES)))
    obs['wrist_rgb'] = _tensor(env, rgb[..., :3])
    if 'ego_camera' in env.scene.sensors:
        ego = env.scene.sensors['ego_camera']
        ego.update(0., force_recompute=True)
        obs['ego_rgb'] = _tensor(env, as_numpy(ego.data.output['rgb'])[..., :3])
    env._arx_joint_target_substeps = []
    return 'obs', obs


def capture_joint_target_substep(env):
    """Record the actual actuator input after each 120 Hz physics step.

    Differential IK recomputes targets four times per 30 Hz pose action. These
    targets are diagnostic evidence, not interchangeable with a single joint
    command held for the whole control period.
    """
    robot = env.scene['robot']
    state_names = getattr(env.cfg, 'arm_state_names', ARM_STATE_NAMES)
    ids, names = robot.find_joints(state_names, preserve_order=True)
    if list(names) != list(state_names):
        raise ValueError('actuator target joint order differs from the state contract')
    env._arx_joint_target_substeps.append(
        as_numpy(robot.actuators.target_command.position)[:, ids].copy())


def capture_post_step(env):
    """Append physical diagnostics and update acceptance after the applied action."""
    cached = getattr(env, '_arx_pre_action_camera', None)
    if cached is None:
        raise RuntimeError('post-step recording requires pre-action camera evidence')
    physical = read_physical_state(env)
    state = acceptance_state(env)
    active = as_numpy(getattr(env, 'active_episode_mask', np.ones(env.num_envs, bool))).astype(bool)
    ids = np.flatnonzero(active)
    state.update(ids, physical['door_joint_pos'][ids], physical['finger_contacts'][ids],
                 physical['arm_contacts'][ids], physical['camera_clearance'][ids],
                 panel_normal=physical['panel_pose'][ids, :3, 1])
    diagnostics = {**physical, **cached,
                   'env_id': np.arange(env.num_envs, dtype=np.int64)[:, None],
                   'physics_timestamp': state.frames[:, None] / 30.,
                   'runtime_accepted': state.accepted()[:, None],
                   'runtime_finite_evidence': state.finite[:, None],
                   'runtime_max_arm_contact_n': state.max_arm[:, None],
                   'runtime_min_camera_clearance_m': state.min_clearance[:, None]}
    targets = getattr(env, '_arx_joint_target_substeps', [])
    if len(targets) != env.cfg.decimation:
        raise RuntimeError(f'expected {env.cfg.decimation} actuator samples, got {len(targets)}')
    diagnostics['joint_target_substeps'] = np.stack(targets, axis=1)
    for env_id in ids:
        if state.frames[env_id] % 300 == 0:
            print(f'ARX_RECORD env={env_id} frames={state.frames[env_id]} '
                  f'door_deg={np.rad2deg(state.final_door[env_id, 0]):.2f} '
                  f'accepted={bool(state.accepted([env_id])[0])}', flush=True)
    env._arx_pre_action_camera = None
    return 'obs/diagnostics', {key: _tensor(env, value) for key, value in diagnostics.items()}


def reset_recording(env, env_ids=None):
    """Reset episode evidence after export, keeping other environments intact."""
    state = acceptance_state(env)
    ids = state._ids(env_ids)
    reset_acceptance(env, ids)
    evidence = getattr(env, '_arx_camera_evidence', {})
    for env_id in ids:
        evidence.pop(int(env_id), None)


@lru_cache(maxsize=1)
def _recorder_classes():
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.utils import configclass

    class ArxPreActionRecorder(RecorderTerm):
        def record_pre_step(self):
            return capture_pre_step(self._env)

    class ArxPostStepRecorder(RecorderTerm):
        def record_post_physics_decimation_step(self):
            capture_joint_target_substep(self._env)
            return None, None

        def record_post_step(self):
            return capture_post_step(self._env)

        def record_post_reset(self, env_ids):
            reset_recording(self._env, env_ids)
            return None, None

    @configclass
    class ArxPreActionRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = ArxPreActionRecorder

    @configclass
    class ArxPostStepRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = ArxPostStepRecorder

    # Config serialization uses module:ClassName. Publish stable names instead
    # of exposing function-local qualified names that cannot be resolved later.
    for cls in (ArxPreActionRecorder, ArxPostStepRecorder,
                ArxPreActionRecorderCfg, ArxPostStepRecorderCfg):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls
    return ArxPreActionRecorderCfg, ArxPostStepRecorderCfg


def __getattr__(name):
    if name in ('ArxPreActionRecorder', 'ArxPostStepRecorder',
                'ArxPreActionRecorderCfg', 'ArxPostStepRecorderCfg'):
        _recorder_classes()
        return globals()[name]
    raise AttributeError(name)


def merge_arena_recorders(arena_cfg):
    """Preserve Arena terms, add official ActionState terms and ARX diagnostics.

    Pass the returned config to AutoData ``setup_env_config(recorder_cfg=...)``.
    No inherited recorder config instance is mutated. The policy term is replaced
    with synchronized capture so every field has exactly one pre-action row.
    """
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import RecorderManagerBaseCfg
    from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

    pre_cfg, post_cfg = _recorder_classes()
    domain = make_configclass('ArxDoorRecorderTermsCfg', [
        ('record_arx_post_step_diagnostics', post_cfg, post_cfg()),
    ])()
    merged = combine_configclass_instances('ArxAutoDataRecorderManagerCfg',
        ActionStateRecorderManagerCfg(), deepcopy(arena_cfg), domain, bases=(RecorderManagerBaseCfg,))
    merged.record_pre_step_flat_policy_observations = pre_cfg()
    return merged
