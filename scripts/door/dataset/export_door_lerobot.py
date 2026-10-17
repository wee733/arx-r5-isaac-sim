#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Export reviewed seeds or physically accepted SkillGen trials to LeRobot v2.1.

Source actions are already absolute controller targets, never reconstructed
from the next observation. Input success/review flags are never modified.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


# One instruction per demonstrated behavior: a knob_hold_open episode never
# releases the knob, so the edge-regrasp text would mislabel it.
TASKS = {
    'edge_regrasp': ('Grasp the round doorknob, turn it counterclockwise 45 degrees, pull a '
                     'gap, release the knob, grasp the door side edge, and pull the door open.'),
    'knob_hold_open': ('Grasp the round doorknob, turn it counterclockwise 45 degrees, and '
                       'keep holding the knob while pulling the door open.'),
}
TASK = TASKS['edge_regrasp']
VIDEO_KEY = 'observation.images.wrist'
# Data v2 adds the body-fixed ego view (arx_door.scene.ego_camera).
VIDEO_KEYS = {'wrist': VIDEO_KEY, 'ego': 'observation.images.ego'}
VIDEO_ARTIFACTS = {'wrist': 'wrist_video', 'ego': 'ego_video'}
STAGE_KEY = 'annotation.human.stage_description'
JOINT_NAMES = [f'joint{i}' for i in range(1, 8)]


def episode_task(args, provenance):
    """Explicit --task wins; otherwise describe the recorded behavior family."""
    if getattr(args, 'task', None):
        return args.task
    result = provenance.get('source_result') or provenance.get('source_episode') or {}
    family = result.get('behavior_family')
    if family is None:
        family = (result.get('expert_parameters') or {}).get('behavior_family', 'edge_regrasp')
    if family not in TASKS:
        raise ValueError(f'no task instruction for behavior family {family!r}')
    return TASKS[family]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(v, allow_nan=False) + '\n' for v in values))


def probe_video(path):
    """Count decoded video frames and verify the actual encoded stream."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,codec_name,pix_fmt',
         '-of', 'json', str(path)], check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1:
        raise ValueError(f'expected one wrist video stream: {path}')
    return streams[0]


def statistics(array):
    """Basic absolute numeric statistics; official GR00T stats remain pending."""
    return {
        'min': array.min(axis=0).tolist(), 'max': array.max(axis=0).tolist(),
        'mean': array.mean(axis=0).tolist(), 'std': array.std(axis=0).tolist(),
        'count': [len(array)],
    }


def inspect_attempt(path, test_only):
    """Fail closed before copying any rejected demonstration."""
    metadata = json.loads((path / 'episode.json').read_text())
    if metadata.get('success') is not True or metadata.get('review_required') is not False:
        raise ValueError(f'REJECTED {path}: success=true and review_required=false required')
    if bool(metadata.get('test_only', False)) != test_only:
        raise ValueError(f'{path}: TEST_ONLY fixtures must be exported separately with --test-only')
    return metadata


def file_hash(path):
    """Hash source artifacts without loading an entire video into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def result_directory(path):
    """Accept a trial directory or its explicit result.json."""
    if path.name == 'result.json':
        path = path.parent
    return path.resolve()


def read_generated_result(path, test_only):
    """Read immutable physical provenance, never imply a new user approval."""
    result = json.loads((path / 'result.json').read_text())
    if result.get('schema') != 'arx-door-skillgen-result-v1':
        raise ValueError(f'{path}: unsupported SkillGen result schema')
    if bool(result.get('test_only', False)) != test_only:
        raise ValueError(f'{path}: TEST_ONLY fixtures must be exported separately with --test-only')
    if not isinstance(result.get('success'), bool):
        raise ValueError(f'{path}: result success must be boolean')
    return result


def generated_sources(args):
    """Resolve batch lineage and exclude failed trials and other splits."""
    selected, excluded, seen = [], [], set()
    split = getattr(args, 'split', None)

    def add(path, result, lineage, allow_failure):
        if path in seen:
            raise ValueError(f'duplicate generated source path: {path}')
        seen.add(path)
        if split is not None and result.get('dataset_split', 'candidate') != split:
            # Holdout trials must never contribute to training normalization.
            return
        provenance = {
            'source_kind': 'skillgen_physical', 'source': str(path),
            'result_sha256': file_hash(path / 'result.json'), 'source_result': result,
            'angle_deg': result.get('angle_deg'), 'user_reviewed': False, **lineage,
        }
        if allow_failure and result['success'] is False:
            if not result.get('failure_stage') or not result.get('failure_reason'):
                raise ValueError(f'{path}: failed trial requires failure stage and reason')
            excluded.append({**provenance, 'reason': 'result.success=false',
                             'failure_stage': result['failure_stage'],
                             'failure_reason': result['failure_reason']})
        else:
            selected.append((path, result, provenance))

    for value in getattr(args, 'generated_attempt', ()) or ():
        path = result_directory(value)
        add(path, read_generated_result(path, args.test_only), {}, False)
    reports = [path.resolve() for path in getattr(args, 'batch_report', ()) or ()]
    if len(set(reports)) != len(reports):
        raise ValueError('duplicate batch report path')
    for report_path in reports:
        report = json.loads(report_path.read_text())
        if report.get('schema') != 'arx-door-skillgen-batch-v1':
            raise ValueError(f'{report_path}: unsupported batch schema')
        trials = report.get('trials')
        if not isinstance(trials, list) or not trials:
            raise ValueError(f'{report_path}: nonempty trials required')
        for trial in trials:
            value = Path(trial['output_dir'])
            path = result_directory(value if value.is_absolute() else report_path.parent / value)
            result = read_generated_result(path, args.test_only)
            snapshot = trial['result']
            # Batch validators add probe/resolved-artifact fields. Compare all
            # original fields against the disk result, not those enrichments.
            if any(snapshot.get(key) != value for key, value in result.items()):
                raise ValueError(f'{path}: batch result snapshot does not match result.json')
            if 'angle_deg' in trial and trial['angle_deg'] != result.get('angle_deg'):
                raise ValueError(f'{path}: batch trial angle mismatch')
            add(path, result, {'batch_report': str(report_path),
                               'batch_report_sha256': file_hash(report_path),
                               'trial_index': trial.get('index'), 'trial_seed': trial.get('seed')}, True)
        successes = sum(trial['result']['success'] is True for trial in trials)
        for key, value in (('attempts', len(trials)), ('successes', successes),
                           ('failures', len(trials) - successes)):
            if key in report and report[key] != value:
                raise ValueError(f'{report_path}: batch {key} count mismatch')
    return selected, excluded


def artifact_path(path, value):
    """Resolve the source's own artifact manifest, supporting relocation."""
    if not isinstance(value, str) or not value:
        raise ValueError(f'{path}: missing artifact path')
    value = Path(value)
    return (value if value.is_absolute() else path / value).resolve()


def validate_camera_se3(path, value, shape, label):
    """Validate finite homogeneous poses, allowing float32 roundoff only."""
    import numpy as np

    try:
        poses = np.asarray(value, dtype='float64')
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path}: camera {label} is not a numeric SE3 transform') from error
    if poses.shape != shape or not np.isfinite(poses).all():
        raise ValueError(f'{path}: camera {label} must be finite {shape}')
    rotation = poses[..., :3, :3]
    if (not np.allclose(poses[..., 3, :], [0., 0., 0., 1.], atol=1e-6, rtol=0)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
        raise ValueError(f'{path}: camera {label} must be a valid SE3 transform')
    return poses


def verify_camera_evidence(path, evidence, data, n):
    """Recompute the full live-mount audit instead of trusting its summary."""
    import numpy as np

    if (evidence.get('schema') != 'arx-wrist-camera-live-mount-v2'
            or evidence.get('mount_link') != 'link6' or evidence.get('use_fabric') is not True
            or evidence.get('camera_prim_path') != '/World/SkillGenWristCamera'
            or type(evidence.get('frames_verified')) is not int or evidence['frames_verified'] != n):
        raise ValueError(f'{path}: camera v2 live link6 mount at /World/SkillGenWristCamera '
                         f'with use_fabric=true and {n} verified frames required')
    validate_camera_se3(path, evidence.get('mount_transform'), (4, 4), 'mount_transform')
    limits = {'max_position_error_m': 1e-4, 'max_rotation_error_rad': 1e-3}
    for key, limit in limits.items():
        value = evidence.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0. <= value < limit:
            raise ValueError(f'{path}: camera {key} must be finite, nonnegative and below {limit}')
    actual = validate_camera_se3(path, data['wrist_camera_pose'], (n, 4, 4), 'actual poses')
    expected = validate_camera_se3(path, data['wrist_camera_expected_pose'], (n, 4, 4), 'expected poses')
    position_error = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=-1)
    # Project float32 rotations onto SO(3) before computing their relative
    # angle. atan2(skew, trace) remains stable near zero; plain arccos(trace)
    # can turn storage roundoff into a false sub-milliradian camera drift.
    rotations = []
    for poses in (actual, expected):
        left, _, right = np.linalg.svd(poses[:, :3, :3])
        rotations.append(left @ right)
    relative = rotations[0] @ rotations[1].swapaxes(-1, -2)
    skew = np.stack((relative[:, 2, 1] - relative[:, 1, 2],
                     relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]), axis=-1)
    sine = np.linalg.norm(skew, axis=-1) / 2.
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.) / 2., -1., 1.)
    rotation_error = np.arctan2(sine, cosine)
    measured = {'max_position_error_m': float(position_error.max()),
                'max_rotation_error_rad': float(rotation_error.max())}
    for key, value in measured.items():
        if not math.isfinite(value) or value >= limits[key]:
            raise ValueError(f'{path}: camera per-frame {key} exceeds the live-mount tolerance')
        if not math.isclose(value, evidence[key], abs_tol=3e-7, rel_tol=0.):
            raise ValueError(f'{path}: camera {key} summary does not match the recorded poses')
    return {'schema': evidence['schema'], 'camera_prim_path': evidence['camera_prim_path'],
            'use_fabric': True, 'frames_verified': n, **measured, 'summary_roundoff_tolerance': 3e-7,
            'actual_pose_source': 'data/demo_0/obs/diagnostics/wrist_camera_pose',
            'expected_pose_source': 'data/demo_0/obs/diagnostics/wrist_camera_expected_pose',
            'mount_verification_source': 'result.json/camera_verification'}


def trim_video(source, destination, start):
    """Re-encode from frame ``start`` with fresh timestamps (crop at export)."""
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vf',
                    f'trim=start_frame={start},setpts=PTS-STARTPTS', '-an', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', '30',
                    str(destination)], check=True, capture_output=True)


SEED_CACHE = {}


def annotate_stages(path, result, hdf5_path):
    """Recover per-frame stages from the reviewed seed; see arx_door.dataset.stages."""
    import h5py

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from arx_door.dataset.stages import annotate_generated, load_seed_skill

    seed_path = artifact_path(path, result.get('source_seed'))
    if seed_path not in SEED_CACHE:
        SEED_CACHE[seed_path] = load_seed_skill(seed_path)
    try:
        with h5py.File(hdf5_path, 'r') as file:
            return annotate_generated(file['data/demo_0'], SEED_CACHE[seed_path])
    except (KeyError, ValueError) as error:
        raise ValueError(f'{path}: stage alignment failed: {error}') from error


def load_generated(path, result, provenance):
    """Read native pre-action data and verify final absolute-command semantics.

    ActionsCfg.arm_action uses JointPositionActionCfg(scale=1.0,
    use_default_offset=False, preserve_order=True). MirroredGripperAction
    clips joint7 to [0, 0.044] metres. The native post-step recorder saves
    processed_actions from the action manager after that exact command is
    applied; diagnostics and wrist RGB are captured before env.step.
    """
    import h5py
    import numpy as np

    if result['success'] is not True or result.get('physics_executed') is not True:
        raise ValueError(f'{path}: success=true and physics_executed=true required')
    if result.get('video_alignment') != 'pre_action_observation_at_30_hz':
        raise ValueError(f'{path}: pre_action_observation_at_30_hz required')
    if (result.get('environment') != 'Isaac-ARX-R5-Door-SkillGen-v0'
            or not isinstance(result.get('angle_deg'), (int, float))
            or not math.isfinite(result['angle_deg'])):
        raise ValueError(f'{path}: invalid environment or trial angle')
    n = result.get('frames')
    if (type(n) is not int or n < 2 or result.get('hdf5_frames') != n
            or result.get('hdf5_episodes') != 1):
        raise ValueError(f'{path}: invalid generated frame counts')
    audit = result.get('physical_audit', {})
    if not isinstance(audit, dict):
        raise ValueError(f'{path}: invalid physical audit object')
    for key in ('knob_grasp_observed', 'unlock_observed', 'gap_with_knob_grasp_observed'):
        if audit.get(key) is not True:
            raise ValueError(f'{path}: invalid physical audit: {key}')
    for key, lower, upper in (
        ('final_door_angle_deg', 29., math.inf), ('max_handle_angle_deg', 43.5, math.inf),
        ('max_latch_m', .0154, math.inf), ('min_camera_clearance_m', .01, math.inf),
        ('max_arm_contact_n', 0., 40.),
    ):
        value = audit.get(key)
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or not lower <= value <= upper):
            raise ValueError(f'{path}: invalid physical audit: {key}')
    if (audit.get('physics_steps') != n
            or result.get('final_door_angle_deg') != audit['final_door_angle_deg']):
        raise ValueError(f'{path}: physical audit does not match result frame count/final angle')
    camera_evidence = result.get('camera_verification')
    if not isinstance(camera_evidence, dict):
        raise ValueError(f'{path}: camera_verification live-mount evidence is required')
    artifacts = result.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError(f'{path}: missing artifact manifest')
    hdf5_path = artifact_path(path, artifacts.get('hdf5'))
    video_path = artifact_path(path, artifacts.get('wrist_video'))
    provenance['video_artifacts'] = {
        camera: str(artifact_path(path, artifacts[key])) for camera, key in VIDEO_ARTIFACTS.items()
        if artifacts.get(key)}
    try:
        with h5py.File(hdf5_path, 'r') as file:
            group = file['data']
            if list(group) != ['demo_0']:
                raise ValueError(f'{path}: expected exactly data/demo_0')
            demo = group['demo_0']
            if not isinstance(demo.attrs.get('success'), (bool, np.bool_)) or not demo.attrs['success']:
                raise ValueError(f'{path}: HDF5 demo success=true required')
            if demo.attrs.get('num_samples') != n or group.attrs.get('total') != n:
                raise ValueError(f'{path}: HDF5 frame count attributes mismatch')
            env = json.loads(group.attrs['env_args'])
            if env.get('env_name') != result['environment']:
                raise ValueError(f'{path}: HDF5 environment mismatch')
            data = {key: demo[field][:] for key, field in {
                'raw_action': 'actions', 'action': 'processed_actions', 'state': 'obs/joint_pos',
                'timestamp': 'obs/diagnostics/timestamp', 'door': 'obs/diagnostics/door_joint_pos',
                'phase': 'obs/diagnostics/transition_index',
                'wrist_camera_pose': 'obs/diagnostics/wrist_camera_pose',
                'wrist_camera_expected_pose': 'obs/diagnostics/wrist_camera_expected_pose',
            }.items()}
    except (OSError, KeyError) as error:
        raise ValueError(f'{path}: invalid HDF5: {error}') from error
    for key, shape in (('raw_action', (n, 7)), ('action', (n, 7)), ('state', (n, 7)),
                       ('timestamp', (n, 1)), ('door', (n, 3)), ('phase', (n, 1))):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'{path}: {key} must be finite {shape}')
    provenance['camera_pose_validation'] = verify_camera_evidence(path, camera_evidence, data, n)
    expected = data['raw_action'].copy()
    expected[:, 6] = np.clip(expected[:, 6], 0., .044)
    if not np.allclose(data['action'], expected, atol=1e-7, rtol=0):
        raise ValueError(f'{path}: processed_actions mismatch native absolute arm / clipped gripper semantics')
    data['timestamp'] = data['timestamp'][:, 0]
    data['phase'] = data['phase'][:, 0]
    if np.any(data['phase'] < 0) or np.any(data['phase'] != np.floor(data['phase'])):
        raise ValueError(f'{path}: transition indices must be nonnegative integers')
    seed_path = artifact_path(path, result.get('source_seed'))
    package = Path(__file__).resolve().parents[3] / 'arx_door/skillgen'
    provenance.update({
        'hdf5': str(hdf5_path), 'hdf5_sha256': file_hash(hdf5_path),
        'wrist_video': str(video_path), 'wrist_video_sha256': file_hash(video_path),
        'source_seed': str(seed_path), 'source_seed_sha256': file_hash(seed_path),
        'action_source': 'data/demo_0/processed_actions',
        'action_semantics_verified': 'raw arm identity; gripper clip [0,0.044] metres',
        'native_action_source_sha256': {name: file_hash(package / name) for name in ('env_cfg.py', 'mdp.py')},
    })
    return data, video_path, provenance


def export(args):
    """Validate all attempts, then create a new dataset without overwriting."""
    attempts = [path.resolve() for path in args.attempt]
    if len(set(attempts)) != len(attempts):
        raise ValueError('duplicate attempt path')
    metadata = [inspect_attempt(path, args.test_only) for path in attempts]
    generated, excluded = generated_sources(args)
    if not attempts and not generated:
        raise ValueError('at least one accepted source required; all batch trials may have failed')
    output = args.output.resolve()
    runtime = Path(__file__).resolve().parents[3] / 'generated/runtime'
    if args.test_only and (runtime not in output.parents or 'TEST_ONLY' not in output.name):
        raise ValueError('test fixture output must be generated/runtime/.../TEST_ONLY...')
    if output.exists():
        raise ValueError(f'output already exists; choose a fresh directory: {output}')

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    cameras = list(dict.fromkeys(getattr(args, 'camera', None) or ['wrist']))
    if cameras[0] != 'wrist' or any(camera not in VIDEO_KEYS for camera in cameras):
        raise ValueError(f'cameras must start with wrist and be among {sorted(VIDEO_KEYS)}')
    stage_language = bool(getattr(args, 'stage_language', False))
    crop_recovery = bool(getattr(args, 'crop_before_recovery', False))
    max_overshoot = getattr(args, 'max_transition_overshoot', None)
    annotate = stage_language or crop_recovery or max_overshoot is not None
    if attempts and (annotate or cameras != ['wrist']):
        raise ValueError('ego video, stage labels and recovery crops require generated sources')
    loaded = []
    fps = float(metadata[0]['fps']) if metadata else 30.0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    for path, meta in zip(attempts, metadata):
        if float(meta['fps']) != fps:
            raise ValueError('all attempts must have the same fps')
        with np.load(path / 'trajectory.npz', allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in ('state', 'action', 'timestamp', 'phase', 'door')}
        n = int(meta['frames'])
        if n < 2:
            raise ValueError(f'{path}: at least two frames required')
        for key in ('state', 'action'):
            if data[key].shape != (n, 7) or not np.isfinite(data[key]).all():
                raise ValueError(f'{path}: {key} must be finite [{n},7]')
        if data['timestamp'].shape != (n,) or not np.isfinite(data['timestamp']).all():
            raise ValueError(f'{path}: invalid timestamps')
        if not np.allclose(data['timestamp'], np.arange(n) / fps, atol=1e-5, rtol=0):
            raise ValueError(f'{path}: timestamps must start at zero and align to video fps')
        if data['phase'].shape != (n,) or data['door'].shape[0] != n:
            raise ValueError(f'{path}: diagnostics do not align to frames')
        video = probe_video(path / 'wrist.mp4')
        if int(video['nb_read_frames']) != n:
            raise ValueError(f'{path}: wrist video frame count does not match trajectory')
        if abs(float(Fraction(video['avg_frame_rate'])) - fps) > 1e-6:
            raise ValueError(f'{path}: encoded video fps mismatch')
        if loaded and any(video[k] != loaded[0][2][k] for k in ('width', 'height', 'codec_name', 'pix_fmt')):
            raise ValueError('all wrist videos must use the same resolution and encoding')
        loaded.append((path, data, video, n, path / 'wrist.mp4', {
            'source_kind': 'reviewed_npz', 'trajectory_sha256': file_hash(path / 'trajectory.npz'),
            'source_episode': meta,
        }))

    for path, result, provenance in generated:
        if fps != 30.0:
            raise ValueError('generated attempts require 30 Hz throughout the dataset')
        data, video_path, provenance = load_generated(path, result, provenance)
        n = result['frames']
        if not np.allclose(data['timestamp'], np.arange(n) / fps, atol=1e-5, rtol=0):
            raise ValueError(f'{path}: timestamps must start at zero and align to video fps')
        video = probe_video(video_path)
        if int(video['nb_read_frames']) != n:
            raise ValueError(f'{path}: wrist video frame count does not match HDF5')
        if abs(float(Fraction(video['avg_frame_rate'])) - fps) > 1e-6:
            raise ValueError(f'{path}: encoded video fps mismatch')
        if (video['width'], video['height'], video['codec_name'], video['pix_fmt']) != (640, 480, 'h264', 'yuv420p'):
            raise ValueError(f'{path}: wrist video must be 640x480 h264 yuv420p')
        if loaded and any(video[k] != loaded[0][2][k] for k in ('width', 'height', 'codec_name', 'pix_fmt')):
            raise ValueError('all wrist videos must use the same resolution and encoding')
        data['videos'] = {'wrist': video_path}
        for camera in cameras[1:]:
            extra = provenance['video_artifacts'].get(camera)
            if extra is None:
                raise ValueError(f'{path}: requested {camera} video is not in the artifact manifest')
            extra = Path(extra)
            probe = probe_video(extra)
            if (int(probe['nb_read_frames']) != n
                    or (probe['width'], probe['height'], probe['codec_name'], probe['pix_fmt'])
                    != (640, 480, 'h264', 'yuv420p')):
                raise ValueError(f'{path}: {camera} video must be {n} frames of 640x480 h264 yuv420p')
            data['videos'][camera] = extra
            provenance[f'{camera}_video_sha256'] = file_hash(extra)
        if annotate:
            annotation = annotate_stages(path, result, Path(provenance['hdf5']))
            alignment = annotation['alignment']
            provenance['stage_alignment'] = {**alignment.to_json(),
                                             'first_transition': annotation['transition'],
                                             'recovery': annotation['recovery']}
            crop = alignment.recovery_start if crop_recovery and alignment.recovery_start else 0
            # Only planned motion kept in the episode is gated: a cropped
            # perturbation transition is not training data.
            kept = [annotation['recovery']] if crop else [annotation['transition'], annotation['recovery']]
            overshoot = max(item['max_overshoot_rad'] for item in kept if item is not None)
            if max_overshoot is not None and overshoot > max_overshoot:
                excluded.append({**provenance, 'reason': 'transition_overshoot',
                                 'max_overshoot_rad': overshoot,
                                 'limit_rad': max_overshoot})
                continue
            data['stage'] = annotation['stage']
            data['frame_source'] = alignment.source
            data['seed_frame'] = alignment.seed_frame
            if crop:
                for key in ('state', 'action', 'phase', 'door', 'stage', 'frame_source',
                            'seed_frame', 'wrist_camera_pose', 'wrist_camera_expected_pose'):
                    data[key] = data[key][crop:]
                n -= crop
                data['timestamp'] = (np.arange(n) / fps).astype(data['timestamp'].dtype)
                provenance['cropped_leading_frames'] = int(crop)
            data['crop'] = int(crop)
        loaded.append((path, data, video, n, video_path, provenance))

    if not loaded:
        raise ValueError('no source passed the export gates; see the excluded reasons')
    seen_content = set()
    for path, data, video, n, video_path, provenance in loaded:
        digest = hashlib.sha256()
        for key in ('state', 'action', 'timestamp'):
            digest.update(np.ascontiguousarray(data[key], dtype='<f4').tobytes())
        content_hash = digest.hexdigest()
        if content_hash in seen_content:
            raise ValueError(f'duplicate source content: {path}')
        seen_content.add(content_hash)
        provenance['trajectory_content_sha256'] = content_hash

    # Build in a separate directory so a partial export is never published as
    # a completed dataset. A failure leaves this clearly named directory for inspection.
    staging = output.with_name(output.name + '.incomplete')
    staging.mkdir(parents=True, exist_ok=False)
    meta_dir = staging / 'meta'
    meta_dir.mkdir()
    (staging / 'diagnostics').mkdir()
    episodes, episode_stats, provenance = [], [], []
    offset = 0
    all_states, all_actions = [], []
    task_texts = [episode_task(args, provenance) for *_, provenance in loaded]
    task_list = list(dict.fromkeys(task_texts))
    if stage_language:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from arx_door.dataset.stages import KNOB_HOLD_OPEN_STAGES, STAGE_NAMES
        if any(text != TASKS['knob_hold_open'] for text in task_texts):
            raise ValueError('stage language is defined for knob_hold_open episodes only')
        stage_texts = [stage.instruction for stage in KNOB_HOLD_OPEN_STAGES]
        task_list = list(dict.fromkeys(task_list + stage_texts))
    for index, (path, data, video, n, video_path, source_provenance) in enumerate(loaded):
        task_index = task_list.index(task_texts[index])
        chunk = index // 1000
        data_dir = staging / f'data/chunk-{chunk:03d}'
        data_dir.mkdir(parents=True, exist_ok=True)
        state, action = data['state'].astype('float32'), data['action'].astype('float32')
        task_indices = pa.array(np.full(n, task_index), type=pa.int64())
        columns = {}
        episode_texts = [task_texts[index]]
        if stage_language:
            stage_index = np.asarray([task_list.index(stage_texts[label]) for label in data['stage']])
            columns[STAGE_KEY] = pa.array(stage_index, type=pa.int64())
            episode_texts += [stage_texts[label] for label in dict.fromkeys(data['stage'].tolist())]
        table = pa.table({
            'observation.state': pa.array(state.tolist(), type=pa.list_(pa.float32(), 7)),
            'action': pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
            'timestamp': pa.array(data['timestamp'], type=pa.float32()),
            'frame_index': pa.array(np.arange(n), type=pa.int64()),
            'episode_index': pa.array(np.full(n, index), type=pa.int64()),
            'index': pa.array(np.arange(offset, offset + n), type=pa.int64()),
            'task_index': task_indices,
            'annotation.human.task_description': task_indices,
            'next.done': pa.array([False] * (n - 1) + [True], type=pa.bool_()),
            'next.reward': pa.array([0.0] * (n - 1) + [1.0], type=pa.float32()),
            **columns,
        })
        filename = f'episode_{index:06d}'
        pq.write_table(table, data_dir / (filename + '.parquet'))
        for camera, camera_path in data.get('videos', {'wrist': video_path}).items():
            video_dir = staging / f'videos/chunk-{chunk:03d}' / VIDEO_KEYS[camera]
            video_dir.mkdir(parents=True, exist_ok=True)
            target = video_dir / (filename + '.mp4')
            if data.get('crop'):
                trim_video(camera_path, target, data['crop'])
                if int(probe_video(target)['nb_read_frames']) != n:
                    raise ValueError(f'{path}: cropped {camera} video frame count mismatch')
            else:
                shutil.copyfile(camera_path, target)
        camera_diagnostics = {key: data[key] for key in ('wrist_camera_pose', 'wrist_camera_expected_pose',
                                                         'stage', 'frame_source', 'seed_frame')
                              if key in data}
        np.savez_compressed(staging / 'diagnostics' / (filename + '.npz'),
                            phase=data['phase'], door=data['door'], timestamp=data['timestamp'],
                            **camera_diagnostics)
        stats = {'observation.state': statistics(state), 'action': statistics(action)}
        episode_stats.append({'episode_index': index, 'stats': stats})
        episodes.append({'episode_index': index, 'tasks': episode_texts, 'length': n})
        provenance.append({
            'episode_index': index, 'source': str(path),
            **source_provenance,
        })
        offset += n
        all_states.append(state)
        all_actions.append(action)

    features = {
        key: {'dtype': 'float32', 'shape': [7], 'names': JOINT_NAMES}
        for key in ('observation.state', 'action')
    }
    video = loaded[0][2]
    for camera in cameras:
        features[VIDEO_KEYS[camera]] = {
            'dtype': 'video', 'shape': [video['height'], video['width'], 3],
            'names': ['height', 'width', 'channels'],
            'info': {'video.height': video['height'], 'video.width': video['width'],
                     'video.codec': video['codec_name'], 'video.pix_fmt': video['pix_fmt'],
                     'video.is_depth_map': False, 'video.fps': fps,
                     'video.channels': 3, 'has_audio': False},
        }
    for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index',
                'annotation.human.task_description', 'next.done', 'next.reward',
                *([STAGE_KEY] if stage_language else [])):
        dtype = ('float32' if key in ('timestamp', 'next.reward') else
                 'bool' if key == 'next.done' else 'int64')
        features[key] = {'dtype': dtype, 'shape': [1], 'names': None}
    write_json(meta_dir / 'info.json', {
        'codebase_version': 'v2.1', 'robot_type': 'arx_r5a',
        'total_episodes': len(loaded), 'total_frames': offset, 'total_tasks': len(task_list),
        'total_videos': len(loaded) * len(cameras), 'total_chunks': (len(loaded) + 999) // 1000,
        'chunks_size': 1000, 'fps': fps,
        'splits': {('train' if getattr(args, 'split', None) in (None, 'candidate')
                    else args.split): f'0:{len(loaded)}'},
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'features': features,
    })
    write_json(meta_dir / 'modality.json', {
        'state': {'single_arm': {'start': 0, 'end': 6}, 'gripper': {'start': 6, 'end': 7}},
        'action': {'single_arm': {'start': 0, 'end': 6}, 'gripper': {'start': 6, 'end': 7}},
        'video': {camera: {'original_key': VIDEO_KEYS[camera]} for camera in cameras},
        'annotation': {'human.task_description': {'original_key': 'annotation.human.task_description'},
                       **({'human.stage_description': {'original_key': STAGE_KEY}}
                          if stage_language else {})},
    })
    write_jsonl(meta_dir / 'episodes.jsonl', episodes)
    write_jsonl(meta_dir / 'tasks.jsonl',
                [{'task_index': index, 'task': task} for index, task in enumerate(task_list)])
    write_jsonl(meta_dir / 'episodes_stats.jsonl', episode_stats)
    write_json(meta_dir / 'stats.json', {
        'observation.state': statistics(np.concatenate(all_states)),
        'action': statistics(np.concatenate(all_actions)),
    })
    write_json(meta_dir / 'door_export_report.json', {
        'test_only': args.test_only, 'official_n1_7_loader_verified': False,
        'statistics': 'basic absolute state/action only; regenerate with the official N1.7 pipeline',
        'action_semantics': 'absolute joint1..joint6 radians and joint7 prismatic metres',
        'alignment': 'source pre-step observation with commanded absolute target; no inferred actions',
        'review_gate': 'NPZ only: success=true AND review_required=false; generated samples are not user-reviewed',
        'generated_gate': 'result success=true AND physics_executed=true AND HDF5 success=true AND valid physical audit AND pre_action_observation_at_30_hz AND v2 per-frame live link6 wrist camera verification at /World/SkillGenWristCamera with Fabric enabled',
        'dataset_split': getattr(args, 'split', None),
        'cameras': cameras, 'stage_language': stage_language,
        'stage_names': list(STAGE_NAMES) if stage_language else None,
        'crop_before_recovery': crop_recovery, 'max_transition_overshoot_rad': max_overshoot,
        'sources': provenance, 'excluded_sources': excluded,
        'schema_source': 'https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md',
    })
    (staging / 'README.md').write_text(
        ('# TEST_ONLY synthetic schema fixture\n\nNOT robot expert data.\n\n' if args.test_only else '# ARX door dataset\n\n')
        + f'RGB views: {", ".join(cameras)}; 7D absolute state and command targets.\n\n'
        + 'Official GR00T N1.7 loader validation and official statistics generation have NOT run. '
        + 'The basic stats.json is incomplete for training. Regenerate official absolute and relative '
        + 'statistics and validate the single-arm data configuration before fine-tuning.\n'
    )
    staging.rename(output)
    print(f'Exported {len(loaded)} episodes / {offset} frames to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, action='append', default=[], help='reviewed NPZ attempt')
    parser.add_argument('--generated-attempt', type=Path, action='append', default=[],
                        help='physical SkillGen trial directory or result.json; repeatable')
    parser.add_argument('--batch-report', type=Path, action='append', default=[],
                        help='SkillGen batch_results.json; failed trials are excluded with reasons')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default=None,
                        help='one instruction for every episode; default picks per behavior family')
    parser.add_argument('--split', choices=['candidate', 'holdout', 'diagnostic'],
                        help='export only trials from this dataset split (recommended)')
    parser.add_argument('--test-only', action='store_true', help='isolated synthetic schema fixture only')
    parser.add_argument('--camera', action='append', choices=sorted(VIDEO_KEYS),
                        help='RGB views to export, wrist first (default: wrist); repeatable')
    parser.add_argument('--stage-language', action='store_true',
                        help='add per-frame stage instructions as annotation.human.stage_description')
    parser.add_argument('--crop-before-recovery', action='store_true',
                        help='drop the deliberate perturbation that precedes a recovery segment')
    parser.add_argument('--max-transition-overshoot', type=float,
                        help='exclude episodes whose kept planned motion (transition, recovery) overshoots by more (rad)')
    args = parser.parse_args()
    try:
        export(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Export refused: {error}\n')


if __name__ == '__main__':
    main()
