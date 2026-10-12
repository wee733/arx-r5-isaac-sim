# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Read-only physical/video quality review, independent of Isaac and GR00T."""

from __future__ import annotations

import hashlib
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import h5py
import numpy as np

from .runtime import ROOT, video_tool
from .contract import EDGE_REGRASP, KNOB_HOLD_OPEN, final_contact_object, normalize_behavior_family


def trajectory_fingerprint(actions: np.ndarray, states: np.ndarray, decimals: int) -> str:
    """One digest of canonical numeric trajectories, never hashes large files."""
    digest = hashlib.sha256()
    for values in (actions, states):
        canonical = np.round(np.asarray(values, dtype=np.float64), decimals).astype('<f8')
        canonical[canonical == 0] = 0  # canonicalize negative zero
        digest.update(str(canonical.shape).encode('ascii'))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def longest_true_run(flags: np.ndarray) -> int:
    padded = np.r_[False, flags, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def motion_summary(actions: np.ndarray, states: np.ndarray, limits: dict) -> dict:
    delta = np.abs(np.diff(actions, axis=0))
    idle = (np.max(delta[:, :6], axis=1) < limits['idle_arm_step_rad']) & (
        delta[:, 6] < limits['idle_gripper_step_m'])
    spans = np.ptp(actions, axis=0)
    return {
        'command_span': spans.tolist(),
        'state_span': np.ptp(states, axis=0).tolist(),
        'active_arm_joints': int(np.sum(spans[:6] >= limits['minimum_joint_span_rad'])),
        'command_idle_fraction': float(idle.mean()) if len(idle) else 1.0,
        'longest_command_idle_s': longest_true_run(idle) / limits['fps'],
        'duration_s': len(actions) / limits['fps'],
    }


def _artifact(directory: Path, result: dict, key: str, fallback: str) -> Path:
    """Resolve migrated artifacts without modifying their historical manifests."""
    value = result.get('artifacts', {}).get(key)
    if value:
        path = Path(value)
        path = path if path.is_absolute() else directory / path
        if path.is_file():
            return path
    path = directory / fallback
    if path.is_file():
        return path
    raise FileNotFoundError(f'{key} artifact is missing in {directory}')


def appearance_summary(directory: Path, result: dict) -> dict:
    """Verify actual scene edits; RNG labels alone never establish appearance."""
    try:
        scene_path = _artifact(directory, result, 'scene_manifest', 'scene.json')
        scene = json.loads(scene_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        scene = {}
    appearance = result.get('appearance')
    visual_seed = result.get('visual_seed', 0)
    if appearance is not None:
        if not isinstance(appearance, dict):
            raise ValueError('appearance metadata must be an object')
        if appearance.get('visual_seed', 0) != visual_seed:
            raise ValueError('result appearance and visual_seed disagree')
        if scene.get('visual_seed', 0) != visual_seed:
            raise ValueError('result and scene visual_seed disagree')
        applied = scene.get('appearance')
        if visual_seed == 0:
            # Historical zero-seed manifests predate the appearance schema.
            if appearance.get('enabled') is True or (isinstance(applied, dict) and applied.get('enabled') is True):
                raise ValueError('nominal visual seed declares enabled appearance edits')
            return {'visual_variant': 'nominal', 'visual_variant_verified': True,
                    'appearance_signature': 'nominal', 'visual_seed': 0}
        if (appearance.get('schema') != 'arx-door-appearance-v1'
                or appearance.get('enabled') is not True
                or applied != appearance):
            raise ValueError('result appearance does not match the recorded scene appearance')
        if not appearance.get('door_shader_targets') or not appearance.get('lights'):
            raise ValueError('appearance has no applied door shader or light targets')
        for label in ('door_rgb', 'background_rgb', 'wall_rgb'):
            color = np.asarray(appearance.get(label), dtype=float)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 1):
                raise ValueError(f'appearance {label} must contain finite RGB values in [0, 1]')
        multiplier = float(appearance['light_intensity_multiplier'])
        temperature = float(appearance['light_temperature_kelvin'])
        if not np.isfinite([multiplier, temperature]).all() or multiplier <= 0 or temperature <= 0:
            raise ValueError('appearance lighting parameters are invalid')
        for light in appearance['lights']:
            if (not light.get('path') or not light['source_intensity'] > 0
                    or not np.isclose(light['intensity'], light['source_intensity'] * multiplier)
                    or light['temperature_kelvin'] != temperature):
                raise ValueError('appearance light edit evidence disagrees with its parameters')
        # Identity derives from applied properties, excluding RNG labels and
        # palette names. No additional file or image hashes are required.
        identity = {key: appearance[key] for key in (
            'door_rgb', 'background_rgb', 'wall_rgb', 'light_intensity_multiplier',
            'light_temperature_kelvin', 'door_shader_targets', 'lights')}
        return {'visual_variant': f'appearance_seed_{visual_seed}', 'visual_variant_verified': True,
                'appearance_signature': json.dumps(identity, sort_keys=True, separators=(',', ':')),
                'visual_seed': visual_seed, 'appearance': appearance}
    if visual_seed != 0 or (scene.get('appearance') or {}).get('enabled') is True:
        raise ValueError('non-nominal scene lacks result appearance provenance')
    # Retain compatibility with explicit external variant metadata while the
    # native preparer uses the appearance schema above.
    visual = result.get('visual_variant')
    if visual is None:
        return {'visual_variant': 'nominal', 'visual_variant_verified': True,
                'appearance_signature': 'nominal', 'visual_seed': 0}
    if not isinstance(visual, dict):
        raise ValueError('visual_variant metadata must be an object')
    evidence = directory / str(visual.get('scene_evidence', ''))
    verified = bool(visual.get('applied') is True and visual.get('parameters') and evidence.is_file())
    if verified:
        manifest = json.loads(evidence.read_text(encoding='utf-8'))
        applied = manifest.get('visual_variant', {})
        verified = bool(applied.get('applied') is True and applied.get('id') == visual.get('id')
                        and applied.get('parameters') == visual.get('parameters'))
    return {'visual_variant': visual.get('id', 'unknown'), 'visual_variant_verified': verified,
            'appearance_signature': json.dumps(visual.get('parameters'), sort_keys=True) if verified else 'unverified',
            'visual_seed': 0}


def _finite_dataset(demo, key: str, count: int, shape: tuple | None = None) -> np.ndarray:
    value = demo[key][:]
    if not len(value) or len(value) != count or (shape and value.shape != (count, *shape)):
        raise ValueError(f'{key}: invalid shape {value.shape}, expected {(count, *(shape or ()))}')
    if not np.isfinite(value).all():
        raise ValueError(f'{key}: non-finite values')
    return value


def _first_index(flags: np.ndarray) -> int | None:
    indices = np.flatnonzero(flags)
    return int(indices[0]) if len(indices) else None


def knob_hold_contact_summary(force_n: np.ndarray, gap_frame: int | None,
                              finger_force_n: float = 3.0) -> dict:
    """Check every recorded row from the first held opening gap to the end."""
    forces = np.asarray(force_n)
    if forces.ndim != 2 or forces.shape[1] != 2 or not np.isfinite(forces).all():
        raise ValueError('knob hold forces must be finite (N, 2) finger norms')
    if gap_frame is None:
        return {'knob_hold_start_frame': None, 'knob_hold_checked_frames': 0,
                'knob_hold_contact_fraction': 0.0, 'knob_hold_first_loss_frame': None,
                'knob_hold_min_finger_force_n': []}
    if not 0 <= gap_frame < len(forces):
        raise ValueError('knob hold gap frame is outside the recorded trajectory')
    held = forces[gap_frame:]
    valid = np.min(held, axis=1) >= finger_force_n
    first_loss = _first_index(~valid)
    return {'knob_hold_start_frame': int(gap_frame), 'knob_hold_checked_frames': len(held),
            'knob_hold_contact_fraction': float(valid.mean()),
            'knob_hold_first_loss_frame': None if first_loss is None else int(gap_frame + first_loss),
            'knob_hold_min_finger_force_n': np.min(held, axis=0).tolist()}


def physical_summary(demo, count: int, limits: dict, *, behavior_family=None) -> tuple[dict, dict]:
    """Recompute physical and pre-action camera evidence from real HDF5 rows."""
    prefix = 'obs/diagnostics/'
    family = normalize_behavior_family(behavior_family if behavior_family is not None
                                       else demo.attrs.get('behavior_family'))
    if ('behavior_family' in demo.attrs
            and normalize_behavior_family(demo.attrs['behavior_family']) != family):
        raise ValueError('HDF5 behavior family differs from the result')
    door = _finite_dataset(demo, prefix + 'door_joint_pos', count, (3,))
    fingers = _finite_dataset(demo, prefix + 'finger_contacts', count, (2, 2, 3))
    arm = _finite_dataset(demo, prefix + 'arm_contacts', count)
    if arm.ndim != 3 or arm.shape[2] != 3:
        raise ValueError('arm contacts must be (N, links, 3)')
    clearance = _finite_dataset(demo, prefix + 'camera_clearance', count, (1,))
    actual = _finite_dataset(demo, prefix + 'wrist_camera_pose', count, (4, 4))
    expected = _finite_dataset(demo, prefix + 'wrist_camera_expected_pose', count, (4, 4))
    panel = _finite_dataset(demo, 'obs/datagen_info/object_pose/panel', count, (4, 4))
    timestamps = _finite_dataset(demo, prefix + 'timestamp', count, (1,))[:, 0]
    transition = _finite_dataset(demo, prefix + 'transition_index', count, (1,))[:, 0]
    # atan2 is stable for identical float32 rotations, unlike acos(trace).
    relative = np.swapaxes(actual[:, :3, :3], 1, 2) @ expected[:, :3, :3]
    antisym = np.stack((relative[:, 2, 1] - relative[:, 1, 2],
                        relative[:, 0, 2] - relative[:, 2, 0],
                        relative[:, 1, 0] - relative[:, 0, 1]), axis=1)
    rotation_error = np.arctan2(np.linalg.norm(antisym, axis=1) / 2,
                               (np.trace(relative, axis1=1, axis2=2) - 1) / 2)
    position_error = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1)
    degrees = np.rad2deg(door[:, 0])
    handle = np.rad2deg(door[:, 1])
    contact = np.linalg.norm(fingers, axis=-1)
    knob_grasp = np.min(contact[:, :, 0], axis=1) >= limits['finger_contact_n']
    unlock = (handle >= limits['unlock_handle_angle_deg']) & (door[:, 2] >= limits['unlock_latch_m'])
    gap = (degrees >= limits['knob_gap_angle_deg']) & knob_grasp
    panel_normal = panel[-1, :3, 1]
    normal_length = np.linalg.norm(panel_normal)
    if normal_length <= 1e-8:
        raise ValueError('panel normal is invalid')
    final_normal_forces = fingers[-1, :, 1] @ (panel_normal / normal_length)
    knob_index = _first_index(knob_grasp)
    unlock_index = _first_index(unlock)
    gap_index = _first_index(gap)
    summary = {
        'behavior_family': family, 'final_contact_object': final_contact_object(family),
        'final_door_angle_deg': float(degrees[-1]),
        'max_handle_angle_deg': float(handle.max()),
        'max_latch_m': float(door[:, 2].max()),
        'knob_grasp_observed': knob_index is not None,
        'unlock_observed': unlock_index is not None,
        'gap_with_knob_grasp_observed': gap_index is not None,
        'grasp_unlock_gap_ordered': bool(
            knob_index is not None and unlock_index is not None and gap_index is not None
            and knob_index <= unlock_index <= gap_index),
        'min_camera_clearance_m': float(clearance.min()),
        'max_arm_contact_n': float(np.linalg.norm(arm, axis=-1).max()),
        'max_camera_position_error_m': float(position_error.max()),
        'max_camera_rotation_error_rad': float(rotation_error.max()),
        'pre_action_timestamp_30_hz': bool(np.allclose(timestamps, np.arange(count) / limits['fps'], atol=1e-4)),
        'camera_pose_rows': count,
        'camera_homogeneous_poses_valid': bool(
            np.allclose(actual[:, 3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(expected[:, 3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(np.swapaxes(actual[:, :3, :3], 1, 2) @ actual[:, :3, :3], np.eye(3), atol=1e-4)
            and np.allclose(np.swapaxes(expected[:, :3, :3], 1, 2) @ expected[:, :3, :3], np.eye(3), atol=1e-4)
        ),
    }
    if family == EDGE_REGRASP:
        summary.update(final_edge_finger_contact_n=contact[-1, :, 1].tolist(),
                       final_edge_opposed_normal_force_n=final_normal_forces.tolist())
    else:
        summary.update(final_knob_finger_contact_n=contact[-1, :, 0].tolist(),
                       final_knob_force_dot_n2=float(np.dot(fingers[-1, 0, 0], fingers[-1, 1, 0])))
        summary.update(knob_hold_contact_summary(
            contact[:, :, 0], _first_index(degrees >= limits['knob_gap_angle_deg']),
            limits['finger_contact_n']))
    phases = {
        'start': 0, 'knob_grasp': knob_index,
        'unlock': unlock_index, 'knob_gap': gap_index,
        'final_open': count - 1,
    }
    if family == EDGE_REGRASP:
        phases['transfer_to_edge'] = _first_index(transition >= 2)
    return summary, {label: index for label, index in phases.items() if index is not None}


def inspect_video(path: Path, phase_indices: dict, *, thumbnail_dir: Path | None = None) -> dict:
    """Decode every frame once; sample phase images without repeated seeks."""
    ffprobe, probe_env = video_tool('ffprobe')
    stream = json.loads(subprocess.check_output([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate,width,height', '-of', 'json', str(path),
    ], env=probe_env, text=True, timeout=30))['streams'][0]
    fps = float(Fraction(stream['avg_frame_rate']))
    width, height = int(stream['width']), int(stream['height'])
    ffmpeg, environment = video_tool('ffmpeg')
    selected = {}
    for label, index in phase_indices.items():
        selected.setdefault(index, []).append(label)
    images = []
    count = 0
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen([
            ffmpeg, '-v', 'error', '-threads', '1', '-i', str(path), '-map', '0:v:0',
            '-vf', 'scale=160:120', '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1',
        ], stdout=subprocess.PIPE, stderr=stderr, env=environment)
        try:
            while True:
                pixels = process.stdout.read(160 * 120 * 3)
                if not pixels:
                    break
                if len(pixels) != 160 * 120 * 3:
                    raise ValueError(f'video returned an incomplete RGB frame: {path}')
                if count in selected:
                    frame = np.frombuffer(pixels, dtype=np.uint8).reshape(120, 160, 3)
                    gray = frame.astype(np.float32) @ np.array([.299, .587, .114], dtype=np.float32)
                    for label in selected[count]:
                        image = {
                            'phase': label, 'frame_index': count,
                            'gray_mean': float(gray.mean()), 'gray_std': float(gray.std()),
                            'black_fraction': float(np.mean(gray <= 3)),
                        }
                        if thumbnail_dir is not None:
                            from PIL import Image
                            thumbnail_dir.mkdir(parents=True, exist_ok=True)
                            destination = thumbnail_dir / f'{label}_{count:05d}.jpg'
                            Image.fromarray(frame).save(destination)
                            image['thumbnail'] = str(destination)
                        images.append(image)
                count += 1
            returncode = process.wait(timeout=30)
            stderr.seek(0)
            error_log = stderr.read().decode('utf-8', errors='replace').strip()
            if returncode != 0 or error_log:
                raise ValueError(f'video decoding failed ({returncode}): {error_log[:500]}')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    return {'path': str(path), 'decoded_frames': count, 'fps': fps,
            'width': width, 'height': height, 'phase_samples': images}


def batch_review_evidence(path: str | Path) -> dict[str, dict]:
    """Retain the batch parent's process outcome when selecting trial directories."""
    path = Path(path).expanduser().resolve()
    batch = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(batch, dict) or batch.get('schema') != 'arx-door-skillgen-batch-v1'
            or not isinstance(batch.get('trials'), list)):
        raise ValueError(f'invalid SkillGen batch report: {path}')
    evidence = {}
    configuration = batch.get('configuration') or {}
    bindings = configuration.get('source_quality') if isinstance(configuration, dict) else None
    if bindings is not None and not isinstance(bindings, list):
        raise ValueError(f'invalid batch source input bindings: {path}')
    for trial in batch['trials']:
        if (not isinstance(trial, dict) or not isinstance(trial.get('output_dir'), str)
                or not trial['output_dir'] or not isinstance(trial.get('result'), dict)):
            raise ValueError(f'invalid trial process evidence in {path}')
        directory = Path(trial['output_dir']).expanduser()
        directory = directory if directory.is_absolute() else path.parent / directory
        key = str(directory.resolve())
        if key in evidence:
            raise ValueError(f'duplicate trial process evidence in {path}: {key}')
        entry = {'batch_path': str(path), 'trial': {**trial, 'output_dir': key}}
        if bindings is not None:
            matched = [binding for binding in bindings if isinstance(binding, dict)
                       and binding.get('input_file') == trial.get('input_file')
                       and binding.get('seed_dir') == trial.get('seed_dir')]
            if len(matched) != 1:
                raise ValueError(f'batch source input/seed pair has no unique binding: {key}')
            entry['source_binding'] = matched[0]
        evidence[key] = entry
    return evidence


def _source_recording_identity(evidence: dict | None, result: dict) -> dict | None:
    """Reuse the verified input-boundary NPZ identity; never hashes another file."""
    if evidence is None or 'source_binding' not in evidence:
        return None
    binding, trial = evidence['source_binding'], evidence['trial']
    if not isinstance(binding, dict):
        raise ValueError('source input binding must be an object')
    identity = binding.get('source_sha256')
    if (binding.get('schema') != 'arx-door-source-input-binding-v1'
            or binding.get('content_verified') is not True
            or binding.get('derived_poses_verified') is not True
            or not isinstance(identity, str) or len(identity) != 64
            or any(character not in '0123456789abcdef' for character in identity)
            or type(binding.get('frames')) is not int or binding['frames'] < 2
            or binding.get('quality_status') not in ('user_approved_seed', 'agent_reviewed_seed')
            or binding.get('behavior_family') not in (EDGE_REGRASP, KNOB_HOLD_OPEN)):
        raise ValueError('source recording identity lacks a verified source input binding')
    for label in ('input_file', 'seed_dir'):
        value = binding.get(label)
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or value != trial.get(label)):
            raise ValueError(f'source input binding does not match the batch pair: {label}')
    if (Path(binding['input_file']).resolve() != Path(str(result.get('source_seed', ''))).resolve()
            or binding['behavior_family'] != normalize_behavior_family(result.get('behavior_family'))):
        raise ValueError('source input binding does not match the recorded source/family')
    if (binding['quality_status'] == 'agent_reviewed_seed'
            and (binding.get('human_review_status') != 'pending'
                 or not isinstance(binding.get('agent_review'), dict))):
        raise ValueError('agent source input binding lacks pending human review provenance')
    return binding


def _process_evidence(directory: Path, result: dict, evidence: dict | None) -> tuple[dict, list[str]]:
    """A parent failure overrides physical success without editing the source result."""
    errors = []
    authoritative = result
    source = 'standalone_result'
    batch_path = None
    disk_returncode = result.get('process_returncode')
    if disk_returncode is not None and type(disk_returncode) is not int:
        raise ValueError('result process returncode must be an integer or unknown')
    if evidence is not None:
        trial = evidence['trial']
        batch_path = evidence['batch_path']
        if Path(trial['output_dir']).expanduser().resolve() != directory:
            raise ValueError('batch process evidence belongs to another trial directory')
        authoritative = trial['result']
        if not isinstance(authoritative, dict) or type(authoritative.get('success')) is not bool:
            raise ValueError('batch process evidence has no explicit success decision')
        source = 'batch_parent'
        identity = {'angle_deg': 'angle_deg', 'seed': 'trial_seed',
                    'visual_seed': 'visual_seed', 'split': 'dataset_split',
                    'input_file': 'source_seed'}
        defaults = {'trial_seed': 0, 'visual_seed': 0, 'dataset_split': 'candidate'}
        comparisons = [(label, trial[label], result.get(field, defaults.get(field)))
                       for label, field in identity.items() if label in trial]
        comparisons.extend((field, authoritative[field], result.get(field, defaults.get(field)))
                           for field in ('angle_deg', 'source_seed', 'dataset_split', 'trial_seed',
                                         'visual_seed', 'behavior_family', 'frames', 'hdf5_frames',
                                         'physics_executed') if field in authoritative)
        for label, expected, actual in comparisons:
            if label in ('input_file', 'source_seed'):
                matches = (isinstance(expected, str) and isinstance(actual, str)
                           and Path(expected).expanduser().resolve() == Path(actual).expanduser().resolve())
            else:
                matches = expected == actual
            if not matches:
                errors.append(f'batch process evidence conflicts with result identity: {label}')
        if ('source_id' in trial
                and trial['source_id'] != Path(str(result.get('source_seed', ''))).stem):
            errors.append('batch process evidence conflicts with result identity: source_id')
        if authoritative['success'] is not True:
            errors.append('batch parent rejected this trial: '
                          f"{authoritative.get('failure_stage', 'process')}/"
                          f"{authoritative.get('failure_reason', 'unsuccessful outcome')}")
        elif result.get('success') is not True or authoritative.get('physics_executed') is not True:
            errors.append('batch parent success conflicts with physical execution evidence')
    returncode = authoritative.get('process_returncode')
    if returncode is not None and type(returncode) is not int:
        raise ValueError('process returncode must be an integer or unknown')
    if (evidence is not None and 'process_returncode' in result
            and result['process_returncode'] != returncode):
        errors.append('batch parent and result process returncodes disagree')
    if returncode is not None and returncode != 0:
        errors.append(f'simulator process exited unsuccessfully: {returncode}')
    return {'source': source, 'batch_path': batch_path, 'returncode': returncode,
            'exit_verified': returncode is not None,
            'recorded_success': authoritative.get('success')}, errors


def gate_errors(physical: dict, motion: dict, limits: dict, family: str) -> list[str]:
    """Return failed quality-gate labels for computed physical/motion summaries."""
    errors = []
    checks = {
        'final door angle below threshold': physical['final_door_angle_deg'] >= limits['door_angle_deg'],
        'knob grasp missing': physical['knob_grasp_observed'],
        'unlock missing': physical['unlock_observed'],
        'knob-held opening gap missing': physical['gap_with_knob_grasp_observed'],
        'grasp/unlock/opening gap order is invalid': physical['grasp_unlock_gap_ordered'],
        'camera clearance below threshold': physical['min_camera_clearance_m'] >= limits['min_camera_clearance_m'],
        'non-finger contact exceeds threshold': physical['max_arm_contact_n'] <= limits['max_arm_contact_n'],
        'camera position mismatch': physical['max_camera_position_error_m'] <= limits['max_camera_position_error_m'],
        'camera orientation mismatch': physical['max_camera_rotation_error_rad'] <= limits['max_camera_rotation_error_rad'],
        'camera pose is not a rigid homogeneous transform': physical['camera_homogeneous_poses_valid'],
        'pre-action timestamp cadence mismatch': physical['pre_action_timestamp_30_hz'],
        'too few active arm joints': motion['active_arm_joints'] >= limits['minimum_active_arm_joints'],
        'insufficient gripper command span': motion['command_span'][6] >= limits['minimum_gripper_span_m'],
        'excessive idle command fraction': motion['command_idle_fraction'] <= limits['max_command_idle_fraction'],
        'excessive uninterrupted idle': motion['longest_command_idle_s'] <= limits['max_command_idle_run_s'],
    }
    if family == EDGE_REGRASP:
        checks.update({
            'final edge grasp missing': min(physical['final_edge_finger_contact_n']) >= limits['finger_contact_n'],
            'final edge normal forces are not opposed': (
                min(abs(value) for value in physical['final_edge_opposed_normal_force_n']) >= limits['finger_contact_n']
                and np.prod(physical['final_edge_opposed_normal_force_n']) < 0),
        })
    else:
        checks.update({
            'final knob grasp missing': min(physical['final_knob_finger_contact_n']) >= limits['finger_contact_n'],
            'final knob force vectors are not opposed': physical['final_knob_force_dot_n2'] < 0,
        })
        if physical['knob_hold_contact_fraction'] != 1.0:
            errors.append(
                'knob contact not continuously held from gap to final: '
                f"first_loss_frame={physical['knob_hold_first_loss_frame']}, "
                f"contact_fraction={physical['knob_hold_contact_fraction']:.9g}")
    errors.extend(label for label, passed in checks.items() if not passed)
    return errors


def video_gate_errors(key: str, video: dict, count: int, limits: dict) -> list[str]:
    """Return failed decoding/cadence/information gates for one inspected video."""
    errors = []
    if video['decoded_frames'] != count or abs(video['fps'] - limits['fps']) > 0.01:
        errors.append(f'{key}: decoded frame count or fps mismatch')
    for image in video['phase_samples']:
        if (image['gray_std'] < limits['minimum_sample_image_std']
                or image['black_fraction'] >= limits['max_sample_black_fraction']):
            errors.append(f'{key}: blank/low-information phase image {image["phase"]}')
    return errors


def review_attempt(directory: str | Path, plan: dict, *, inspect_videos: bool = True,
                   thumbnail_dir: Path | None = None,
                   batch_trial: dict | None = None) -> dict[str, Any]:
    """Return quality decisions without editing samples or training artifacts."""
    directory = Path(directory).expanduser().resolve()
    limits = plan['quality_gates']
    record: dict[str, Any] = {'attempt_dir': str(directory), 'errors': [], 'warnings': []}
    errors = record['errors']
    try:
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        if not isinstance(result, dict):
            raise ValueError('result.json must be an object')
        process, process_errors = _process_evidence(directory, result, batch_trial)
        record['process_evidence'] = process
        errors.extend(process_errors)
        binding = _source_recording_identity(batch_trial, result)
        record['source_recording_identity'] = binding['source_sha256'] if binding else None
        if binding:
            record['source_input_binding'] = binding
        else:
            record['warnings'].append('no verified source recording identity; this sample cannot count as an independent holdout')
        if not process['exit_verified']:
            record['warnings'].append('simulator process exit is unknown; physical/video evidence does not establish its returncode')
        family = normalize_behavior_family(result.get('behavior_family'))
        if ('final_contact_object' in result
                and result['final_contact_object'] != final_contact_object(family)):
            raise ValueError('result final contact object disagrees with behavior family')
        record.update(behavior_family=family, final_contact_object=final_contact_object(family))
        if not isinstance(result.get('angle_deg'), (int, float)) or not np.isfinite(result['angle_deg']):
            raise ValueError('angle_deg must be a finite number')
        source = Path(str(result.get('source_seed', 'unknown'))).stem
        registry = plan.get('sources', {}).get(source, {})
        split = result.get('dataset_split', 'candidate')
        registry_file = registry.get('seed_file')
        registry_path = Path(registry_file).expanduser() if isinstance(registry_file, str) else None
        if registry_path is not None and not registry_path.is_absolute():
            registry_path = ROOT / registry_path
        registry_matches = bool(binding and registry_path is not None
                                and registry_path.resolve() == Path(binding['input_file']).resolve())
        record.update({
            'angle_deg': result.get('angle_deg'), 'source_id': source,
            'source_seed': result.get('source_seed'), 'split': split,
            'independence_group': registry.get('independence_group', source),
            'behavior_group': registry.get('behavior_group') if registry.get('reviewed_distinct_behavior') else None,
            'trial_seed': result.get('trial_seed', 0),
            'runtime_profile': (result.get('runtime') or {}).get('profile', 'legacy_unrecorded'),
            'source_registry_verified': bool(
                registry_matches and registry.get('reviewed_distinct_behavior') is True
                and registry.get('independence_group')
                and registry.get('assigned_split') == split),
        })
        if split not in ('candidate', 'holdout', 'diagnostic'):
            errors.append('unsupported dataset split')
        if split in ('candidate', 'holdout') and registry.get('assigned_split') and registry['assigned_split'] != split:
            errors.append('source split disagrees with the preassigned source registry')
        if not source or source == 'unknown':
            errors.append('source seed provenance is missing')
        if not registry:
            record['warnings'].append('source has no reviewed behavior/independence registry entry')
        if record['runtime_profile'] == 'legacy_unrecorded':
            record['warnings'].append('historical sample has no per-result Isaac runtime profile')
        record.update(appearance_summary(directory, result))
        if not record['visual_variant_verified']:
            record['warnings'].append('declared visual variant has no applied parameters/scene evidence')
        if result.get('success') is not True or result.get('physics_executed') is not True:
            errors.append('physical execution did not report success')
        path = _artifact(directory, result, 'hdf5', 'generated.hdf5')
        with h5py.File(path, 'r') as file:
            episodes = file['data']
            if len(episodes) != 1:
                raise ValueError('expected one HDF5 episode per attempt')
            demo = episodes[next(iter(episodes))]
            if family == KNOB_HOLD_OPEN and demo.attrs.get('behavior_family') != family:
                raise ValueError('knob family requires matching HDF5 behavior provenance')
            actions = demo['actions'][:]
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 2 or not np.isfinite(actions).all():
                raise ValueError('actions must be finite (N, 7), with at least two frames')
            count = len(actions)
            states = _finite_dataset(demo, 'obs/joint_pos', count, (7,))
            if not bool(demo.attrs.get('success', False)):
                errors.append('HDF5 episode is not marked physically successful')
            if result.get('frames') != count or result.get('hdf5_frames') != count:
                errors.append('result/HDF5 frame counts disagree')
            record['frames'] = count
            record['trajectory_fingerprint'] = trajectory_fingerprint(actions, states, limits['duplicate_decimals'])
            record['motion'] = motion_summary(actions, states, limits)
            physical, phases = physical_summary(demo, count, limits, behavior_family=family)
        record['physical'] = physical
        record['phase_indices'] = phases
        errors.extend(gate_errors(physical, record['motion'], limits, family))
        camera = result.get('camera_verification', {})
        audit = result.get('physical_audit', {})
        if (result.get('video_alignment') != 'pre_action_observation_at_30_hz'
                or camera.get('schema') != 'arx-wrist-camera-live-mount-v2'
                or camera.get('use_fabric') is not True
                or camera.get('mount_link') != 'link6'
                or camera.get('frames_verified') != count
                or audit.get('camera_frames_verified') != count
                or audit.get('physics_steps') != count):
            errors.append('complete pre-action camera/physics coverage evidence is missing')
        record['alignment_evidence'] = 'Declared pre-action recording plus per-frame camera poses and 30 Hz diagnostics; image/state simultaneity is not independently reconstructable from MP4 timestamps.'
        record['image_review_scope'] = 'Phase samples verify decoding, brightness and texture. Semantic target/gripper visibility still requires visual review.'
        record['videos_verified'] = inspect_videos
        if inspect_videos:
            record['videos'] = {}
            for key, fallback in (('wrist_video', 'wrist.mp4'), ('overview_video', 'overview.mp4')):
                video_path = _artifact(directory, result, key, fallback)
                samples = phases if key == 'wrist_video' else {}
                video = inspect_video(video_path, samples, thumbnail_dir=thumbnail_dir if samples else None)
                record['videos'][key] = video
                errors.extend(video_gate_errors(key, video, count, limits))
        else:
            errors.append('video decoding was skipped; data usability is unverified')
    except (OSError, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, subprocess.SubprocessError) as error:
        errors.append(f'{type(error).__name__}: {error}')
    record['quality_usable'] = not errors
    record['numerically_unique'] = None
    return record


def summarize_reviews(records: list[dict], plan: dict) -> dict:
    """Separate usable data counts from diversity and independent evaluation."""
    owners = {}
    views = {}
    split_by_trajectory = {}
    candidate_recordings = {row['source_recording_identity'] for row in records
                            if row.get('split') == 'candidate' and row.get('source_recording_identity')}
    holdout_recordings = {row['source_recording_identity'] for row in records
                          if row.get('split') == 'holdout' and row.get('source_recording_identity')}
    leaking_recordings = candidate_recordings & holdout_recordings
    for record in records:
        if record.get('trajectory_fingerprint') and record.get('split') in ('candidate', 'holdout'):
            split_by_trajectory.setdefault(record['trajectory_fingerprint'], set()).add(record['split'])
    leaking_trajectories = {key for key, splits in split_by_trajectory.items() if len(splits) > 1}
    for record in records:
        fingerprint = record.get('trajectory_fingerprint')
        if fingerprint is None or not record['quality_usable']:
            continue
        record['numerically_unique'] = fingerprint not in owners
        appearance = record.get('appearance_signature', record.get('visual_variant', 'nominal'))
        view_key = (fingerprint, appearance)
        if (record.get('split') == 'holdout'
                and record.get('source_recording_identity') in leaking_recordings):
            record['quality_usable'] = False
            record['errors'].append('source recording leaks between candidate and holdout despite source/group labels')
            continue
        if record.get('split') == 'holdout' and fingerprint in leaking_trajectories:
            record['quality_usable'] = False
            record['errors'].append('numerical trajectory leaks between candidate and holdout')
            continue
        if view_key in views:
            record['duplicate_of'] = views[view_key]
            record['quality_usable'] = False
            record['errors'].append('duplicate numerical trajectory and appearance')
        else:
            views[view_key] = record['attempt_dir']
            if fingerprint in owners:
                record['shared_trajectory_with'] = owners[fingerprint]
            else:
                owners[fingerprint] = record['attempt_dir']
    usable = [row for row in records if row['quality_usable']]
    candidate = [row for row in usable if row['split'] == 'candidate']
    holdout = [row for row in usable if row['split'] == 'holdout']
    # Report leakage even when a leaked/duplicate holdout was already rejected.
    candidate_sources = {row['source_id'] for row in records if row.get('split') == 'candidate' and row.get('source_id')}
    holdout_sources = {row['source_id'] for row in records if row.get('split') == 'holdout' and row.get('source_id')}
    candidate_groups = {row['independence_group'] for row in records if row.get('split') == 'candidate' and row.get('independence_group')}
    holdout_groups = {row['independence_group'] for row in records if row.get('split') == 'holdout' and row.get('independence_group')}
    leaks = sorted(candidate_sources & holdout_sources | candidate_groups & holdout_groups)
    behavior = sorted({row['behavior_group'] for row in candidate if row.get('behavior_group')})
    visual_by_properties = {
        row.get('appearance_signature', row['visual_variant']): row['visual_variant']
        for row in candidate if row.get('visual_variant_verified')
    }
    visual = sorted(visual_by_properties.values())
    angle_counts = {
        str(angle): len({row['trajectory_fingerprint'] for row in candidate
                        if abs(float(row['angle_deg']) - angle) <= plan['angle_tolerance_deg']})
        for angle in plan['required_geometry_angles_deg']
    }
    current = sum(row['runtime_profile'] == plan['current_runtime_profile'] for row in candidate)
    candidate_trajectories = {row['trajectory_fingerprint'] for row in candidate}
    holdout_trajectories = {row['trajectory_fingerprint'] for row in holdout}
    candidate_identity_complete = all(row.get('source_recording_identity') for row in candidate)
    independent_holdout = [row for row in holdout if row.get('source_recording_identity')
                           and candidate_identity_complete
                           and row.get('source_registry_verified')
                           and row['source_id'] not in candidate_sources
                           and row['independence_group'] not in candidate_groups
                           and row['source_recording_identity'] not in candidate_recordings]
    independent_holdout_trajectories = {row['trajectory_fingerprint'] for row in independent_holdout}
    requirements = {
        'candidate_count': len(candidate) >= plan['minimum_candidate_episodes'],
        'candidate_unique_trajectory_count': len(candidate_trajectories) >= plan.get('minimum_unique_candidate_trajectories', plan['minimum_candidate_episodes']),
        'current_profile_candidate_count': current >= plan['minimum_current_profile_candidates'],
        'independent_holdout_count': len(independent_holdout_trajectories) >= plan['minimum_independent_holdout_episodes'],
        'behavior_diversity': len(behavior) >= plan['minimum_candidate_behavior_groups'],
        'visual_diversity': len(visual) >= plan['minimum_candidate_visual_variants'],
        'geometry_coverage': all(count >= plan['minimum_episodes_per_required_angle'] for count in angle_counts.values()),
        'source_split_independence': not leaks and not leaking_recordings,
        'trajectory_split_independence': not leaking_trajectories,
        'source_provenance_complete': all(row.get('source_registry_verified') for row in candidate + holdout),
    }
    return {
        'schema': 'arx-door-quality-review-v1',
        'threshold_basis': plan['threshold_basis'],
        'attempts_reviewed': len(records), 'usable_unique_episodes': len(usable),
        'candidate_episodes': len(candidate), 'holdout_episodes': len(holdout),
        'usable_unique_trajectories': len({row['trajectory_fingerprint'] for row in usable}),
        'candidate_unique_trajectories': len(candidate_trajectories),
        'holdout_unique_trajectories': len(holdout_trajectories),
        'independent_holdout_unique_trajectories': len(independent_holdout_trajectories),
        'holdout_source_recording_count': len({row['source_recording_identity'] for row in independent_holdout}),
        'holdout_independence_groups': sorted({row['independence_group'] for row in independent_holdout}),
        'candidate_source_recording_identity_complete': candidate_identity_complete,
        'usable_frames': sum(row['frames'] for row in usable),
        'current_profile_candidate_episodes': current,
        'source_ids': sorted({row['source_id'] for row in usable}),
        'candidate_behavior_groups': behavior, 'candidate_visual_variants': visual,
        'observed_angles_deg': sorted({float(row['angle_deg']) for row in candidate}),
        'required_angle_counts': angle_counts,
        'split_leakage_sources_or_groups': leaks,
        'split_leakage_source_recording_count': len(leaking_recordings),
        'split_leakage_trajectory_count': len(leaking_trajectories),
        'duplicate_attempts': [row['attempt_dir'] for row in records if row.get('duplicate_of')],
        'readiness_requirements': requirements,
        'ready_to_discuss_fine_tuning': all(requirements.values()),
        'target_candidate_count_reached': len(candidate) >= plan['target_candidate_episodes'],
        'missing_readiness_requirements': [key for key, passed in requirements.items() if not passed],
        'generalization_verified': False,
        'records': records,
    }


def review_collection(attempts: list[str | Path], plan: dict, *, thumbnail_root: Path | None = None,
                      inspect_videos: bool = True,
                      batch_trials: dict[str, dict] | None = None) -> dict:
    directories = [Path(path).expanduser().resolve() for path in attempts]
    if len(set(directories)) != len(directories):
        raise ValueError('review inputs contain duplicate attempt directories')
    evidence = batch_trials or {}
    if set(evidence) - {str(path) for path in directories}:
        raise ValueError('batch process evidence references an unselected trial directory')
    records = [review_attempt(path, plan, inspect_videos=inspect_videos,
                              thumbnail_dir=thumbnail_root / f'attempt_{index:04d}' if thumbnail_root else None,
                              batch_trial=evidence.get(str(path)))
               for index, path in enumerate(directories)]
    return summarize_reviews(records, plan)
