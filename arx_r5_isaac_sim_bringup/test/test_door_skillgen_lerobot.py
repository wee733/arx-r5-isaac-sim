# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Real HDF5 and encoded-video tests for generated door exports."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('door_export', ROOT / 'scripts/export_door_lerobot.py')
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


@pytest.fixture
def workspace():
    """Keep synthetic outputs in the exporter's explicit isolation root."""
    runtime = ROOT / 'generated/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='TEST_ONLY_lerobot_', dir=runtime) as directory:
        yield Path(directory)


def source(root, name='trial', success=True, video_frames=3):
    """Create the native schema, including gripper action processing."""
    path = root / name
    path.mkdir()
    seed = root / 'seed.hdf5'
    if not seed.exists():
        with h5py.File(seed, 'w') as file:
            file.create_dataset('seed', data=[1])
    states = np.array([[.1, .2, .3, .4, .5, .6, .02],
                       [.11, .21, .31, .41, .51, .61, .021],
                       [.12, .22, .32, .42, .52, .62, .022]], dtype='float32')
    actions = states + np.array([.3, .3, .3, .3, .3, .3, .03], dtype='float32')
    processed = actions.copy()
    processed[:, 6] = .044
    camera_poses = np.repeat(np.eye(4, dtype='float32')[None], 3, axis=0)
    camera_poses[:, 0, 3] = [1., 1.1, 1.2]
    with h5py.File(path / 'generated.hdf5', 'w') as file:
        data = file.create_group('data')
        data.attrs['total'] = 3
        data.attrs['env_args'] = json.dumps({'env_name': 'Isaac-ARX-R5-Door-SkillGen-v0'})
        demo = data.create_group('demo_0')
        demo.attrs['success'] = success
        demo.attrs['num_samples'] = 3
        demo.create_dataset('actions', data=actions)
        demo.create_dataset('processed_actions', data=processed)
        demo.create_dataset('obs/joint_pos', data=states)
        demo.create_dataset('obs/diagnostics/timestamp', data=np.arange(3)[:, None] / 30.)
        demo.create_dataset('obs/diagnostics/door_joint_pos', data=np.zeros((3, 3)))
        demo.create_dataset('obs/diagnostics/transition_index', data=[[0], [1], [1]])
        demo.create_dataset('obs/diagnostics/wrist_camera_pose', data=camera_poses)
        demo.create_dataset('obs/diagnostics/wrist_camera_expected_pose', data=camera_poses)
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=blue:s=640x480:r=30', '-frames:v', str(video_frames),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path / 'wrist.mp4')], check=True)
    result = {
        'schema': 'arx-door-skillgen-result-v1', 'environment': 'Isaac-ARX-R5-Door-SkillGen-v0',
        'test_only': True, 'angle_deg': 3., 'physics_executed': True, 'success': success,
        'failure_stage': None if success else 'side_grasp_ik',
        'failure_reason': None if success else 'unreachable entry',
        'frames': 3, 'hdf5_frames': 3, 'hdf5_episodes': 1, 'final_door_angle_deg': 31.,
        'source_seed': str(seed), 'video_alignment': 'pre_action_observation_at_30_hz',
        'physical_audit': {'physics_steps': 3, 'final_door_angle_deg': 31.,
                           'max_handle_angle_deg': 44., 'max_latch_m': .0158,
                           'min_camera_clearance_m': .06, 'max_arm_contact_n': 0.,
                           'knob_grasp_observed': True, 'unlock_observed': True,
                           'gap_with_knob_grasp_observed': True},
        'camera_verification': {
            'schema': 'arx-wrist-camera-live-mount-v2', 'mount_link': 'link6',
            'camera_prim_path': '/World/SkillGenWristCamera',
            'mount_transform': np.eye(4).tolist(), 'use_fabric': True, 'frames_verified': 3,
            'max_position_error_m': 0., 'max_rotation_error_rad': 0.,
        },
        'artifacts': {'hdf5': 'generated.hdf5', 'wrist_video': 'wrist.mp4'},
    }
    (path / 'result.json').write_text(json.dumps(result))
    return path, states, processed


def arguments(workspace, attempts=(), batches=()):
    return argparse.Namespace(attempt=[], generated_attempt=list(attempts),
                              batch_report=list(batches), output=workspace / 'TEST_ONLY_dataset',
                              task='Open the door using the knob then the edge.', test_only=True)


def test_generated_export_keeps_pre_action_state_and_processed_absolute_targets(workspace):
    """Using next state or raw gripper commands would corrupt these labels."""
    path, states, actions = source(workspace)
    before = (path / 'result.json').read_bytes()
    args = arguments(workspace, [path / 'result.json'])
    EXPORTER.export(args)
    table = pq.read_table(args.output / 'data/chunk-000/episode_000000.parquet').to_pydict()
    np.testing.assert_allclose(table['observation.state'], states)
    np.testing.assert_allclose(table['action'], actions)
    np.testing.assert_allclose(table['timestamp'], [0., 1./30., 2./30.])
    assert table['annotation.human.task_description'] == [0, 0, 0]
    assert json.loads((args.output / 'meta/tasks.jsonl').read_text())['task'] == args.task
    modality = json.loads((args.output / 'meta/modality.json').read_text())
    assert modality['action']['single_arm'] == {'start': 0, 'end': 6}
    assert list(modality['video']) == ['wrist']
    report = json.loads((args.output / 'meta/door_export_report.json').read_text())
    assert report['official_n1_7_loader_verified'] is False
    assert report['sources'][0]['source_kind'] == 'skillgen_physical'
    assert report['sources'][0]['angle_deg'] == 3.
    assert report['sources'][0]['source_seed_sha256']
    assert report['sources'][0]['hdf5_sha256']
    assert report['sources'][0]['user_reviewed'] is False
    assert before == (path / 'result.json').read_bytes()


def test_batch_excludes_failed_trial_and_records_reason(workspace):
    """Failed automatic trials must not become rewarded training episodes."""
    good, _, _ = source(workspace, 'good')
    failed, _, _ = source(workspace, 'failed', success=False)
    batch = workspace / 'batch.json'
    batch.write_text(json.dumps({'schema': 'arx-door-skillgen-batch-v1', 'trials': [
        {'output_dir': str(path), 'angle_deg': 3., 'seed': 0,
         'result': json.loads((path / 'result.json').read_text())} for path in [good, failed]]}))
    args = arguments(workspace, batches=[batch])
    EXPORTER.export(args)
    info = json.loads((args.output / 'meta/info.json').read_text())
    assert info['total_episodes'] == 1
    report = json.loads((args.output / 'meta/door_export_report.json').read_text())
    assert report['excluded_sources'][0]['failure_reason'] == 'unreachable entry'
    assert report['sources'][0]['trial_seed'] == 0


@pytest.mark.parametrize('trial_angle', [None, 4.])
def test_native_batch_accepts_missing_trial_fields_but_rejects_explicit_angle_conflict(
        workspace, trial_angle):
    """The native runner stores angle in result, not necessarily in trial."""
    path, _, _ = source(workspace)
    trial = {'output_dir': str(path), 'result': json.loads((path / 'result.json').read_text())}
    if trial_angle is not None:
        trial['angle_deg'] = trial_angle
    batch = workspace / 'native_batch.json'
    batch.write_text(json.dumps({'schema': 'arx-door-skillgen-batch-v1', 'trials': [trial]}))
    args = arguments(workspace, batches=[batch])
    if trial_angle is not None:
        with pytest.raises(ValueError, match='trial angle mismatch'):
            EXPORTER.export(args)
        assert not args.output.exists()
    else:
        EXPORTER.export(args)
        report = json.loads((args.output / 'meta/door_export_report.json').read_text())
        assert report['sources'][0]['angle_deg'] == 3.
        assert report['sources'][0]['trial_index'] is None
        assert report['sources'][0]['trial_seed'] is None


@pytest.mark.parametrize('corruption,match', [
    ('video_frames', 'frame count'), ('timestamp', 'timestamps'),
    ('hdf5_success', 'HDF5.*success'), ('processing', 'processed_actions'),
    ('audit', 'physical audit'), ('alignment', 'pre_action_observation'),
    ('nonfinite', 'finite'), ('failed_direct', 'success=true'),
])
def test_generated_rejects_corrupt_or_unqualified_sources(workspace, corruption, match):
    """Each mutation catches a distinct acceptance or label-alignment bug."""
    path, _, _ = source(workspace, video_frames=2 if corruption == 'video_frames' else 3)
    if corruption in ('timestamp', 'hdf5_success', 'processing', 'nonfinite'):
        with h5py.File(path / 'generated.hdf5', 'r+') as file:
            demo = file['data/demo_0']
            if corruption == 'timestamp':
                demo['obs/diagnostics/timestamp'][1, 0] = .9
            elif corruption == 'hdf5_success':
                demo.attrs['success'] = False
            elif corruption == 'processing':
                demo['processed_actions'][0, 0] = 2.
            else:
                demo['obs/joint_pos'][0, 0] = np.nan
    else:
        result = json.loads((path / 'result.json').read_text())
        if corruption == 'audit':
            result['physical_audit']['unlock_observed'] = False
        elif corruption == 'alignment':
            result['video_alignment'] = 'post_action'
        elif corruption == 'failed_direct':
            result['success'] = False
        (path / 'result.json').write_text(json.dumps(result))
    args = arguments(workspace, [path])
    with pytest.raises(ValueError, match=match):
        EXPORTER.export(args)
    assert not args.output.exists()


def test_generated_rejects_duplicate_content_under_different_paths(workspace):
    """Renaming a successful source must not duplicate its training weight."""
    path, _, _ = source(workspace)
    clone = workspace / 'clone'
    shutil.copytree(path, clone)
    with pytest.raises(ValueError, match='duplicate.*content'):
        EXPORTER.export(arguments(workspace, [path, clone]))


def test_old_npz_review_gate_remains_required(workspace):
    """Generated acceptance must not bypass the legacy user review gate."""
    path = workspace / 'legacy'
    path.mkdir()
    (path / 'episode.json').write_text(json.dumps({'success': True, 'review_required': True}))
    args = arguments(workspace)
    args.attempt = [path]
    with pytest.raises(ValueError, match='review_required=false'):
        EXPORTER.export(args)


def test_reviewed_npz_still_exports_original_absolute_actions(workspace):
    """The original NPZ path must not acquire generated-only requirements."""
    path, states, processed = source(workspace)
    metadata = {'success': True, 'review_required': False, 'test_only': True, 'fps': 30., 'frames': 3}
    (path / 'episode.json').write_text(json.dumps(metadata))
    np.savez(path / 'trajectory.npz', state=states, action=processed,
             timestamp=np.arange(3) / 30., phase=[0, 1, 1], door=np.zeros((3, 3)))
    args = arguments(workspace)
    args.attempt = [path]
    EXPORTER.export(args)
    table = pq.read_table(args.output / 'data/chunk-000/episode_000000.parquet').to_pydict()
    np.testing.assert_allclose(table['action'], processed)
    report = json.loads((args.output / 'meta/door_export_report.json').read_text())
    assert report['sources'][0]['source_episode'] == metadata
    assert report['sources'][0]['trajectory_sha256']


def test_batch_rejects_result_snapshot_mismatch(workspace):
    """A stale or altered batch approval cannot override the disk result."""
    path, _, _ = source(workspace)
    snapshot = json.loads((path / 'result.json').read_text())
    snapshot['success'] = False
    batch = workspace / 'batch.json'
    batch.write_text(json.dumps({'schema': 'arx-door-skillgen-batch-v1', 'trials': [
        {'output_dir': str(path), 'angle_deg': 3., 'result': snapshot}]}))
    with pytest.raises(ValueError, match='snapshot'):
        EXPORTER.export(arguments(workspace, batches=[batch]))


def test_cli_accepts_generated_sources_and_keeps_fixture_isolation(workspace):
    """CLI parsing must expose generated input without requiring --attempt."""
    path, _, _ = source(workspace)
    args = arguments(workspace)
    command = [sys.executable,
               str(ROOT / 'scripts/export_door_lerobot.py'), '--generated-attempt', str(path),
               '--output', str(args.output)]
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert 'TEST_ONLY' in rejected.stderr
    accepted = subprocess.run(command + ['--test-only'], capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert (args.output / 'meta/info.json').is_file()


def test_hdf5_shape_mismatch_fails_before_creating_dataset(workspace):
    """Frame count checks must also reject a diagnostic with fewer rows."""
    path, _, _ = source(workspace)
    with h5py.File(path / 'generated.hdf5', 'r+') as file:
        del file['data/demo_0/obs/diagnostics/door_joint_pos']
        file.create_dataset('data/demo_0/obs/diagnostics/door_joint_pos', data=np.zeros((2, 3)))
    args = arguments(workspace, [path])
    with pytest.raises(ValueError, match='finite.*3, 3'):
        EXPORTER.export(args)
    assert not args.output.exists()


@pytest.mark.parametrize('field', ['physical_audit', 'artifacts'])
def test_malformed_result_objects_are_rejected_cleanly(workspace, field):
    """Malformed JSON objects must yield an export refusal, not a traceback."""
    path, _, _ = source(workspace)
    result = json.loads((path / 'result.json').read_text())
    result[field] = None
    (path / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='physical audit|artifact manifest'):
        EXPORTER.export(arguments(workspace, [path]))


@pytest.mark.parametrize('corruption', [
    'missing_evidence', 'legacy_schema', 'fabric', 'old_camera_parent',
    'frames', 'nonfinite_summary', 'invalid_mount',
    'missing_pose', 'invalid_pose', 'position_drift', 'rotation_drift', 'summary_mismatch',
])
def test_generated_camera_evidence_is_required_and_recomputed(workspace, corruption):
    """A claimed camera pass cannot hide missing, stale or invalid poses."""
    path, _, _ = source(workspace)
    result = json.loads((path / 'result.json').read_text())
    evidence = result['camera_verification']
    if corruption == 'missing_evidence':
        del result['camera_verification']
    elif corruption == 'legacy_schema':
        evidence['schema'] = 'arx-wrist-camera-live-mount-v1'
        evidence['use_fabric'] = False
    elif corruption == 'fabric':
        evidence['use_fabric'] = False
    elif corruption == 'old_camera_parent':
        evidence['camera_prim_path'] = '/R5a/link6/TeachingWristCamera'
    elif corruption == 'frames':
        evidence['frames_verified'] = 2
    elif corruption == 'nonfinite_summary':
        evidence['max_position_error_m'] = float('nan')
    elif corruption == 'invalid_mount':
        evidence['mount_transform'][0][0] = -1.
    elif corruption == 'summary_mismatch':
        evidence['max_position_error_m'] = .00005
    else:
        with h5py.File(path / 'generated.hdf5', 'r+') as file:
            field = 'data/demo_0/obs/diagnostics/wrist_camera_pose'
            if corruption == 'missing_pose':
                del file[field]
            elif corruption == 'invalid_pose':
                file[field][1, 3, 0] = .1
            elif corruption == 'position_drift':
                file[field][1, 0, 3] += .001
            else:
                angle = .005
                matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                file[field][1, :3, :3] = matrix
    (path / 'result.json').write_text(json.dumps(result))
    args = arguments(workspace, [path])
    with pytest.raises(ValueError, match='camera'):
        EXPORTER.export(args)
    assert not args.output.exists()


def test_camera_summary_tolerates_float32_pose_storage_and_retains_source(workspace):
    """Sub-microradian quantization must not reject genuinely aligned mounts."""
    path, _, _ = source(workspace)
    result = json.loads((path / 'result.json').read_text())
    # This non-identity rotation avoids a misleading trace/arccos residual
    # from float32 matrices that are only approximately orthonormal.
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    with h5py.File(path / 'generated.hdf5', 'r+') as file:
        for field in ('wrist_camera_pose', 'wrist_camera_expected_pose'):
            file[f'data/demo_0/obs/diagnostics/{field}'][:, :3, :3] = rotation
        file['data/demo_0/obs/diagnostics/wrist_camera_pose'][1, 0, 3] += .00003
    result['camera_verification']['max_position_error_m'] = .00003
    result['camera_verification']['max_rotation_error_rad'] = .0000001
    (path / 'result.json').write_text(json.dumps(result))
    args = arguments(workspace, [path])
    EXPORTER.export(args)
    report = json.loads((args.output / 'meta/door_export_report.json').read_text())
    assert report['sources'][0]['source_result']['camera_verification'] == result['camera_verification']
    assert report['sources'][0]['camera_pose_validation']['frames_verified'] == 3
    assert report['sources'][0]['camera_pose_validation']['max_position_error_m'] < .0001
    assert report['sources'][0]['camera_pose_validation']['max_rotation_error_rad'] < .001
    with np.load(args.output / 'diagnostics/episode_000000.npz', allow_pickle=False) as diagnostics:
        assert diagnostics['wrist_camera_pose'].shape == (3, 4, 4)
        np.testing.assert_allclose(diagnostics['wrist_camera_expected_pose'][:, 0, 3], [1., 1.1, 1.2])
        np.testing.assert_allclose(diagnostics['wrist_camera_pose'][1, 0, 3], 1.10003, atol=1e-7, rtol=0)
