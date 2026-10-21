#!/usr/bin/env python3
"""Export only AutoData episodes with successful pose AND joint-action replays.

The source of training actions is recorded actuator targets, never next state.
Stage language is inherited by exact monotonic alignment to the reviewed seed.
No legacy dataset or old SkillGen result is accepted by this entry point.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.dataset.stages import KNOB_HOLD_OPEN_STAGES
from export_door_lerobot import probe_video, statistics, write_json, write_jsonl


def local_path(value):
    value = str(value)
    for prefix in ('/outputs/', '/workspaces/arx/generated/door_autodata/'):
        if value.startswith(prefix):
            return ROOT / 'generated/door_autodata' / value[len(prefix):]
    if value.startswith('/workspaces/arx/'):
        return ROOT / value[len('/workspaces/arx/'):]
    return Path(value)


def verified_replay(summary_path, run, demo, actions):
    summary = json.loads(summary_path.read_text())
    for index, row in enumerate(summary['episodes']):
        if local_path(row['run']).resolve() != run.resolve() or row['demo'] != demo:
            continue
        folder = summary_path.parent / f'{index:03d}_{run.name}_{demo}' / 'rollout'
        result = json.loads((folder / 'result.json').read_text())
        if not row['success'] or not result['success'] or result['steps'] != len(actions):
            raise ValueError(f'{run}/{demo}: full independent replay did not pass')
        with np.load(folder / 'trajectory.npz') as replay:
            np.testing.assert_allclose(replay['command'], actions, rtol=0, atol=1e-7)
        return str(folder / 'result.json')
    raise ValueError(f'No independent replay for {run}/{demo} in {summary_path}')


def episode_door_pose(episode):
    """Measured closed-door pose in the robot frame, independent of actions."""
    from scipy.spatial.transform import Rotation
    def matrix(name):
        pose = episode[f'initial_state/articulation/{name}/root_pose'][0]
        result = np.eye(4)
        result[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
        result[:3, 3] = pose[:3]
        return result
    return np.linalg.inv(matrix('robot')) @ matrix('door')


def stage_alignment(actions, seed, generated_door_pose=None):
    """Match the entire skill after its measured object-relative rigid transform.

    Labels remain source-frame labels: a changed pose never permits a partial
    match, a temporal guess, or an arbitrary fit to the generated action stream.
    """
    from scipy.spatial.transform import Rotation
    source = seed['actions'][:]
    signal = seed['obs/datagen_info/subtask_start_signals/knob_interaction'][:].reshape(-1)
    start = int(np.flatnonzero(signal)[0])
    transformed = source.copy()
    delta = np.eye(4)
    if generated_door_pose is not None:
        reference = seed['obs/datagen_info/object_pose/door'][start]
        delta = generated_door_pose @ np.linalg.inv(reference)
        transformed[:, :3] = source[:, :3] @ delta[:3, :3].T + delta[:3, 3]
        transformed[:, 3:7] = (Rotation.from_matrix(delta[:3, :3])
                               * Rotation.from_quat(source[:, 3:7])).as_quat()
    segment = transformed[start:]
    count = len(segment)
    # The official generator may append final hold samples. All interaction
    # samples must match in order; never derive labels from elapsed time.
    match = None
    for offset in range(max(0, len(actions) - count - 8), len(actions) - count + 1):
        part = actions[offset:offset + count]
        position = np.max(np.abs(part[:, :3] - segment[:, :3]))
        rotation = np.max(1 - np.abs(np.sum(part[:, 3:7] * segment[:, 3:7], axis=1)))
        grip = np.max(np.abs(part[:, 7] - segment[:, 7]))
        if position <= 1e-4 and rotation <= 1e-5 and grip <= 1e-6:
            match = offset
            break
    if match is None:
        raise ValueError('Cannot align the full skill to source; stage language not fabricated')
    names = json.loads(seed['obs/diagnostics'].attrs['phase_names'])
    phase_map = {phase: i for i, stage in enumerate(KNOB_HOLD_OPEN_STAGES) for phase in stage.seed_phases}
    source_stage = np.asarray([phase_map[names[int(i)]] for i in seed['obs/diagnostics/phase_id'][:].reshape(-1)])
    labels = np.zeros(len(actions), dtype=np.int64)
    labels[match:match + count] = source_stage[start:]
    if match + count < len(actions):
        np.testing.assert_allclose(actions[match + count:], np.broadcast_to(transformed[-1], actions[match + count:].shape), atol=1e-5)
        labels[match + count:] = source_stage[-1]
    return labels, {'source_start': start, 'generated_start': match, 'matched_frames': count,
                    'object_relative_transform': delta.tolist(),
                    'transform_evidence': 'recorded_initial_robot_and_door_poses' if generated_door_pose is not None else 'identity'}


def encode_video(dataset, path):
    height, width = dataset.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
           '-video_size', f'{width}x{height}', '-framerate', '30', '-i', '-', '-an',
           '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '18',
           '-pix_fmt', 'yuv420p', str(path)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        # Official HDF5 chunks span many frames. Read each time slab once;
        # frame-by-frame reads repeatedly decompress the same large chunk set.
        batch = dataset.chunks[0] if dataset.chunks else 32
        for start in range(0, len(dataset), batch):
            for frame in dataset[start:start + batch]:
                process.stdin.write(np.asarray(frame[..., :3], dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f'Encoding failed: {path}')
    probe = probe_video(path)
    if int(probe['nb_read_frames']) != len(dataset):
        raise ValueError(f'Video frame count mismatch: {path}')
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='JSON list: run, split, pose_replay_summary, joint_replay_summary')
    parser.add_argument('--split', choices=('train', 'holdout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    # Neither a generation seed nor an initial condition may cross splits.
    split_groups = {}
    for entry in entries:
        run = local_path(entry['run']).resolve()
        metadata = json.loads((run / 'run.json').read_text())
        condition = {k: v for k, v in metadata['scene_parameters'].items() if k != 'condition_id'}
        for group in (('seed', metadata['seed']), ('condition', json.dumps(condition, sort_keys=True))):
            if group in split_groups and split_groups[group] != entry['split']:
                raise ValueError('seed/scene condition leaks across train and holdout')
            split_groups[group] = entry['split']
    selected = [entry for entry in entries if entry['split'] == args.split]
    if not selected or args.output.exists():
        raise ValueError('requires nonempty selection and fresh output')
    staging = args.output.with_name(args.output.name + '.incomplete')
    staging.mkdir(parents=True, exist_ok=False)
    meta = staging / 'meta'
    meta.mkdir()
    episodes, stats_rows, lineage, all_states, all_actions = [], [], [], [], []
    texts = [stage.instruction for stage in KNOB_HOLD_OPEN_STAGES]
    offset = 0
    features = {}
    for entry in selected:
        run = local_path(entry['run']).resolve()
        metadata = json.loads((run / 'run.json').read_text())
        if metadata['schema'] != 'arx-autodata-run-v1':
            raise ValueError('only current official AutoData runs are supported')
        with h5py.File(run / 'accepted.hdf5', 'r') as source, h5py.File((run / metadata['input']).resolve(), 'r') as seeds:
            if list(seeds['data']) != ['demo_0']:
                raise ValueError('pilot stage alignment requires exactly one reviewed source')
            for demo, data in source['data'].items():
                n = len(data['actions'])
                if n < 16 or not bool(data['obs/diagnostics/runtime_accepted'][-1].item()):
                    raise ValueError('episode lacks final physical acceptance')
                actions = data['obs/diagnostics/joint_target_substeps'][:, 0, :].astype(np.float32)
                state = data['obs/joint_pos'][:].astype(np.float32)
                if state.shape != (n, 7) or actions.shape != (n, 7) or not np.isfinite([state, actions]).all():
                    raise ValueError('invalid numeric state/action contract')
                if np.min(actions[:, 6]) < 0 or np.max(actions[:, 6]) > .044001:
                    raise ValueError('gripper target must be single-finger metres in [0, .044]')
                np.testing.assert_allclose(data['obs/diagnostics/timestamp'][:].reshape(-1), np.arange(n) / 30, atol=1e-5)
                np.testing.assert_array_equal(data['obs/diagnostics/action_index'][:].reshape(-1), np.arange(n))
                pose_receipt = verified_replay(local_path(entry['pose_replay_summary']), run, demo, data['actions'][:])
                joint_receipt = verified_replay(local_path(entry['joint_replay_summary']), run / 'joint_replay_candidate', demo, actions)
                with np.load(Path(joint_receipt).parent / 'trajectory.npz') as replay:
                    drift = replay['state'] - state
                    controller_comparison = {
                        'state_max_abs_error_per_joint': np.max(np.abs(drift), axis=0).tolist(),
                        'state_rmse_per_joint': np.sqrt(np.mean(drift ** 2, axis=0)).tolist(),
                        'controllers_identical': False,
                        'joint_replay_physical_acceptance_passed': True,
                    }
                randomized = metadata['scene_parameters'].get('expected_skill_transform_robot')
                measured_door = episode_door_pose(data) if randomized is not None else None
                labels, alignment = stage_alignment(data['actions'][:], seeds['data/demo_0'], measured_door)
                if randomized is not None:
                    np.testing.assert_allclose(alignment['object_relative_transform'], randomized, atol=1e-5,
                                               err_msg='declared scene change did not reach the simulator')
                for old_state, old_action in zip(all_states, all_actions):
                    if state.shape == old_state.shape and np.array_equal(state, old_state) and np.array_equal(actions, old_action):
                        raise ValueError('duplicate trajectory; not augmentation')
                index = len(episodes)
                numeric = {'observation.state': state, 'action': actions}
                columns = {k: pa.array(v.tolist(), type=pa.list_(pa.float32(), 7)) for k, v in numeric.items()}
                columns.update(timestamp=pa.array(np.arange(n) / 30, type=pa.float32()),
                               frame_index=pa.array(np.arange(n)), episode_index=pa.array(np.full(n, index)),
                               index=pa.array(np.arange(offset, offset + n)), task_index=pa.array(labels),
                               **{'annotation.human.stage_description': pa.array(labels)})
                destination = staging / f'data/chunk-000/episode_{index:06d}.parquet'
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table(columns), destination)
                for camera in ('wrist', 'ego'):
                    rgb = data[f'obs/{camera}_rgb']
                    if len(rgb) != n or rgb.dtype != np.uint8:
                        raise ValueError('pre-action RGB rows must match action rows')
                    key = f'observation.images.{camera}'
                    probe = encode_video(rgb, staging / f'videos/chunk-000/{key}/episode_{index:06d}.mp4')
                    h, w = int(probe['height']), int(probe['width'])
                    features[key] = {'dtype': 'video', 'shape': [h, w, 3], 'names': ['height', 'width', 'channels'],
                                     'info': {'video.height': h, 'video.width': w, 'video.codec': 'h264',
                                              'video.pix_fmt': 'yuv420p', 'video.fps': 30, 'video.channels': 3,
                                              'video.is_depth_map': False, 'has_audio': False}}
                episodes.append({'episode_index': index, 'tasks': [texts[i] for i in sorted(set(labels))], 'length': n})
                stats_rows.append({'episode_index': index, 'stats': {k: statistics(v) for k, v in numeric.items()}})
                lineage.append({'episode_index': index, 'run': str(run), 'demo': demo, **metadata,
                                'pose_replay': pose_receipt, 'joint_replay': joint_receipt,
                                'controller_comparison': controller_comparison, 'stage_alignment': alignment})
                all_states.append(state); all_actions.append(actions); offset += n
    if not episodes:
        raise ValueError('no eligible AutoData episodes')
    for key in ('observation.state', 'action'):
        features[key] = {'dtype': 'float32', 'shape': [7], 'names': [f'joint{i}' for i in range(1, 8)]}
    for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'annotation.human.stage_description'):
        features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
    write_json(meta / 'info.json', {'codebase_version': 'v2.1', 'robot_type': 'arx_r5a', 'fps': 30,
        'total_episodes': len(episodes), 'total_frames': offset, 'total_tasks': len(texts),
        'total_videos': len(episodes) * 2, 'total_chunks': 1, 'chunks_size': 1000,
        'splits': {args.split: f'0:{len(episodes)}'}, 'features': features,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'})
    write_json(meta / 'modality.json', {'state': {'single_arm': {'start': 0, 'end': 6}, 'gripper': {'start': 6, 'end': 7}},
        'action': {'single_arm': {'start': 0, 'end': 6}, 'gripper': {'start': 6, 'end': 7}},
        'video': {k: {'original_key': f'observation.images.{k}'} for k in ('wrist', 'ego')},
        'annotation': {'human.stage_description': {'original_key': 'annotation.human.stage_description'}}})
    write_jsonl(meta / 'episodes.jsonl', episodes)
    write_jsonl(meta / 'tasks.jsonl', [{'task_index': i, 'task': t} for i, t in enumerate(texts)])
    write_jsonl(meta / 'episodes_stats.jsonl', stats_rows)
    write_jsonl(meta / 'autodata_lineage.jsonl', lineage)
    write_json(meta / 'door_export_report.json', {'schema': 'arx-autodata-lerobot-export-v1',
        'episodes': len(episodes), 'frames': offset, 'source': 'official_autodata',
        'physical_acceptance_and_independent_replays_verified': True,
        'official_n1_7_loader_verified': False, 'sufficient_for_finetuning': False,
        'lineage': 'autodata_lineage.jsonl', 'split': args.split})
    write_json(meta / 'stats.json', {k: statistics(np.concatenate(v)) for k, v in [('observation.state', all_states), ('action', all_actions)]})
    write_json(meta / 'export_contract.json', {'source': 'official_autodata', 'split': args.split,
        'action': 'first actual IK substep joint target held at 30 Hz; independently replayed',
        'units': ['rad'] * 6 + ['m'], 'gripper': 'joint7 displacement; not total jaw opening; no pre-normalization',
        'alignment': 'RGB/state before action, diagnostics after action',
        'official_loader_verified': False, 'sufficient_for_finetuning': False})
    staging.rename(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
