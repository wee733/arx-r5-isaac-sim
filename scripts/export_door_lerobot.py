#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Export reviewed successful door attempts to single-wrist LeRobot v2.1.

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


TASK = ('Grasp the round doorknob, turn it counterclockwise 45 degrees, pull a '
        'gap, release the knob, grasp the door side edge, and pull the door open.')
VIDEO_KEY = 'observation.images.wrist'
JOINT_NAMES = [f'joint{i}' for i in range(1, 8)]


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


def export(args):
    """Validate all attempts, then create a new dataset without overwriting."""
    attempts = [path.resolve() for path in args.attempt]
    if len(set(attempts)) != len(attempts):
        raise ValueError('duplicate attempt path')
    metadata = [inspect_attempt(path, args.test_only) for path in attempts]
    output = args.output.resolve()
    runtime = Path(__file__).resolve().parents[1] / 'generated/runtime'
    if args.test_only and (runtime not in output.parents or 'TEST_ONLY' not in output.name):
        raise ValueError('test fixture output must be generated/runtime/.../TEST_ONLY...')
    if output.exists():
        raise ValueError(f'output already exists; choose a fresh directory: {output}')

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    loaded = []
    fps = float(metadata[0]['fps'])
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
        loaded.append((path, data, video, n))

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
    for index, (path, data, video, n) in enumerate(loaded):
        chunk = index // 1000
        data_dir = staging / f'data/chunk-{chunk:03d}'
        video_dir = staging / f'videos/chunk-{chunk:03d}' / VIDEO_KEY
        data_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        state, action = data['state'].astype('float32'), data['action'].astype('float32')
        table = pa.table({
            'observation.state': pa.array(state.tolist(), type=pa.list_(pa.float32(), 7)),
            'action': pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
            'timestamp': pa.array(data['timestamp'], type=pa.float32()),
            'frame_index': pa.array(np.arange(n), type=pa.int64()),
            'episode_index': pa.array(np.full(n, index), type=pa.int64()),
            'index': pa.array(np.arange(offset, offset + n), type=pa.int64()),
            'task_index': pa.array(np.zeros(n), type=pa.int64()),
            'annotation.human.task_description': pa.array(np.zeros(n), type=pa.int64()),
            'next.done': pa.array([False] * (n - 1) + [True], type=pa.bool_()),
            'next.reward': pa.array([0.0] * (n - 1) + [1.0], type=pa.float32()),
        })
        filename = f'episode_{index:06d}'
        pq.write_table(table, data_dir / (filename + '.parquet'))
        shutil.copyfile(path / 'wrist.mp4', video_dir / (filename + '.mp4'))
        np.savez_compressed(staging / 'diagnostics' / (filename + '.npz'),
                            phase=data['phase'], door=data['door'], timestamp=data['timestamp'])
        stats = {'observation.state': statistics(state), 'action': statistics(action)}
        episode_stats.append({'episode_index': index, 'stats': stats})
        episodes.append({'episode_index': index, 'tasks': [args.task], 'length': n})
        provenance.append({
            'episode_index': index, 'source': str(path),
            'trajectory_sha256': hashlib.sha256((path / 'trajectory.npz').read_bytes()).hexdigest(),
            'source_episode': metadata[index],
        })
        offset += n
        all_states.append(state)
        all_actions.append(action)

    features = {
        key: {'dtype': 'float32', 'shape': [7], 'names': JOINT_NAMES}
        for key in ('observation.state', 'action')
    }
    video = loaded[0][2]
    features[VIDEO_KEY] = {
        'dtype': 'video', 'shape': [video['height'], video['width'], 3],
        'names': ['height', 'width', 'channels'],
        'info': {'video.height': video['height'], 'video.width': video['width'],
                 'video.codec': video['codec_name'], 'video.pix_fmt': video['pix_fmt'],
                 'video.is_depth_map': False, 'video.fps': fps,
                 'video.channels': 3, 'has_audio': False},
    }
    for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index',
                'annotation.human.task_description', 'next.done', 'next.reward'):
        dtype = ('float32' if key in ('timestamp', 'next.reward') else
                 'bool' if key == 'next.done' else 'int64')
        features[key] = {'dtype': dtype, 'shape': [1], 'names': None}
    write_json(meta_dir / 'info.json', {
        'codebase_version': 'v2.1', 'robot_type': 'arx_r5a',
        'total_episodes': len(loaded), 'total_frames': offset, 'total_tasks': 1,
        'total_videos': len(loaded), 'total_chunks': (len(loaded) + 999) // 1000,
        'chunks_size': 1000, 'fps': fps, 'splits': {'train': f'0:{len(loaded)}'},
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'features': features,
    })
    write_json(meta_dir / 'modality.json', {
        'state': {'single_arm': {'start': 0, 'end': 6}, 'gripper': {'start': 6, 'end': 7}},
        'action': {'single_arm': {'start': 0, 'end': 6}, 'gripper': {'start': 6, 'end': 7}},
        'video': {'wrist': {'original_key': VIDEO_KEY}},
        'annotation': {'human.task_description': {'original_key': 'annotation.human.task_description'}},
    })
    write_jsonl(meta_dir / 'episodes.jsonl', episodes)
    write_jsonl(meta_dir / 'tasks.jsonl', [{'task_index': 0, 'task': args.task}])
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
        'review_gate': 'success=true AND review_required=false', 'sources': provenance,
        'schema_source': 'https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md',
    })
    (staging / 'README.md').write_text(
        ('# TEST_ONLY synthetic schema fixture\n\nNOT robot expert data.\n\n' if args.test_only else '# ARX door dataset\n\n')
        + 'Single wrist RGB; 7D absolute state and command targets.\n\n'
        + 'Official GR00T N1.7 loader validation and official statistics generation have NOT run. '
        + 'The basic stats.json is incomplete for training. Regenerate official absolute and relative '
        + 'statistics and validate the single-arm data configuration before fine-tuning.\n'
    )
    staging.rename(output)
    print(f'Exported {len(loaded)} episodes / {offset} frames to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default=TASK)
    parser.add_argument('--test-only', action='store_true', help='isolated synthetic schema fixture only')
    args = parser.parse_args()
    try:
        export(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Export refused: {error}\n')


if __name__ == '__main__':
    main()
