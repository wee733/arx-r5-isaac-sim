#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Compute fresh official GR00T statistics and exercise its real episode loader.

No model or training GPU is needed. Run with Python 3.12, torch 2.9 and
torchcodec 0.8; pass a clean checkout of the pinned official source. Every
episode's numeric arrays and relative chunks are checked; three RGB frames per
episode are decoded through the official torchcodec backend.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


PINNED_COMMIT = '9c7e746b2cd37a810070a98ef41d290a07e806c2'
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Publish a JSON report atomically."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_config_contract(config):
    """Reject sampling or conversion options outside the approved contract.

    The episode loader exposes raw absolute columns even when a downstream
    processor will make a gripper relative. Validate the actual official
    dataclasses before generating statistics or claiming processor semantics.
    """
    from gr00t.data.types import (
        ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
    )

    expected = {
        'video': (['wrist'], [0]),
        'state': (['single_arm', 'gripper'], [0]),
        'action': (['single_arm', 'gripper'], list(range(16))),
        'language': (['annotation.human.task_description'], [0]),
    }
    if type(config) is not dict or set(config) != set(expected):
        raise ValueError('config requires exactly video, state, action and language modalities')
    modality_fields = {'delta_indices', 'modality_keys', 'sin_cos_embedding_keys',
                       'mean_std_embedding_keys', 'action_configs'}
    validated = {}
    for category, (keys, indices) in expected.items():
        modality = config[category]
        if type(modality) is not ModalityConfig or set(vars(modality)) != modality_fields:
            raise ValueError(f'config {category} has unknown modality fields or type')
        if modality.modality_keys != keys:
            raise ValueError(f'config {category} requires ordered keys {keys}')
        if (modality.delta_indices != indices
                or any(type(index) is not int for index in modality.delta_indices)):
            raise ValueError(f'config {category} requires delta_indices {indices}')
        if (modality.sin_cos_embedding_keys not in (None, [])
                or modality.mean_std_embedding_keys not in (None, [])):
            raise ValueError(f'config {category} embedding changes are not approved')
        if category != 'action' and modality.action_configs is not None:
            raise ValueError(f'config {category} must not define action conversion')
        validated[category] = {'modality_keys': keys, 'delta_indices': indices}
    actions = config['action'].action_configs
    if type(actions) is not list or len(actions) != 2:
        raise ValueError('config action requires exactly arm and gripper conversion configs')
    action_fields = {'rep', 'type', 'format', 'state_key'}
    conversions = []
    for key, action, representation in zip(
        expected['action'][0], actions,
        (ActionRepresentation.RELATIVE, ActionRepresentation.ABSOLUTE),
    ):
        if type(action) is not ActionConfig or set(vars(action)) != action_fields:
            raise ValueError(f'config action {key} has unknown conversion fields or type')
        if (action.rep is not representation or action.type is not ActionType.NON_EEF
                or action.format is not ActionFormat.DEFAULT):
            raise ValueError(f'config action {key} requires {representation.name} NON_EEF DEFAULT')
        if action.state_key not in (None, key):
            raise ValueError(f'config action {key} requires reference state None or {key}')
        conversions.append({'key': key, 'rep': action.rep.name, 'type': action.type.name,
                            'format': action.format.name, 'state_key': action.state_key})
    validated['action']['action_configs'] = conversions
    return validated


def validate(dataset, groot_root, config_path):
    """Validate the single-wrist dataset using unmodified official code."""
    meta = dataset / 'meta'
    # A failed revalidation must never leave an earlier acceptance flag true.
    report_path = meta / 'n17_validation.json'
    write_json(report_path, {'official_n1_7_loader_verified': False, 'status': 'running'})
    export_path = meta / 'door_export_report.json'
    if export_path.exists():
        export_report = json.loads(export_path.read_text())
        export_report['official_n1_7_loader_verified'] = False
        write_json(export_path, export_report)
        if export_report.get('eligible_for_training') is False:
            raise ValueError('dataset is quarantined; loader validation cannot lift visual rejection')
    episodes = [json.loads(line) for line in (meta / 'episodes.jsonl').read_text().splitlines()
                if line.strip()]
    if not episodes or any(ep['length'] < 16 for ep in episodes):
        raise ValueError('every episode requires at least 16 frames for the action horizon')
    commit = subprocess.check_output(
        ['git', '-C', str(groot_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f'expected official source {PINNED_COMMIT}, got {commit}')
    dirty = subprocess.check_output(
        ['git', '-C', str(groot_root), 'status', '--porcelain', '--', 'gr00t'], text=True)
    if dirty:
        raise ValueError('official gr00t source must be unmodified')

    sys.path.insert(0, str(groot_root))
    import numpy as np
    import pandas as pd
    import torch
    import torchcodec
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import RelativeActionLoader, generate_rel_stats, generate_stats

    spec = importlib.util.spec_from_file_location('arx_door_runtime_config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.door_wrist_config
    config_contract = validate_config_contract(config)
    info = json.loads((meta / 'info.json').read_text())
    modality = json.loads((meta / 'modality.json').read_text())
    if info['codebase_version'] != 'v2.1' or info['fps'] != 30:
        raise ValueError('expected LeRobot v2.1 at 30 Hz')
    if set(modality['video']) != {'wrist'} or config['video'].modality_keys != ['wrist']:
        raise ValueError('this validator requires the approved wrist-only contract')
    if info['total_episodes'] != len(episodes):
        raise ValueError('episode metadata count mismatch')

    # Upstream fingerprints schema/config, not parquet content. Always regenerate;
    # preserve old statistics for inspection instead of trusting a stale cache.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = meta / 'stats_history' / stamp
    for name in ('stats.json', 'relative_stats.json'):
        path = meta / name
        if path.exists():
            backup.mkdir(parents=True, exist_ok=True)
            path.replace(backup / name)
    generate_stats(dataset)
    generate_rel_stats(dataset, EmbodimentTag.NEW_EMBODIMENT)
    loader = LeRobotEpisodeLoader(dataset, config, decoder_kwargs={'num_ffmpeg_threads': 1})
    relative_loader = RelativeActionLoader(dataset, EmbodimentTag.NEW_EMBODIMENT, 'single_arm')
    stats = loader.get_dataset_statistics()
    for group, dim in (('single_arm', 6), ('gripper', 1)):
        for category in ('state', 'action'):
            for key in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
                value = np.asarray(stats[category][group][key])
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f'invalid official statistics {category}/{group}/{key}')
    for key in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
        value = np.asarray(stats['relative_action']['single_arm'][key])
        if value.shape != (16, 6) or not np.isfinite(value).all():
            raise ValueError(f'invalid official relative statistics: {key}')

    checks = []
    input_hashes = {}
    offset = 0
    for position, ep in enumerate(episodes):
        index, count = ep['episode_index'], ep['length']
        if index != position:
            raise ValueError('episode indices must be contiguous from zero')
        parquet = dataset / info['data_path'].format(
            episode_chunk=index // info['chunks_size'], episode_index=index)
        raw = pd.read_parquet(parquet)
        data = loader._load_parquet_data(index)
        if len(data) != count or len(raw) != count:
            raise ValueError(f'episode {index} length mismatch')
        state = np.stack(raw['observation.state'])
        action = np.stack(raw['action'])
        if state.shape != (count, 7) or action.shape != state.shape:
            raise ValueError(f'episode {index} must contain seven-dimensional rows')
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f'episode {index} contains non-finite values')
        np.testing.assert_array_equal(raw['frame_index'], np.arange(count))
        np.testing.assert_array_equal(raw['index'], np.arange(offset, offset + count))
        np.testing.assert_array_equal(raw['episode_index'], np.full(count, index))
        np.testing.assert_allclose(raw['timestamp'], np.arange(count) / 30, atol=1e-5, rtol=0)
        for key, expected in (
            ('state.single_arm', state[:, :6]), ('state.gripper', state[:, 6:]),
            ('action.single_arm', action[:, :6]), ('action.gripper', action[:, 6:]),
        ):
            np.testing.assert_array_equal(np.stack(data[key]), expected)
        language = data['language.annotation.human.task_description']
        if any(text not in ep['tasks'] for text in language):
            raise ValueError(f'episode {index} task text failed official decoding')
        chunks = np.asarray(relative_loader.load_relative_actions(position))
        starts = np.arange(count - 15)
        expected = action[starts[:, None] + np.arange(16), :6] - state[starts, None, :6]
        np.testing.assert_allclose(chunks, expected, atol=1e-6, rtol=0)
        frames = np.array(sorted({0, count // 2, count - 1}))
        videos = loader._load_video_data(index, frames)
        if videos['wrist'].shape != (len(frames), 480, 640, 3):
            raise ValueError(f'episode {index} decoded wrist shape {videos["wrist"].shape}')
        checks.append({'episode_index': index, 'frames': count,
                       'decoded_frames': frames.tolist(), 'relative_chunks': len(chunks)})
        input_hashes[str(parquet.relative_to(dataset))] = hashlib.sha256(parquet.read_bytes()).hexdigest()
        video = dataset / info['video_path'].format(
            episode_chunk=index // info['chunks_size'], episode_index=index,
            video_key=modality['video']['wrist']['original_key'])
        input_hashes[str(video.relative_to(dataset))] = hashlib.sha256(video.read_bytes()).hexdigest()
        offset += count
        print(f'PASS episode {index}: {count} rows, wrist RGB, absolute targets, relative chunks', flush=True)
    if offset != info['total_frames']:
        raise ValueError('total frame count mismatch')
    for name in ('info.json', 'modality.json', 'episodes.jsonl', 'tasks.jsonl',
                 'stats.json', 'relative_stats.json'):
        input_hashes[f'meta/{name}'] = hashlib.sha256((meta / name).read_bytes()).hexdigest()
    config_dir = dataset / 'training'
    config_dir.mkdir(exist_ok=True)
    config_copy = config_dir / 'door_wrist_n17.py'
    if config_copy.resolve() != config_path.resolve():
        shutil.copyfile(config_path, config_copy)
    input_hashes['training/door_wrist_n17.py'] = hashlib.sha256(config_copy.read_bytes()).hexdigest()
    report = {
        'official_n1_7_loader_verified': True, 'status': 'passed',
        'official_commit': commit, 'verified_at_utc': stamp,
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'torchcodec': torchcodec.__version__, 'device': 'cpu',
        'episodes': len(checks), 'frames': offset, 'checks': checks,
        'input_sha256': input_hashes, 'model_training_executed': False,
        'validated_config_contract': config_contract,
        'scope': 'official statistics, every numeric row/relative chunk, three decoded RGB frames per episode',
    }
    write_json(report_path, report)
    if export_path.exists():
        export_report['official_n1_7_loader_verified'] = True
        export_report['statistics'] = 'fresh official absolute and 16-step relative statistics'
        export_report['official_validation_report'] = 'meta/n17_validation.json'
        write_json(export_path, export_report)
    (dataset / 'README.md').write_text(
        '# ARX door — single-wrist GR00T N1.7 dataset\n\n'
        f'{len(checks)} successful generated episodes, {offset} frames at 30 Hz.\n\n'
        'LeRobot v2.1; wrist RGB only; absolute joint1..joint6 radians and joint7 metres. '
        'Use training/door_wrist_n17.py: relative arm targets, absolute gripper, horizon 16.\n\n'
        f'Official loader and statistics verified against NVIDIA Isaac-GR00T {commit}. '
        'See meta/n17_validation.json for runtime, input hashes and validation coverage.\n\n'
        'No model training was performed. This small, narrow-position pilot is not '
        'a generalization benchmark or a claim of trained policy success. Source seed '
        'lineage and physical acceptance are in meta/door_export_report.json.\n'
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--groot-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/door_wrist_n17.py')
    args = parser.parse_args()
    try:
        report = validate(args.dataset.resolve(), args.groot_root.resolve(), args.config.resolve())
    except (ValueError, KeyError, OSError, AssertionError, ImportError, subprocess.SubprocessError) as error:
        report_path = args.dataset / 'meta/n17_validation.json'
        if report_path.parent.is_dir():
            write_json(report_path, {'official_n1_7_loader_verified': False,
                                    'status': 'failed', 'error': str(error)})
        parser.exit(1, f'N1.7 validation failed: {error}\n')
    print(json.dumps({'episodes': report['episodes'], 'frames': report['frames'], 'status': 'passed'}))


if __name__ == '__main__':
    main()
