#!/usr/bin/env python3
"""Check v3 video/chunks and ARX mappings with LingBot's unmodified data classes.

This exercises data loading, not a model or training. Run with the isolated
lingbot_data/packages directory on PYTHONPATH and the data runtime's FFmpeg.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np


def check_normalization(dataset, stats_path, fit):
    """Train-only exact quantiles, then exercise the official numeric normalizer.

    Like LingBot compute_norm, relative arm statistics retain the chunk axis;
    absolute effector statistics use the first action of each chunk.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lingbot_dataset_api.transform import Normalizer
    table = pa.concat_tables([pq.read_table(p) for p in sorted((dataset / 'data').rglob('*.parquet'))]).to_pandas().sort_values('index')
    values = {key: [] for key in ('observation.state.arm.position',
              'observation.state.effector.position', 'action.arm.position', 'action.effector.position')}
    for _, episode in table.groupby('episode_index', sort=True):
        state = np.stack(episode['observation.state']).astype(np.float32)
        action = np.stack(episode['action']).astype(np.float32)
        indices = np.minimum(np.arange(len(state))[:, None] + np.arange(16), len(state) - 1)
        values['observation.state.arm.position'].append(state[:, :6])
        values['observation.state.effector.position'].append(state[:, 6:])
        values['action.arm.position'].append(action[indices, :6] - state[:, None, :6])
        values['action.effector.position'].append(action[indices, 6:])
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    if fit:
        if stats_path.exists():
            raise ValueError('normalization output already exists')
        stats = {}
        for key, data in values.items():
            if key == 'action.effector.position':
                data = data[:, 0]
            data = data.astype(np.float64)
            stats[key] = {'mean': data.mean(axis=0).tolist(), 'std': data.std(axis=0).tolist(),
                          'min': data.min(axis=0).tolist(), 'max': data.max(axis=0).tolist()}
            for label, q in (('q01', .01), ('q99', .99), ('q02', .02), ('q98', .98)):
                stats[key][label] = np.quantile(data, q, axis=0).tolist()
            if key.endswith('effector.position'):
                # Physical single-finger stroke, not full jaw aperture or a
                # narrow observed range. Storage remains metres.
                for label in ('min', 'q01', 'q02'):
                    stats[key][label] = [0.0]
                for label in ('max', 'q99', 'q98'):
                    stats[key][label] = [.044]
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps({'norm_stats': stats, 'count': len(table),
            'provenance': {'fit_dataset': str(dataset.resolve()), 'chunk_size': 16,
                'quantiles': 'exact numpy quantiles over train only',
                'effector_bounds_override_m': [0, .044]}}, indent=2) + '\n')
    stats = json.loads(stats_path.read_text())['norm_stats']
    normalizer = Normalizer(stats, norm_type={key: 'bounds_99' for key in values})
    normalized = normalizer.normalize(values)
    restored = normalizer.unnormalize(normalized)
    max_error = {}
    for key in values:
        if not np.isfinite(normalized[key]).all():
            raise ValueError('nonfinite normalized data')
        np.testing.assert_allclose(restored[key], values[key], rtol=1e-5, atol=1e-6)
        max_error[key] = float(np.max(np.abs(restored[key] - values[key])))
    probe = normalizer.normalize({'action.effector.position': np.asarray([[0], [.022], [.044]], np.float32)})
    np.testing.assert_allclose(probe['action.effector.position'].ravel(), [-1, 0, 1], atol=5e-5)
    return {'passed': True, 'stats': str(stats_path.resolve()), 'fit_on_this_dataset': fit,
            'all_numeric_rows_and_16_step_chunks': True, 'max_roundtrip_error': max_error,
            'gripper_normalized_closed_half_open': probe['action.effector.position'].ravel().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--lingbot-root', type=Path, required=True)
    parser.add_argument('--robot-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-v21', type=Path, help='verify every converted numeric row against this original')
    parser.add_argument('--norm-stats', type=Path, help='train statistics used for both splits')
    parser.add_argument('--fit-norm-stats', action='store_true', help='fit only on the train split')
    args = parser.parse_args()
    if args.fit_norm_stats and (not args.norm_stats or args.dataset.name != 'train'):
        raise ValueError('fit normalization only on the train dataset with --norm-stats')
    commit = subprocess.check_output(['git', '-C', str(args.lingbot_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '4eb34b7693a0565c67433f8fac9c59a2e67eb60b':
        raise ValueError('unreviewed LingBot source revision')
    subprocess.run(['git', '-C', str(args.lingbot_root), 'diff', '--quiet', 'HEAD', '--'], check=True)
    # Import the genuine data/config modules without package initializers that
    # eagerly import every policy model (and require unrelated training extras).
    # No loader, feature transform or config implementation is replaced.
    import lerobot
    for name in ('policies', 'policies.pi0'):
        key = 'lerobot.' + name
        module = types.ModuleType(key)
        module.__path__ = [str(Path(lerobot.__file__).parent.joinpath(*name.split('.')))]
        sys.modules[key] = module
    root = args.lingbot_root.resolve() / 'lingbotvla/data/vla_data'
    spec = importlib.util.spec_from_file_location('lingbot_dataset_api', root / '__init__.py',
                                                submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from lingbot_dataset_api.base_dataset import LeRobotDataset
    from lingbot_dataset_api.utils import FeatureTransform
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        raise ValueError('LingBot delivery requires the converted v3 dataset')
    dataset = LeRobotDataset('arx/local_pilot', root=args.dataset.resolve(),
                             delta_timestamps={'action': [i / info['fps'] for i in range(16)]},
                             video_backend='pyav')
    numeric_rows_preserved = False
    if args.source_v21:
        import pyarrow as pa
        import pyarrow.parquet as pq
        def numeric_table(root):
            files = sorted((root / 'data').rglob('*.parquet'))
            return pa.concat_tables([pq.read_table(p) for p in files]).to_pandas().sort_values('index')
        original, converted = numeric_table(args.source_v21), numeric_table(args.dataset)
        for key in original.columns:
            np.testing.assert_array_equal(np.stack(original[key]), np.stack(converted[key]))
        tasks = [json.loads(line) for line in (args.source_v21 / 'meta/tasks.jsonl').read_text().splitlines()]
        for task in tasks:
            if dataset.meta.tasks.iloc[task['task_index']].name != task['task']:
                raise ValueError('v3 task mapping changed')
        numeric_rows_preserved = True
    config = types.SimpleNamespace(joints=["{'arm.position': 6}", "{'effector.position': 1}"],
                                   cameras=['camera_top', 'camera_wrist_left'])
    transform = FeatureTransform(args.robot_config, config, None, None,
                                 do_nomalize=False, chunk_size=16, load_image=True)
    checks = []
    for episode in dataset.meta.episodes:
        start, stop = int(episode['dataset_from_index']), int(episode['dataset_to_index'])
        for index in sorted({start, (start + stop) // 2, stop - 1}):
            raw = dataset[index]
            mapped = transform.apply(dict(raw))
            np.testing.assert_allclose(mapped['action.arm.position'],
                                       raw['action'][:, :6] - raw['observation.state'][:6])
            np.testing.assert_allclose(mapped['action.effector.position'], raw['action'][:, 6:7])
            for key in ('camera_top', 'camera_wrist_left'):
                image = mapped['observation.images.' + key]
                if image.shape[-3] != 3 or not np.isfinite(image.numpy()).all():
                    raise ValueError('invalid decoded camera frame')
            if not raw['task']:
                raise ValueError('missing stage language')
            checks.append({'index': index, 'task': raw['task'], 'action_shape': list(raw['action'].shape)})
    normalization = check_normalization(args.dataset, args.norm_stats, args.fit_norm_stats) if args.norm_stats else None
    args.output.write_text(json.dumps({'passed': True, 'lingbot_commit': commit,
        'dataset': str(args.dataset.resolve()), 'episodes': info['total_episodes'],
        'checks': checks, 'all_numeric_rows_preserved_from_v21': numeric_rows_preserved,
        'normalization_tested': normalization is not None, 'normalization': normalization,
        'model_or_training_tested': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
