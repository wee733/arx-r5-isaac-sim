# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Deterministic batch layout and artifact validation for door SkillGen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from ..contract import FIRST_BATCH_ANGLES_DEG
from ..files import write_json


@dataclass(frozen=True)
class TrialSpec:
    """One explicit physical generation attempt."""

    index: int
    angle_deg: float
    seed: int
    output_name: str
    output_dir: Path
    source_id: str = 'default'
    input_file: Path | None = None
    seed_dir: Path | None = None
    split: str = 'candidate'
    visual_seed: int = 0


def trial_manifest(trial) -> dict:
    """Persist the requested trial identity independently of its outcome."""
    return json.loads(json.dumps(asdict(trial), default=str))


def _angle_name(angle_deg: float) -> str:
    sign = 'p' if angle_deg >= 0 else 'm'
    magnitude = abs(float(angle_deg))
    whole = int(magnitude)
    fraction = int(round((magnitude - whole) * 10))
    if fraction == 10:
        whole += 1
        fraction = 0
    return f'angle_{sign}{whole:03d}_{fraction}'


def build_trial_specs(
    angles_deg: Sequence[float],
    output_root: str | Path,
    *,
    seed: int,
    input_files: Sequence[str | Path] | None = None,
    seed_dirs: Sequence[str | Path] | None = None,
    trial_seeds: Sequence[int] | None = None,
    visual_seeds: Sequence[int] = (0,),
    split: str = 'candidate',
) -> tuple[TrialSpec, ...]:
    """Build ordered, unique trial specs without silently deduplicating."""
    root = Path(output_root).expanduser().resolve()
    angles = tuple(float(value) for value in angles_deg)
    if not angles:
        raise ValueError('at least one angle is required')
    if not all(math.isfinite(value) for value in angles):
        raise ValueError('angles must be finite')
    names = tuple(_angle_name(value) for value in angles)
    if len(set(names)) != len(names):
        raise ValueError('angles collide after output-name normalization')
    sources = tuple(Path(path).expanduser().resolve() for path in input_files or ())
    seed_paths = tuple(Path(path).expanduser().resolve() for path in seed_dirs or ())
    if len(sources) != len(seed_paths):
        raise ValueError('each input HDF5 requires its corresponding '
                         'approved seed directory')
    if len(set(sources)) != len(sources):
        raise ValueError('duplicate input seed files are not separate demonstrations')
    source_ids = tuple(path.stem for path in sources)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('input seeds must have distinct file stems')
    seeds = tuple(int(value) for value in trial_seeds) if trial_seeds is not None else (int(seed),)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not 0 <= value < 2**32 for value in seeds)):
        raise ValueError('trial seeds must be distinct unsigned 32-bit integers')
    variants = tuple(int(value) for value in visual_seeds)
    if (not variants or len(set(variants)) != len(variants)
            or any(not 0 <= value < 2**32 for value in variants)):
        raise ValueError('visual seeds must be distinct unsigned 32-bit integers')
    if split not in ('candidate', 'holdout', 'diagnostic'):
        raise ValueError('unsupported split')
    source_pairs = tuple(zip(source_ids, sources, seed_paths)) or (('default', None, None),)
    trials = []
    nested = len(source_pairs) > 1 or len(seeds) > 1 or seeds != (0,)
    for source_id, input_file, seed_dir in source_pairs:
        for trial_seed in seeds:
            prefix = Path(f'source_{source_id}') / f'trial_seed_{trial_seed}' if nested else Path()
            for visual_seed in variants:
                variant = Path(f'visual_seed_{visual_seed}') if visual_seed else Path()
                for angle, name in zip(angles, names):
                    output_name = (prefix / variant / name).as_posix()
                    trials.append(TrialSpec(
                        len(trials), angle, trial_seed, output_name, root / output_name,
                        source_id, input_file, seed_dir, split, visual_seed))
    return tuple(trials)


def summarize_trials(records: Sequence[dict]) -> dict:
    """Count retained attempts by angle, input demonstration and trial RNG seed."""
    def counts(items):
        successes = sum(item['result'].get('success') is True for item in items)
        return {'attempts': len(items), 'successes': successes, 'failures': len(items) - successes}
    summary = counts(records)
    for label, key in (('by_angle', 'angle_deg'), ('by_source', 'source_id'),
                       ('by_trial_seed', 'seed'), ('by_visual_seed', 'visual_seed')):
        groups = {}
        for item in records:
            value = item.get(key, 0) if key == 'visual_seed' else item[key]
            groups.setdefault(str(value), []).append(item)
        summary[label] = {name: counts(items) for name, items in groups.items()}
    return summary


def process_result(result: dict, returncode: int) -> dict:
    """A failed simulator process cannot publish an accepted success."""
    result = {**result, 'process_returncode': returncode}
    if returncode != 0:
        if (result.get('success') is False and result.get('failure_stage')
                and result.get('failure_reason')):
            return result
        return {**result, 'success': False, 'failure_stage': 'simulator_process',
                'failure_reason': f'simulator exited with code {returncode}',
                'reported_physical_success': result.get('success')}
    if result.get('success') is True and result.get('physics_executed') is not True:
        return {**result, 'success': False, 'failure_stage': 'artifact_validation',
                'failure_reason': 'success was reported without physical execution'}
    return result


def failed_process_result(trial: TrialSpec, stage: str, reason: str) -> dict:
    """Preserve partial physics evidence when a process cannot finish cleanly."""
    result = {}
    path = trial.output_dir / 'result.json'
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                result = value
        except (OSError, ValueError):
            pass
    # A launched simulator with no final result may have already stepped.
    # Unknown is more honest than claiming no physics occurred on interruption.
    physics = result.get('physics_executed')
    if physics is None and not (trial.output_dir / 'generation.log').exists():
        physics = False
    return {**result, 'angle_deg': trial.angle_deg, 'success': False,
            'physics_executed': physics, 'failure_stage': stage,
            'failure_reason': reason}


def _relative_artifact(output_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} artifact path is missing')
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else output_dir / candidate
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'{label} artifact is missing or empty: {path}')
    return path


def validate_trial_output(trial: TrialSpec) -> dict:
    """Validate one success or failure as an auditable physical attempt."""
    result_path = trial.output_dir / 'result.json'
    if not result_path.is_file():
        raise ValueError(f'missing result.json for {trial.output_name}')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('schema') != 'arx-door-skillgen-result-v1':
        raise ValueError(f'unsupported result schema for {trial.output_name}')
    if float(result.get('angle_deg')) != trial.angle_deg:
        raise ValueError(f'angle mismatch for {trial.output_name}')
    if 'trial_seed' in result and result['trial_seed'] != trial.seed:
        raise ValueError(f'trial seed mismatch for {trial.output_name}')
    if 'visual_seed' in result and result['visual_seed'] != trial.visual_seed:
        raise ValueError(f'visual seed mismatch for {trial.output_name}')
    if result.get('physics_executed') is not True:
        raise ValueError(f'physics was not executed for {trial.output_name}')
    if not isinstance(result.get('frames'), int) or result['frames'] <= 0:
        raise ValueError(f'invalid frame count for {trial.output_name}')
    if not isinstance(result.get('final_door_angle_deg'), (float, int)):
        raise ValueError(f'missing final door angle for {trial.output_name}')
    success = result.get('success')
    if not isinstance(success, bool):
        raise ValueError(f'success must be boolean for {trial.output_name}')
    if not success and (
        not result.get('failure_stage') or not result.get('failure_reason')
    ):
        raise ValueError(f'failed trial lacks stage/reason for {trial.output_name}')
    artifacts = result.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError(f'artifact manifest is missing for {trial.output_name}')
    resolved = {
        key: str(_relative_artifact(trial.output_dir, artifacts.get(key), key))
        for key in ('scene', 'scene_manifest', 'cuboids', 'hdf5', 'wrist_video', 'overview_video')
    }
    if 'ego_video' in artifacts:  # data v2 body-fixed second view
        resolved['ego_video'] = str(_relative_artifact(trial.output_dir, artifacts['ego_video'], 'ego_video'))
    if trial.visual_seed:
        scene_manifest = json.loads(Path(resolved['scene_manifest']).read_text())
        appearance = scene_manifest.get('appearance', {})
        if (scene_manifest.get('visual_seed') != trial.visual_seed
                or appearance.get('visual_seed') != trial.visual_seed
                or appearance.get('enabled') is not True
                or appearance.get('schema') != 'arx-door-appearance-v1'):
            raise ValueError(f'scene appearance mismatch for {trial.output_name}')
    try:
        with h5py.File(resolved['hdf5'], 'r') as file:
            episodes = file['data']
            if len(episodes) != 1:
                raise ValueError('expected exactly one HDF5 episode per trial')
            demo = episodes[next(iter(episodes))]
            actions = demo['actions'][:]
            states = demo['obs/joint_pos'][:]
            if actions.ndim != 2 or actions.shape[1] != 7 or states.shape != actions.shape:
                raise ValueError('HDF5 action/state shapes must both be (N, 7)')
            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                raise ValueError('HDF5 contains non-finite actions or states')
            count = len(actions)
            if count <= 0 or result.get('hdf5_frames') != count:
                raise ValueError('HDF5 frame count does not match result.json')
            if success and result['frames'] != count:
                raise ValueError('successful HDF5 and video frame counts differ')
    except (OSError, KeyError) as error:
        raise ValueError(f'invalid HDF5 for {trial.output_name}: {error}') from error
    return {**result, 'artifacts_resolved': resolved}


def validate_batch_output(
    output_root: str | Path,
    trials: Iterable[TrialSpec],
) -> dict:
    """Validate all requested attempts and return an aggregate report."""
    root = Path(output_root).expanduser().resolve()
    trial_tuple = tuple(trials)
    results = [validate_trial_output(trial) for trial in trial_tuple]
    successes = sum(bool(result['success']) for result in results)
    return {
        'schema': 'arx-door-skillgen-batch-v1',
        'output_root': str(root),
        'attempts': len(results),
        'successes': successes,
        'failures': len(results) - successes,
        'trials': [
            {
                **asdict(trial),
                'output_dir': str(trial.output_dir),
                'result': result,
            }
            for trial, result in zip(trial_tuple, results)
        ],
    }


def write_batch_report(output_root: str | Path, report: dict) -> Path:
    """Atomically publish the aggregate batch result."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return write_json(root / 'batch_results.json', report, allow_nan=True, default=str)


__all__ = [
    'FIRST_BATCH_ANGLES_DEG',
    'TrialSpec',
    'build_trial_specs',
    'validate_batch_output',
    'validate_trial_output',
    'write_batch_report',
    'process_result',
    'summarize_trials',
]
