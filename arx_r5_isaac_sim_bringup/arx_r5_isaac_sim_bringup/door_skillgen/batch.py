# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Deterministic batch layout and artifact validation for door SkillGen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from .contract import FIRST_BATCH_ANGLES_DEG


@dataclass(frozen=True)
class TrialSpec:
    """One explicit physical generation attempt."""

    index: int
    angle_deg: float
    seed: int
    output_name: str
    output_dir: Path


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
) -> tuple[TrialSpec, ...]:
    """Build ordered, unique trial specs without silently deduplicating."""
    root = Path(output_root).expanduser().resolve()
    angles = tuple(float(value) for value in angles_deg)
    if not angles:
        raise ValueError('at least one angle is required')
    names = tuple(_angle_name(value) for value in angles)
    if len(set(names)) != len(names):
        raise ValueError('angles collide after output-name normalization')
    return tuple(
        TrialSpec(index, angle, int(seed), name, root / name)
        for index, (angle, name) in enumerate(zip(angles, names))
    )


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
    destination = root / 'batch_results.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, destination)
    return destination


__all__ = [
    'FIRST_BATCH_ANGLES_DEG',
    'TrialSpec',
    'build_trial_specs',
    'validate_batch_output',
    'validate_trial_output',
    'write_batch_report',
]
