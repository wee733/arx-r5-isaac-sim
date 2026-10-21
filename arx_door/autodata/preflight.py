# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Read-only dependency and annotated-source checks, without starting Kit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from . import DEPENDENCY, source_checkouts


def source_evidence(path: str | Path) -> dict:
    """Check a convert_seed.py source against the ARX AutoData contract; not physical acceptance."""
    import h5py
    import numpy as np

    episodes = []
    with h5py.File(path, 'r') as dataset:
        for name, episode in dataset['data'].items():
            actions = episode['actions'][:]
            if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
                raise ValueError(f'{name}: actions must be finite [T,8] absolute link6 pose + gripper; '
                                 'convert seeds with scripts/door/autodata/convert_seed.py')
            info = episode['obs/datagen_info']
            for field, keys in (
                ('eef_pose', ('link6',)), ('target_eef_pose', ('link6',)),
                ('object_pose', ('door',)),
            ):
                for key in keys:
                    values = info[f'{field}/{key}'][:]
                    if values.shape != (len(actions), 4, 4) or not np.isfinite(values).all():
                        raise ValueError(f'{name}: invalid {field}/{key}')
            boundaries = {}
            for field in ('subtask_start_signals', 'subtask_term_signals'):
                signal = info[f'{field}/knob_interaction'][:].reshape(-1)
                if len(signal) != len(actions) or not np.isin(signal, (0, 1)).all():
                    raise ValueError(f'{name}: invalid {field}/knob_interaction')
                rises = np.flatnonzero(np.diff(signal.astype(int)) > 0) + 1
                if len(rises) != 1:
                    raise ValueError(f'{name}: expected exactly one rising {field} edge')
                boundaries[field] = int(rises[0])
            if boundaries['subtask_start_signals'] >= boundaries['subtask_term_signals']:
                raise ValueError(f'{name}: non-increasing skill boundaries')
            episodes.append({'name': name, 'frames': len(actions), **boundaries})
    if not episodes:
        raise ValueError('source has no episodes')
    return {'source': str(Path(path).resolve()), 'episodes': episodes,
            'scope': 'schema_and_signals_only', 'physical_validation': False}


def _command(argv: list[str]) -> str | None:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def checkout_evidence(autodata_root: str | Path | None = None) -> dict:
    """Check pinned Git checkouts without importing a simulator or inspecting Docker."""
    problems = []
    pins = {}
    root = None if autodata_root is None else Path(autodata_root).resolve()
    for name, path, expected in source_checkouts(root):
        # A missing submodule directory must not resolve git HEAD from its parent repo.
        actual = _command(['git', '-C', str(path), 'rev-parse', 'HEAD']) if (path / '.git').exists() else None
        pins[name] = {'expected': expected, 'actual': actual}
        if actual != expected:
            problems.append(f'{name}: missing or different pinned checkout')
        elif _command(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=normal']) != '':
            problems.append(f'{name}: vendor checkout has local changes or cannot be inspected')
    return {'pins': pins, 'problems': problems, 'source_ready': not problems}


def runtime_evidence(autodata_root: str | Path | None = None) -> dict:
    """Report exact source pins and available resources without loading models."""
    source = checkout_evidence(autodata_root)
    problems = list(source['problems'])
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key] = int(value.split()[0]) * 1024
    if memory.get('SwapTotal', 0) and memory.get('SwapFree', 0) == 0:
        problems.append('swap is full; inspect memory pressure before simulator startup')
    gpu = _command(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'])
    image_name = DEPENDENCY['runtime']['image']
    image = _command(['docker', 'image', 'inspect', image_name, '--format', '{{.Id}}'])
    if image is None:
        problems.append(f'pinned AutoData cuRobo Docker image is absent: {image_name}')
    else:
        for name, _, expected in source_checkouts():
            label = _command(['docker', 'image', 'inspect', image_name, '--format',
                              '{{index .Config.Labels "arx.' + name + '_commit"}}'])
            if label != expected:
                problems.append(f'Docker image has missing or different {name} revision label')
    return {'pins': source['pins'], 'memory_bytes': memory, 'gpu_memory_mib': gpu,
            'docker_image': image_name,
            'docker_image_available': image is not None, 'problems': problems,
            'ready_for_simulation': not problems}


def write_report(path: str | Path, report: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
