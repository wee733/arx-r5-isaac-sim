#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Replay accepted AutoData episodes through the policy evaluation rollout.

Each accepted episode of a ``scripts/door/autodata/generate.sh`` run replays its
recorded actions open loop in ``rollout.py``, built from the run's own asset
bundle and in the action mode it was generated with, so generation and
evaluation share physics, cameras, action mapping and acceptance. Every replay
must pass before a closed-loop policy failure says anything about the model.
No policy server is needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.files import write_json
from arx_door.policy.config import load_config
from arx_door.autodata.runtime import PYTHON, simulation_environment, replay_inputs

DEFAULT_CONFIG = ROOT / 'configs/policy/arx_knob_v2_n17.json'


def run_episodes(runs: list[Path]) -> list[tuple[Path, str, int]]:
    """Return ``(run, demo, frames)`` for every accepted episode of each run."""
    import h5py
    episodes = []
    for run in runs:
        accepted = run / 'accepted.hdf5'
        if not accepted.is_file():
            raise FileNotFoundError(f'{accepted} missing; run scripts/door/autodata/accept.py {run}')
        with h5py.File(accepted, 'r') as file:
            episodes += [(run, name, len(file['data'][name]['actions'])) for name in file['data']]
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='output directories of generate.sh')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, help='replay at most this many episodes')
    args = parser.parse_args()
    sim_env = simulation_environment()
    config = load_config(args.config)
    episodes = run_episodes([run.absolute() for run in args.runs])[:args.limit]
    if not episodes:
        parser.error('no accepted episodes to replay')
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = (args.output or Path('/outputs/policy') / f'replay_check_{timestamp}').absolute()
    output.mkdir(parents=True, exist_ok=False)
    python = PYTHON
    rows = []
    for index, (run, demo, frames) in enumerate(episodes):
        assets, physics = replay_inputs(run)
        episode = output / f'{index:03d}_{run.name}_{demo}'
        episode.mkdir()
        argv = [str(python), str(ROOT / 'scripts/door/policy/rollout.py'),
                '--config', str(args.config.absolute()), '--assets', str(assets),
                '--physics', physics,
                '--output', str(episode / 'rollout'),
                '--replay-hdf5', str(run / 'accepted.hdf5'), '--replay-demo', demo,
                # AutoData ends on the first accepted step. Requiring another
                # 30 recorded steps rejects an otherwise identical replay.
                '--success-hold-steps', '1', '--max-steps', str(frames), '--headless']
        write_json(episode / 'command.json', argv)
        with (episode / 'simulation.log').open('w') as log:
            code = subprocess.call(argv, env=sim_env, stdout=log, stderr=subprocess.STDOUT)
        result_file = episode / 'rollout/result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {
            'status': 'startup_error', 'success': False, 'steps': 0}
        metrics = result.get('physical_metrics', {})
        rows.append({
            'run': str(run), 'demo': demo, 'physics': physics, 'exit_code': code, 'status': result['status'],
            'success': bool(result.get('success')), 'stop_reason': result.get('stop_reason'),
            'steps': result.get('steps'), 'generated_frames': frames,
            'initial_state_max_abs_error': result.get('replay', {}).get('initial_state_max_abs_error'),
            'final_door_angle_deg': metrics.get('final_door_angle_deg'),
            'error': result.get('error'),
        })
        print(f'REPLAY {run.name}/{demo}: success={rows[-1]["success"]} stop={rows[-1]["stop_reason"]} '
              f'steps={rows[-1]["steps"]}/{frames}', flush=True)
        write_json(output / 'summary.json', {
            'schema': 'arx-door-replay-check-v1', 'config': str(args.config.absolute()),
            'policy_checkpoint_used': False, 'video_keys': config.get('video_keys', ['wrist']),
            'requested': len(episodes), 'attempted': len(rows),
            'successes': sum(row['success'] for row in rows),
            'all_passed': len(rows) == len(episodes) and all(row['success'] for row in rows),
            'episodes': rows,
        })
    summary = json.loads((output / 'summary.json').read_text())
    print(json.dumps({key: summary[key] for key in ('requested', 'successes', 'all_passed')}))
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
