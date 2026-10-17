#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Replay generated training episodes through the policy evaluation rollout.

Each trial's recorded absolute actions run open loop in ``rollout.py`` with
the trial's own scene, so generation and evaluation share physics, cameras,
action mapping and acceptance. Every replay must pass before a closed-loop
policy failure says anything about the model. No policy server is needed.
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
from arx_door.runtime import clean_environment, simulation_python

DEFAULT_CONFIG = ROOT / 'configs/policy/arx_knob_v2_n17.json'


def trial_dirs(paths: list[Path]) -> list[Path]:
    """Accept trial directories or batch_results.json files (successful trials only)."""
    trials = []
    for path in paths:
        if path.is_file():
            report = json.loads(path.read_text())
            trials += [Path(item['output_dir']) for item in report['trials']
                       if item.get('result', {}).get('success')]
        else:
            trials.append(path)
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials', type=Path, nargs='+', help='trial directories or batch_results.json')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, help='replay at most this many trials')
    args = parser.parse_args()
    config = load_config(args.config)
    trials = trial_dirs(args.trials)[:args.limit]
    if not trials:
        parser.error('no successful trials to replay')
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = (args.output or ROOT / 'generated/door_policy/runs' / f'replay_check_{timestamp}').absolute()
    output.mkdir(parents=True, exist_ok=False)
    base_env = clean_environment()
    base_env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
    python = simulation_python()
    sim_env = json.loads(subprocess.check_output(
        [str(python), '-c', ('import json;from arx_door.runtime import simulation_environment;'
                            'print(json.dumps(simulation_environment()))')],
        env=base_env, text=True))
    sim_env.update(ISAAC_SIM_PYTHON=str(python), PYTHONUNBUFFERED='1')
    rows = []
    for index, trial in enumerate(trials):
        trial = trial.absolute()
        source = json.loads((trial / 'result.json').read_text())
        episode = output / f'{index:03d}_{trial.name}'
        episode.mkdir()
        argv = [str(python), str(ROOT / 'scripts/door/policy/rollout.py'),
                '--config', str(args.config.absolute()), '--scene', str(trial / 'scene.usd'),
                '--cuboids', str(trial / 'cuboids.json'), '--output', str(episode / 'rollout'),
                '--replay-hdf5', str(trial / 'generated.hdf5'),
                '--max-steps', str(min(2700, int(source['frames']) + 30)), '--headless']
        write_json(episode / 'command.json', argv)
        with (episode / 'simulation.log').open('w') as log:
            code = subprocess.call(argv, env=sim_env, stdout=log, stderr=subprocess.STDOUT)
        result_file = episode / 'rollout/result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {
            'status': 'startup_error', 'success': False, 'steps': 0}
        metrics = result.get('physical_metrics', {})
        rows.append({
            'trial': str(trial), 'exit_code': code, 'status': result['status'],
            'success': bool(result.get('success')), 'stop_reason': result.get('stop_reason'),
            'steps': result.get('steps'), 'generated_frames': source['frames'],
            'initial_state_max_abs_error': result.get('replay', {}).get('initial_state_max_abs_error'),
            'final_door_angle_deg': metrics.get('final_door_angle_deg'),
            'generation_final_door_angle_deg': source.get('final_door_angle_deg'),
            'error': result.get('error'),
        })
        print(f'REPLAY {trial.name}: success={rows[-1]["success"]} stop={rows[-1]["stop_reason"]} '
              f'steps={rows[-1]["steps"]}/{source["frames"]}', flush=True)
        write_json(output / 'summary.json', {
            'schema': 'arx-door-replay-check-v1', 'config': str(args.config.absolute()),
            'policy_checkpoint_used': False, 'video_keys': config.get('video_keys', ['wrist']),
            'requested': len(trials), 'attempted': len(rows),
            'successes': sum(row['success'] for row in rows),
            'all_passed': len(rows) == len(trials) and all(row['success'] for row in rows),
            'episodes': rows,
        })
    summary = json.loads((output / 'summary.json').read_text())
    print(json.dumps({key: summary[key] for key in ('requested', 'successes', 'all_passed')}))
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
