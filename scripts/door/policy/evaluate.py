#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Evaluate fixed or seeded randomized scenes against an already-running policy server."""

from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.files import write_json
from arx_door.policy.config import DEFAULT_CONFIG, load_config, resource
from arx_door.autodata.runtime import PYTHON, simulation_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--scenarios', nargs='*')
    parser.add_argument('--scenario-config', type=Path,
                        default=ROOT / 'configs/policy/evaluation_scenarios.json')
    parser.add_argument('--random-episodes', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    sim_env = simulation_environment()
    config = load_config(args.config)
    scenario_config = json.loads(args.scenario_config.read_text())
    scenarios = {item['name']: item for item in scenario_config['trials']}
    names = args.scenarios if args.scenarios is not None else (
        [] if args.random_episodes else ['baseline'])
    unknown = set(names) - set(scenarios)
    if unknown or args.random_episodes < 0:
        parser.error(f'unknown scenarios: {sorted(unknown)}; random count must be nonnegative')
    if args.max_steps is not None and not 1 <= args.max_steps <= 2700:
        parser.error('max-steps must be 1..2700, within the original environment cap')
    selected = [scenarios[name] for name in names]
    rng_config = scenario_config['randomization']
    rng = random.Random(rng_config['seed'])
    selected += [{'name': f'random_{i:03d}',
                  'angle_deg': rng.uniform(*rng_config['angle_deg']),
                  'radius_m': rng.uniform(*rng_config['radius_m']),
                  'visual_seed': rng.choice(rng_config['visual_seeds'])}
                 for i in range(args.random_episodes)]
    if not selected:
        parser.error('select at least one fixed or random episode')
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = (args.output or Path('/outputs/policy') / f'evaluation_{timestamp}').absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'scenarios.json', selected)
    write_json(output / 'policy_config.json', config)
    write_json(output / 'evaluation_config.json', scenario_config)
    try:
        with socket.create_connection((config['host'], config['port']), timeout=2):
            pass
    except OSError as exc:
        write_json(output / 'summary.json', {
            'status': 'policy_server_unavailable', 'evaluation_complete': False,
            'error': str(exc), 'requested': len(selected), 'completed': 0,
        })
        print('Start run_policy.sh serve first.', file=sys.stderr)
        return 2
    free_mib = int(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        text=True).strip().splitlines()[0])
    if free_mib < 6144:
        write_json(output / 'summary.json', {
            'scope': 'simulation_closed_loop', 'evaluation_complete': False,
            'status': 'insufficient_gpu_memory', 'free_mib': free_mib,
            'required_free_mib': 6144, 'requested': len(selected), 'completed': 0,
        })
        print(f'Need 6144 MiB free for rendering after the server is loaded; available {free_mib}',
              file=sys.stderr)
        return 2
    python = PYTHON
    results = []
    for item in selected:
        trial = output / item['name']
        trial.mkdir()
        scene = resource(config, 'baseline_scene')
        if item['name'] != 'baseline':
            scene = trial / 'scene.usd'
            argv = ['bash', str(ROOT / 'scripts/door/scene/run_prepare_door_scene.sh'),
                    '--source', str(resource(config, 'source_scene')),
                    '--seed', str(resource(config, 'seed_dir')), '--output', str(scene),
                    '--angle-deg', str(item['angle_deg']), '--radius-m', str(item['radius_m']),
                    '--visual-seed', str(item['visual_seed']), '--allow-agent-reviewed-seeds']
            with (trial / 'prepare.log').open('w') as log:
                subprocess.run(argv, env=sim_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # The evaluation env is built from an asset bundle, exactly like AutoData generation.
        assets = trial / 'assets'
        with (trial / 'assets.log').open('w') as log:
            subprocess.run([str(python), str(ROOT / 'scripts/door/autodata/prepare.py'),
                            '--scene', str(scene), '--output', str(assets)],
                           env=sim_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        argv = [str(python), str(Path(__file__).with_name('rollout.py')),
                '--config', str(args.config.absolute()), '--assets', str(assets / 'manifest.json'),
                '--output', str(trial / 'rollout'), '--max-steps',
                str(args.max_steps or scenario_config['max_steps']),
                '--success-hold-steps', str(scenario_config['success_hold_steps'])]
        if args.headless:
            argv.append('--headless')
        print(f'EVALUATING {item["name"]}: {trial}', flush=True)
        write_json(trial / 'command.json', argv)
        with (trial / 'simulation.log').open('w') as log:
            code = subprocess.call(argv, env=sim_env, stdout=log, stderr=subprocess.STDOUT)
        result_file = trial / 'rollout/result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {
            'status': 'startup_error', 'success': False, 'steps': 0}
        if 'ERROR_OUT_OF_DEVICE_MEMORY' in (trial / 'simulation.log').read_text(errors='replace'):
            result.update(status='resource_error', success=False,
                          error='RTX ERROR_OUT_OF_DEVICE_MEMORY')
            write_json(trial / 'rollout/result.json', result)
            code = code or 1
        elif code and result['status'] in ('running', 'initializing'):
            result.update(status='process_interrupted', success=False)
            if result_file.parent.exists():
                write_json(result_file, result)
        results.append({'scenario': item, 'exit_code': code, 'result': result})
        write_json(output / 'summary.json', {
            'scope': 'simulation_closed_loop', 'episodes': results,
            'attempted': len(results),
            'completed': sum(row['result']['status'] == 'completed' for row in results),
            'requested': len(selected),
            'successes': sum(bool(row['result']['success']) for row in results),
            'evaluation_complete': (len(results) == len(selected)
                                    and all(row['result']['status'] == 'completed' for row in results)),
        })
        print(f'EPISODE_RESULT {item["name"]}: status={result["status"]} '
              f'success={result["success"]} steps={result["steps"]}', flush=True)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
