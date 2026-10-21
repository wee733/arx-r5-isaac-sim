#!/usr/bin/env python3
"""Generate and independently verify one real trajectory per sampled condition.

Run inside the pinned AutoData container. This correctness-first campaign builds
the planner from each condition's actual USD geometry. It is not a parallel
throughput benchmark. Completed stages may be resumed; incomplete runs are never
silently counted or overwritten. The output export_manifest.json feeds the
existing VLA exporter on the host.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = '/isaac-sim/python.sh'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    conditions = json.loads(args.conditions.read_text())[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / 'campaign.json'
    report = json.loads(summary_path.read_text()) if summary_path.exists() else {
        'schema': 'arx-autodata-conditions-v1', 'episodes': [],
        'count_unit': 'one physically executed trajectory per distinct scene condition',
        'source': str(args.input.resolve()), 'vla_training_executed': False}
    records = {row['condition_id']: row for row in report['episodes']}
    export = []
    os.environ['PYTHONPATH'] = f'{ROOT}:/workspaces/autodata'

    def command(argv, log):
        start = time.monotonic()
        with log.open('w') as stream:
            code = subprocess.call(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT)
        return {'exit_code': code, 'wall_seconds': time.monotonic() - start, 'log': str(log)}

    for index, condition in enumerate(conditions):
        name = condition['condition_id']
        run = args.output / name
        row = records.setdefault(name, {'condition_id': name, 'run': str(run), 'stages': {}})
        # Hold out an entire scene condition and random seed, never adjacent frames.
        split = 'holdout' if len(conditions) > 1 and index == len(conditions)-1 else 'train'
        row['split'] = split
        try:
            if not (run / 'result.json').exists():
                if run.exists() and any(run.iterdir()):
                    raise RuntimeError(f'incomplete generation retained at {run}; choose a fresh run directory')
                row['stages']['generation'] = command([
                    'bash', ROOT / 'scripts/door/autodata/generate.sh', '--assets', condition['assets'],
                    '--input', args.input, '--output', run, '--num-envs', '1', '--trials', '1',
                    '--seed', str(condition['seed'])], args.output / f'{name}.generation.log')
            result = json.loads((run / 'result.json').read_text())
            row['attempts'] = result['num_attempts']
            row['generator_successes'] = result['num_success']
            if not (run / 'accepted.json').exists():
                row['stages']['acceptance'] = command([
                    PYTHON, ROOT / 'scripts/door/autodata/accept.py', run], args.output / f'{name}.accept.log')
            accepted = json.loads((run / 'accepted.json').read_text())
            row['physical_successes'] = len(accepted['accepted'])
            row['physical_details'] = accepted['episode_details']
            if row['physical_successes'] == 0:
                raise RuntimeError('no physically accepted trajectory')
            for mode in ('pose', 'joint'):
                replay_run = run
                if mode == 'joint':
                    replay_run = run / 'joint_replay_candidate'
                    if not replay_run.exists():
                        preparation = command([PYTHON, ROOT / 'scripts/door/autodata/prepare_joint_replay.py', run],
                                              args.output / f'{name}.joint_prepare.log')
                        if preparation['exit_code']:
                            raise RuntimeError('joint action preparation failed')
                output = run / f'{mode}_replay'
                if not (output / 'summary.json').exists():
                    row['stages'][f'{mode}_replay'] = command([
                        PYTHON, ROOT / 'scripts/door/policy/replay_check.py', replay_run,
                        '--output', output], args.output / f'{name}.{mode}_replay.log')
                replay = json.loads((output / 'summary.json').read_text())
                row[f'{mode}_replay_successes'] = replay['successes']
                if not replay['all_passed']:
                    raise RuntimeError(f'{mode} replay failed')
            row['eligible_for_export'] = True
            row.pop('error', None)
            export.append({'run': str(run), 'split': split,
                           'pose_replay_summary': str(run / 'pose_replay/summary.json'),
                           'joint_replay_summary': str(run / 'joint_replay/summary.json')})
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            row['eligible_for_export'] = False
            row['error'] = str(error)
        report['episodes'] = list(records.values())
        report['counts'] = {key: sum(r.get(key, 0) for r in report['episodes']) for key in
                            ('attempts', 'generator_successes', 'physical_successes',
                             'pose_replay_successes', 'joint_replay_successes', 'eligible_for_export')}
        report['counts']['distinct_accepted_conditions'] = sum(r.get('physical_successes', 0) > 0 for r in report['episodes'])
        summary_path.write_text(json.dumps(report, indent=2) + '\n')
        (args.output / 'export_manifest.json').write_text(json.dumps(export, indent=2) + '\n')
        print(json.dumps({'condition': name, 'counts': report['counts'], 'error': row.get('error')}), flush=True)
    return 0 if len(export) == len(conditions) else 1


if __name__ == '__main__':
    raise SystemExit(main())
