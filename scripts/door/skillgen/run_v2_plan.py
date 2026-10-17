#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run or inspect a data v2 production plan, one SkillGen batch per plan entry.

``status`` reads every batch report and prints counts; ``run`` launches the
missing trials of each batch through ``run_batch.sh``, resuming batches that
already have a report. Interrupting is safe: rerun the same command. A batch
whose saved plan differs from the plan file is refused, never overwritten.
Standard library only; any python3 can run it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = ROOT / 'configs/skillgen/v2_first_success_plan.json'


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get('schema') != 'arx-door-v2-production-plan-v1':
        raise ValueError(f'{path}: not a v2 production plan')
    names = [batch['name'] for batch in plan['batches']]
    if len(set(names)) != len(names):
        raise ValueError('batch names must be unique')
    for batch in plan['batches']:
        if batch['source'] not in plan['sources']:
            raise ValueError(f'{batch["name"]}: unknown source {batch["source"]}')
        if batch['split'] not in ('candidate', 'holdout'):
            raise ValueError(f'{batch["name"]}: split must be candidate or holdout')
    holdout_sources = {b['source'] for b in plan['batches'] if b['split'] == 'holdout'}
    train_sources = {b['source'] for b in plan['batches'] if b['split'] == 'candidate'}
    if holdout_sources & train_sources:
        raise ValueError(f'seeds {sorted(holdout_sources & train_sources)} feed both train and holdout')
    return plan


def resolve(path: str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def batch_root(plan: dict, batch: dict) -> Path:
    return resolve(plan['output_root']) / batch['name']


def planned_trials(batch: dict) -> int:
    return len(batch['angles']) * len(batch['trial_seeds']) * len(batch['visual_seeds'])


def batch_status(plan: dict, batch: dict) -> dict:
    report_path = batch_root(plan, batch) / 'batch_results.json'
    row = {'name': batch['name'], 'split': batch['split'], 'planned': planned_trials(batch),
           'attempted': 0, 'succeeded': 0, 'recovery': 0, 'failures': {}}
    if not report_path.is_file():
        return row
    report = json.loads(report_path.read_text())
    for trial in report['trials']:
        result = trial.get('result', {})
        row['attempted'] += 1
        if result.get('success'):
            row['succeeded'] += 1
            recovery = result.get('physical_audit', {}).get('recovery') or {}
            row['recovery'] += recovery.get('status') == 'recovery_connected'
        else:
            reason = (result.get('failure_reason') or 'unknown')[:80]
            row['failures'][reason] = row['failures'].get(reason, 0) + 1
    return row


def command(plan: dict, batch: dict, *, resume: bool, max_trials: int | None) -> list[str]:
    source = plan['sources'][batch['source']]
    argv = ['bash', str(ROOT / 'scripts/door/skillgen/run_batch.sh'),
            '--output-root', str(batch_root(plan, batch)),
            '--angles', *[str(value) for value in batch['angles']],
            '--source-scene', str(resolve(source['source_scene'])),
            '--seed-dir', str(resolve(source['seed_dir'])),
            '--input-file', str(resolve(source['input_file'])),
            '--allow-agent-reviewed-seeds',
            '--trial-seeds', *[str(value) for value in batch['trial_seeds']],
            '--visual-seeds', *[str(value) for value in batch['visual_seeds']],
            '--split', batch['split'], '--radius-m', str(batch['radius_m']),
            '--physics-profile', plan['physics_profile'], '--timeout-s', str(plan['timeout_s']),
            '--generation-profile', str(resolve(plan['generation_profile'])),
            '--urdf', str(resolve(plan['urdf']))]
    if resume:
        argv.append('--resume')
    if max_trials is not None:
        argv += ['--max-trials', str(max_trials)]
    return argv


def print_status(plan: dict) -> list[dict]:
    rows = [batch_status(plan, batch) for batch in plan['batches']]
    for row in rows:
        print(f'{row["name"]:<16} {row["split"]:<9} attempted {row["attempted"]:>3}/{row["planned"]:<3} '
              f'success {row["succeeded"]:>3}  recovery {row["recovery"]:>3}')
        for reason, count in sorted(row['failures'].items(), key=lambda item: -item[1]):
            print(f'    {count:>3} x {reason}')
    totals = {split: sum(row['succeeded'] for row in rows if row['split'] == split)
              for split in ('candidate', 'holdout')}
    print(f'successful train episodes {totals["candidate"]}, holdout {totals["holdout"]}')
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'run', 'print-commands'])
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--batch', action='append', help='limit to these batch names')
    parser.add_argument('--max-trials', type=int, help='per batch, for this invocation')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    batches = [batch for batch in plan['batches'] if not args.batch or batch['name'] in args.batch]
    if args.batch and len(batches) != len(set(args.batch)):
        parser.error(f'unknown batch in {args.batch}')
    if args.action == 'status':
        print_status(plan)
        return 0
    for batch in batches:
        status = batch_status(plan, batch)
        resume = (batch_root(plan, batch) / 'batch_results.json').is_file()
        argv = command(plan, batch, resume=resume, max_trials=args.max_trials)
        if args.action == 'print-commands':
            print(' '.join(argv))
            continue
        if status['attempted'] >= status['planned']:
            print(f'{batch["name"]}: complete ({status["succeeded"]}/{status["planned"]} succeeded)')
            continue
        print(f'{batch["name"]}: {"resuming" if resume else "starting"} '
              f'({status["attempted"]}/{status["planned"]} attempted)', flush=True)
        # ROS or Isaac Python paths from the caller's shell must not leak into Kit.
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
        code = subprocess.call(argv, cwd=ROOT, env=environment)
        # run_batch exits 1 when any trial failed; that is a normal, recorded outcome.
        if code not in (0, 1):
            print(f'{batch["name"]}: batch runner exited {code}; stopping', file=sys.stderr)
            return code
    if args.action == 'run':
        print_status(plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
