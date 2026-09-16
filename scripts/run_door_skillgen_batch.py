#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Run sequential physical SkillGen trials and retain every attempt's result."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from arx_r5_isaac_sim_bringup.door_skillgen.batch import build_trial_specs, validate_trial_output


ROOT = Path(__file__).resolve().parents[1]


def usd_environment() -> dict[str, str]:
    """Find the USD libraries bundled with the selected Isaac Sim Python."""
    spec = importlib.util.find_spec('isaacsim')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('run this script with the Isaac Lab Python environment')
    sim_root = Path(next(iter(spec.submodule_search_locations)))
    libraries = sorted(
        path for path in (sim_root / 'extscache').glob('omni.usd.libs-*')
        if (path / 'pxr').is_dir()
    )
    if not libraries:
        raise RuntimeError(f'cannot find bundled USD libraries under {sim_root}')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (
        str(ROOT / 'arx_r5_isaac_sim_bringup'), str(libraries[-1]),
        environment.get('PYTHONPATH'),
    )))
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (
        str(libraries[-1] / 'bin'), sysconfig.get_config_var('LIBDIR'),
        environment.get('LD_LIBRARY_PATH'),
    )))
    return environment


def main() -> int:
    """Prepare scenes, run isolated simulator processes, and summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--angles', type=float, nargs='+', default=[0, -3, 3, -5, 5])
    parser.add_argument('--source-scene', type=Path,
                        default=ROOT / 'generated/door_teaching/upright02/live_scene.usd')
    parser.add_argument('--seed-dir', type=Path,
                        default=ROOT / 'generated/door_teaching/upright02/candidate_006')
    parser.add_argument('--input-file', type=Path,
                        default=ROOT / 'generated/door_skillgen/seed/candidate_006.hdf5')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timeout-s', type=float, default=900)
    args = parser.parse_args()
    trials = build_trial_specs(args.angles, args.output_root, seed=0)
    if any(trial.output_dir.exists() for trial in trials):
        parser.error('use a fresh output root; existing attempts are never overwritten')
    environment = usd_environment()
    sim_environment = os.environ.copy()
    sim_environment['PYTHONPATH'] = os.pathsep.join(filter(None, (
        str(ROOT / 'arx_r5_isaac_sim_bringup'), sim_environment.get('PYTHONPATH'),
    )))
    sim_environment.setdefault('TERM', 'xterm')
    report = {'schema': 'arx-door-skillgen-batch-v1', 'trials': []}
    report_path = args.output_root.resolve() / 'batch_results.json'
    for trial in trials:
        trial.output_dir.mkdir(parents=True)
        scene = trial.output_dir / 'scene.usd'
        cuboids = trial.output_dir / 'cuboids.json'
        try:
            with (trial.output_dir / 'prepare.log').open('w') as log:
                for command in (
                    [sys.executable, str(ROOT / 'scripts/prepare_door_skillgen_scene.py'),
                     '--source', str(args.source_scene.resolve()),
                     '--seed', str(args.seed_dir.resolve()), '--output', str(scene),
                     '--angle-deg', str(trial.angle_deg)],
                    [sys.executable, str(ROOT / 'scripts/export_door_cuboids.py'),
                     '--scene', str(scene), '--output', str(cuboids)],
                ):
                    subprocess.run(command, env=environment, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'Starting physical trial at {trial.angle_deg:+g} degrees', flush=True)
            with (trial.output_dir / 'generation.log').open('w') as log:
                process = subprocess.run([
                    sys.executable, '-u', str(ROOT / 'scripts/generate_door_skillgen.py'),
                    '--scene', str(scene), '--scene-manifest', str(scene.with_suffix('.json')),
                    '--cuboids', str(cuboids), '--input-file', str(args.input_file.resolve()),
                    '--output-file', str(trial.output_dir / 'generated.hdf5'),
                    '--artifact-dir', str(trial.output_dir), '--angle-deg', str(trial.angle_deg),
                    '--timeout-s', str(args.timeout_s), '--device', args.device,
                    '--headless', '--enable_cameras',
                ], env=sim_environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result_path = trial.output_dir / 'result.json'
            if not result_path.is_file():
                raise RuntimeError(f'simulator exited {process.returncode} without a result')
            # Kit may exit with zero even on an application error; use the
            # explicitly recorded physical outcome, never the exit code alone.
            result = json.loads(result_path.read_text())
            if result.get('physics_executed'):
                try:
                    result = validate_trial_output(trial)
                except ValueError as error:
                    result = {**result, 'success': False,
                              'failure_stage': 'artifact_validation',
                              'failure_reason': str(error)}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result = {'angle_deg': trial.angle_deg, 'success': False,
                      'physics_executed': False, 'failure_stage': 'batch_setup_or_process',
                      'failure_reason': str(error)}
        report['trials'].append({'output_dir': str(trial.output_dir), 'result': result})
        report['attempts'] = len(report['trials'])
        report['successes'] = sum(item['result'].get('success') is True
                                  for item in report['trials'])
        report['failures'] = report['attempts'] - report['successes']
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'angle_deg': trial.angle_deg, 'success': result.get('success'),
                          'failure_reason': result.get('failure_reason')}), flush=True)
    return 0 if report['failures'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
