#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Run sequential physical SkillGen trials and retain every attempt's result."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door import paths  # noqa: E402

from arx_door.skillgen.batch import (  # noqa: E402
    build_trial_specs, failed_process_result, process_result, summarize_trials,
    trial_manifest, validate_trial_output, write_batch_report,
)
from arx_door.runtime import (  # noqa: E402
    planning_urdf, runtime_provenance, simulation_environment, usd_environment,
    validate_runtime,
)
from arx_door.skillgen.physics_profile import (  # noqa: E402
    PHYSICS_PROFILES, configuration_identity, physics_device,
)


def runtime_identity(runtime: dict) -> dict:
    """Keep a batch on one validated simulator, source revision and numerical stack."""
    keys = ('profile', 'isaac_sim', 'isaac_lab', 'isaac_lab_revision', 'torch', 'numpy')
    if any(not runtime.get(key) for key in keys) or not runtime.get('python'):
        raise ValueError('batch runtime identity is incomplete; use a fresh output root')
    return {**{key: runtime[key] for key in keys},
            'python_minor': '.'.join(runtime['python'].split('.')[:2])}


def open_batch(output_root: Path, trials, *, resume: bool, runtime: dict,
               configuration: dict | None = None) -> dict:
    """Resume the same explicit plan without overwriting completed or interrupted trials."""
    plan = [trial_manifest(trial) for trial in trials]
    report_path = output_root / 'batch_results.json'
    if resume:
        if not report_path.is_file():
            raise ValueError('--resume requires an existing batch_results.json')
        report = json.loads(report_path.read_text())
        saved_plan = [{**item, 'visual_seed': item.get('visual_seed', 0)}
                      for item in report.get('plan', [])]
        if saved_plan != plan:
            raise ValueError('resume plan differs; use the original seed, angle '
                             'and split arguments')
        if configuration_identity(report.get('configuration')) != configuration_identity(configuration):
            raise ValueError('resume source scene or planning configuration differs')
        if runtime_identity(report.get('runtime', {})) != runtime_identity(runtime):
            raise ValueError('resume runtime differs; keep this batch on its original '
                             'Isaac profile')
        if len({item['output_dir'] for item in report['trials']}) != len(report['trials']):
            raise ValueError('existing report has duplicate attempts')
        report['plan'] = saved_plan
        return report
    if output_root.exists():
        raise ValueError('use a fresh output root; existing attempts are never overwritten')
    output_root.mkdir(parents=True)
    return {'schema': 'arx-door-skillgen-batch-v1', 'output_root': str(output_root),
            'plan': plan, 'configuration': configuration, 'runtime': runtime,
            'trials': [], **summarize_trials([])}


def record_result(report: dict, trial, result: dict, runtime: dict) -> None:
    """Save progress atomically after each attempted trial, including every failure."""
    item = {**trial_manifest(trial), 'runtime': runtime, 'result': result}
    scene_manifest = trial.output_dir / 'scene.json'
    if scene_manifest.is_file():
        try:
            metadata = json.loads(scene_manifest.read_text())
            item['appearance'] = metadata.get('appearance')
        except (OSError, ValueError, AttributeError) as error:
            item['appearance_metadata_error'] = str(error)
            if result.get('success') is True:
                result.update(success=False, failure_stage='artifact_validation',
                              failure_reason=f'cannot read scene appearance metadata: {error}')
    report['trials'].append(item)
    report.update(summarize_trials(report['trials']))
    report['pending'] = len(report['plan']) - report['attempts']
    write_batch_report(report['output_root'], report)


def run_process(command, *, env, cwd, stdout, stderr, timeout, check=False):
    """Own a process group so a timeout or interrupt also stops native planner children."""
    process = subprocess.Popen(command, env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                               start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)


def run_trial(trial, args, environment: dict, sim_environment: dict) -> dict:
    """Prepare one new scene and supervise its simulator process."""
    scene = trial.output_dir / 'scene.usd'
    cuboids = trial.output_dir / 'cuboids.json'
    prepare = [sys.executable, str(paths.PREPARE_SCENE),
               '--source', str(args.source_scene.resolve()), '--seed', str(trial.seed_dir),
               '--output', str(scene), '--angle-deg', str(trial.angle_deg),
               '--visual-seed', str(trial.visual_seed)]
    if args.allow_agent_reviewed_seeds:
        prepare.append('--allow-agent-reviewed-seeds')
    if args.radius_m is not None:
        prepare.extend(['--radius-m', str(args.radius_m)])
    with (trial.output_dir / 'prepare.log').open('w') as log:
        for command in (
            prepare,
            [sys.executable, str(paths.EXPORT_CUBOIDS),
             '--scene', str(scene), '--output', str(cuboids)],
        ):
            run_process(command, env=environment, cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, check=True, timeout=180)
    command = [
        sys.executable, '-u', str(paths.GENERATE_SKILLGEN),
        '--scene', str(scene), '--scene-manifest', str(scene.with_suffix('.json')),
        '--cuboids', str(cuboids), '--input-file', str(trial.input_file),
        '--output-file', str(trial.output_dir / 'generated.hdf5'),
        '--artifact-dir', str(trial.output_dir), '--angle-deg', str(trial.angle_deg),
        '--trial-seed', str(trial.seed), '--dataset-split', trial.split,
        '--visual-seed', str(trial.visual_seed),
        '--render-schedule', args.render_schedule,
        '--physics-profile', args.physics_profile,
        '--timeout-s', str(args.timeout_s), '--device',
        physics_device(args.physics_profile, args.device),
        '--headless', '--enable_cameras',
    ]
    if args.urdf:
        command.extend(['--urdf', str(args.urdf.resolve())])
    if getattr(args, 'generation_profile', None):
        command.extend(['--generation-profile', str(args.generation_profile.resolve())])
    with (trial.output_dir / 'generation.log').open('w') as log:
        process = run_process(command, env=sim_environment, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, timeout=args.timeout_s + 180)
    result_path = trial.output_dir / 'result.json'
    if not result_path.is_file():
        raise RuntimeError(f'simulator exited {process.returncode} without a result')
    raw_result = json.loads(result_path.read_text())
    if not isinstance(raw_result, dict):
        raise ValueError('simulator result must be a JSON object')
    result = process_result(raw_result, process.returncode)
    if process.returncode == 0 and result.get('physics_executed'):
        try:
            result = {**validate_trial_output(trial), 'process_returncode': process.returncode}
        except (TypeError, ValueError) as error:
            result = {**result, 'success': False, 'failure_stage': 'artifact_validation',
                      'failure_reason': str(error)}
    if result.get('success') is True:
        camera = result.get('camera_verification', {})
        if (camera.get('schema') != 'arx-wrist-camera-live-mount-v2'
                or camera.get('use_fabric') is not True
                or camera.get('frames_verified') != result.get('frames')
                or not 0 <= camera.get('max_position_error_m', float('inf')) <= 0.0001
                or not 0 <= camera.get('max_rotation_error_rad', float('inf')) <= 0.001):
            result = {**result, 'success': False, 'failure_stage': 'camera_validation',
                      'failure_reason': 'successful trial lacks complete camera v2 coverage'}
    return result


def reviewed_source_identity(seed_dir: Path, input_file: Path, *,
                             allow_agent_reviewed: bool = False,
                             urdf: Path | None = None) -> dict:
    """Bind both approval kinds and complete HDF5 contents before starting Kit."""
    from arx_door.demos.source_identity import bind_seed_hdf5
    from arx_door.kinematics import ArmKinematics

    return bind_seed_hdf5(seed_dir, input_file, ArmKinematics(planning_urdf(urdf)),
                         allow_agent_reviewed=allow_agent_reviewed)


def main() -> int:
    """Prepare scenes, run isolated simulator processes, and summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--angles', type=float, nargs='+', default=[0, -3, 3, -5, 5])
    parser.add_argument('--source-scene', type=Path,
                        default=ROOT / 'generated/door_teaching/upright02/live_scene.usd')
    parser.add_argument('--seed-dir', type=Path, action='append',
                        help='approved source directory; repeat in the same order as --input-file')
    parser.add_argument('--input-file', type=Path, action='append',
                        help='SkillGen seed HDF5; repeat for multiple distinct demonstrations')
    parser.add_argument('--allow-agent-reviewed-seeds', action='store_true',
                        help='accept verified agent-reviewed sources and preserve pending human review')
    parser.add_argument('--trial-seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--visual-seeds', type=int, nargs='+', default=[0],
                        help='appearance variants; zero preserves the original scene')
    parser.add_argument('--split', choices=['candidate', 'holdout', 'diagnostic'],
                        default='candidate')
    parser.add_argument('--resume', action='store_true', help='continue the original saved plan')
    parser.add_argument('--max-trials', type=int,
                        help='limit new attempts, including retained unreported directories')
    parser.add_argument('--urdf', type=Path, help='URDF from the selected ARX_DESCRIPTION_DIR package')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--physics-profile', choices=PHYSICS_PROFILES, default='lab_default',
                        help='source_cpu explicitly selects source CPU PhysX; default preserves Lab physics')
    parser.add_argument('--radius-m', type=float,
                        help='positive horizontal handle-to-base radius; default preserves each source')
    parser.add_argument('--render-schedule', choices=['legacy', 'pre_action'], default='pre_action',
                        help='camera render timing; legacy is available for historical replay')
    parser.add_argument('--timeout-s', type=float, default=900)
    parser.add_argument('--generation-profile', type=Path,
                        help='data v2 profile (e.g. configs/skillgen/data_v2.json)')
    args = parser.parse_args()
    if args.max_trials is not None and args.max_trials <= 0:
        parser.error('--max-trials must be positive')
    if not 0 < args.timeout_s < float('inf'):
        parser.error('--timeout-s must be finite and positive')
    if args.radius_m is not None and (not math.isfinite(args.radius_m) or args.radius_m <= 0):
        parser.error('--radius-m must be finite and positive')
    args.output_root = args.output_root.expanduser().resolve()
    input_files = args.input_file or [ROOT / 'generated/door_skillgen/seed/candidate_006.hdf5']
    seed_dirs = args.seed_dir or [ROOT / 'generated/door_teaching/upright02/candidate_006']
    try:
        args.device = physics_device(args.physics_profile, args.device)
        trials = build_trial_specs(args.angles, args.output_root, seed=0,
                                   input_files=input_files, seed_dirs=seed_dirs,
                                   trial_seeds=args.trial_seeds, visual_seeds=args.visual_seeds,
                                   split=args.split)
        for path in [args.source_scene, *(trial.input_file for trial in trials),
                     *(trial.seed_dir / 'episode.json' for trial in trials)]:
            if not path.is_file():
                raise ValueError(f'required input does not exist: {path}')
        args.urdf = planning_urdf(args.urdf)
        if not args.urdf.is_file():
            raise ValueError(f'planning URDF does not exist: {args.urdf}')
        source_quality = [reviewed_source_identity(
            seed, file, allow_agent_reviewed=args.allow_agent_reviewed_seeds, urdf=args.urdf)
            for seed, file in zip(seed_dirs, input_files)]
        runtime = runtime_provenance()
        runtime['profile'] = validate_runtime(runtime)
        environment, sim_environment = usd_environment(), simulation_environment()
        configuration = {'source_scene': str(args.source_scene.resolve()),
                         'urdf': str(args.urdf),
                         'device': args.device, 'timeout_s': args.timeout_s,
                         'render_schedule': args.render_schedule,
                         'physics_profile': args.physics_profile,
                         'source_quality': source_quality}
        if args.allow_agent_reviewed_seeds:
            configuration['allow_agent_reviewed_seeds'] = True
        if args.radius_m is not None:
            configuration['radius_m'] = args.radius_m
        if args.generation_profile:
            from arx_door.skillgen.recovery import load_generation_profile
            configuration['generation_profile'] = {
                'path': str(args.generation_profile.resolve()),
                'contents': load_generation_profile(args.generation_profile)}
        report = open_batch(args.output_root, trials, resume=args.resume, runtime=runtime,
                            configuration=configuration)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    report['pending'] = len(trials) - report['attempts']
    write_batch_report(args.output_root, report)
    completed = {item['output_dir'] for item in report['trials']}
    started = 0
    for trial in trials:
        if str(trial.output_dir) in completed:
            continue
        if args.max_trials is not None and started >= args.max_trials:
            break
        # Retaining an orphan is a newly reported failure and consumes this
        # invocation's quota, even though no simulator may be launched for it.
        started += 1
        interrupted = False
        try:
            # An unreported directory may be a killed run. Retain it as a failure;
            # a retry needs a new root so logs and partial artifacts survive.
            if trial.output_dir.exists():
                raise RuntimeError('unreported existing attempt retained as interrupted; '
                                   'not rerun')
            trial.output_dir.mkdir(parents=True)
            print(f'Starting {trial.source_id}, RNG seed {trial.seed}, '
                  f'visual seed {trial.visual_seed}, angle {trial.angle_deg:+g} degrees '
                  f'({trial.split})', flush=True)
            result = run_trial(trial, args, environment, sim_environment)
        except KeyboardInterrupt:
            interrupted = True
            result = failed_process_result(trial, 'interrupted', 'batch interrupted')
        except (OSError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result = failed_process_result(trial, 'batch_setup_or_process', str(error))
        record_result(report, trial, result, runtime)
        print(json.dumps({'angle_deg': trial.angle_deg, 'trial_seed': trial.seed,
                          'visual_seed': trial.visual_seed,
                          'success': result.get('success'),
                          'failure_reason': result.get('failure_reason')}), flush=True)
        if interrupted:
            return 130
    return 0 if report['failures'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
