#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Turn a finished data v2 production plan into a training bundle.

Exports train (candidate) and holdout separately, validates both with the
official N1.7 loader, audits the inference stage prompter, enforces the
plan's acceptance numbers and packages the full fine-tune scripts. Writes
``<output>/`` plus ``<output>.tar.zst`` (or ``.tar.gz``). Refuses to package
when any check fails; ``build_report.json`` always says why.
Run with the GR00T data venv: scripts/door/dataset/run_dataset_tool.sh build_v2_bundle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.files import write_json

PLAN = ROOT / 'configs/skillgen/v2_first_success_plan.json'
GROOT_ROOT = ROOT / 'generated/runtime/groot_n17_source'
TOOLS = Path(__file__).resolve().parent


def resolve(path: str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def run(argv: list[str], log: Path) -> None:
    with log.open('w') as handle:
        code = subprocess.call(argv, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    if code:
        raise RuntimeError(f'{Path(argv[1]).name} failed ({code}); see {log}')


def export_split(plan: dict, split: str, destination: Path, logs: Path) -> dict:
    reports = [resolve(plan['output_root']) / batch['name'] / 'batch_results.json'
               for batch in plan['batches'] if batch['split'] == split]
    missing = [str(path) for path in reports if not path.is_file()]
    if missing:
        raise RuntimeError(f'{split}: batches not run yet: {missing}')
    options = plan['export']
    argv = [sys.executable, str(TOOLS / 'export_door_lerobot.py'), '--output', str(destination),
            '--split', split]
    for report in reports:
        argv += ['--batch-report', str(report)]
    for camera in options['cameras']:
        argv += ['--camera', camera]
    if options['stage_language']:
        argv.append('--stage-language')
    if options['crop_before_recovery']:
        argv.append('--crop-before-recovery')
    if options.get('max_transition_overshoot_rad') is not None:
        argv += ['--max-transition-overshoot', str(options['max_transition_overshoot_rad'])]
    run(argv, logs / f'export_{split}.log')
    low, high = options['gripper_state_range_m']
    run([sys.executable, str(TOOLS / 'validate_door_n17.py'), '--dataset', str(destination),
         '--groot-root', str(GROOT_ROOT), '--config', str(resolve(options['modality_config'])),
         '--gripper-state-range', str(low), str(high)], logs / f'validate_{split}.log')
    export = json.loads((destination / 'meta/door_export_report.json').read_text())
    recovered = sum(bool(source.get('cropped_leading_frames')) for source in export['sources'])
    excluded = {}
    for item in export.get('excluded_sources', []):
        reason = item.get('reason') or item.get('failure_reason') or 'unknown'
        excluded[reason] = excluded.get(reason, 0) + 1
    info = json.loads((destination / 'meta/info.json').read_text())
    return {'episodes': info['total_episodes'], 'frames': info['total_frames'],
            'recovery_episodes': recovered,
            'recovery_fraction': recovered / max(1, info['total_episodes']),
            'excluded': excluded}


def acceptance(plan: dict, train: dict, holdout: dict, prompter: dict) -> list[str]:
    limits, problems = plan['acceptance'], []
    if train['episodes'] < limits['min_train_episodes']:
        problems.append(f'train has {train["episodes"]} episodes, needs {limits["min_train_episodes"]}')
    if holdout['episodes'] < limits['min_holdout_episodes']:
        problems.append(f'holdout has {holdout["episodes"]}, needs {limits["min_holdout_episodes"]}')
    if not limits['min_recovery_fraction'] <= train['recovery_fraction'] <= limits['max_recovery_fraction']:
        problems.append(f'train recovery fraction {train["recovery_fraction"]:.2f} outside '
                        f'[{limits["min_recovery_fraction"]}, {limits["max_recovery_fraction"]}]')
    if prompter['frame_agreement_min'] < limits['min_prompter_agreement']:
        problems.append(f'stage prompter agreement {prompter["frame_agreement_min"]:.3f} below '
                        f'{limits["min_prompter_agreement"]}')
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def package(bundle: Path) -> Path:
    if shutil.which('zstd'):
        archive = bundle.with_name(bundle.name + '.tar.zst')
        argv = ['tar', '-I', 'zstd -T0 -3', '-cf', str(archive), '-C', str(bundle.parent), bundle.name]
    else:
        archive = bundle.with_name(bundle.name + '.tar.gz')
        argv = ['tar', '-czf', str(archive), '-C', str(bundle.parent), bundle.name]
    subprocess.run(argv, check=True)
    return archive


README = """# ARX door data v2 — full fine-tune bundle

Contents: `datasets/train` ({train_episodes} episodes, {train_frames} frames) and
`datasets/holdout` ({holdout_episodes} episodes, never a training input), the
N1.7 modality config `configs/door_v2_n17.py` (wrist + ego RGB, per-frame stage
language, 16-step relative arm + absolute gripper), and the full fine-tune
scripts. `manifest.json` holds counts and SHA-256 of the small files;
`build_report.json` holds every check.

## Train (all modules: LLM, vision, projector, diffusion head)

```bash
cd <this bundle>
# edit the two paths and #SBATCH lines in scripts/train_v2_full_n17.sbatch, then
sbatch scripts/train_v2_full_n17.sbatch
# or interactively:
GR00T_REPO=<Isaac-GR00T at d2b7e75> N17_OUTPUT_DIR=<new dir> bash scripts/train_v2_full_n17.sh
```

Defaults: 4 GPUs, global batch 64, 15000 steps, checkpoint every 2500,
learning rate 5e-5 (not an official value). Full fine-tuning needs 80 GB+ per
GPU per NVIDIA's guidance; the script refuses smaller GPUs. With smaller
GPUs, lower `N17_GLOBAL_BATCH_SIZE` and raise
`N17_GRADIENT_ACCUMULATION_STEPS` so their product stays 64.

## Return

Bring back every `checkpoint-*` directory (not only the last): offline loss
does not predict closed-loop success, so 5000/10000/15000 are all evaluated
in simulation. Also return `slurm-*.out` and `experiment_cfg/`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--output', type=Path, required=True, help='new bundle directory')
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    bundle = args.output.absolute()
    if bundle.exists():
        parser.error(f'{bundle} exists; choose a new bundle directory')
    (bundle / 'datasets').mkdir(parents=True)
    logs = bundle / 'logs'
    logs.mkdir()
    report = {'schema': 'arx-door-v2-bundle-report-v1', 'plan': str(args.plan.absolute()),
              'plan_contents': plan, 'status': 'building'}
    write_json(bundle / 'build_report.json', report)
    try:
        report['train'] = export_split(plan, 'candidate', bundle / 'datasets/train', logs)
        report['holdout'] = export_split(plan, 'holdout', bundle / 'datasets/holdout', logs)
        from audit_stage_prompter import audit
        report['stage_prompter'] = audit(bundle / 'datasets/train')
        report['problems'] = acceptance(plan, report['train'], report['holdout'],
                                        report['stage_prompter'])
        if report['problems']:
            raise RuntimeError('acceptance failed: ' + '; '.join(report['problems']))
        (bundle / 'configs').mkdir()
        (bundle / 'scripts').mkdir()
        shutil.copy2(resolve(plan['export']['modality_config']), bundle / 'configs/door_v2_n17.py')
        for name in ('train_v2_full_n17.sh', 'train_v2_full_n17.sbatch'):
            shutil.copy2(ROOT / 'scripts/door/training' / name, bundle / 'scripts' / name)
        (bundle / 'README.md').write_text(README.format(
            train_episodes=report['train']['episodes'], train_frames=report['train']['frames'],
            holdout_episodes=report['holdout']['episodes']))
        hashed = ['configs/door_v2_n17.py', 'scripts/train_v2_full_n17.sh',
                  'datasets/train/meta/info.json', 'datasets/train/meta/stats.json',
                  'datasets/train/meta/modality.json', 'datasets/train/meta/tasks.jsonl',
                  'datasets/train/meta/n17_validation.json']
        write_json(bundle / 'manifest.json', {
            'schema': 'arx-door-v2-bundle-v1', 'train': report['train'], 'holdout': report['holdout'],
            'video_keys': plan['export']['cameras'],
            'language_key': 'annotation.human.stage_description',
            'sha256': {name: sha256(bundle / name) for name in hashed}})
        report['status'] = 'ready'
        if not args.no_archive:
            report['archive'] = str(package(bundle))
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report.update(status='refused', error=str(error))
        write_json(bundle / 'build_report.json', report)
        print(f'BUNDLE REFUSED: {error}', file=sys.stderr)
        return 1
    write_json(bundle / 'build_report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'train', 'holdout')}, indent=2))
    print(f'archive: {report.get("archive", "(skipped)")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
