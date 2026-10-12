#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Build a CPU-only queue from the finite allocation and verified pilot receipts.

Pending pilots are reserved, never counted as accepted or scheduled twice.
Accepted receipts must reference completed batch, quality and visual evidence.
Failed receipts preserve their old attempt and enter a fresh output root.
This command only reads JSON and writes a new queue; it never launches Kit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.scene.workcell import sample_base_pose  # noqa: E402


def _read(path):
    return json.loads(Path(path).expanduser().read_text(encoding='utf-8'))


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _number_equal(actual, expected):
    return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9))


def _replace_flag(argv, flag, values, *, multiple=False):
    start = argv.index(flag) + 1
    end = start + 1
    if multiple:
        while end < len(argv) and not argv[end].startswith('--'):
            end += 1
    argv[start:end] = [str(value) for value in values]


def _validate_accepted(context, receipt, template, allocation):
    directory = Path(receipt['attempt_dir']).expanduser().resolve()
    result, scene = _read(directory / 'result.json'), _read(directory / 'scene.json')
    batch_path = Path(receipt['batch_report']).expanduser().resolve()
    batch = _read(batch_path)
    matches = [row for row in batch.get('trials', [])
               if Path(row.get('output_dir', '')).expanduser().resolve() == directory]
    if len(matches) != 1:
        raise ValueError('pilot has no unique completed batch trial')
    trial = matches[0]
    authoritative = trial.get('result', {})
    if (result.get('success') is not True or result.get('physics_executed') is not True
            or authoritative.get('success') is not True
            or authoritative.get('physics_executed') is not True
            or type(authoritative.get('process_returncode')) is not int
            or authoritative['process_returncode'] != 0):
        raise ValueError('pilot requires successful physical execution and verified zero process exit')
    configuration = batch.get('configuration') or {}
    if (configuration.get('render_schedule') != allocation['render_schedule']
            or (batch.get('runtime') or {}).get('profile') != allocation['runtime_profile']):
        raise ValueError('pilot runtime/render schedule differs from the allocation')
    for record in (result, trial):
        source = Path(str(record.get('source_seed', record.get('input_file', '')))).stem
        if (source != context['source_id']
                or not _number_equal(record.get('angle_deg'), context['angle_deg'])
                or record.get('visual_seed', 0) != context['visual_seed']
                or record.get('trial_seed', record.get('seed', 0)) != context['trial_seed']
                or record.get('dataset_split', record.get('split', 'candidate')) != 'candidate'):
            raise ValueError('pilot source/angle/visual/RNG/split differs from its primary context')
    if result.get('behavior_family') != context['behavior_family']:
        raise ValueError('pilot behavior family differs from its primary context')
    config = scene.get('config') or {}
    placement = scene.get('placement') or {}
    if (not _number_equal(config.get('radius_m'), context['radius_m'])
            or not _number_equal(config.get('base_height_m'), allocation['base_height_m'])
            or config.get('yaw_mode') != allocation['yaw_mode']
            or not _number_equal(placement.get('angle_deg'), context['angle_deg'])
            or scene.get('visual_seed', 0) != context['visual_seed']):
        raise ValueError('pilot scene radius/angle/yaw/visual differs from its allocated placement')
    expected = sample_base_pose(scene['handle_world_xyz'], angle_deg=context['angle_deg'], **config)
    xyz, quaternion = placement.get('position'), placement.get('quaternion_xyzw')
    if (np.shape(xyz) != (3,) or np.shape(quaternion) != (4,)
            or not np.allclose(xyz, expected.position, rtol=0, atol=1e-6)
            or not (np.allclose(quaternion, expected.quaternion_xyzw, rtol=0, atol=1e-6)
                    or np.allclose(quaternion, -np.asarray(expected.quaternion_xyzw), rtol=0, atol=1e-6))):
        raise ValueError('pilot world placement does not implement its radius and face_handle angle')
    argv = template['argv']
    expected_input = str(Path(_flag(argv, '--input-file')).resolve())
    expected_seed = str(Path(_flag(argv, '--seed-dir')).resolve())
    if (str(Path(str(trial.get('input_file', ''))).resolve()) != expected_input
            or str(Path(str(trial.get('seed_dir', ''))).resolve()) != expected_seed
            or str(Path(str(result.get('source_seed', ''))).resolve()) != expected_input):
        raise ValueError('pilot source input pair differs from the scheduled binding')
    quality = _read(receipt['quality_report'])
    records = [row for row in quality.get('records', [])
               if Path(row.get('attempt_dir', '')).expanduser().resolve() == directory]
    if len(records) != 1:
        raise ValueError('pilot has no unique quality review record')
    record = records[0]
    process = record.get('process_evidence') or {}
    binding = record.get('source_input_binding') or {}
    if (record.get('quality_usable') is not True or record.get('errors')
            or record.get('visual_variant_verified') is not True
            or record.get('source_registry_verified') is not True
            or record.get('source_id') != context['source_id']
            or record.get('split') != 'candidate'
            or not _number_equal(record.get('angle_deg'), context['angle_deg'])
            or record.get('visual_seed') != context['visual_seed']
            or process.get('exit_verified') is not True or process.get('returncode') != 0
            or binding.get('input_file') != expected_input or binding.get('seed_dir') != expected_seed
            or binding.get('content_verified') is not True
            or binding.get('derived_poses_verified') is not True
            or not isinstance(record.get('trajectory_fingerprint'), str)
            or not record['trajectory_fingerprint']):
        raise ValueError('pilot requires a complete matching usable quality/source-binding review')
    matching_bindings = [row for row in configuration.get('source_quality', [])
                         if row.get('input_file') == expected_input and row.get('seed_dir') == expected_seed]
    if (len(matching_bindings) != 1
            or matching_bindings[0].get('source_sha256') != binding.get('source_sha256')):
        raise ValueError('pilot quality source identity differs from its batch input binding')
    visual = receipt.get('visual_review') or {}
    evidence = visual.get('evidence')
    if (visual.get('confirmed') is not True or not isinstance(visual.get('notes'), str)
            or len(visual['notes'].strip()) < 12 or not isinstance(evidence, list) or not evidence
            or not all(isinstance(path, str) and Path(path).expanduser().is_file() for path in evidence)):
        raise ValueError('pilot needs actual semantic visual review notes and existing evidence images')
    return {'context_id': context['context_id'], 'attempt_dir': str(directory),
            'batch_report': str(batch_path), 'quality_report': str(Path(receipt['quality_report']).resolve()),
            'trajectory_fingerprint': record['trajectory_fingerprint'], 'visual_review': visual}


def build_queue(allocation, *, queue_id, resolutions=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', queue_id):
        raise ValueError('queue_id must be a short letters/numbers/underscore/hyphen identifier')
    contexts = allocation['candidate_contexts']
    if len({row['context_id'] for row in contexts}) != len(contexts):
        raise ValueError('allocation contains repeated context IDs')
    resolutions = resolutions or {}
    unknown = set(resolutions) - {row['context_id'] for row in contexts}
    if unknown:
        raise ValueError(f'unknown pilot resolution contexts: {sorted(unknown)}')
    templates = {(row['source_id'], row['radius_m'], row['visual_seed']): row
                 for row in allocation['candidate_batch_commands']}
    report = {'schema': 'arx-door-augmentation-queue-v1', 'queue_id': queue_id,
              'runtime_profile': allocation['runtime_profile'], 'render_schedule': allocation['render_schedule'],
              'simulation_launched': False, 'candidate_target': allocation['candidate_count_target'],
              'claimed': [], 'pilot_pending': [], 'failed_attempts_preserved': [], 'batches': []}
    queued = defaultdict(list)
    for context in contexts:
        key = (context['source_id'], context['radius_m'], context['visual_seed'])
        if key not in templates:
            raise ValueError(f'no command template for context {context["context_id"]}')
        receipt = resolutions.get(context['context_id'], {})
        status = receipt.get('status', 'pending' if context.get('physical_status') == 'pilot_pending' else 'queued')
        if status == 'accepted':
            report['claimed'].append(_validate_accepted(context, receipt, templates[key], allocation))
        elif status == 'pending':
            report['pilot_pending'].append(copy.deepcopy(context))
        elif status in ('failed', 'queued'):
            if status == 'failed':
                directory = Path(receipt['attempt_dir']).expanduser().resolve()
                if not directory.is_dir() or not receipt.get('reason'):
                    raise ValueError('failed pilot needs an existing preserved attempt and reason')
                report['failed_attempts_preserved'].append({'context_id': context['context_id'],
                                                          'attempt_dir': str(directory), 'reason': receipt['reason']})
            queued[key].append(context)
        else:
            raise ValueError(f'unsupported pilot resolution status: {status}')
    for key, rows in queued.items():
        template = templates[key]
        source, radius, visual = key
        original = Path(template['output_root'])
        output = original.parent / f'{source}_r{radius:.3f}_v{visual}_{queue_id}'
        retry = 1
        while output.exists():
            output = original.parent / f'{source}_r{radius:.3f}_v{visual}_{queue_id}_fresh{retry:02d}'
            retry += 1
        argv = copy.deepcopy(template['argv'])
        _replace_flag(argv, '--angles', [row['angle_deg'] for row in rows], multiple=True)
        _replace_flag(argv, '--output-root', [output])
        report['batches'].append({'source_id': source, 'radius_m': radius, 'visual_seed': visual,
                                 'context_ids': [row['context_id'] for row in rows],
                                 'expected_attempts': len(rows), 'output_root': str(output), 'argv': argv,
                                 'launch_gate': template['launch_gate']})
    fingerprints = [row['trajectory_fingerprint'] for row in report['claimed']]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('claimed pilots repeat a numerical trajectory; only one unique trajectory may count')
    report.update(queued_contexts=sum(row['expected_attempts'] for row in report['batches']),
                  pilot_pending_contexts=len(report['pilot_pending']),
                  claimed_unique_trajectories=len(fingerprints))
    if report['queued_contexts'] + report['pilot_pending_contexts'] + len(report['claimed']) != len(contexts):
        raise ValueError('queue lost or duplicated a context')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--queue-id', required=True)
    parser.add_argument('--pilot-resolutions', type=Path,
                        help='JSON {context_id: {status, attempt_dir, batch_report, quality_report, visual_review}}')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_queue(_read(args.allocation), queue_id=args.queue_id,
                             resolutions=_read(args.pilot_resolutions) if args.pilot_resolutions else None)
        report['allocation'] = str(args.allocation.expanduser().resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ('queued_contexts', 'pilot_pending_contexts', 'claimed_unique_trajectories')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
