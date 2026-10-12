#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run preassigned holdouts serially after the candidate campaign releases the GPU."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from arx_door import paths  # noqa: E402

import h5py  # noqa: E402
from arx_door.contract import EDGE_REGRASP, KNOB_HOLD_OPEN  # noqa: E402
import run_door_augmentation_campaign as candidate  # noqa: E402
from run_door_augmentation_campaign import (  # noqa: E402
    next_batch_argv, one, quality, read_json, runtime_provenance, snapshot,
    trial_plan, validate_batch, validate_runtime, values, write_json,
)


def resolved(value):
    return str((ROOT / value).resolve())


def file_stamp(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def source_evidence(batch, registry):
    """Check existing input-boundary identities; the batch binds actual arrays before Kit."""
    directory, source = Path(batch['seed_dir']), Path(batch['input_file'])
    metadata = read_json(directory / 'episode.json')
    review = read_json(directory / 'agent_review.json')
    identity = batch['source_recording_identity_sha256']
    if (not isinstance(identity, str) or len(identity) != 64
            or any(char not in '0123456789abcdef' for char in identity)
            or registry.get('source_recording_sha256') != identity
            or metadata.get('success') is not True
            or metadata.get('quality_status') != 'agent_reviewed_seed'
            or metadata.get('human_review_status') != 'pending'
            or metadata.get('runtime_profile') != batch['runtime_profile']
            or validate_runtime(metadata['runtime']) != batch['runtime_profile']
            or review.get('schema') != 'arx-door-agent-seed-review-v1'
            or review.get('decision') != 'agent_reviewed_seed'
            or review.get('human_review_status') != 'pending'
            or review.get('trajectory', {}).get('sha256') != identity
            or review.get('physical_evidence', {}).get('passed') is not True
            or review.get('visual_review', {}).get('confirmed') is not True):
        raise ValueError('holdout source lacks matching recorded identity and agent review')
    with h5py.File(source, 'r') as data:
        if list(data['data']) != ['demo_0']:
            raise ValueError('holdout source must contain exactly one bound demonstration')
        demo = data['data/demo_0']
        if (demo.attrs.get('source_sha256') != identity
                or demo.attrs.get('source_quality_status') != 'agent_reviewed_seed'
                or json.loads(demo.attrs.get('source_agent_review', '{}')) != review
                or demo.attrs.get('num_samples') != metadata.get('frames')
                or demo.attrs.get('num_samples') != review['trajectory'].get('frames')):
            raise ValueError('holdout HDF5 and source review identify different recordings')
    cpu = read_json(batch['cpu_report'])
    if (cpu.get('source_own_trajectory_used') is not True or cpu.get('proxy_source_used') is not False
            or resolved(cpu.get('source', '')) != str(directory.resolve())
            or cpu.get('radius_m') != batch['radius_m'] or cpu.get('angles_deg') != batch['angles_deg']
            or len(cpu.get('contexts', [])) != len(batch['angles_deg'])
            or any(row.get('all_required_channels_strict_ik') is not True
                   or row.get('all_required_channels_margin_002') is not True for row in cpu['contexts'])):
        raise ValueError('holdout CPU evidence differs from its own preassigned contexts')
    paths = [source, directory / 'trajectory.npz', directory / 'episode.json',
             directory / 'agent_review.json', Path(batch['source_scene']), Path(batch['cpu_report'])]
    if batch.get('numeric_equality_report'):
        paths.append(Path(batch['numeric_equality_report']))
    return {str(path.resolve()): file_stamp(path) for path in paths}


def validate_queue(queue, plan):
    if (queue.get('schema') != 'arx-door-fresh-holdout20-queue-v1'
            or queue.get('all_sources_ready') is not True
            or queue.get('expected_attempts') != 20
            or queue.get('ready_expected_attempts') != 20
            or queue.get('render_schedule') != 'pre_action'
            or queue.get('runtime_profile') != plan['current_runtime_profile']):
        raise ValueError('holdout queue is not a ready preassigned 20-trial campaign')
    roots, contexts, identities, groups, aliases, stamps = [], [], set(), set(), {}, {}
    for batch in queue['batches']:
        argv = batch['argv']
        if '--resume' in argv or '--max-trials' in argv:
            raise ValueError('holdout queue must retain the original complete batch plan')
        expected = trial_plan(batch)
        if (len(expected) != batch['expected_attempts']
                or batch.get('split') != 'holdout'
                or batch.get('all_source_launch_gates_passed') is not True
                or batch.get('source_review_passed') is not True
                or batch.get('own_source_cpu_screen_passed') is not True
                or argv[:3] != ['env', f'ISAAC_SIM_PYTHON={ROOT}/generated/runtime/{queue["runtime_profile"]}/.venv/bin/python',
                                str(paths.SKILLGEN_BATCH_LAUNCHER)]
                or argv.count('--allow-agent-reviewed-seeds') != 1
                or one(argv, '--split') != 'holdout'
                or one(argv, '--render-schedule') != queue['render_schedule']
                or resolved(one(argv, '--output-root')) != batch['output_root']
                or float(one(argv, '--radius-m')) != batch['radius_m']
                or [float(value) for value in values(argv, '--angles')] != batch['angles_deg']
                or any(item['source_id'] != batch['source_id'] or item['visual_seed'] != batch['visual_seed']
                       or item['seed'] != batch['trial_seed'] for item in expected)):
            raise ValueError('holdout queue argv disagrees with its preassigned contexts')
        for field, option in (('source_scene', '--source-scene'), ('seed_dir', '--seed-dir'), ('input_file', '--input-file')):
            if resolved(one(argv, option)) != batch[field]:
                raise ValueError(f'holdout argv changed {field}')
        registry = plan['sources'].get(batch['source_id'], {})
        canonical = batch['canonical_source_id']
        if (registry.get('assigned_split') != 'holdout'
                or registry.get('reviewed_distinct_behavior') is not True
                or registry.get('independence_group') != batch['independence_group']
                or registry.get('behavior_group') != batch['behavior_group']
                or registry.get('canonical_source_id', batch['source_id']) != canonical
                or resolved(registry.get('seed_file', '')) != batch['input_file']
                or resolved(registry.get('source_recording_dir', '')) != batch['seed_dir']
                or plan['sources'].get(canonical, {}).get('assigned_split') != 'holdout'):
            raise ValueError('holdout source differs from its preassigned registry')
        if canonical != batch['source_id']:
            aliases[batch['source_id']] = canonical
            duplicate = read_json(batch['numeric_equality_report'])
            if (duplicate.get('all_fields_exact_equal') is not True
                    or resolved(duplicate.get('new_source_npz', '')) != resolved(batch['seed_dir'] + '/trajectory.npz')
                    or resolved(duplicate.get('old_source_npz', '')) != resolved(
                        plan['sources'][canonical]['source_recording_dir'] + '/trajectory.npz')
                    or registry.get('source_recording_sha256') != plan['sources'][canonical].get('source_recording_sha256')):
                raise ValueError('holdout source alias lacks canonical numerical identity evidence')
        stamps[batch['source_id']] = source_evidence({**batch, 'runtime_profile': queue['runtime_profile']}, registry)
        roots.append(batch['output_root'])
        contexts.extend((canonical, batch['radius_m'], item['angle_deg']) for item in expected)
        identities.add(batch['source_recording_identity_sha256']); groups.add(batch['independence_group'])
    if (len(set(roots)) != len(roots) or len(set(contexts)) != len(contexts)
            or sum(batch['expected_attempts'] for batch in queue['batches']) != 20
            or len(identities) != queue['numerical_source_recording_identity_count']
            or sorted(groups) != queue['canonical_holdout_independence_groups']
            or len(groups) != queue['canonical_holdout_independence_group_count']
            or aliases != queue['source_aliases']):
        raise ValueError('holdout contexts or canonical source counts disagree with the complete queue')
    return stamps


def check_cached_record(cached, entry, plan):
    record = cached['record']
    if cached['snapshot'] != snapshot(entry) or cached.get('quality_policy') != plan['quality_gates']:
        raise ValueError('cached review parent/config/artifacts or quality policy changed')
    result = cached['snapshot']['disk_result'] or entry['trial']['result']
    registry = plan['sources'].get(record.get('source_id'), {})
    process, errors = quality._process_evidence(Path(record['attempt_dir']), result, entry)
    binding = (quality._source_recording_identity(entry, result)
               if result.get('source_seed') else None)
    if (registry.get('assigned_split') != record.get('split')
            or registry.get('reviewed_distinct_behavior') is not True
            or registry.get('independence_group') != record.get('independence_group')
            or registry.get('behavior_group') != record.get('behavior_group')
            or resolved(registry.get('seed_file', '')) != entry['trial']['input_file']
            or (record.get('process_evidence') is not None and record['process_evidence'] != process)
            or (record.get('source_input_binding') is not None and record['source_input_binding'] != binding)
            or (record.get('source_recording_identity') is not None
                and (binding is None or record['source_recording_identity'] != binding['source_sha256']))
            or (record.get('quality_usable') and (errors or binding is None
                                                or record.get('source_recording_identity') != binding['source_sha256']
                                                or record.get('runtime_profile') != plan['current_runtime_profile']
                                                or record.get('source_registry_verified') is not True))):
        raise ValueError('cached review no longer matches authoritative source/split/process evidence')


def candidate_records(root, plan, *, require_complete=False):
    """Read authoritative cached candidate reviews without hashing or decoding again."""
    root = Path(root).resolve()
    queue, original_plan = read_json(root / 'queue.json'), read_json(root / 'plan.json')
    candidate.validate_queue(queue, plan)
    if candidate.plan_identity(plan, queue) != candidate.plan_identity(original_plan, queue):
        raise ValueError('candidate registry/configuration changed from its cached campaign')
    cache = read_json(root / 'reviews_cache.json')
    if (len(set(cache['review_order'])) != len(cache['review_order'])
            or set(cache['review_order']) != set(cache['entries'])):
        raise ValueError('candidate cache order/entries disagree')
    allowed = {item['output_dir'] for batch in queue['batches'] for item in trial_plan(batch)}
    allowed.update(resolved(item['attempt_dir']) for item in queue.get('claimed', []))
    records = []
    for key in cache['review_order']:
        cached = cache['entries'][key]
        if key not in allowed or cached['record'].get('attempt_dir') != key:
            raise ValueError('candidate cache includes an unplanned trial')
        entry = quality.batch_review_evidence(cached['snapshot']['evidence']['batch_path'])[key]
        check_cached_record(cached, entry, plan)
        if cached['record'].get('split') != 'candidate':
            raise ValueError('candidate cache contains another split')
        records.append(deepcopy(cached['record']))
    summary = quality.summarize_reviews(deepcopy(records), plan)
    progress = read_json(root / 'progress.json')
    if require_complete:
        published = read_json(root / 'campaign_quality.json')
        if (progress.get('status') != 'complete' or progress.get('active') is not None
                or progress.get('queued_remaining') != 0
                or summary['candidate_unique_trajectories'] < 100
                or summary['current_profile_candidate_episodes'] < 100
                or summary['candidate_unique_trajectories'] != progress.get('candidate_unique_trajectories')
                or any(published.get(key) != value for key, value in summary.items())
                or type(progress.get('pid')) is not int or progress['pid'] <= 0
                or Path(f'/proc/{progress["pid"]}').exists()):
            raise ValueError('candidate100 must be complete, fully reviewed, inactive, and its process exited before holdout launch')
    return records


class HoldoutCampaign(candidate.Campaign):
    """Separate heldout cache; shared serial launch and owned-process cleanup only."""

    def __init__(self, queue_path, output_root, plan_path, candidate_root, *, resume=False):
        self.root = Path(output_root).resolve()
        self.queue_path, self.plan_path = Path(queue_path).resolve(), Path(plan_path).resolve()
        self.candidate_root = Path(candidate_root).resolve()
        current_queue, current_plan = read_json(self.queue_path), read_json(self.plan_path)
        if resolved(current_queue['source_split_registry']) != str(self.plan_path):
            raise ValueError('holdout queue names a different split registry')
        current_sources = validate_queue(current_queue, current_plan)
        if validate_runtime(runtime_provenance()) != current_queue['runtime_profile']:
            raise ValueError('holdout Python differs from the approved runtime profile')
        self.candidates = candidate_records(self.candidate_root, current_plan)
        candidate_groups = {row['independence_group'] for row in self.candidates}
        candidate_identities = {row.get('source_recording_identity') for row in self.candidates}
        if any(batch['independence_group'] in candidate_groups
               or batch['source_recording_identity_sha256'] in candidate_identities for batch in current_queue['batches']):
            raise ValueError('preassigned holdout source/group leaks into the candidate campaign')
        for batch in current_queue['batches']:
            next_batch_argv(batch, current_queue)
        if resume:
            self.queue, self.plan = read_json(self.root / 'queue.json'), read_json(self.root / 'plan.json')
            self.cache = read_json(self.root / 'reviews_cache.json')
            if (current_queue != self.queue or current_plan != self.plan
                    or read_json(self.root / 'source_evidence.json') != current_sources
                    or read_json(self.root / 'configuration.json') != self.configuration()):
                raise ValueError('holdout original queue/plan/configuration/source evidence changed')
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            self.queue, self.plan = current_queue, current_plan
            self.cache = {'schema': 'arx-door-holdout-review-cache-v1', 'review_order': [], 'entries': {}}
            for name, value in (('queue.json', self.queue), ('plan.json', self.plan),
                                ('source_evidence.json', current_sources), ('configuration.json', self.configuration()),
                                ('reviews_cache.json', self.cache)):
                write_json(self.root / name, value)
        self.active = None
        allowed = {item['output_dir'] for batch in self.queue['batches'] for item in trial_plan(batch)}
        if (len(set(self.cache['review_order'])) != len(self.cache['review_order'])
                or set(self.cache['review_order']) != set(self.cache['entries'])):
            raise ValueError('holdout cache order/entries disagree')
        for key, cached in self.cache['entries'].items():
            if key not in allowed or cached['record'].get('attempt_dir') != key:
                raise ValueError('holdout cache includes an unplanned trial')
            entry = quality.batch_review_evidence(cached['snapshot']['evidence']['batch_path'])[key]
            check_cached_record(cached, entry, self.plan)
            self.check_source_binding(entry)
        self.refresh('prepared')

    def configuration(self):
        return {'candidate_campaign': str(self.candidate_root), 'queue_path': str(self.queue_path),
                'plan_path': str(self.plan_path), 'failure_limit': 5, 'expected_attempts': 20}

    def check_source_binding(self, entry):
        row = entry['trial']
        batch = next(batch for batch in self.queue['batches'] if row['output_dir'] in
                     {item['output_dir'] for item in trial_plan(batch)})
        # A preparation/process failure may have no result.json. Validate the
        # input binding against the planned pair without inventing task success.
        source_result = {'source_seed': batch['input_file'],
                         'behavior_family': (KNOB_HOLD_OPEN if batch['behavior_group'] == KNOB_HOLD_OPEN
                                             else EDGE_REGRASP)}
        binding = quality._source_recording_identity(entry, source_result)
        if binding is None or binding['source_sha256'] != batch['source_recording_identity_sha256']:
            raise ValueError('holdout authoritative input binding differs from its preassigned source identity')

    def refresh(self, status, *, error=None):
        records = [deepcopy(self.cache['entries'][key]['record']) for key in self.cache['review_order']]
        combined = quality.summarize_reviews([*deepcopy(self.candidates), *records], self.plan)
        combined['candidate_campaign'] = str(self.candidate_root)
        combined['holdout_campaign_queue'] = str(self.queue_path)
        write_json(self.root / 'combined_quality.json', combined)
        held = {key: deepcopy(value) for key, value in combined.items() if key != 'records'}
        held['schema'] = 'arx-door-holdout-campaign-quality-v1'
        held['records'] = [row for row in combined['records'] if row.get('split') == 'holdout']
        held['attempts_reviewed'] = len(held['records'])
        held['usable_unique_episodes'] = combined['holdout_episodes']
        held['usable_unique_trajectories'] = combined['holdout_unique_trajectories']
        held['usable_frames'] = sum(row['frames'] for row in held['records'] if row['quality_usable'])
        held['candidate_records_reference'] = str(self.candidate_root / 'reviews_cache.json')
        held['candidate_records_rehashed_or_decoded'] = False
        write_json(self.root / 'heldout_quality.json', held)
        stage, consecutive = self._failure_streak(held['records'])
        self.progress = {'schema': 'arx-door-holdout-campaign-progress-v1',
            'updated_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'status': status,
            'candidate_campaign': str(self.candidate_root), 'expected_attempts': 20,
            'reviewed_attempts': len(self.cache['entries']), 'remaining_attempts': 20 - len(self.cache['entries']),
            'holdout_episodes': combined['holdout_episodes'],
            'independent_holdout_unique_trajectories': combined['independent_holdout_unique_trajectories'],
            'holdout_source_recording_count': combined['holdout_source_recording_count'],
            'holdout_independence_groups': combined['holdout_independence_groups'],
            'consecutive_failure_stage': stage, 'consecutive_failure_count': consecutive,
            'active': self.active, 'failed_quality_attempts': [row['attempt_dir'] for row in held['records']
                                                            if not row['quality_usable']],
            'human_review_status': 'pending', 'generalization_verified': False, 'error': error}
        write_json(self.root / 'progress.json', self.progress)
        return self.progress

    def review_batch(self, index, batch):
        path = Path(batch['output_root']) / 'batch_results.json'
        report = read_json(path)
        validate_batch(batch, report, self.queue)
        for key, entry in quality.batch_review_evidence(path).items():
            self.check_source_binding(entry)
            cached = self.cache['entries'].get(key)
            if cached:
                check_cached_record(cached, entry, self.plan)
                continue
            thumbnail = self.root / 'phase_images' / f'batch_{index:03d}' / f"trial_{entry['trial']['index']:03d}"
            record = quality.review_attempt(key, self.plan, thumbnail_dir=thumbnail, batch_trial=entry)
            # Keep failed attempts identifiable even when no result was written.
            row, registry = entry['trial'], self.plan['sources'][entry['trial']['source_id']]
            for field, value in {'source_id': row['source_id'], 'split': row['split'], 'angle_deg': row['angle_deg'],
                                 'independence_group': registry['independence_group'],
                                 'behavior_group': registry['behavior_group'], 'frames': 0}.items():
                record.setdefault(field, value)
            self.save_review(record, entry)
            self.refresh('reviewing')
        self.refresh('batch_reviewed')
        return len(report['trials']) == batch['expected_attempts']

    def run(self):
        self.candidates = candidate_records(self.candidate_root, self.plan, require_complete=True)
        self.refresh('ready_to_run')
        if self.progress['consecutive_failure_count'] >= 5:
            self.refresh('stopped_failure_streak')
            return 1
        for index, batch in enumerate(self.queue['batches']):
            # Review already reported rows first; launch at most one remaining
            # trial from the complete immutable batch plan, then review it once.
            report_path = Path(batch['output_root']) / 'batch_results.json'
            if report_path.is_file():
                self.review_batch(index, batch)
            while (argv := next_batch_argv(batch, self.queue)) is not None:
                if self.progress['consecutive_failure_count'] >= 5:
                    self.refresh('stopped_failure_streak')
                    return 1
                current_queue, current_plan = read_json(self.queue_path), read_json(self.plan_path)
                if (current_queue != self.queue or current_plan != self.plan
                        or validate_queue(current_queue, current_plan) != read_json(self.root / 'source_evidence.json')):
                    raise ValueError('holdout source evidence changed before launch')
                self.candidates = candidate_records(self.candidate_root, self.plan, require_complete=True)
                prior = len(read_json(report_path)['trials']) if report_path.is_file() else 0
                argv = [*argv, '--max-trials', '1']
                returncode = self.run_batch(index, batch, argv)
                write_json(self.root / f'batch_{index:03d}_launch_{prior:03d}_process.json', {'argv': argv, 'returncode': returncode})
                if not report_path.is_file() or len(read_json(report_path)['trials']) <= prior:
                    raise RuntimeError(f'holdout batch {index} made no reported progress; attempts preserved')
                self.review_batch(index, batch)
            if self.progress['consecutive_failure_count'] >= 5:
                self.refresh('stopped_failure_streak')
                return 1
        complete = self.progress['independent_holdout_unique_trajectories'] >= 20
        self.refresh('complete' if complete else 'complete_with_shortfall')
        return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/door_augmentation_plan.json')
    parser.add_argument('--candidate-campaign', type=Path, required=True,
                        help='read-only candidate cache; actual launch requires complete100 and its process exited')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='validate/cache only; never launch a simulator')
    args = parser.parse_args()
    worker = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        worker = HoldoutCampaign(args.queue, args.output_root, args.plan, args.candidate_campaign, resume=args.resume)
        return 0 if args.prepare_only else worker.run()
    except KeyboardInterrupt:
        if worker:
            worker.refresh('interrupted', error='interrupted; own batch and simulator groups cleaned')
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        if worker:
            worker.refresh('blocked', error=str(error))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
