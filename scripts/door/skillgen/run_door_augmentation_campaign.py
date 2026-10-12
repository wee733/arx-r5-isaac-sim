#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run an approved candidate queue serially and review each completed trial once."""

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
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
from arx_door import quality  # noqa: E402
from arx_door.skillgen.batch import build_trial_specs, trial_manifest  # noqa: E402
from arx_door.files import read_json, write_json  # noqa: E402
from arx_door.runtime import runtime_provenance, validate_runtime  # noqa: E402


def values(argv, option):
    if argv.count(option) != 1:
        raise ValueError(f'queue must declare {option} exactly once')
    start = argv.index(option) + 1
    end = next((index for index in range(start, len(argv)) if argv[index].startswith('--')), len(argv))
    return argv[start:end]


def one(argv, option):
    result = values(argv, option)
    if len(result) != 1:
        raise ValueError(f'queue must declare one value for {option}')
    return result[0]


def trial_plan(batch):
    argv = batch['argv']
    specs = build_trial_specs(
        [float(value) for value in values(argv, '--angles')], batch['output_root'], seed=0,
        input_files=values(argv, '--input-file'), seed_dirs=values(argv, '--seed-dir'),
        trial_seeds=[int(value) for value in values(argv, '--trial-seeds')],
        visual_seeds=[int(value) for value in values(argv, '--visual-seeds')], split=one(argv, '--split'))
    return [trial_manifest(spec) for spec in specs]


def claimed_source_id(claim, plan):
    """Bind a claim to its own small result, including pre-source_id queues."""
    result = read_json(Path(claim['attempt_dir']).resolve() / 'result.json')
    seed_file = result.get('source_seed') if isinstance(result, dict) else None
    if (not isinstance(seed_file, str) or not seed_file.strip()
            or not Path(seed_file).is_absolute()
            or result.get('success') is not True
            or result.get('dataset_split', 'candidate') != 'candidate'):
        raise ValueError('claimed trial lacks a successful candidate source identity')
    source_id = Path(seed_file).stem
    if 'source_id' in claim and claim['source_id'] != source_id:
        raise ValueError('claimed source_id differs from its recorded source_seed')
    registry = plan['sources'].get(source_id, {})
    if (registry.get('assigned_split') != 'candidate'
            or registry.get('reviewed_distinct_behavior') is not True
            or not registry.get('independence_group')
            or (ROOT / registry.get('seed_file', '')).resolve() != Path(seed_file).resolve()):
        raise ValueError('claimed source is not bound to its preassigned source registry')
    return source_id


def plan_identity(plan, queue):
    source_ids = {batch['source_id'] for batch in queue['batches']}
    source_ids.update(claimed_source_id(claim, plan) for claim in queue.get('claimed', []))
    # Holdout recordings may advance independently while this candidate job runs.
    return {**{key: value for key, value in plan.items() if key not in ('sources', 'phases')},
            'sources': {key: plan['sources'][key] for key in sorted(source_ids)}}


def validate_queue(queue, plan):
    if (queue.get('schema') != 'arx-door-augmentation-queue-v1'
            or queue.get('pilot_pending') or queue.get('render_schedule') != 'pre_action'
            or queue.get('runtime_profile') != plan['current_runtime_profile']):
        raise ValueError('queue is not a ready, current-profile candidate campaign')
    contexts = [item['context_id'] for item in queue.get('claimed', [])]
    for claim in queue.get('claimed', []):
        claimed_source_id(claim, plan)
    roots = []
    for batch in queue['batches']:
        argv = batch['argv']
        if '--resume' in argv or '--max-trials' in argv:
            raise ValueError('queue argv must contain the original complete batch plan')
        expected = trial_plan(batch)
        if len(expected) != batch['expected_attempts'] or len(expected) != len(batch['context_ids']):
            raise ValueError('queue batch context count differs from its actual trial plan')
        if (str(Path(one(argv, '--output-root')).resolve()) != batch['output_root']
                or one(argv, '--split') != 'candidate'
                or one(argv, '--render-schedule') != queue['render_schedule']
                or float(one(argv, '--radius-m')) != batch['radius_m']
                or any(item['source_id'] != batch['source_id'] for item in expected)
                or any(item['visual_seed'] != batch['visual_seed'] for item in expected)):
            raise ValueError('queue argv disagrees with its declared candidate context')
        registry = plan['sources'].get(batch['source_id'], {})
        registry_file = ROOT / registry.get('seed_file', '')
        if (registry.get('assigned_split') != 'candidate'
                or registry.get('reviewed_distinct_behavior') is not True
                or not registry.get('independence_group')
                or registry_file.resolve() != Path(one(argv, '--input-file')).resolve()):
            raise ValueError('candidate source is not bound to its preassigned source registry')
        roots.append(batch['output_root'])
        contexts.extend(batch['context_ids'])
    if len(set(roots)) != len(roots) or len(set(contexts)) != len(contexts):
        raise ValueError('queue contains repeated output roots or context identities')
    if sum(batch['expected_attempts'] for batch in queue['batches']) != queue['queued_contexts']:
        raise ValueError('queue attempt total differs from its batch plans')


def validate_batch(batch, report, queue):
    if (report.get('output_root') != batch['output_root']
            or report.get('plan') != trial_plan(batch)
            or report.get('runtime', {}).get('profile') != queue['runtime_profile']):
        raise ValueError('authoritative batch plan/runtime does not match the campaign queue')
    if validate_runtime(report['runtime']) != queue['runtime_profile']:
        raise ValueError('authoritative batch numerical/runtime identity changed')
    configuration = report.get('configuration', {})
    for field, option in (('source_scene', '--source-scene'), ('urdf', '--urdf')):
        if configuration.get(field) != str(Path(one(batch['argv'], option)).resolve()):
            raise ValueError(f'authoritative batch changed {field}')
    if (configuration.get('radius_m') != batch['radius_m']
            or configuration.get('render_schedule') != queue['render_schedule']
            or configuration.get('timeout_s') != float(one(batch['argv'], '--timeout-s'))):
        raise ValueError('authoritative batch changed placement/render/planning configuration')
    expected = {item['output_dir']: item for item in trial_plan(batch)}
    seen = set()
    for row in report['trials']:
        directory = row['output_dir']
        if directory in seen or directory not in expected:
            raise ValueError('authoritative batch contains an unexpected or repeated trial')
        if any(row.get(key) != value for key, value in expected[directory].items()):
            raise ValueError('authoritative trial identity changed from the campaign plan')
        seen.add(directory)


def next_batch_argv(batch, queue):
    report_path = Path(batch['output_root']) / 'batch_results.json'
    if report_path.exists():
        report = read_json(report_path)
        validate_batch(batch, report, queue)
        return None if len(report['trials']) == batch['expected_attempts'] else [*batch['argv'], '--resume']
    if Path(batch['output_root']).exists():
        raise ValueError('unreported existing batch directory is preserved; choose a new approved root')
    return list(batch['argv'])


def snapshot(entry):
    directory = Path(entry['trial']['output_dir'])
    batch = read_json(entry['batch_path'])
    artifacts = {}
    for filename in ('generated.hdf5', 'wrist.mp4', 'overview.mp4'):
        path = directory / filename
        stat = path.stat() if path.is_file() else None
        artifacts[filename] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} if stat else None
    return {'evidence': entry, 'batch_configuration': batch.get('configuration'),
            'batch_runtime': batch.get('runtime'), 'artifacts': artifacts,
            'disk_result': read_json(directory / 'result.json') if (directory / 'result.json').is_file() else None,
            'scene_manifest': read_json(directory / 'scene.json') if (directory / 'scene.json').is_file() else None}


def validate_imported_review(record, entry, review_path, plan):
    directory = Path(record['attempt_dir']).resolve()
    reviewed_at = Path(review_path).stat().st_mtime_ns
    for filename in ('result.json', 'scene.json', 'generated.hdf5', 'wrist.mp4', 'overview.mp4'):
        if (directory / filename).stat().st_mtime_ns > reviewed_at:
            raise ValueError(f'artifact changed after its cached quality review: {directory / filename}')
    result = read_json(directory / 'result.json')
    registry = plan['sources'].get(record.get('source_id'), {})
    if (registry.get('assigned_split') != record.get('split')
            or registry.get('reviewed_distinct_behavior') is not True
            or registry.get('independence_group') != record.get('independence_group')
            or registry.get('behavior_group') != record.get('behavior_group')
            or (ROOT / registry.get('seed_file', '')).resolve() != Path(str(result.get('source_seed', ''))).resolve()):
        raise ValueError('cached review does not match the current candidate source registry')
    if validate_runtime(result['runtime']) != plan['current_runtime_profile']:
        raise ValueError('cached review numerical/runtime identity changed')
    process, errors = quality._process_evidence(directory, result, entry)
    binding = quality._source_recording_identity(entry, result)
    if (errors or process != record.get('process_evidence') or binding is None
            or binding != record.get('source_input_binding')
            or binding['source_sha256'] != record.get('source_recording_identity')
            or record.get('videos_verified') is not True or record.get('quality_usable') is not True
            or record.get('source_registry_verified') is not True
            or result.get('runtime', {}).get('profile') != record.get('runtime_profile')):
        raise ValueError(f'cached review lacks matching authoritative evidence: {directory}')
    for key, result_key, default in (('angle_deg', 'angle_deg', None), ('split', 'dataset_split', 'candidate'),
                                     ('trial_seed', 'trial_seed', 0), ('visual_seed', 'visual_seed', 0),
                                     ('frames', 'frames', None), ('source_seed', 'source_seed', None)):
        if record.get(key) != result.get(result_key, default):
            raise ValueError(f'cached review identity differs: {key}')
    appearance = quality.appearance_summary(directory, result)
    if any(record.get(key) != appearance[key] for key in ('appearance_signature', 'visual_seed')):
        raise ValueError('cached review appearance changed')
    with h5py.File(directory / 'generated.hdf5', 'r') as data:
        demo = data['data/demo_0']
        if (demo['actions'].shape != (record['frames'], 7)
                or demo['obs/joint_pos'].shape != demo['actions'].shape
                or result.get('hdf5_frames') != record['frames']):
            raise ValueError('cached review stored frame/shape evidence changed')
        # Older reports omit their gate policy. Recompute only CPU physical/motion
        # summaries once on import, without a fingerprint hash or video decode.
        limits = plan['quality_gates']
        physical, _ = quality.physical_summary(demo, record['frames'], limits, behavior_family=record['behavior_family'])
        motion = quality.motion_summary(demo['actions'][:], demo['obs/joint_pos'][:], limits)
        if physical != record['physical'] or motion != record['motion']:
            raise ValueError('cached review summaries differ under the current quality policy')
    errors = quality.gate_errors(physical, motion, limits, record['behavior_family'])
    for key, video in record['videos'].items():
        errors.extend(quality.video_gate_errors(key, video, record['frames'], limits))
    if errors:
        raise ValueError('cached review no longer passes the current quality gates')


def observe_groups(root_pid, groups):
    """Remember only descendants of our own batch, including separate Sim sessions."""
    processes = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(directory.name)] = (int(fields[1]), int(fields[2]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    while True:
        added = {pid for pid, (parent, _, _) in processes.items() if parent in descendants}
        if added <= descendants:
            break
        descendants.update(added)
    for pid in descendants:
        if pid in processes:
            _, group, started = processes[pid]
            groups.setdefault(group, {})[pid] = started


def kill_owned_groups(groups):
    for group, members in groups.items():
        if group == os.getpgrp():
            continue
        for pid, started in members.items():
            try:
                fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                if fields[19] == started and int(fields[2]) == group:
                    os.killpg(group, signal.SIGKILL)
                    break
            except (OSError, ValueError, IndexError):
                continue


class Campaign:
    def __init__(self, queue_path, output_root, plan_path, *, resume=False, reuse_reviews=()):
        self.root = Path(output_root).resolve()
        self.queue_path, self.plan_path = Path(queue_path).resolve(), Path(plan_path).resolve()
        current_queue, current_plan = read_json(self.queue_path), read_json(self.plan_path)
        validate_queue(current_queue, current_plan)
        if validate_runtime(runtime_provenance()) != current_queue['runtime_profile']:
            raise ValueError('campaign Python differs from the approved runtime profile')
        if resume:
            self.queue, self.plan = read_json(self.root / 'queue.json'), read_json(self.root / 'plan.json')
            if (current_queue != self.queue
                    or plan_identity(current_plan, self.queue) != plan_identity(self.plan, self.queue)):
                raise ValueError('campaign queue/candidate registry/quality gates changed')
            self.cache = read_json(self.root / 'reviews_cache.json')
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            self.queue, self.plan = current_queue, current_plan
            self.cache = {'schema': 'arx-door-campaign-review-cache-v1', 'review_order': [], 'entries': {}}
            write_json(self.root / 'queue.json', self.queue)
            write_json(self.root / 'plan.json', self.plan)
            write_json(self.root / 'reviews_cache.json', self.cache)
        self.active = None
        for batch in self.queue['batches']:
            next_batch_argv(batch, self.queue)  # validates existing batch0 before any launch
        allowed = {item['output_dir'] for batch in self.queue['batches'] for item in trial_plan(batch)}
        claims = {str(Path(item['attempt_dir']).resolve()): item for item in self.queue.get('claimed', [])}
        allowed.update(claims)
        for key, cached in self.cache['entries'].items():
            if key not in allowed:
                raise ValueError('campaign cache includes an unplanned trial')
            entry = quality.batch_review_evidence(cached['snapshot']['evidence']['batch_path'])[key]
            if cached['snapshot'] != snapshot(entry):
                raise ValueError(f'cached trial parent/config/artifacts changed: {key}')
            if cached.get('quality_policy') != self.plan.get('quality_gates', {}):
                if not cached.get('reused_from'):
                    raise ValueError('cached review lacks its original quality policy')
                validate_imported_review(cached['record'], entry, cached['reused_from'], self.plan)
                cached['quality_policy'] = deepcopy(self.plan.get('quality_gates', {}))
        write_json(self.root / 'reviews_cache.json', self.cache)
        imports = list(reuse_reviews)
        if not resume:
            imports.extend(item['quality_report'] for item in self.queue.get('claimed', []))
        for path in imports:
            for record in read_json(path)['records']:
                key = str(Path(record['attempt_dir']).resolve())
                if key not in allowed:
                    raise ValueError('reuse report includes a trial outside the approved campaign')
                if (key in claims and record.get('trajectory_fingerprint') != claims[key]['trajectory_fingerprint']):
                    raise ValueError('claimed primary fingerprint differs from its accepted review')
                if key in self.cache['entries']:
                    continue
                batch_path = record.get('process_evidence', {}).get('batch_path')
                if not batch_path:
                    raise ValueError('reuse report has no authoritative batch parent evidence')
                entry = quality.batch_review_evidence(batch_path)[key]
                validate_imported_review(record, entry, path, self.plan)
                self.save_review(record, entry, reused_from=str(Path(path).resolve()))
        self.refresh('prepared')

    def save_review(self, record, entry, *, reused_from=None):
        key = str(Path(record['attempt_dir']).resolve())
        stored = deepcopy(record)
        for field in ('duplicate_of', 'shared_trajectory_with', 'numerically_unique'):
            stored.pop(field, None)
        stored['numerically_unique'] = None
        self.cache['entries'][key] = {'record': stored, 'snapshot': snapshot(entry), 'reused_from': reused_from,
                                     'quality_policy': deepcopy(self.plan.get('quality_gates', {}))}
        self.cache['review_order'].append(key)
        write_json(self.root / 'reviews_cache.json', self.cache)

    def _failure_streak(self, records):
        """Count trailing failures at the same stage in review order."""
        stage, consecutive = None, 0
        for row in records:
            if row['quality_usable']:
                stage, consecutive = None, 0
            else:
                parent = self.cache['entries'][row['attempt_dir']]['snapshot']['evidence']['trial']['result']
                next_stage = parent.get('failure_stage') or ('duplicate' if row.get('duplicate_of') else 'quality_validation')
                consecutive = consecutive + 1 if stage == next_stage else 1
                stage = next_stage
        return stage, consecutive

    def refresh(self, status, *, error=None):
        records = [deepcopy(self.cache['entries'][key]['record']) for key in self.cache['review_order']]
        summary = quality.summarize_reviews(records, self.plan)
        summary['campaign_queue'] = str(self.queue_path)
        write_json(self.root / 'campaign_quality.json', summary)
        qualified = [row for row in summary['records'] if row['quality_usable'] and row.get('split') == 'candidate']
        by_family, by_color, by_geometry = {}, {}, {}
        for row in qualified:
            by_family.setdefault(row['behavior_family'], set()).add(row['trajectory_fingerprint'])
            evidence = self.cache['entries'][row['attempt_dir']]['snapshot']
            palette = ((evidence.get('disk_result') or {}).get('appearance') or {}).get('palette', 'source_nominal')
            by_color[palette] = by_color.get(palette, 0) + 1
            scene = evidence.get('scene_manifest') or {}
            key = f"{row['angle_deg']:g}deg/r{scene.get('config', {}).get('radius_m', 'unknown')}"
            by_geometry.setdefault(key, set()).add(row['trajectory_fingerprint'])
        stage, consecutive = self._failure_streak(summary['records'])
        planned = {item['output_dir'] for batch in self.queue['batches'] for item in trial_plan(batch)}
        reviewed = set(self.cache['entries']) & planned
        self.progress = {'schema': 'arx-door-augmentation-campaign-progress-v1',
            'updated_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'status': status,
            'queue_id': self.queue['queue_id'], 'queued_attempts': self.queue['queued_contexts'],
            'queued_reviewed': len(reviewed), 'queued_remaining': len(planned - reviewed),
            'candidate_episodes': summary['candidate_episodes'],
            'candidate_unique_trajectories': summary['candidate_unique_trajectories'],
            'current_profile_candidates': summary['current_profile_candidate_episodes'],
            'candidate_target': self.queue['candidate_target'],
            'families_unique_trajectories': {key: len(value) for key, value in by_family.items()},
            'colors_qualified_episodes': by_color,
            'geometry_unique_trajectories': {key: len(value) for key, value in by_geometry.items()},
            'required_angle_counts': summary['required_angle_counts'], 'active': self.active,
            'consecutive_failure_stage': stage, 'consecutive_failure_count': consecutive,
            'failed_quality_attempts': [row['attempt_dir'] for row in summary['records'] if not row['quality_usable']],
            'human_review_status': 'pending', 'generalization_verified': False, 'error': error}
        write_json(self.root / 'progress.json', self.progress)
        return self.progress

    def run_batch(self, index, batch, argv):
        logs = self.root / 'logs'; logs.mkdir(exist_ok=True)
        log_path = logs / f'batch_{index:03d}.log'
        groups = {}
        with log_path.open('a') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + batch['expected_attempts'] * (float(one(batch['argv'], '--timeout-s')) + 540) + 180
            try:
                observe_groups(process.pid, groups)
                self.active = {'batch_index': index, 'batch_pid': process.pid, 'batch_log': str(log_path),
                               'output_root': batch['output_root'], 'argv': argv}
                self.refresh('running')
                while True:
                    observe_groups(process.pid, groups)
                    try:
                        returncode = process.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() > deadline:
                            raise TimeoutError('campaign batch deadline exceeded')
                        report_path = Path(batch['output_root']) / 'batch_results.json'
                        if report_path.is_file():
                            report = read_json(report_path)
                            finished = {row['output_dir'] for row in report['trials']}
                            active = next((row for row in report['plan'] if row['output_dir'] not in finished
                                           and Path(row['output_dir']).exists()), None)
                            self.active['current_episode_log'] = str(Path(active['output_dir']) / 'generation.log') if active else None
                            self.active['batch_completed_rows'] = len(finished)
                        self.progress.update(updated_at=datetime.now(timezone.utc).isoformat(), active=self.active)
                        write_json(self.root / 'progress.json', self.progress)
            except (KeyboardInterrupt, TimeoutError):
                observe_groups(process.pid, groups)
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                raise
            finally:
                # A progress-write failure can occur while the batch is spawning
                # a simulator in another session; collect those children too.
                observe_groups(process.pid, groups)
                kill_owned_groups(groups)
                if process.poll() is None:
                    process.wait(timeout=30)
                self.active = None
        return returncode

    def review_batch(self, index, batch):
        report_path = Path(batch['output_root']) / 'batch_results.json'
        report = read_json(report_path)
        validate_batch(batch, report, self.queue)
        evidence = quality.batch_review_evidence(report_path)
        for key, entry in evidence.items():
            cached = self.cache['entries'].get(key)
            if cached:
                if cached['snapshot'] != snapshot(entry):
                    raise ValueError(f'cached trial parent/config/artifacts changed; review is not reused: {key}')
                continue
            thumbnail = self.root / 'phase_images' / f'batch_{index:03d}' / f"trial_{entry['trial']['index']:03d}"
            record = quality.review_attempt(key, self.plan, thumbnail_dir=thumbnail, batch_trial=entry)
            self.save_review(record, entry)
            self.refresh('reviewing')
        self.refresh('batch_reviewed')
        print(json.dumps({'batch_index': index, **{key: self.progress[key] for key in
              ('queued_reviewed', 'candidate_episodes', 'candidate_unique_trajectories',
               'current_profile_candidates', 'consecutive_failure_count', 'failed_quality_attempts')}}, ensure_ascii=False), flush=True)
        return len(report['trials']) == batch['expected_attempts']

    def run(self, *, pause_after_batch=None, failure_limit=5):
        for index, batch in enumerate(self.queue['batches']):
            argv = next_batch_argv(batch, self.queue)
            if argv is not None:
                returncode = self.run_batch(index, batch, argv)
                write_json(self.root / f'batch_{index:03d}_process.json', {'argv': argv, 'returncode': returncode})
            if not self.review_batch(index, batch):
                raise RuntimeError(f'batch {index} ended before completing its original plan; attempts preserved')
            if self.progress['consecutive_failure_count'] >= failure_limit:
                self.refresh('stopped_failure_streak')
                return 1
            pause_file = self.root / 'pause_after_batch'
            if index == pause_after_batch or pause_file.exists():
                pause_file.unlink(missing_ok=True)
                self.refresh('yielded')
                return 0
        self.refresh('complete' if self.progress['candidate_unique_trajectories'] >= self.queue['candidate_target'] else 'complete_with_shortfall')
        return 0 if self.progress['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/door_augmentation_plan.json')
    parser.add_argument('--reuse-review', type=Path, action='append', default=[])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='validate/import cache without launching simulation')
    parser.add_argument('--pause-after-batch', type=int, help='yield the GPU after this zero-based batch')
    args = parser.parse_args()
    campaign = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        campaign = Campaign(args.queue, args.output_root, args.plan, resume=args.resume, reuse_reviews=args.reuse_review)
        return 0 if args.prepare_only else campaign.run(pause_after_batch=args.pause_after_batch)
    except KeyboardInterrupt:
        if campaign:
            campaign.refresh('interrupted', error='interrupted; own batch and simulator groups cleaned')
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        if campaign:
            campaign.refresh('blocked', error=str(error))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
