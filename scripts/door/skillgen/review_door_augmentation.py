#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Review physical SkillGen HDF5 and real videos; never edits source samples."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.files import write_json  # noqa: E402
from arx_door.quality import (  # noqa: E402
    batch_review_evidence, review_collection,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/door_augmentation_plan.json')
    parser.add_argument('--manifest', type=Path, help='migration sample manifest')
    parser.add_argument('--batch-report', type=Path, action='append', help='batch_results.json; repeat for collections')
    parser.add_argument('--attempt-dir', type=Path, action='append', help='single trial directory; repeat as needed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thumbnails', action='store_true', help='save phase JPGs beside the review report')
    parser.add_argument('--skip-videos', action='store_true', help='diagnostic only; never counts as usable data')
    args = parser.parse_args()
    attempts = list(args.attempt_dir or [])
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        attempts.extend(ROOT / path for path in manifest['generated_attempts'])
    batch_trials = {}
    for batch_path in args.batch_report or []:
        evidence = batch_review_evidence(batch_path)
        if set(batch_trials) & set(evidence):
            parser.error('batch reports contain duplicate trial directories')
        batch_trials.update(evidence)
        attempts.extend(Path(path) for path in evidence)
    if not attempts:
        parser.error('provide --manifest, --batch-report, or --attempt-dir')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    output = args.output.expanduser().resolve()
    thumbnails = output.parent / (output.stem + '_phase_images') if args.thumbnails else None
    report = review_collection(attempts, plan, thumbnail_root=thumbnails,
                               inspect_videos=not args.skip_videos, batch_trials=batch_trials)
    report['augmentation_plan'] = str(args.plan.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({key: report[key] for key in (
        'attempts_reviewed', 'usable_unique_episodes', 'candidate_episodes', 'holdout_episodes',
        'ready_to_discuss_fine_tuning', 'missing_readiness_requirements')}, indent=2))
    print(f'Review: {output}')
    # Readiness is a later engineering decision; reject only unusable attempts.
    return 0 if all(row['quality_usable'] for row in report['records']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
