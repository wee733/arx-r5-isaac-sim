#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run an approved candidate queue serially and review each completed trial once."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.skillgen.campaign import Campaign  # noqa: E402


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
