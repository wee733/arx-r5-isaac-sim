#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Validate a pending source and explicitly record agent visual review.

This records agent-reviewed source eligibility. Human review remains pending.
Run only after actually inspecting knob grasp, unlock, gap, edge grasp and hold.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.demos.seed import load_attempt
from arx_door.demos.review import (
    decoded_seed_videos, mark_agent_reviewed_seed, physical_seed_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--confirm-agent-visual-review', action='store_true',
                        help='affirm the agent actually inspected the complete task sequence')
    parser.add_argument('--visual-evidence', type=Path, action='append',
                        help='existing review image, absolute or relative to the source; repeat as needed')
    parser.add_argument('--visual-review-notes',
                        help='describe the observed grasp, unlock, gap, transfer and final hold')
    args = parser.parse_args()
    try:
        if args.confirm_agent_visual_review:
            if not args.visual_evidence or not args.visual_review_notes:
                parser.error('confirmation requires --visual-evidence and --visual-review-notes')
            report = mark_agent_reviewed_seed(
                args.input, confirm_visual_review=True, visual_evidence=args.visual_evidence,
                visual_review_notes=args.visual_review_notes,
            )
        else:
            if args.visual_evidence or args.visual_review_notes:
                parser.error('visual declarations require --confirm-agent-visual-review')
            seed = load_attempt(args.input)
            report = {'schema': 'arx-door-source-evidence-check-v1', 'review_written': False,
                      'physical_evidence': physical_seed_evidence(seed),
                      'video_evidence': decoded_seed_videos(seed),
                      'human_review_status': 'pending'}
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
