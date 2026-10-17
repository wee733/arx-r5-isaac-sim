#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Run GPU policy verification or its local server in an isolated environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.policy.config import DEFAULT_CONFIG, load_config, policy_environment, resource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'serve'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        help='serve this checkpoint-* directory instead of the config entry')
    parser.add_argument('--dataset', type=Path,
                        help='holdout LeRobot dataset for the warm-up inference check')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint or args.dataset:
        # The server owns --output; keep the derived config beside it.
        if args.output.exists():
            parser.error(f'{args.output} exists; use a new output directory')
        if args.checkpoint:
            config['checkpoint'] = str(args.checkpoint.absolute())
        if args.dataset:
            config['dataset'] = str(args.dataset.absolute())
        derived = args.output.absolute().with_name(args.output.name + '.policy_config.json')
        derived.write_text(json.dumps(config, indent=2) + '\n')
        args.config = derived
    python = resource(config, 'policy_python')
    if not python.is_file():
        parser.error('GPU policy environment is missing; see scripts/door/policy/README.md')
    argv = [str(python), str(Path(__file__).with_name('server.py')), args.mode,
            '--config', str(args.config.absolute()), '--output', str(args.output.absolute())]
    os.execve(str(python), argv, policy_environment(config))


if __name__ == '__main__':
    main()
