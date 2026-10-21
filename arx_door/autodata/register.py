# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Entry points that AutoData's official ``scripts/generate_dataset.py`` loads.

    --external_callback arx_door.autodata.register:register
    --recorder_cfg      arx_door.autodata.register:recorder_cfg

``register`` runs after the simulation app starts. It receives the environment
arguments through ``sys.argv`` (``--task``, ``--device``, ``--num_envs``,
``--enable_cameras`` from AutoData, plus ``--assets``/``--physics`` from us), builds
the door environment in ``ik_abs`` mode and registers it under ``--task``. It also
registers the ARX cuRobo planner profile, which the official script builds after
this callback returns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REGISTERED: dict = {}


def parse_environment_args(argv: list[str]) -> argparse.Namespace:
    """Parse the callback's own argv; unknown options are an error, never ignored."""
    parser = argparse.ArgumentParser(prog='arx_door.autodata.register', allow_abbrev=False)
    parser.add_argument('--task', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--enable_cameras', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--assets', type=Path, required=True,
                        help='manifest.json of an asset bundle from scripts/door/autodata/prepare.py')
    parser.add_argument('--physics', choices=('source_cpu', 'lab_default'), default='source_cpu')
    args = parser.parse_args(argv)
    if not args.assets.is_file():
        parser.error(f'asset manifest not found: {args.assets}')
    if args.physics == 'source_cpu' and args.device != 'cpu':
        parser.error('source_cpu physics needs --device cpu on generate_dataset.py')
    if not args.enable_cameras:
        parser.error('the ARX recorders capture the wrist camera; pass --enable_cameras')
    return args


def register() -> list[str]:
    """Register the door env and the ARX planner profile; return unconsumed argv."""
    from arx_door.envs.door import build
    from .planner import register_planner_profile

    args = parse_environment_args(sys.argv[1:])
    env_id, cfg, _ = build(args.assets, args.num_envs, action='ik_abs',
                           physics=args.physics, name=args.task, ego_camera={})
    register_planner_profile()
    _REGISTERED.update(env_id=env_id, cfg=cfg, assets=args.assets.resolve(), physics=args.physics)
    return []


def recorder_cfg():
    """ARX wrist-camera and physical-acceptance recorders merged with the env's own."""
    from arx_door.envs.recorders import merge_arena_recorders

    if 'cfg' not in _REGISTERED:
        raise RuntimeError('recorder_cfg needs --external_callback arx_door.autodata.register:register')
    return merge_arena_recorders(_REGISTERED['cfg'].recorders)
