#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Generate ARX door data in the official AutoData container, using Arena."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--cuboids', type=Path, required=True)
    parser.add_argument('--worker-config', type=Path, required=True,
                        help='JSON containing a command argv list for the isolated native planner')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', type=Path, default=ROOT / 'configs/autodata/knob_hold_open.yaml')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=16)
    # Help remains available on the host, before importing simulator packages.
    if any(flag in sys.argv for flag in ('--help', '-h')):
        parser.print_help()
        return 0
    if not Path('/isaac-sim/python.sh').is_file() or not os.environ.get('ISAACLAB_PATH'):
        parser.error('run inside the prepared official AutoData container; see run_container.sh')
    from arx_door.autodata.preflight import checkout_evidence
    # Keep the parent repository and its .git/modules together in the mount:
    # a submodule's .git file cannot resolve through a standalone vendor bind.
    source = checkout_evidence()
    if not source['source_ready']:
        parser.error('; '.join(source['problems']))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.trials < 1 or args.max_attempts < args.trials:
        parser.error('num-envs/trials must be positive and max-attempts >= trials')
    for path in (args.assets, args.input, args.urdf, args.cuboids, args.worker_config, args.task):
        if not path.is_file():
            parser.error(f'missing input: {path}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a fresh directory to preserve run isolation')
    args.output.mkdir(parents=True, exist_ok=True)
    args.device = 'cpu'
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    loop = None
    gathered = None
    stats = {'num_attempts': 0, 'num_success': 0, 'num_failures': 0,
             'num_official_success': 0, 'num_cancelled': 0, 'generation_seconds': 0.0}
    started = time.monotonic()
    error = None
    completed = False
    try:
        # Kit must exist before importing Arena, sensor classes or AutoData.
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.managers import TerminationTermCfg
        from autodata_core import DataGenerator
        from autodata_interfaces.env import env_loop, setup_env_config
        from autodata_interfaces.tasks.task_descriptor import TaskDescriptor
        from arx_door.autodata.acceptance import domain_success
        from arx_door.autodata.adapter import ArxAdapter
        from arx_door.autodata.algorithm import DoorSkillGen
        from arx_door.autodata.arena.factory import build_arena_environment
        from arx_door.autodata.datastream import DoorDatastream
        from arx_door.autodata.planner import DoorPlanner
        from arx_door.autodata.recorders import merge_arena_recorders
        from arx_door.autodata.runner import run_attempts, supervise_attempts, target_reached
        from arx_door.autodata.preflight import source_evidence, write_report
        from arx_door.autodata import AUTODATA_COMMIT, ARENA_COMMIT, LAB_COMMIT
        from arx_door.skillgen.physics_profile import (
            physics_material_bindings, physics_runtime_evidence, validate_physics_runtime,
        )

        descriptor = TaskDescriptor.from_yaml(args.task)
        if descriptor.name != 'arx_knob_hold_open':
            raise ValueError('first migration stage supports the reviewed knob_hold_open task only')
        policy = descriptor.get_generation_policy()
        policy.num_trials = args.trials
        policy.keep_failed = True
        policy.use_skillgen = True
        policy.guarantee_success = False  # First validation counts attempts, not unlimited retries.
        random.seed(policy.seed)
        np.random.seed(policy.seed)
        torch.manual_seed(policy.seed)
        name, arena_cfg, env_kwargs = build_arena_environment(args.assets, args.num_envs, 'source_cpu')
        recorder_cfg = merge_arena_recorders(arena_cfg.recorders)
        cfg, _ = setup_env_config(
            env_name=name, output_dir=str(args.output), output_file_name='generated',
            num_envs=args.num_envs, device='cpu', generation_policy_params=policy,
            recorder_cfg=recorder_cfg, enable_cameras=True,
        )
        success = TerminationTermCfg(func=domain_success)
        env = gym.make(name, cfg=cfg, **env_kwargs).unwrapped
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        construction_bindings = physics_material_bindings(stage)
        env.reset()
        physics_evidence = physics_runtime_evidence(
            stage, cfg.sim, env.device, 'source_cpu', source_settings=cfg.arx_source_settings,
            original_bindings=construction_bindings,
        )
        physics_evidence['material_comparison_origin'] = 'after_arena_construction_before_first_reset'
        write_report(args.output / 'physics_runtime.json', physics_evidence)
        validate_physics_runtime(physics_evidence)
        adapter = ArxAdapter(args.urdf)
        stream = DoorDatastream(
            env=env, task_descriptor=descriptor, embodiment_adapter=adapter,
            source_dataset_path=str(args.input), uses_start_signals=True,
            cuboids_path=str(args.cuboids),
        )
        worker = json.loads(args.worker_config.read_text())
        command = worker['command']
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError('worker command must be a nonempty argv list')
        planners = {i: DoorPlanner(
            stream, worker_command=command, env_id=i, urdf_path=str(args.urdf),
            cuboids_path=str(args.cuboids), artifact_dir=args.output / 'planner' / f'env_{i}',
        ) for i in range(args.num_envs)}
        generator = DataGenerator(datastream=stream, algorithm=DoorSkillGen(planners))
        write_report(args.output / 'configuration.json', {
            'schema': 'arx-autodata-run-v1', 'args': {k: str(v) if isinstance(v, Path) else v
                                                   for k, v in vars(args).items()},
            'source': source_evidence(args.input),
            'pins': {'autodata': AUTODATA_COMMIT, 'arena': ARENA_COMMIT, 'lab': LAB_COMMIT},
            'scope': 'knob_hold_open', 'physics_profile': 'source_cpu',
            'human_review_approved': False,
        })
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        reset_queue, action_queue = asyncio.Queue(), asyncio.Queue()
        jobs = [loop.create_task(run_attempts(
            env=env, env_id=i, generator=generator, success_term=success,
            reset_queue=reset_queue, action_queue=action_queue, stats=stats,
            output=args.output, planner=planners[i], max_attempts=args.max_attempts,
        )) for i in range(args.num_envs)]
        gathered = loop.create_task(supervise_attempts(jobs, stats, policy, args.max_attempts))
        env_loop(env, reset_queue, action_queue, loop,
                 generation_policy_params=policy, stats=stats, data_gen_tasks=gathered)
        # The official loop returns False when its stop future resolves while
        # waiting for actions. Our coordinated future also covers normal target
        # completion, so use the actual counters rather than that boolean.
        completed = target_reached(stats, policy)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, file=sys.stderr)
    finally:
        if gathered is not None:
            gathered.cancel()
            try:
                loop.run_until_complete(gathered)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if error is None:
                    error = f'cleanup failed: {type(exc).__name__}: {exc}'
        if env is not None:
            env.close()
        if loop is not None:
            loop.close()
        from arx_door.autodata.preflight import write_report
        write_report(args.output / 'result.json', {
            'schema': 'arx-autodata-run-result-v1', 'generation_completed': completed,
            'stats': stats, 'wall_seconds': time.monotonic() - started,
            'error': error, 'physical_acceptance_passed': bool(completed and stats['num_success'] > 0 and error is None),
            'human_review_approved': False,
        })
        app.close()
    return 0 if completed and stats['num_success'] > 0 and error is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
