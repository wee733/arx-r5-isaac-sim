# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Entry-point orchestration around the unmodified official generation loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .preflight import write_report


def target_reached(stats: dict, policy) -> bool:
    """Use the same target semantics as the official loop, with domain counters."""
    count = stats['num_success'] if policy.guarantee_success else stats['num_attempts']
    return count >= policy.num_trials


async def supervise_attempts(jobs, stats: dict, policy, max_attempts: int):
    """Complete the official loop's stop future without abandoning one batch slot.

    A worker exiting alone would leave env_loop waiting for num_envs actions.
    Coordinate every exit, cancel and await all producers, then resolve this
    future normally. The caller decides whether the trial target was reached.
    """
    try:
        while True:
            await asyncio.sleep(0)
            for job in jobs:
                if job.done() and not job.cancelled() and job.exception() is not None:
                    raise job.exception()
            if (target_reached(stats, policy) or stats['num_attempts'] >= max_attempts
                    or any(job.done() for job in jobs)):
                return
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        outcomes = await asyncio.gather(*jobs, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                raise RuntimeError(f'attempt finalization failed: {type(outcome).__name__}: {outcome}') from outcome


def finalize_attempt(env, env_id: int, attempt_id: int, output: Path, *,
                     result=None, error: str | None = None, planner=None) -> dict:
    """Set the exported label from terminal domain evidence, including early failures."""
    import torch
    from .acceptance import acceptance_state
    from .recorders import camera_evidence

    state = acceptance_state(env)
    physical = state.summary(env_id)
    camera = camera_evidence(env, env_id)
    camera_ok = (camera.get('pose_verified') is True
                 and camera.get('capture_protocol_verified') is True
                 and camera.get('timestamp_action_index_mapping_complete') is True
                 and camera.get('valid_rgb_frames', 0) > 0
                 and camera.get('valid_rgb_frames') == camera.get('frames_verified')
                 and camera.get('valid_rgb_frames') == physical.get('frames'))
    official_success = bool(result is not None and result.success)
    accepted = bool(error is None and official_success and state.accepted([env_id])[0] and camera_ok)
    episode = env.recorder_manager.get_episode(env_id)
    # InitialStateRecorder already writes at reset. is_empty() therefore does
    # not tell us whether a single action has actually been executed.
    actions = episode.data.get('actions', [])
    action_count = len(actions)
    has_data = action_count > 0
    accepted = accepted and has_data
    env_ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    label = torch.tensor([[accepted]], dtype=torch.bool, device=env.device)
    env.recorder_manager.set_success_to_episodes(env_ids, label)
    demo_id = (attempt_id - 1) * env.num_envs + env_id
    filename = getattr(getattr(env.recorder_manager, 'cfg', None), 'dataset_filename', 'generated')
    dataset_file = f'{filename}{"" if accepted else "_failed"}.hdf5' if has_data else None
    report = {
        'schema': 'arx-autodata-attempt-v1', 'env_id': env_id, 'attempt_id': attempt_id,
        'official_generation_success': official_success, 'domain_qc_accepted': accepted,
        'exported_partial_or_full_episode': False, 'empty_trajectory': not has_data,
        'action_count': action_count, 'has_initial_state': 'initial_state' in episode.data,
        'dataset_file': dataset_file, 'demo_key': f'demo_{demo_id}' if has_data else None,
        'physical': physical, 'camera_verification': camera,
        'human_review_approved': False, 'error': error,
        'planner_request': getattr(planner, 'last_request', None),
        'planner_result': getattr(planner, 'last_result', None),
    }
    if not accepted and error is None:
        report['error'] = 'domain_or_camera_qc_rejected' if official_success else 'generation_failed'
    manifest_path = output / 'attempts' / f'env_{env_id:03d}_attempt_{attempt_id:06d}.json'
    # Preserve evidence before I/O that could fail; a reset-only episode has
    # a manifest but never masquerades as a training trajectory.
    report['export_status'] = 'pending' if has_data else 'no_actions'
    write_report(manifest_path, report)
    if has_data:
        try:
            env.recorder_manager.export_episodes(env_ids, demo_ids=[demo_id])
        except Exception as exc:
            report['domain_qc_accepted'] = False
            report['export_status'] = 'failed'
            report['error'] = f'export failed: {type(exc).__name__}: {exc}'
            write_report(manifest_path, report)
            raise
        report['exported_partial_or_full_episode'] = True
        report['export_status'] = 'completed'
        write_report(manifest_path, report)
    return report


async def run_attempts(*, env, env_id, generator, success_term, reset_queue, action_queue,
                       stats: dict, output: Path, planner, max_attempts: int):
    """Tally domain-accepted outcomes; every attempt has a failure manifest."""
    sequence = 0
    while stats['num_attempts'] < max_attempts:
        sequence += 1
        result = None
        error = None
        started = time.monotonic()
        try:
            result = await generator.generate(
                env_id=env_id, success_term=success_term,
                env_reset_queue=reset_queue, env_action_queue=action_queue,
                export_demo=False,
            )
        except asyncio.CancelledError:
            finalize_attempt(env, env_id, sequence, output, error='cancelled', planner=planner)
            stats['num_cancelled'] += 1
            raise
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            # Preserve the trajectory before failing the run. Repeated configuration
            # errors must not become an endless sequence of successful-looking retries.
        report = finalize_attempt(env, env_id, sequence, output,
                                  result=result, error=error, planner=planner)
        stats['num_attempts'] += 1
        stats['num_success'] += int(report['domain_qc_accepted'])
        stats['num_failures'] += int(not report['domain_qc_accepted'])
        stats['num_official_success'] += int(report['official_generation_success'])
        stats['generation_seconds'] += time.monotonic() - started
        if error is not None:
            raise RuntimeError(error)
        await asyncio.sleep(0)
