#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Collect one physical door expert in an owned, disposable teaching session."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door import paths  # noqa: E402
from arx_door.runtime import clean_environment, simulation_python  # noqa: E402


def terminate_owned_group(process, *, grace_s=5):
    """Reap this child's group, including workers left by an exited leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may already have exited while a cuMotion worker remains.
        # Never enumerate or kill other GPU processes or other sessions.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def start_logged_process(command, log_path, environment):
    with log_path.open('w', encoding='utf-8') as log:
        return subprocess.Popen(
            [str(value) for value in command], cwd=ROOT, env=environment,
            stdout=log, stderr=log, start_new_session=True,
        )


def assert_port_available(port):
    """Match HTTPServer reuse after close while refusing a live listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # HTTPServer enables SO_REUSEADDR. The preflight must permit its
        # previous session's TIME_WAIT sockets under the same rule. This
        # does not enable SO_REUSEPORT or displace an existing listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


def teaching_request(port, operation, *, timeout=5):
    payload = operation if isinstance(operation, dict) else {'op': operation}
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def is_own_session(state, output):
    wrist = state.get('images', {}).get('wrist')
    return wrist is not None and Path(wrist).parent.resolve() == output.resolve()


def wait_teaching_ready(process, log_path, output, *, timeout_s=180):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'teaching process exited before ready: {process.returncode}')
        if log_path.is_file():
            for line in log_path.read_text(errors='replace').splitlines():
                if line.startswith('TEACHING_READY '):
                    try:
                        state = json.loads(line[len('TEACHING_READY '):])
                    except json.JSONDecodeError:
                        continue
                    if not is_own_session(state, output):
                        raise RuntimeError('teaching readiness belongs to a different output directory')
                    if process.poll() is not None:
                        raise RuntimeError('teaching process exited after printing ready')
                    return state
        time.sleep(0.2)
    raise TimeoutError(f'teaching session did not become ready within {timeout_s:g}s')


def shutdown_session(process, port, output, *, ready, episode_name=None, grace_s=20):
    """Request shutdown only from our verified session, then bound its exit."""
    shutdown = {'requested': False}
    if ready and process.poll() is not None:
        shutdown['premature_exit'] = True
    if ready and process.poll() is None:
        try:
            observed = teaching_request(port, 'observe')
            if not is_own_session(observed, output):
                raise RuntimeError('port now belongs to a different teaching output')
            if episode_name is not None:
                episode = output / episode_name
                if episode.is_dir() and not (episode / 'episode.json').is_file():
                    # A killed collector cannot send its normal failed `end`.
                    # Flush an interrupted recording as a failed, pending
                    # episode while the verified teacher is still responsive.
                    try:
                        finalized = teaching_request(port, {
                            'op': 'end', 'success': False,
                            'notes': 'Expert runner stopped before collector end; human review pending.',
                        })
                        shutdown['failed_episode_finalized'] = finalized.get('ok') is True
                    except Exception as exc:
                        shutdown['failed_episode_error'] = str(exc)
            response = teaching_request(port, 'shutdown')
            if response.get('ok') is not True:
                raise RuntimeError('owned teaching session rejected shutdown')
            shutdown['requested'] = True
        except Exception as exc:
            shutdown['error'] = str(exc)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            shutdown['forced'] = True
    terminate_owned_group(process)
    shutdown['process_returncode'] = process.returncode
    return shutdown


def prepare_scene(args, environment):
    """Flatten USD references offline and preserve the paired placement manifest."""
    from arx_door.runtime import usd_environment

    command = [args.simulation_python, '-c', (
        'import json,sys; from pathlib import Path; '
        'from arx_door.scene.usd_assets import copy_flattened_scene; '
        'source=Path(sys.argv[1]); output=Path(sys.argv[2]); '
        'asset_resolution=copy_flattened_scene(source, output / "scene.usd"); '
        'manifest=json.loads(source.with_suffix(".json").read_text()); '
        'manifest["asset_resolution"]=asset_resolution; '
        '(output / "scene.json").write_text(json.dumps(manifest,indent=2)+"\\n")'
    ), str(args.scene), str(args.output_root)]
    usd_env = usd_environment()
    # usd_environment derives from our already cleaned, selected interpreter.
    process = start_logged_process(command, args.output_root / 'prepare.log', usd_env)
    try:
        returncode = process.wait(timeout=180)
        if returncode:
            raise RuntimeError(f'offline scene preparation failed: exit {returncode}')
    finally:
        terminate_owned_group(process)


def collect_expert(args, environment, runtime):
    """Own every spawned process while preserving failed episodes and logs."""
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'arx-door-expert-run-v1', 'name': args.name,
        'scene_source': str(args.scene), 'output_root': str(output),
        'simulation_python': str(args.simulation_python), 'runtime': runtime,
        'parameters': {key: getattr(args, key) for key in (
            'behavior_family',
            'edge_height', 'edge_yaw', 'edge_pregrasp_distance', 'retreat_distance',
            'edge_x', 'final_angle', 'gripper_force', 'gripper_stiffness', 'mimic_frequency',
        )},
        'success': False, 'review_status': 'pending', 'video_encoded': False,
    }
    report['parameters']['unlock_command_continuity'] = bool(
        getattr(args, 'unlock_command_continuity', False))
    teacher = None
    collector = None
    ready = False
    stage = 'prepare_scene'
    returncode = 1
    try:
        prepare_scene(args, environment)
        stage = 'teaching_startup'
        assert_port_available(args.port)
        teacher_command = [args.simulation_python, paths.TEACH_SESSION,
                           '--scene', output / 'scene.usd', '--output', output,
                           '--port', args.port, '--gripper-force', args.gripper_force,
                           '--gripper-stiffness', args.gripper_stiffness,
                           '--mimic-frequency', args.mimic_frequency]
        teacher = start_logged_process(teacher_command, output / 'teacher.log', environment)
        state = wait_teaching_ready(teacher, output / 'teacher.log', output,
                                    timeout_s=args.ready_timeout_s)
        ready = True
        if not is_own_session(teaching_request(args.port, 'observe'), output):
            raise RuntimeError('collector cannot verify the owned teaching session')
        report['teaching_runtime'] = state.get('runtime')
        stage = 'collection'
        command = [args.simulation_python, paths.EXPERT_COLLECTOR,
                   '--session-output', output, '--name', args.name, '--port', args.port]
        for parameter in ('edge_height', 'edge_yaw', 'edge_pregrasp_distance',
                          'retreat_distance', 'edge_x', 'final_angle', 'behavior_family'):
            command += ['--' + parameter.replace('_', '-'), getattr(args, parameter)]
        if report['parameters']['unlock_command_continuity']:
            command.append('--unlock-command-continuity')
        collector = start_logged_process(command, output / 'collector.log', environment)
        code = collector.wait(timeout=args.timeout_s)
        report['collector_returncode'] = code
        if code:
            raise RuntimeError(f'expert collector failed: exit {code}')
        if teacher.poll() is not None:
            raise RuntimeError(f'teacher exited before controlled shutdown: {teacher.returncode}')
        episode = json.loads((output / args.name / 'episode.json').read_text())
        if episode.get('success') is not True:
            raise RuntimeError('collector exited successfully without a successful physical episode')
        report['physics_executed'] = True
        report['physical_success'] = True
        returncode = 0
    except KeyboardInterrupt:
        report.update(failure_stage=stage, failure_reason='interrupted by user')
        returncode = 130
    except Exception as exc:
        report.update(failure_stage=stage, failure_reason=str(exc))
    finally:
        if collector is not None:
            try:
                terminate_owned_group(collector)
                report.setdefault('collector_returncode', collector.returncode)
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append(f'collector: {exc}')
                returncode = 1
        if teacher is not None:
            try:
                report['shutdown'] = shutdown_session(
                    teacher, args.port, output, ready=ready, episode_name=args.name,
                )
                report['teacher_returncode'] = teacher.returncode
                shutdown = report['shutdown']
                if returncode == 0 and (teacher.returncode != 0
                        or not shutdown['requested'] or shutdown.get('error')
                        or shutdown.get('forced') or shutdown.get('premature_exit')):
                    report.update(failure_stage='teaching_shutdown',
                                  failure_reason='teacher did not complete an acknowledged owned shutdown')
                    returncode = 1
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append(f'teacher: {exc}')
                returncode = 1

    # The simulator is already closed. Failed recordings may also be encoded
    # for diagnosis, retaining physical failure and pending human review.
    episode_dir = output / args.name
    if (episode_dir / 'episode.json').is_file():
        try:
            metadata = json.loads((episode_dir / 'episode.json').read_text())
            report['physical_success'] = metadata.get('success') is True
            report['episode_frames'] = metadata.get('frames')
            should_encode = metadata.get('success') is True or args.encode_failed
            if should_encode:
                if (not (episode_dir / 'trajectory.npz').is_file()
                        or type(metadata.get('frames')) is not int or metadata['frames'] < 2):
                    raise RuntimeError('recorded episode lacks an encodable trajectory and frame count')
                encoder = start_logged_process(
                    [args.simulation_python, paths.ENCODE_ATTEMPT, episode_dir],
                    output / 'encode.log', environment,
                )
                try:
                    code = encoder.wait(timeout=args.timeout_s)
                    report['encoder_returncode'] = code
                    if code:
                        raise RuntimeError(f'video encoder failed: exit {code}')
                    if any(not (episode_dir / filename).is_file()
                           or (episode_dir / filename).stat().st_size == 0
                           for filename in ('wrist.mp4', 'overview.mp4', 'review.mp4')):
                        raise RuntimeError('video encoder exited without all three nonempty videos')
                    report['video_encoded'] = True
                finally:
                    terminate_owned_group(encoder)
        except KeyboardInterrupt:
            report.update(failure_stage='encoding', failure_reason='interrupted by user')
            returncode = 130
        except Exception as exc:
            report['encoding_error'] = str(exc)
            if returncode == 0:
                report.update(failure_stage='encoding', failure_reason=str(exc))
                returncode = 1
    report['success'] = (returncode == 0 and report.get('physical_success') is True
                         and report['video_encoded'])
    report['returncode'] = returncode
    (output / 'expert_run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output_root': str(output), 'success': report['success'],
                      'review_status': 'pending', 'returncode': returncode}), flush=True)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--simulation-python', type=Path)
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--behavior-family', choices=['edge_regrasp', 'knob_hold_open'],
                        default='edge_regrasp')
    parser.add_argument('--edge-height', type=float, default=1.01)
    parser.add_argument('--edge-yaw', type=float, default=160)
    parser.add_argument('--edge-pregrasp-distance', type=float, default=.20)
    parser.add_argument('--retreat-distance', type=float, default=.08)
    parser.add_argument('--edge-x', type=float, default=.740)
    parser.add_argument('--final-angle', type=float, default=33)
    parser.add_argument('--gripper-force', type=float, default=60)
    parser.add_argument('--gripper-stiffness', type=float, default=3000)
    parser.add_argument('--mimic-frequency', type=float, default=0)
    parser.add_argument('--ready-timeout-s', type=float, default=180)
    parser.add_argument('--timeout-s', type=float, default=1800)
    parser.add_argument('--encode-failed', action='store_true',
                        help='Also encode a recorded failed episode for diagnosis; it remains failed.')
    parser.add_argument('--unlock-command-continuity', action='store_true',
                        help='Experimental collector continuation from the previous completed '
                             'unlock command target; defaults to the measured-start protocol.')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        parser.error('--name must be a single episode directory name')
    args.scene = args.scene.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.scene.is_file() or not args.scene.with_suffix('.json').is_file():
        parser.error('--scene requires an existing USD and its matching .json manifest')
    if args.output_root.exists():
        parser.error('--output-root must be a fresh directory')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    if (not math.isfinite(args.ready_timeout_s) or not 0 < args.ready_timeout_s <= 180
            or not math.isfinite(args.timeout_s) or args.timeout_s <= 0):
        parser.error('ready timeout must be 0..180 seconds and collection timeout positive')
    try:
        args.simulation_python = simulation_python(args.simulation_python)
        environment = clean_environment()
        if (Path(sys.executable).absolute() != args.simulation_python
                or os.environ.get('ARX_DOOR_EXPERT_CLEAN_RUNTIME') != '1'):
            environment['ARX_DOOR_EXPERT_CLEAN_RUNTIME'] = '1'
            os.execve(str(args.simulation_python),
                      [str(args.simulation_python), str(Path(__file__).resolve()), *sys.argv[1:]],
                      environment)
        # This branch uses exactly the selected runtime, without starting Kit.
        os.environ.clear()
        os.environ.update(environment)
        from arx_door.runtime import (
            runtime_provenance, simulation_environment, validate_runtime,
        )
        runtime = runtime_provenance()
        runtime['profile'] = validate_runtime(runtime)
        environment = simulation_environment()
        os.environ.update(environment)
        def interrupted(signum, unused_frame):
            # Independent teacher sessions survive a default parent SIGTERM;
            # convert it to a controlled failure so ownership cleanup runs.
            raise InterruptedError(f'expert runner received {signal.Signals(signum).name}')
        signal.signal(signal.SIGTERM, interrupted)
        return collect_expert(args, environment, runtime)
    except Exception as exc:
        print(f'expert setup failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
