#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Create the isolated stage-4 runtime: GR00T data venv plus pinned GR00T source.

This environment only exports LeRobot data and runs the official N1.7 loader
checks on CPU. Fine-tuning itself needs the full official GR00T environment
and a large-memory GPU; see docs/pipeline.md.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.runtime import VIDEO_TOOL_LIBRARIES  # noqa: E402

GROOT_URL = 'https://github.com/NVIDIA/Isaac-GR00T.git'
GROOT_COMMIT = '9c7e746b2cd37a810070a98ef41d290a07e806c2'
TORCH = ('torch==2.9.0', 'torchvision==0.24.0')
TORCH_INDEX = 'https://download.pytorch.org/whl/cpu'


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def checkout_groot(target: Path) -> None:
    """Fetch only the pinned commit; skip LFS objects (no model weights needed)."""
    environment = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}
    if not (target / '.git').is_dir():
        run(['git', 'init', '-q', target])
        run(['git', '-C', target, 'remote', 'add', 'origin', GROOT_URL])
    head = subprocess.run(['git', '-C', target, 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
    if head != GROOT_COMMIT:
        run(['git', '-C', target, 'fetch', '-q', '--depth', '1', 'origin', GROOT_COMMIT],
            env=environment)
        run(['git', '-C', target, 'checkout', '-q', 'FETCH_HEAD'], env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default='3.12',
                        help='uv-managed interpreter version for the data venv')
    args = parser.parse_args()
    uv = shutil.which('uv')
    if not uv:
        parser.error('uv is required to create the isolated data environment')
    runtime = ROOT / 'generated/runtime'
    venv = runtime / 'groot_data/.venv'
    python = venv / 'bin/python'
    if not python.is_file():
        run([uv, 'venv', '-q', '--python', args.python, venv])
    run([uv, 'pip', 'install', '-q', '--python', python, '--index-url', TORCH_INDEX, *TORCH])
    run([uv, 'pip', 'install', '-q', '--python', python,
         '-r', ROOT / 'dependencies/python/groot-data.txt', 'pytest==9.1.1'])
    checkout_groot(runtime / 'groot_n17_source')
    # torchcodec loads FFmpeg shared libraries; use the project's copy when
    # the host has none (scripts/door/setup/setup_door_runtime.py installs it).
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('PYTHONPATH', 'PYTHONHOME')}
    if VIDEO_TOOL_LIBRARIES.is_dir():
        environment['LD_LIBRARY_PATH'] = str(VIDEO_TOOL_LIBRARIES)
    run([python, '-c', 'import pyarrow, torchcodec, tyro; print("groot data runtime ok")'],
        env=environment)
    print(f'Run stage-4 tools with {ROOT / "scripts/door/dataset/run_dataset_tool.sh"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
