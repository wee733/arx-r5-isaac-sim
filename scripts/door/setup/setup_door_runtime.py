#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Create a small project overlay over the installed, pinned Isaac Lab stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from arx_door.runtime import (  # noqa: E402
    LOCAL_RUNTIME_PATH, VIDEO_TOOL_LIBRARIES, VIDEO_TOOLS, runtime_profiles,
)


def install_video_tools() -> None:
    """Extract Ubuntu video tools locally when the host has no FFmpeg."""
    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        return
    if not shutil.which('apt-get') or not shutil.which('dpkg-deb'):
        raise RuntimeError('install ffmpeg and ffprobe before creating the simulation runtime')
    tools = VIDEO_TOOLS
    packages = tools / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    subprocess.run(['apt-get', 'download', 'ffmpeg', 'libavdevice60', 'libavfilter9',
                    'libpostproc57', 'libopenal1'], cwd=packages, check=True)
    for package in packages.glob('*.deb'):
        subprocess.run(['dpkg-deb', '-x', str(package), str(tools)], check=True)
    environment = {**os.environ, 'LD_LIBRARY_PATH': str(VIDEO_TOOL_LIBRARIES)}
    subprocess.run([str(tools / 'usr/bin/ffprobe'), '-version'], env=environment,
                   stdout=subprocess.DEVNULL, check=True)


def main() -> int:
    """Keep source runtimes intact; isolate fixes and write resolved paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-lab-root', type=Path, required=True)
    parser.add_argument('--profile', choices=('isaac51', 'isaac61'), default='isaac61')
    parser.add_argument('--source-python', type=Path,
                        help='Installed matching Sim Python; defaults to Lab/.venv/bin/python.')
    parser.add_argument('--keep-default', action='store_true',
                        help='Register this runtime without changing the default runtime selection.')
    args = parser.parse_args()
    lab_root = args.isaac_lab_root.resolve()
    source_python = (args.source_python or lab_root / '.venv/bin/python').absolute()
    if not source_python.is_file():
        parser.error(f'installed Isaac Lab Python not found: {source_python}')
    expected = runtime_profiles()[args.profile]
    release = (lab_root / 'VERSION').read_text().strip()
    revision = subprocess.check_output(['git', '-C', str(lab_root), 'rev-parse', 'HEAD'], text=True).strip()
    if release != expected['isaac_lab'] or revision != expected['isaac_lab_revision']:
        parser.error(f'Lab source does not match {args.profile}: {release} / {revision}')
    installed = json.loads(subprocess.check_output([
        str(source_python), '-c',
        'import json,sys,importlib.metadata as m; '
        'print(json.dumps({"python":"%d.%d" % sys.version_info[:2], '
        '"isaac_sim":m.version("isaacsim"),"torch":m.version("torch")}))',
    ], text=True))
    for key in ('python', 'isaac_sim', 'torch'):
        if installed[key] != expected[key]:
            parser.error(f'{args.profile} requires {key}={expected[key]}, found {installed[key]}')
    site_path = subprocess.check_output([
        str(source_python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))',
    ], text=True).strip()
    target = ROOT / 'generated/runtime' / args.profile / '.venv'
    uv = shutil.which('uv')
    if not uv:
        parser.error('uv is required to install isolated project dependencies')
    if not (target / 'bin/python').is_file():
        subprocess.run([uv, 'venv', '--python', str(source_python), str(target)], check=True)
    python = target / 'bin/python'
    target_site = Path(subprocess.check_output([
        str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))',
    ], text=True).strip())
    registrations = [str(ROOT)]
    registrations.extend(str(path) for path in sorted((lab_root / 'source').glob('isaaclab*'))
                         if path.is_dir() and (path / path.name).is_dir())
    # An independently installed old stack can already be the target. Avoid
    # recursively adding its own site directory through a .pth callback.
    if Path(site_path).resolve() != target_site.resolve():
        registrations.insert(0, f'import site; site.addsitedir({site_path!r})')
    (target_site / 'arx_isaac_dependencies.pth').write_text('\n'.join(registrations) + '\n')
    subprocess.run([uv, 'pip', 'install', '--python', str(python),
                    f'numpy=={expected["numpy"]}'], check=True)
    install_video_tools()
    config = {
        'schema': 'arx-local-runtime-v1', 'profile': args.profile,
        'isaac_lab_root': str(lab_root), 'simulation_python': str(python),
        'source_site_packages': site_path,
    }
    (target.parent / 'selection.json').write_text(json.dumps(config, indent=2) + '\n')
    if not args.keep_default:
        LOCAL_RUNTIME_PATH.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
