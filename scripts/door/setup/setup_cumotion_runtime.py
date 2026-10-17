#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Install the pinned official cuMotion wheel into a project-local Python venv.

Uses only the native planner's dependencies, never Isaac Sim or ROS packages.
Pass --wheel on another machine, or set CUMOTION_WHEEL. An existing matching
Isaac ROS source distribution is discovered relative to this workspace.
"""

import argparse
from email.parser import Parser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/runtime/cumotion_runtime.json'


def resolve_wheel(explicit, config):
    """Select a local official distribution without downloading a replacement."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = ROOT.parent / 'isaac_ros_source/isaac_ros_nitros/isaac_ros_nitros/lib/cumotion'
    candidate /= f"x86_64_cuda_{config['cuda_variant'].replace('.', '_')}"
    candidate /= 'python_wheels'
    candidate /= f"cumotion-{config['cumotion_version']}-{config['wheel_tag']}.whl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError('official cuMotion wheel not found; pass --wheel or set CUMOTION_WHEEL')


def validate_wheel(wheel, config):
    """Reject incompatible native wheels before changing an environment."""
    expected = f"cumotion-{config['cumotion_version']}-{config['wheel_tag']}.whl"
    if wheel.name != expected:
        raise ValueError(f'expected {expected}, received {wheel.name}')
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('wheel must contain exactly one package metadata file')
        package = Parser().parsestr(archive.read(metadata[0]).decode('utf-8'))
    if package['Name'] != 'cumotion' or package['Version'] != config['cumotion_version']:
        raise ValueError('cuMotion wheel metadata does not match the pinned runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', default=os.environ.get('CUMOTION_WHEEL'))
    parser.add_argument('--python', default=shutil.which('python3.12') or sys.executable)
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--offline', action='store_true', help='use uv cache only')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    wheel = resolve_wheel(args.wheel, config)
    validate_wheel(wheel, config)
    runtime = (args.runtime_dir or ROOT / config['runtime_dir']).expanduser().resolve()
    python_minor = subprocess.check_output(
        [args.python, '-I', '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
        text=True,
    ).strip()
    if python_minor != config['python_minor']:
        raise ValueError(f"cuMotion requires Python {config['python_minor']}, received {python_minor}")
    uv = shutil.which('uv')
    if uv is None:
        raise FileNotFoundError('uv is required to install into the isolated runtime')
    runtime.mkdir(parents=True, exist_ok=True)
    environment = runtime / '.venv'
    python = environment / 'bin/python'
    clean_env = {
        key: value for key, value in os.environ.items()
        if key not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV')
    }
    if not python.exists():
        subprocess.run([args.python, '-I', '-m', 'venv', '--without-pip', str(environment)],
                       env=clean_env, check=True)
    command = [uv, 'pip', 'install', '--python', str(python),
               str(wheel), f"numpy=={config['numpy_version']}"]
    if args.offline:
        command.append('--offline')
    subprocess.run(command, env=clean_env, check=True)
    # This inspects package metadata; native CUDA initialization is left to the
    # single concentrated planning diagnostic / real planning request.
    installed = json.loads(subprocess.check_output(
        [str(python), '-I', '-c',
         'import json,sys,importlib.metadata as m; '
         'print(json.dumps({"python":sys.version.split()[0],'
         '"cumotion":m.version("cumotion"),"numpy":m.version("numpy")}))'],
        text=True, env=clean_env,
    ))
    if installed['cumotion'] != config['cumotion_version'] or installed['numpy'] != config['numpy_version']:
        raise RuntimeError(f'installed planner dependencies do not match the pins: {installed}')
    manifest = {
        'schema_version': 1,
        'role': 'native_planning_only',
        'python_executable': str(python),
        'installed': installed,
        'wheel_path': str(wheel),
        'wheel_tag': config['wheel_tag'],
        'cuda_variant': config['cuda_variant'],
        'installation_reference': config['installation_reference'],
        'simulation_packages_installed': False,
    }
    manifest_path = runtime / 'runtime.json'
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    print(json.dumps({'runtime_manifest': str(manifest_path), **installed}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
