# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Discover and validate the selected Isaac runtime without launching Kit."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / 'configs/runtime/isaac_runtime_profiles.json'
LOCAL_RUNTIME_PATH = ROOT / 'generated/runtime/local.json'
VIDEO_TOOLS = ROOT / 'generated/runtime/tools'
VIDEO_TOOL_LIBRARIES = VIDEO_TOOLS / 'usr/lib/x86_64-linux-gnu'
URDF_RELATIVE = 'urdf/r5a_cumotion.urdf'
XRDF_RELATIVE = 'xrdf/r5a.xrdf'


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def isaac_lab_root() -> Path:
    """Find the loaded source tree; subpackage versions are not Lab releases."""
    explicit = os.environ.get('ISAAC_LAB_ROOT')
    spec = importlib.util.find_spec('isaaclab')
    candidates = list(Path(spec.origin).resolve().parents) if spec and spec.origin else []
    for candidate in candidates:
        if (candidate / 'VERSION').is_file() and (candidate / 'source/isaaclab').is_dir():
            if explicit and Path(explicit).expanduser().resolve() != candidate:
                raise RuntimeError('ISAAC_LAB_ROOT differs from the loaded Isaac Lab source: '
                                   f'{explicit} != {candidate}')
            return candidate
    raise RuntimeError('Isaac Lab source tree was not found; set ISAAC_LAB_ROOT')


def runtime_provenance() -> dict:
    """Describe actual imports, including source release and revision."""
    lab_root = isaac_lab_root()
    try:
        revision = subprocess.check_output(
            ['git', '-C', str(lab_root), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        source_changes = subprocess.check_output(
            ['git', '-C', str(lab_root), 'status', '--porcelain', '--', 'source', 'apps'],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision = None
        source_changes = ['source revision could not be inspected']
    return {
        'python': '.'.join(map(str, sys.version_info[:3])),
        'python_executable': sys.executable,
        'isaac_sim': _version('isaacsim'),
        'isaac_lab': (lab_root / 'VERSION').read_text().strip(),
        'isaac_lab_package': _version('isaaclab'),
        'isaac_lab_root': str(lab_root),
        'isaac_lab_revision': revision,
        'isaac_lab_source_changes': source_changes,
        'torch': _version('torch'),
        'numpy': _version('numpy'),
        'h5py': _version('h5py'),
    }


def runtime_profiles() -> dict[str, dict]:
    """Return the pinned runtime profiles keyed by profile name."""
    return json.loads(PROFILE_PATH.read_text())['profiles']


def validate_runtime(provenance: dict | None = None) -> str:
    """Accept the two pinned stacks; fail before physics on version drift."""
    actual = provenance or runtime_provenance()
    if actual.get('isaac_lab_source_changes'):
        raise RuntimeError('Isaac Lab source has unvalidated local changes: '
                           + '; '.join(actual['isaac_lab_source_changes']))
    for name, expected in runtime_profiles().items():
        if actual['isaac_sim'] != expected['isaac_sim']:
            continue
        errors = []
        for key in ('isaac_lab', 'isaac_lab_revision', 'numpy', 'torch'):
            if actual.get(key) != expected[key]:
                errors.append(f'{key}: expected {expected[key]}, found {actual.get(key)}')
        if not actual['python'].startswith(expected['python'] + '.'):
            errors.append(f"python: expected {expected['python']}, found {actual['python']}")
        if errors:
            raise RuntimeError('Isaac runtime differs from its pinned profile: ' + '; '.join(errors))
        return name
    raise RuntimeError(f"Unsupported Isaac Sim version: {actual.get('isaac_sim')}; see {PROFILE_PATH}")


def simulation_environment() -> dict[str, str]:
    """Add project/Mimic sources while keeping ROS Python out of Kit."""
    environment = os.environ.copy()
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH'):
        paths = environment.get(key, '').split(os.pathsep)
        if any('/opt/ros/' in path or '/isaac-ros/' in path for path in paths):
            raise RuntimeError(f'{key} contains ROS runtime paths; use a clean simulation shell')
    lab_root = isaac_lab_root()
    paths = [str(ROOT), str(lab_root / 'source/isaaclab_mimic')]
    paths.extend(filter(None, environment.get('PYTHONPATH', '').split(os.pathsep)))
    environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    environment.setdefault('TERM', 'xterm')
    # Match this host's existing Isaac activation script for the installed Kit.
    environment.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    tools = VIDEO_TOOLS / 'usr/bin'
    if tools.is_dir():
        environment['PATH'] = str(tools) + os.pathsep + environment.get('PATH', '')
        environment['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (
            str(VIDEO_TOOL_LIBRARIES), environment.get('LD_LIBRARY_PATH'),
        )))
    return environment


def clean_environment() -> dict[str, str]:
    """Remove sourced ROS overlays without modifying the calling shell."""
    environment = os.environ.copy()
    overlay_roots = [path for key in ('AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH')
                     for path in environment.get(key, '').split(os.pathsep) if path]
    for key in list(environment):
        if key.startswith(('ROS_', 'AMENT_', 'COLCON_', 'RMW_', 'CATKIN_')) or key in (
            'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH',
        ):
            environment.pop(key, None)
    environment['PATH'] = os.pathsep.join(
        path for path in environment.get('PATH', '').split(os.pathsep)
        if path and '/opt/ros' not in path and '/isaac-ros/' not in path
        and '/isaac_ros' not in path
        and not any(path == root or path.startswith(root + '/') for root in overlay_roots)
    )
    return environment


def simulation_python(explicit: str | Path | None = None) -> Path:
    """Select the explicit, ISAAC_SIM_PYTHON, or locally registered Sim Python."""
    value = explicit or os.environ.get('ISAAC_SIM_PYTHON')
    if not value:
        if not LOCAL_RUNTIME_PATH.is_file():
            raise RuntimeError('set --simulation-python or create generated/runtime/local.json')
        value = json.loads(LOCAL_RUNTIME_PATH.read_text())['simulation_python']
    # Do not resolve this symlink: executing its resolved base interpreter
    # would lose the selected venv and its isolated simulation dependencies.
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f'simulation Python is unavailable: {path}')
    return path


def video_tool(name: str) -> tuple[str, dict[str, str]]:
    """Prefer host FFmpeg tools, else the project's extracted Ubuntu packages."""
    environment = os.environ.copy()
    installed = shutil.which(name)
    if installed:
        return installed, environment
    local = VIDEO_TOOLS / 'usr/bin' / name
    if not local.is_file():
        raise FileNotFoundError(f'{name} is required for real video decoding')
    environment['LD_LIBRARY_PATH'] = str(VIDEO_TOOL_LIBRARIES)
    return str(local), environment


def close_app(app, exit_code: int) -> None:
    """Report the exit code where SimulationApp.close accepts it (Sim 6+)."""
    parameters = inspect.signature(app.close).parameters
    if 'exit_code' in parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        app.close(exit_code=exit_code)
    else:
        app.close()


def usd_environment() -> dict[str, str]:
    """Expose the selected Sim's USD libraries for standalone scene tools."""
    environment = simulation_environment()
    spec = importlib.util.find_spec('isaacsim')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('selected Python does not contain Isaac Sim')
    sim_root = Path(next(iter(spec.submodule_search_locations)))
    candidates = sorted(
        path for path in (sim_root / 'extscache').glob('omni.usd.libs-*')
        if (path / 'pxr').is_dir()
    )
    if not candidates:
        raise RuntimeError(f'USD libraries were not found under {sim_root}')
    libraries = candidates[-1]
    environment['PYTHONPATH'] = os.pathsep.join((str(libraries), environment['PYTHONPATH']))
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (
        str(libraries / 'bin'), sysconfig.get_config_var('LIBDIR'),
        environment.get('LD_LIBRARY_PATH'),
    )))
    return environment


def arx_description_dir() -> Path:
    """Resolve an explicit or migrated ARX model without relying on old mounts."""
    explicit = os.environ.get('ARX_DESCRIPTION_DIR')
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not (candidate / URDF_RELATIVE).is_file():
            raise RuntimeError(f'ARX_DESCRIPTION_DIR does not contain the ARX URDF: {candidate}')
        return candidate
    candidates = (
        ROOT.parent / 'arx-r5-moveit/isaac_ros_manipulation_arx_r5a_robot_description',
        ROOT.parent / 'B2ARX_Train_12DOF/Isaac_Ros_CuMotion_ArxR5a/isaac_ros_manipulation_arx_r5a_robot_description',
    )
    for candidate in candidates:
        if (candidate / URDF_RELATIVE).is_file():
            return candidate.resolve()
    raise RuntimeError('ARX robot description not found; set ARX_DESCRIPTION_DIR')


def planning_urdf(explicit: str | Path | None = None) -> Path:
    """Keep Cartesian IK and the native URDF/XRDF worker on one robot model."""
    description = arx_description_dir()
    expected = (description / URDF_RELATIVE).resolve()
    selected = Path(explicit).expanduser().resolve() if explicit else expected
    if selected != expected:
        raise RuntimeError('Planning URDF differs from the native cuMotion model; '
                           'set ARX_DESCRIPTION_DIR to the matching URDF/XRDF package: '
                           f'{selected} != {expected}')
    if not (description / XRDF_RELATIVE).is_file():
        raise RuntimeError(f'Native cuMotion XRDF not found: {description / XRDF_RELATIVE}')
    return selected
