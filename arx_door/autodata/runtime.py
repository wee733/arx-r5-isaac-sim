# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Select the same pinned container for generation and Arena policy evaluation."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from . import source_checkouts

PYTHON = Path('/isaac-sim/python.sh')
AUTODATA = Path('/workspaces/autodata')
ARENA = AUTODATA / 'submodules/IsaacLab-Arena'
LAB = ARENA / 'submodules/IsaacLab'


def runtime_provenance() -> dict:
    """Validate loaded packages; source revisions come from launcher-checked image labels.

    The official Dockerfile excludes Git metadata. Image build pins are recorded
    explicitly as such, not reported as a successful Git inspection of imports.
    """
    if not PYTHON.is_file():
        raise RuntimeError('Arena evaluation requires the AutoData container. Use '
                           'scripts/door/autodata/run_container.sh --exec '
                           '/isaac-sim/python.sh <script> --output /outputs/<run>')
    pins = {name: os.environ.get(f'ARX_{name.upper()}_COMMIT')
            for name, _, _ in source_checkouts()}
    for name, _, expected in source_checkouts():
        if pins[name] != expected:
            raise RuntimeError(f'Missing or different {name} image build pin; use run_container.sh')
    for module, root in (('isaaclab', LAB), ('isaaclab_arena', ARENA), ('autodata_core', AUTODATA)):
        spec = importlib.util.find_spec(module)
        if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
            raise RuntimeError(f'{module} is not loaded from the pinned image: {spec}')
    # NVIDIA's binary container has no isaacsim pip distribution metadata.
    sim_build = (PYTHON.parent / 'VERSION').read_text().strip()
    if sim_build.split('-', 1)[0] != '6.1.0':
        raise RuntimeError(f'Isaac Sim container version differs: {sim_build}')
    versions = {name: importlib.metadata.version(name)
                for name in ('numpy', 'torch', 'autodata', 'isaaclab-arena')}
    expected = {'numpy': '2.3.1', 'torch': '2.11.0+cu128', 'autodata': '0.1.0',
                'isaaclab-arena': '0.3.0'}
    if versions != expected or sys.version_info[:2] != (3, 12):
        raise RuntimeError(f'AutoData runtime versions differ: {versions}, Python {sys.version}')
    return {'profile': 'autodata', 'python': sys.version.split()[0],
            'python_executable': sys.executable, 'packages': versions, 'isaac_sim_build': sim_build,
            'isaac_lab': (LAB / 'VERSION').read_text().strip(),
            'isaac_lab_root': str(LAB), 'source_evidence': 'launcher_checked_image_build_labels',
            'image_build_pins': pins}


def simulation_environment() -> dict[str, str]:
    """Retain Isaac Sim's container activation, never select the host local.json."""
    runtime_provenance()
    environment = os.environ.copy()
    environment.update(ISAAC_SIM_PYTHON=str(PYTHON), PYTHONUNBUFFERED='1')
    return environment


def replay_inputs(run: Path) -> tuple[Path, str]:
    """Require generation metadata rather than silently choosing another physics profile."""
    from arx_door.envs.physics import PHYSICS_PROFILES

    metadata = json.loads((run / 'run.json').read_text())
    physics = metadata['physics']
    if physics not in PHYSICS_PROFILES:
        raise ValueError(f'Invalid generation physics profile: {physics}')
    return (run / metadata['assets']).resolve(), physics
