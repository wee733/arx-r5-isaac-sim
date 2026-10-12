# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""ROS simulator model discovery and standalone Sim launch ABI contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arx_r5_isaac_sim_bringup.simulation import find_description_share


ROOT = Path(__file__).resolve().parents[2]


def _model(path):
    (path / 'urdf').mkdir(parents=True)
    (path / 'urdf/r5a_cumotion.urdf').write_text('<robot name="fixture"/>')
    return path


def test_explicit_model_precedes_discovery_and_invalid_choice_fails(tmp_path, monkeypatch):
    model = _model(tmp_path / 'chosen-model')
    monkeypatch.setenv('ARX_DESCRIPTION_DIR', str(model))
    assert find_description_share(None) == model
    cli_model = _model(tmp_path / 'cli-model')
    assert find_description_share(str(cli_model)) == cli_model
    monkeypatch.setenv('ARX_DESCRIPTION_DIR', str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='ARX_DESCRIPTION_DIR'):
        find_description_share(None)


@pytest.fixture
def sim_python(tmp_path):
    """Use an inert executable to inspect the wrapper without starting Kit."""
    executable = tmp_path / 'selected-python'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'if sys.argv[1:] == ["-"]:\n'
        '    sys.stdin.read()\n'
        '    print("\\t".join(os.environ[name] for name in '
        '("TEST_SIM_ROOT", "TEST_SIM_VERSION", "TEST_PYTHON_VERSION")))\n'
        'else:\n'
        '    print(json.dumps({"args": sys.argv[1:], '
        '"ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")}))\n',
    )
    executable.chmod(0o755)
    sim_root = tmp_path / 'isaacsim'
    bridge = sim_root / 'exts/isaacsim.ros2.bridge/jazzy/lib'
    bridge.mkdir(parents=True)
    core_bridge = sim_root / 'exts/isaacsim.ros2.core/jazzy/lib'
    core_bridge.mkdir(parents=True)
    environment = {
        **os.environ,
        'ISAAC_SIM_PYTHON': str(executable),
        'TEST_SIM_ROOT': str(sim_root),
        'TEST_SIM_VERSION': '6.1.0.0',
        'TEST_PYTHON_VERSION': '3.12',
        'PYTHONPATH': '',
        'LD_LIBRARY_PATH': '',
        'ROS_DISTRO': 'jazzy',
    }
    return environment, {'5.1.0.0': bridge, '6.1.0.0': core_bridge}


@pytest.mark.parametrize('sim,python,path,accepted', [
    ('5.1.0.0', '3.11', '', True),
    ('5.1.0.0', '3.11', '/foreign/python3.12/site-packages', False),
    ('5.1.0.0', '3.12', '', False),
    ('6.1.0.0', '3.12', '/selected/python3.12/site-packages', True),
    ('6.1.0.0', '3.11', '', False),
])
def test_standalone_launcher_selects_abi_by_actual_sim(sim_python, sim, python, path, accepted):
    environment, _ = sim_python
    environment.update({
        'TEST_SIM_VERSION': sim,
        'TEST_PYTHON_VERSION': python,
        'PYTHONPATH': path,
    })
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/run_isaac_sim.sh'), '--no-ros', '--headless'],
        env=environment, text=True, capture_output=True,
    )
    assert (result.returncode == 0) is accepted, result.stderr
    if accepted:
        assert json.loads(result.stdout)['args'] == [
            '-m', 'arx_r5_isaac_sim_bringup.simulation', '--no-ros', '--headless',
        ]
    else:
        assert 'Python 3.' in result.stderr


@pytest.mark.parametrize('sim,python', [('5.1.0.0', '3.11'), ('6.1.0.0', '3.12')])
def test_standalone_launcher_keeps_selected_ros_bridge(sim_python, sim, python):
    environment, bridges = sim_python
    environment.update({'TEST_SIM_VERSION': sim, 'TEST_PYTHON_VERSION': python})
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/run_isaac_sim.sh'), '--headless'],
        env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ld_library_path'] == str(bridges[sim])
