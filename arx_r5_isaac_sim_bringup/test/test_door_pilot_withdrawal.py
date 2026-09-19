# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""A rejected collection recipe must not send commands to a simulator."""
import importlib.util
from pathlib import Path
import sys

import pytest


def test_rejected_pilot_stops_before_simulator_access(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'scripts/collect_door_pilot.py'
    spec = importlib.util.spec_from_file_location('door_pilot_under_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', [str(script), '--session-output', str(tmp_path)])

    def unexpected_request(*args, **kwargs):
        pytest.fail('withdrawn recipe attempted to access the simulator')

    monkeypatch.setattr(module.urllib.request, 'urlopen', unexpected_request)
    with pytest.raises(RuntimeError, match='withdrawn'):
        module.main()
