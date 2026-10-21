# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Resolve policy resources without changing other project runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from arx_door.runtime import ROOT, clean_environment

DEFAULT_CONFIG = ROOT / 'configs/policy/arx_knob_v2_n17.json'


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def resource(config: dict, key: str) -> Path:
    path = Path(config[key]).expanduser()
    return path if path.is_absolute() else ROOT / path


def policy_environment(config: dict) -> dict[str, str]:
    environment = clean_environment()
    tools = ROOT / 'generated/runtime/tools/usr'
    environment.update({
        'PYTHONPATH': os.pathsep.join((str(ROOT), str(resource(config, 'official_source')))),
        'HF_HOME': str(resource(config, 'hf_home')),
        'HF_HUB_CACHE': str(resource(config, 'hf_home') / 'hub'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'GROOT_PATCH_MISTRAL': '1',
        'NO_ALBUMENTATIONS_UPDATE': '1', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': '4', 'MPLBACKEND': 'Agg',
    })
    if tools.is_dir():
        environment['PATH'] = str(tools / 'bin') + os.pathsep + environment['PATH']
        environment['LD_LIBRARY_PATH'] = str(tools / 'lib/x86_64-linux-gnu')
    return environment
