# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""ARX extension points for the pinned NVIDIA AutoData submodule.

``register`` (env + planner profile, loaded by the official generate_dataset.py),
``planner`` (ARX cuRobo profile), ``source`` (expert seed -> AutoData source),
``accept`` (strict physical acceptance of a run), ``preflight`` (pins, inputs).
The environment itself lives in :mod:`arx_door.envs`. Simulator imports stay lazy.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY = json.loads((PROJECT_ROOT / 'dependencies/autodata.lock.json').read_text())
AUTODATA_ROOT = PROJECT_ROOT / DEPENDENCY['source']['path']
AUTODATA_COMMIT = DEPENDENCY['source']['commit']
ARENA_COMMIT = DEPENDENCY['submodules']['arena']['commit']
LAB_COMMIT = DEPENDENCY['submodules']['lab']['commit']


def source_checkouts(root: Path | None = None) -> tuple[tuple[str, Path, str], ...]:
    """Pair each official checkout with its expected revision, outermost first."""
    root = AUTODATA_ROOT if root is None else Path(root)
    return (
        ('autodata', root, AUTODATA_COMMIT),
        ('arena', root / DEPENDENCY['submodules']['arena']['path'], ARENA_COMMIT),
        ('lab', root / DEPENDENCY['submodules']['lab']['path'], LAB_COMMIT),
    )
