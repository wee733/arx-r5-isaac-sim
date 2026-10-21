# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Repository locations shared by the door pipeline (stdlib only).

Command-line tools launch each other as separate processes because each stage
needs a different interpreter (Isaac Sim venv, cuMotion venv). Keeping their
paths here means a script move only changes this table.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts/door'
GENERATED = ROOT / 'generated'
CONFIGS = ROOT / 'configs'
# Project assets: door/ (source door), arx_r5a/ (arm + cuMotion model),
# b2arx_robot/ (B2 with the mounted arm).
ASSETS = ROOT / 'assets'
ARX_DESCRIPTION = ASSETS / 'arx_r5a'

# Stage 1: scene
EXPORT_CUBOIDS = SCRIPTS / 'scene/export_door_cuboids.py'
# Stage 2: expert demonstrations (seeds); planned by the native cuMotion worker
TEACH_SESSION = SCRIPTS / 'demos/door_teach_session.py'
EXPERT_COLLECTOR = SCRIPTS / 'demos/collect_door_upright.py'
ENCODE_ATTEMPT = SCRIPTS / 'demos/encode_door_attempt.py'
CUMOTION_LAUNCHER = SCRIPTS / 'planner/run_cumotion.sh'
# Stage 3: augmentation runs AutoData's official generator (scripts/door/autodata)
AUTODATA_SCRIPTS = SCRIPTS / 'autodata'
