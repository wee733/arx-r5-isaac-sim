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
# Recorded scenes reference textures under this historical package path.
ASSETS = ROOT / 'arx_r5_isaac_sim_bringup/assets'

# Stage 1: scene
PREPARE_SCENE = SCRIPTS / 'scene/prepare_door_skillgen_scene.py'
EXPORT_CUBOIDS = SCRIPTS / 'scene/export_door_cuboids.py'
# Stage 2: expert demonstrations
TEACH_SESSION = SCRIPTS / 'demos/door_teach_session.py'
EXPERT_COLLECTOR = SCRIPTS / 'demos/collect_door_upright.py'
ENCODE_ATTEMPT = SCRIPTS / 'demos/encode_door_attempt.py'
# Stage 3: augmentation
GENERATE_SKILLGEN = SCRIPTS / 'skillgen/generate_door_skillgen.py'
# Campaign queues persist this exact launcher path in their argv; the old
# location (scripts/run_door_skillgen.sh) is kept as a forwarding wrapper.
SKILLGEN_BATCH_LAUNCHER = ROOT / 'scripts/run_door_skillgen.sh'
# Native planner (isolated cuMotion venv)
CUMOTION_LAUNCHER = SCRIPTS / 'planner/run_cumotion.sh'
