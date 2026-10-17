# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""ARX R5A door-opening data pipeline.

Stages, in order (CLIs live in ``scripts/door/<stage>/``):

1. ``arx_door.scene``    – door/robot scene assets and placement sampling
2. ``arx_door.demos``    – agent-recorded expert demonstrations and their review
3. ``arx_door.autodata`` – primary AutoData/Arena augmentation integration
   (runtime migration pending; ``skillgen`` retains the existing baseline and shared components)
4. dataset export        – LeRobot/GR00T N1.7 tools in ``scripts/door/dataset``
   (separate GR00T venv, so they import nothing from this package)

Shared across stages: ``contract`` (task semantics), ``quality`` (physical and
video gates), ``kinematics``, ``runtime`` (pinned Isaac stacks and
interpreters), ``paths`` and ``files``.
"""
