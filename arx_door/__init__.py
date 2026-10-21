# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""ARX R5A door-opening data pipeline.

Stages, in order (CLIs live in ``scripts/door/<stage>/``):

1. ``arx_door.scene``    – door/robot scene placement and USD assets
2. ``arx_door.demos``    – agent-recorded expert demonstrations (seeds) and their review
3. AutoData augmentation – the pinned official generator (``third_party/AutoData``) runs the
   door environment ``arx_door.envs``; ``arx_door.autodata`` holds its extension points
4. ``arx_door.dataset``  – LeRobot/GR00T N1.7 export contracts
5. ``arx_door.policy``   – closed-loop evaluation in the same ``arx_door.envs`` environment

Shared across stages: ``contract`` (task semantics), ``quality`` (physical and
video gates), ``kinematics``, ``runtime`` (pinned Isaac stacks and
interpreters), ``paths`` and ``files``. Assets live in the project ``assets/``.
"""
