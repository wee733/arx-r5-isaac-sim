# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""ARX R5A door-opening data pipeline.

Stages, in order:

1. ``arx_door.scene``    – door/robot scene assets and placement sampling
2. ``arx_door.demos``    – agent-recorded expert demonstrations and their review
3. ``arx_door.skillgen`` – Isaac Lab SkillGen/Mimic augmentation of approved demos
4. ``arx_door.quality``  – physical/video quality gates for generated attempts

Shared: ``arx_door.contract`` (task semantics), ``arx_door.kinematics``,
``arx_door.runtime`` (pinned Isaac stacks and interpreters), ``arx_door.files``.
"""
