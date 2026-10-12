# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Gym registration for the project-local ARX door SkillGen task."""

from __future__ import annotations

import gymnasium as gym

from ..demos.seed import ENV_ID


def register() -> str:
    """Register idempotently without importing Isaac Lab before AppLauncher."""
    if ENV_ID not in gym.registry:
        gym.register(
            id=ENV_ID,
            entry_point=(
                'arx_door.skillgen.env:'
                'DoorSkillGenEnv'
            ),
            disable_env_checker=True,
            kwargs={
                'env_cfg_entry_point': (
                    'arx_door.skillgen.env_cfg:'
                    'DoorSkillGenEnvCfg'
                ),
            },
        )
    return ENV_ID


__all__ = ['ENV_ID', 'register']
