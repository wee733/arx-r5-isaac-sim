# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""ARX door Policy API boundary, with already-decoded absolute targets."""

from __future__ import annotations

import numpy as np

DEFAULT_LANGUAGE_KEY = 'annotation.human.task_description'


def observation(rgb, state: np.ndarray, language: str, *,
                language_key: str = DEFAULT_LANGUAGE_KEY) -> dict:
    """Keep RGB views and one 6-radian/1-metre state in training order.

    ``rgb`` is the wrist frame or a ``{view: frame}`` mapping that includes it.
    """
    views = dict(rgb) if isinstance(rgb, dict) else {'wrist': rgb}
    if 'wrist' not in views:
        raise ValueError('the wrist view is required')
    for name, frame in views.items():
        frame = np.asarray(frame)
        if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
            raise ValueError(f'{name} RGB must be uint8 (480, 640, 3)')
        views[name] = np.ascontiguousarray(frame)[None, None]
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('state must be finite (7,): six radians, joint7 metres')
    if not isinstance(language, str) or not language.strip():
        raise ValueError('language must be nonempty text')
    return {
        'video': views,
        'state': {'single_arm': state[None, None, :6].copy(),
                  'gripper': state[None, None, 6:].copy()},
        'language': {language_key: [[language]]},
    }


def action_chunk(reply: dict) -> np.ndarray:
    """Validate and concatenate decoded absolute actions; never add state again."""
    if set(reply) != {'single_arm', 'gripper'}:
        raise ValueError('expected single_arm and gripper action streams')
    arm, grip = np.asarray(reply['single_arm']), np.asarray(reply['gripper'])
    if arm.shape != (1, 16, 6) or grip.shape != (1, 16, 1):
        raise ValueError(f'expected (1,16,6) and (1,16,1), got {arm.shape}, {grip.shape}')
    result = np.concatenate((arm[0], grip[0]), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('model produced nonfinite actions')
    # Allow tiny regression overshoot; the existing simulator gripper term
    # clamps to [0, .044] m. Larger values indicate a broken unit/scale contract.
    if np.any(result[:, 6] < -.001) or np.any(result[:, 6] > .045):
        raise ValueError('gripper commands exceed the metre-valued ARX range')
    return result
