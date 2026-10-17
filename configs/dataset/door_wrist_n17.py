# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Single-wrist ARX door modality for pinned GR00T N1.7.

On disk both state and action are absolute: six arm radians and joint7 metres.
The official processor computes each future arm target relative to current state.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)


door_wrist_config = {
    'video': ModalityConfig(delta_indices=[0], modality_keys=['wrist']),
    'state': ModalityConfig(delta_indices=[0], modality_keys=['single_arm', 'gripper']),
    'action': ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=['single_arm', 'gripper'],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE,
                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    'language': ModalityConfig(
        delta_indices=[0], modality_keys=['annotation.human.task_description']),
}

register_modality_config(door_wrist_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
