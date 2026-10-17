# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Data v2 ablation: wrist RGB only, per-frame stage language.

State and action on disk are absolute: six arm radians and joint7 metres.
The official processor makes each future arm target relative to current state.
Validate with scripts/door/dataset/validate_door_n17.py --config <this file>.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)


door_config = {
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
    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.stage_description']),
}

register_modality_config(door_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
