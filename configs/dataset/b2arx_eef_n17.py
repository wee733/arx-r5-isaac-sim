"""30 Hz live R5a mount-frame pose control; gripper is one finger in metres."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

door_config = {
    'video': ModalityConfig(delta_indices=[0], modality_keys=['wrist', 'ego']),
    'state': ModalityConfig(delta_indices=[0], modality_keys=['eef', 'gripper', 'joints']),
    'action': ModalityConfig(delta_indices=list(range(16)), modality_keys=['eef', 'gripper'], action_configs=[
        ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,
                     format=ActionFormat.XYZ_ROTVEC, state_key='eef'),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                     format=ActionFormat.DEFAULT, state_key='gripper'),
    ]),
    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description']),
}
register_modality_config(door_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
