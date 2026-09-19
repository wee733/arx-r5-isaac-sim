# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Guard the official-loader entrypoint before any statistics are rewritten."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_validator():
    """Load the entrypoint without importing its heavy official runtime."""
    spec = importlib.util.spec_from_file_location('door_n17_validator', ROOT / 'scripts/validate_door_n17.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def official_registry():
    """Restore the upstream process-global registry after each real import."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    original = dict(MODALITY_CONFIGS)
    MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
    try:
        yield
    finally:
        MODALITY_CONFIGS.clear()
        MODALITY_CONFIGS.update(original)


@pytest.fixture
def official_config(official_registry):
    """Use the production configuration's real upstream dataclasses."""
    spec = importlib.util.spec_from_file_location('door_n17_test_config', ROOT / 'configs/door_wrist_n17.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.door_wrist_config


def test_short_episode_is_rejected_before_official_runtime_import(tmp_path):
    """A six-frame recording cannot supply a 16-step action chunk."""
    meta = tmp_path / 'meta'
    meta.mkdir()
    (meta / 'info.json').write_text(json.dumps({'total_episodes': 1}))
    (meta / 'n17_validation.json').write_text(json.dumps(
        {'official_n1_7_loader_verified': True}))
    (meta / 'episodes.jsonl').write_text(json.dumps(
        {'episode_index': 0, 'length': 6, 'tasks': ['open the door']}) + '\n')
    command = [sys.executable, str(ROOT / 'scripts/validate_door_n17.py'),
               '--dataset', str(tmp_path), '--groot-root', str(tmp_path / 'absent')]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'at least 16 frames' in result.stderr
    assert not (meta / 'stats.json').exists()
    assert json.loads((meta / 'n17_validation.json').read_text())[
        'official_n1_7_loader_verified'] is False


def test_quarantined_visual_data_cannot_be_rehabilitated_by_loader(tmp_path):
    """A numeric loader pass must never erase a separate visual rejection."""
    meta = tmp_path / 'meta'
    meta.mkdir()
    (meta / 'door_export_report.json').write_text(json.dumps({
        'official_n1_7_loader_verified': True, 'eligible_for_training': False,
    }))
    warning = 'DO NOT TRAIN: visual quarantine\n'
    (tmp_path / 'README.md').write_text(warning)
    with pytest.raises(ValueError, match='quarantined'):
        load_validator().validate(tmp_path, tmp_path / 'absent', tmp_path / 'absent.py')
    assert (tmp_path / 'README.md').read_text() == warning
    report = json.loads((meta / 'door_export_report.json').read_text())
    assert report['eligible_for_training'] is False
    assert report['official_n1_7_loader_verified'] is False


def test_relative_gripper_is_rejected_before_statistics_are_rewritten(tmp_path, official_registry):
    """A config accepted by the raw loader must not relabel gripper targets."""
    meta = tmp_path / 'meta'
    meta.mkdir()
    (meta / 'episodes.jsonl').write_text(json.dumps(
        {'episode_index': 0, 'length': 16, 'tasks': ['open the door']}) + '\n')
    (meta / 'info.json').write_text(json.dumps(
        {'codebase_version': 'v2.1', 'fps': 30, 'total_episodes': 1}))
    (meta / 'modality.json').write_text(json.dumps({'video': {'wrist': {}}}))
    sentinel = '{"unmodified_statistics": true}\n'
    (meta / 'stats.json').write_text(sentinel)
    (meta / 'door_export_report.json').write_text(json.dumps(
        {'official_n1_7_loader_verified': True}))
    config_path = tmp_path / 'relative_gripper.py'
    config_path.write_text((ROOT / 'configs/door_wrist_n17.py').read_text().replace(
        'ActionRepresentation.ABSOLUTE', 'ActionRepresentation.RELATIVE'))
    with pytest.raises(ValueError, match='config.*gripper'):
        load_validator().validate(tmp_path, ROOT / 'generated/runtime/groot_n17_source', config_path)
    assert (meta / 'stats.json').read_text() == sentinel
    assert not (meta / 'stats_history').exists()
    assert json.loads((meta / 'door_export_report.json').read_text())[
        'official_n1_7_loader_verified'] is False


@pytest.mark.parametrize('explicit_state_keys', [False, True])
def test_approved_official_configuration_and_same_name_state_keys_are_accepted(
        explicit_state_keys, official_config):
    """Explicitly naming the same reference state preserves arm semantics."""
    config = official_config
    if explicit_state_keys:
        for key, action in zip(config['action'].modality_keys, config['action'].action_configs):
            action.state_key = key
    load_validator().validate_config_contract(config)


@pytest.mark.parametrize('mutation', [
    'video_time', 'state_time', 'action_order', 'action_horizon', 'language_key',
    'arm_representation', 'gripper_type', 'arm_format', 'state_key', 'embedding', 'unknown_field',
])
def test_changed_sampling_or_action_conversion_contract_is_rejected(mutation, official_config):
    """The acceptance report must cover the configuration actually consumed."""
    from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType

    config = official_config
    if mutation == 'video_time':
        config['video'].delta_indices = [-1, 0]
    elif mutation == 'state_time':
        config['state'].delta_indices = [-1]
    elif mutation == 'action_order':
        config['action'].modality_keys.reverse()
    elif mutation == 'action_horizon':
        config['action'].delta_indices = list(range(15))
    elif mutation == 'language_key':
        config['language'].modality_keys = ['annotation.language.other']
    elif mutation == 'arm_representation':
        config['action'].action_configs[0].rep = ActionRepresentation.ABSOLUTE
    elif mutation == 'gripper_type':
        config['action'].action_configs[1].type = ActionType.EEF
    elif mutation == 'arm_format':
        config['action'].action_configs[0].format = ActionFormat.XYZ_ROTVEC
    elif mutation == 'state_key':
        config['action'].action_configs[0].state_key = 'gripper'
    elif mutation == 'embedding':
        config['state'].sin_cos_embedding_keys = ['single_arm']
    else:
        config['action'].future_conversion_option = True
    with pytest.raises(ValueError, match='config'):
        load_validator().validate_config_contract(config)
