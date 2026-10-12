# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Bind a reviewed NPZ source to its converted HDF5 before starting Kit.

Source paths can change during migration. Identity comes from the NPZ digest
at this input boundary and actual converted values, including derived poses
and segmentation. This module neither imports Kit nor rewrites any input.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .seed import (
    EEF_NAME, ENV_ID, ForwardKinematics, _base_pose, _door_object_poses,
    _macro_annotations, _validate_phase_order, load_seed, trajectory_sha256,
)
from ..contract import behavior_family_from_metadata, final_contact_object, normalize_behavior_family


POSE_ROUNDING_ATOL = 1e-6


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else value


def _json_attribute(group, key):
    try:
        return json.loads(_text(group.attrs[key]))
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise ValueError(f'input HDF5 has invalid or missing {group.name} attribute {key}') from error


def _same_array(demo, path, expected, *, atol=0):
    try:
        dataset = demo[path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError('field is not a dataset')
        actual = dataset[:]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'input HDF5 is missing numeric dataset {path}') from error
    if (actual.shape != expected.shape or actual.dtype.kind not in 'fiu'
            or not np.isfinite(actual).all()):
        raise ValueError(f'input HDF5 field {path} has invalid shape or numeric values')
    equal = (np.allclose(actual, expected, rtol=0, atol=atol)
             if atol else np.array_equal(actual, expected))
    if not equal:
        raise ValueError(f'input HDF5 field {path} does not match the reviewed NPZ source; reconvert')


def _same_keys(demo, path, expected):
    try:
        group = demo[path]
        valid = isinstance(group, h5py.Group) and set(group) == set(expected)
    except KeyError:
        valid = False
    if not valid:
        raise ValueError(f'input HDF5 group {path} does not match the source contract')


def bind_seed_hdf5(seed_dir: str | Path, input_file: str | Path,
                   kinematics: ForwardKinematics, *,
                   allow_agent_reviewed: bool = False) -> dict:
    """Check approval and complete content identity once per source/batch.

    The agent opt-in retains the existing full physical/video/visual review
    check in load_seed. Its checked digest is reused; user-approved sources
    receive one NPZ hash here. No HDF5, video, image or URDF hash is added.
    """
    seed = load_seed(seed_dir, allow_agent_reviewed=allow_agent_reviewed)
    quality = seed.metadata['quality_status']
    review = seed.metadata.get('agent_review') if quality == 'agent_reviewed_seed' else None
    source_hash = (review['trajectory']['sha256'] if review is not None else
                   trajectory_sha256(seed))
    count = len(seed.state)
    base = _base_pose(seed.metadata)
    actual_pose = np.asarray([base @ kinematics.fk(q[:6]) for q in seed.state], dtype=np.float32)
    target_pose = np.asarray([base @ kinematics.fk(q[:6]) for q in seed.action], dtype=np.float32)
    objects = _door_object_poses(seed)
    family = behavior_family_from_metadata(seed.metadata)
    boundaries, starts, terms = _macro_annotations(seed.phase, family)
    phase_names = _validate_phase_order(seed.phase, family)
    phase_index = {phase: index for index, phase in enumerate(phase_names)}
    fields = {
        'actions': seed.action.astype(np.float32),
        'obs/diagnostics/joint_state': seed.state.astype(np.float32),
        'obs/diagnostics/timestamp': seed.timestamp.astype(np.float64),
        'obs/diagnostics/door_joint_pos': seed.door.astype(np.float32),
        'obs/diagnostics/finger_contacts': seed.contacts.astype(np.float32),
        'obs/diagnostics/camera_clearance': seed.camera_clearance.astype(np.float32),
        'obs/diagnostics/arm_contacts': seed.arm_contacts.astype(np.float32),
        'obs/diagnostics/phase_id': np.asarray(
            [phase_index[str(value)] for value in seed.phase], dtype=np.uint8),
    }
    path = Path(input_file).expanduser().resolve()
    with h5py.File(path, 'r') as dataset:
        if ('data' not in dataset or not isinstance(dataset['data'], h5py.Group)
                or set(dataset['data']) != {'demo_0'}):
            raise ValueError('input HDF5 must contain exactly one source episode data/demo_0')
        data = dataset['data']
        demo = data['demo_0']
        if not isinstance(demo, h5py.Group):
            raise ValueError('input HDF5 data/demo_0 must be a group')
        if (_text(demo.attrs.get('schema')) != 'arx-door-skillgen-seed-v1'
                or demo.attrs.get('success') != True
                or demo.attrs.get('num_samples') != count or data.attrs.get('total') != count):
            raise ValueError('input HDF5 schema, success or episode frame count does not match its source')
        env = _json_attribute(data, 'env_args')
        if (not isinstance(env, dict) or env.get('env_name') != ENV_ID
                or env.get('type') != 2 or env.get('env_kwargs') != {}):
            raise ValueError('input HDF5 environment does not match the door seed contract')
        if _text(demo.attrs.get('source_sha256')) != source_hash:
            raise ValueError('input HDF5 NPZ source hash does not match its reviewed seed_dir')
        if (normalize_behavior_family(_text(demo.attrs.get('behavior_family'))) != family
                or ('behavior_family' in data.attrs
                    and normalize_behavior_family(_text(data.attrs['behavior_family'])) != family)):
            raise ValueError('input HDF5 behavior family does not match its source')
        if ('final_contact_object' in demo.attrs
                and _text(demo.attrs['final_contact_object']) != final_contact_object(family)):
            raise ValueError('input HDF5 final contact object does not match its source')
        stored_quality = _text(demo.attrs.get('source_quality_status'))
        # Migrated user-approved v1 seeds predate this attribute. Agent inputs
        # must always carry the full review declaration and pending human gate.
        if stored_quality != quality and not (quality == 'user_approved_seed' and stored_quality is None):
            raise ValueError('input HDF5 does not match its source approval status')
        if review is not None:
            if (_text(demo.attrs.get('human_review_status')) != 'pending'
                    or _json_attribute(demo, 'source_agent_review') != review):
                raise ValueError('input HDF5 does not match its agent-reviewed source evidence')
        if _json_attribute(demo, 'macro_boundaries') != {key: list(value) for key, value in boundaries.items()}:
            raise ValueError('input HDF5 macro boundaries do not match its source phases')
        try:
            diagnostics = demo['obs/diagnostics']
            if not isinstance(diagnostics, h5py.Group):
                raise TypeError('diagnostics is not a group')
        except (KeyError, TypeError) as error:
            raise ValueError('input HDF5 is missing its diagnostics group') from error
        if _json_attribute(diagnostics, 'phase_names') != list(phase_names):
            raise ValueError('input HDF5 phase names do not match its source phases')
        for name, expected in fields.items():
            _same_array(demo, name, expected)
        poses = {'eef_pose': {EEF_NAME: actual_pose},
                 'target_eef_pose': {EEF_NAME: target_pose}, 'object_pose': objects}
        for group, values in poses.items():
            prefix = f'obs/datagen_info/{group}'
            _same_keys(demo, prefix, values)
            for name, expected in values.items():
                _same_array(demo, f'{prefix}/{name}', expected, atol=POSE_ROUNDING_ATOL)
        for group, values in (('subtask_start_signals', starts), ('subtask_term_signals', terms)):
            prefix = f'obs/datagen_info/{group}'
            _same_keys(demo, prefix, values)
            for name, expected in values.items():
                _same_array(demo, f'{prefix}/{name}', expected)
        historical_path = _text(demo.attrs.get('source_path'))
    identity = {
        'schema': 'arx-door-source-input-binding-v1',
        'seed_dir': str(seed.path), 'input_file': str(path), 'quality_status': quality,
        'source_sha256': source_hash, 'frames': count,
        'behavior_family': family, 'final_contact_object': final_contact_object(family),
        'content_verified': True, 'derived_poses_verified': True,
        'pose_rounding_atol': POSE_ROUNDING_ATOL,
        'hdf5_recorded_source_path': historical_path,
        'source_path_matches_current_location': historical_path == str(seed.path),
    }
    if review is not None:
        identity.update(human_review_status='pending', agent_review=review)
    return identity
