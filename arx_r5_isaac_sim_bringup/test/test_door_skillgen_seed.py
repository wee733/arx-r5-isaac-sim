# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Tests for converting approved door attempts into SkillGen seeds."""

import json

import numpy as np
import pytest

from arx_r5_isaac_sim_bringup.door_skillgen.seed import (
    ENV_ID,
    build_skillgen_arrays,
    load_seed,
    write_skillgen_hdf5,
)


PHASES = np.asarray([
    'cumotion_approach_knob',
    'approach_knob',
    'grasp_knob',
    'unlock_round_knob',
    'pull_knob_for_gap',
    'release_knob',
    'retreat_from_knob',
    'cumotion_transfer_to_upright_edge',
    'approach_door_edge',
    'grasp_door_edge',
    'pull_door_by_edge',
    'hold_door_open',
])


class LinearKinematics:
    """Small deterministic FK implementation for converter contract tests."""

    def fk(self, joints):
        """Map the first three joints to translation with identity rotation."""
        result = np.eye(4)
        result[:3, 3] = np.asarray(joints)[:3]
        return result


def make_attempt(path, *, approved=True):
    """Write one complete, compact attempt fixture."""
    path.mkdir()
    count = len(PHASES)
    state = np.zeros((count, 7), dtype=np.float32)
    action = np.zeros((count, 7), dtype=np.float64)
    state[:, 0] = np.arange(count, dtype=np.float32) / 100
    action[:, 1] = np.arange(count, dtype=np.float64) / 50
    state[:, 6] = 0.028
    action[:, 6] = 0.028
    door = np.zeros((count, 3), dtype=np.float32)
    door[:, 0] = np.linspace(0, np.deg2rad(30), count)
    np.savez_compressed(
        path / 'trajectory.npz',
        state=state,
        action=action,
        timestamp=np.arange(count) / 30,
        phase=PHASES,
        door=door,
        contacts=np.zeros((count, 2, 2, 3), dtype=np.float32),
        camera_clearance=np.full(count, 0.1),
        arm_contacts=np.zeros((count, 6, 3), dtype=np.float32),
    )
    metadata = {
        'schema': 'arx-door-teaching-attempt-v1',
        'success': True,
        'quality_status': 'user_approved_seed' if approved else 'pending',
        'frames': count,
        'fps': 30,
        'scene_placement': {
            'seed': 7,
            'handle_world_xyz': [0.313, -0.3752, 0.928],
            'placement': {
                'position': [1.0, 2.0, 3.0],
                'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0],
                'angle_deg': 0.0,
            },
        },
    }
    (path / 'episode.json').write_text(json.dumps(metadata))
    return path


def test_unapproved_attempt_cannot_become_a_skillgen_seed(tmp_path):
    """Removing approval must stop data from entering source generation."""
    attempt = make_attempt(tmp_path / 'attempt', approved=False)

    with pytest.raises(ValueError, match='user-approved'):
        load_seed(attempt)


def test_seed_maps_to_two_non_overlapping_macro_skills(tmp_path):
    """Changing macro endpoints must not insert planning inside contact."""
    attempt = make_attempt(tmp_path / 'attempt')

    arrays = build_skillgen_arrays(load_seed(attempt), LinearKinematics())

    assert arrays.macro_boundaries == {
        'knob_interaction': (1, 7),
        'edge_interaction': (8, 12),
    }
    assert arrays.eef_pose['link6'].shape == (12, 4, 4)
    assert arrays.target_eef_pose['link6'].shape == (12, 4, 4)
    np.testing.assert_allclose(arrays.eef_pose['link6'][0, :3, 3], [1, 2, 3])
    np.testing.assert_allclose(arrays.target_eef_pose['link6'][1, :3, 3], [1, 2.02, 3])
    np.testing.assert_array_equal(
        arrays.subtask_start_signals['knob_interaction'].reshape(-1),
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    np.testing.assert_array_equal(
        arrays.subtask_term_signals['knob_interaction'].reshape(-1),
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    )
    np.testing.assert_array_equal(
        arrays.subtask_start_signals['edge_interaction'].reshape(-1),
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    )
    assert arrays.phase_names == tuple(PHASES.tolist())


def test_hdf5_matches_isaac_lab_mimic_layout(tmp_path):
    """Renaming nested datagen fields must break the loader-facing contract."""
    h5py = pytest.importorskip('h5py')
    attempt = make_attempt(tmp_path / 'attempt')
    arrays = build_skillgen_arrays(load_seed(attempt), LinearKinematics())
    output = tmp_path / 'seed.hdf5'

    write_skillgen_hdf5(arrays, output, env_name=ENV_ID)

    with h5py.File(output) as dataset:
        assert json.loads(dataset['data'].attrs['env_args'])['env_name'] == ENV_ID
        demo = dataset['data/demo_0']
        assert demo.attrs['num_samples'] == 12
        assert demo.attrs['success']
        np.testing.assert_allclose(demo['actions'][:], arrays.actions)
        assert demo['obs/datagen_info/eef_pose/link6'].shape == (12, 4, 4)
        assert demo['obs/datagen_info/object_pose/knob'].shape == (12, 4, 4)
        assert demo['obs/datagen_info/subtask_start_signals/edge_interaction'].shape == (12, 1)
        assert demo['obs/diagnostics/phase_id'].shape == (12,)
