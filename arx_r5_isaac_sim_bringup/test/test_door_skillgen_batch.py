# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Manifest and validation tests for the first SkillGen batch."""

import json

import h5py
import numpy as np
import pytest

from arx_r5_isaac_sim_bringup.door_skillgen.batch import (
    FIRST_BATCH_ANGLES_DEG,
    build_trial_specs,
    validate_batch_output,
)


def test_default_first_batch_is_fixed_and_reproducible(tmp_path):
    """Trial names do not depend on locale or random iteration order."""
    trials = build_trial_specs(FIRST_BATCH_ANGLES_DEG, tmp_path, seed=0)
    assert [trial.angle_deg for trial in trials] == [0.0, -3.0, 3.0, -5.0, 5.0]
    assert [trial.output_name for trial in trials] == [
        'angle_p000_0',
        'angle_m003_0',
        'angle_p003_0',
        'angle_m005_0',
        'angle_p005_0',
    ]
    assert len({trial.output_dir for trial in trials}) == 5


def test_validation_keeps_failures_but_rejects_incomplete_results(tmp_path):
    """A failed physics trial remains valid when its reason is structured."""
    root = tmp_path / 'batch'
    trials = build_trial_specs((0.0, -3.0), root, seed=0)
    for trial, success in zip(trials, (True, False)):
        trial.output_dir.mkdir(parents=True)
        for name in ('scene.usd', 'scene.json', 'cuboids.json', 'wrist.mp4', 'overview.mp4'):
            (trial.output_dir / name).write_bytes(b'x')
        with h5py.File(trial.output_dir / 'generated.hdf5', 'w') as file:
            demo = file.create_group('data/demo_0')
            demo.create_dataset('actions', data=np.zeros((20, 7)))
            demo.create_dataset('obs/joint_pos', data=np.zeros((20, 7)))
        result = {
            'schema': 'arx-door-skillgen-result-v1',
            'angle_deg': trial.angle_deg,
            'physics_executed': True,
            'success': success,
            'failure_stage': None if success else 'edge_interaction',
            'failure_reason': None if success else 'door_angle_below_threshold',
            'frames': 20,
            'hdf5_frames': 20,
            'hdf5_episodes': 1,
            'final_door_angle_deg': 30.0 if success else 17.0,
            'artifacts': {
                'scene': 'scene.usd',
                'scene_manifest': 'scene.json',
                'cuboids': 'cuboids.json',
                'hdf5': 'generated.hdf5',
                'wrist_video': 'wrist.mp4',
                'overview_video': 'overview.mp4',
            },
        }
        (trial.output_dir / 'result.json').write_text(json.dumps(result))
    report = validate_batch_output(root, trials)
    assert report['attempts'] == 2
    assert report['successes'] == 1
    assert report['failures'] == 1

    (trials[0].output_dir / 'generated.hdf5').write_bytes(b'not a dataset')
    with pytest.raises(ValueError, match='HDF5'):
        validate_batch_output(root, trials)

    (trials[1].output_dir / 'result.json').unlink()
    with pytest.raises(ValueError, match='result.json'):
        validate_batch_output(root, trials[1:])
