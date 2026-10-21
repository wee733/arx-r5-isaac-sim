#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Measure declared initial-state variation and actual trajectory diversity."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from export_autodata_lerobot import local_path, stage_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    rows, trajectories = [], []
    baseline = None
    for entry in entries:
        run = local_path(entry['run']).resolve()
        metadata = json.loads((run / 'run.json').read_text())
        with h5py.File(run / 'accepted.hdf5') as file, h5py.File(run / metadata['input']) as seed:
            for demo, group in file['data'].items():
                actions = group['actions'][:]
                state = group['obs/joint_pos'][:]
                joint = group['obs/diagnostics/joint_target_substeps'][:, 0]
                _, alignment = stage_alignment(actions, seed['data/demo_0'])
                prefix = alignment['generated_start']
                images = {name: group[f'obs/{name}_rgb'][:prefix] for name in ('wrist', 'ego')}
                if baseline is None:
                    baseline = (actions, state, joint, images)
                duplicate = [label for label, old_state, old_joint in trajectories
                             if state.shape == old_state.shape and np.array_equal(state, old_state)
                             and np.array_equal(joint, old_joint)]
                if duplicate:
                    raise ValueError(f'{run.name}/{demo} duplicates {duplicate}')
                observed_delta = float(state[0, 0] - baseline[1][0, 0])
                intended_delta = float(metadata['scene_parameters'].get('initial_joint1_delta_rad', 0))
                if not np.isclose(observed_delta, intended_delta, atol=1e-7, rtol=0):
                    raise ValueError(f'{run.name}: initial condition did not reach the simulator')
                rows.append({'run': str(run), 'demo': demo, 'split': entry['split'], 'seed': metadata['seed'],
                    'frames': len(actions), 'initial_joint1_delta_rad_declared': intended_delta,
                    'initial_joint1_delta_rad_measured': observed_delta,
                    'planned_prefix_frames': prefix, 'reused_source_skill_frames': alignment['matched_frames'],
                    'source_skill_fraction': alignment['matched_frames'] / len(actions),
                    'state_max_difference_from_baseline_per_joint': np.abs(state-baseline[1]).max(axis=0).tolist(),
                    'joint_target_max_difference_from_baseline_per_joint': np.abs(joint-baseline[2]).max(axis=0).tolist(),
                    'planned_prefix_camera_mean_abs_difference_0_255': {
                        k: float(np.abs(v.astype(np.int16)-baseline[3][k].astype(np.int16)).mean())
                        for k, v in images.items()},
                    'exact_duplicate': False})
                trajectories.append((run.name + '/' + demo, state, joint))
    args.output.write_text(json.dumps({'episodes': rows, 'exact_duplicates': 0,
        'expert_sources': 1, 'new_expert_skill_demonstrations': 0,
        'variation_scope': 'initial joint1 and planner transit only; same K001 interaction',
        'sufficient_for_vla_finetuning': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
