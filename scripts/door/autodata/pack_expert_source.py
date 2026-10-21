#!/usr/bin/env python3
"""Make a lightweight annotated source without loading RGB into AutoData's pool.

The full accepted expert with synchronized images and diagnostics is retained.
This source is an input to generation, not a replacement training dataset.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    result = json.loads((run/'result.json').read_text())
    if not result['success']:
        raise ValueError('cannot pack a failed expert')
    with h5py.File(run/'accepted.hdf5', 'r') as src, h5py.File(run/'source.hdf5', 'x') as dst:
        data = dst.create_group('data')
        for key, value in src['data'].attrs.items():
            data.attrs[key] = value
        for name, original in src['data'].items():
            if not bool(original['obs/diagnostics/runtime_accepted'][-1].item()):
                raise ValueError('expert has no final acceptance')
            episode = data.create_group(name)
            for key, value in original.attrs.items():
                episode.attrs[key] = value
            for key in ('actions', 'initial_state', 'obs/datagen_info', 'obs/joint_pos'):
                parent, _, leaf = key.rpartition('/')
                src.copy(original[key], episode.require_group(parent) if parent else episode, name=leaf)
            diagnostics = episode.require_group('obs/diagnostics')
            diagnostics.attrs['phase_names'] = original['obs/diagnostics'].attrs['phase_names']
            for key in ('phase_id', 'timestamp', 'action_index'):
                src.copy(original['obs/diagnostics/'+key], diagnostics, name=key)
            actions = original['actions'][:]
            if actions.shape[1] != 8 or not np.isfinite(actions).all():
                raise ValueError('invalid root-frame absolute-pose commands')
    report = dict(schema='arx-fresh-expert-source-v1', full_recording='accepted.hdf5',
                  source_demonstrations=[], strategy=result['strategy'], seed=result['seed'],
                  frames=result['frames'], action_frame='robot_root', quaternion_order='xyzw',
                  gripper_unit='single_finger_metres', physics_hz=120, control_hz=30,
                  images_retained_in='accepted.hdf5', independent_replay_required=True)
    (run/'source.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
