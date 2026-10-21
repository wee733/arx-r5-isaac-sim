#!/usr/bin/env python3
"""Prepare a separate joint-command replay candidate; this does not approve it.

Take the first of the four actual IK actuator targets for each 30 Hz command.
Holding that target differs from recomputing differential IK at 120 Hz, so a
fresh replay_check.py must pass before these commands can be training actions.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / 'joint_replay_candidate'
    output.mkdir(exist_ok=False)
    metadata = json.loads((run / 'run.json').read_text())
    metadata.update(assets=os.path.relpath((run / metadata['assets']).resolve(), output),
                    input=os.path.relpath((run / metadata['input']).resolve(), output),
                    action_mapping='first_recorded_ik_substep_target_held_at_30hz',
                    source_run='..', training_approved=False)
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    # The filename is the input expected by replay_check, not an export approval.
    with h5py.File(run / 'accepted.hdf5', 'r') as src, h5py.File(output / 'accepted.hdf5', 'x') as dst:
        data = dst.create_group('data')
        for key, value in src['data'].attrs.items():
            data.attrs[key] = value
        for name, episode in src['data'].items():
            targets = episode['obs/diagnostics/joint_target_substeps'][:]
            if targets.shape != (len(episode['actions']), 4, 7) or not np.isfinite(targets).all():
                raise ValueError(f'{name}: missing finite four-substep target record')
            group = data.create_group(name)
            group['actions'] = targets[:, 0].astype(np.float32)
            group['obs/joint_pos'] = episode['obs/joint_pos'][:]
            rows.append({'demo': name, 'frames': len(targets),
                         'max_within_step_target_change': np.max(np.abs(targets - targets[:, :1]), axis=(0, 1)).tolist()})
    (output / 'mapping.json').write_text(json.dumps({'episodes': rows, 'independent_replay_required': True}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
