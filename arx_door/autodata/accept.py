# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Keep only physically accepted episodes of an AutoData run.

AutoData latches the env ``success`` term, which is
:meth:`arx_door.envs.acceptance.RuntimeAcceptance.accepted`. Acceptance
violations are irreversible within an episode, so the last row of
``obs/diagnostics/runtime_accepted`` is the episode verdict; an episode latched
successful earlier but violating afterwards is dropped here.
"""

from __future__ import annotations

import json
from pathlib import Path

VERDICT = 'obs/diagnostics/runtime_accepted'


def physical_summary(episode) -> dict:
    """Explain the recorded verdict using the same gates, including pre-step extrema."""
    import numpy as np
    from arx_door.envs.acceptance import RuntimeAcceptance
    diagnostics = episode['obs/diagnostics']
    arrays = {key: diagnostics[key][:] for key in
              ('door_joint_pos', 'finger_contacts', 'arm_contacts', 'camera_clearance', 'panel_pose')}
    state = RuntimeAcceptance(1)
    for i in range(len(episode['actions'])):
        state.update([0], arrays['door_joint_pos'][i:i+1], arrays['finger_contacts'][i:i+1],
                     arrays['arm_contacts'][i:i+1], arrays['camera_clearance'][i:i+1],
                     panel_normal=arrays['panel_pose'][i:i+1, :3, 1])
    state.max_arm[0] = max(state.max_arm[0], float(np.max(diagnostics['runtime_max_arm_contact_n'][:])))
    state.min_clearance[0] = min(state.min_clearance[0], float(np.min(diagnostics['runtime_min_camera_clearance_m'][:])))
    result = state.summary(0)
    result['recorded_runtime_accepted'] = episode_verdict(episode)
    if result['accepted'] != result['recorded_runtime_accepted']:
        raise ValueError('diagnostic reconstruction disagrees with recorded physical verdict')
    return result


def episode_verdict(episode) -> bool:
    if VERDICT not in episode:
        raise ValueError(f'{episode.name} lacks {VERDICT}; was it recorded with the ARX recorders?')
    values = episode[VERDICT][:].reshape(-1)
    return bool(len(values) and values[-1])


def filter_accepted(generated: str | Path, output: str | Path) -> dict:
    """Copy accepted episodes to ``output`` (never overwritten) and return a report."""
    import h5py
    generated, output = Path(generated), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = {'schema': 'arx-autodata-acceptance-v1', 'generated': str(generated.resolve()),
              'accepted': [], 'rejected': [], 'episode_details': [], 'generation_failed': []}
    partial = output.with_name(output.name + '.partial')
    with h5py.File(generated, 'r') as source, h5py.File(partial, 'w') as target:
        data = target.create_group('data')
        for key, value in source['data'].attrs.items():
            data.attrs[key] = value
        for name in source['data']:
            accepted = episode_verdict(source['data'][name])
            report['accepted' if accepted else 'rejected'].append(name)
            report['episode_details'].append({'file': generated.name, 'demo': name,
                                               **physical_summary(source['data'][name])})
            if accepted:
                source.copy(source['data'][name], data, name=name)
        data.attrs['total'] = sum(len(data[name]['actions']) for name in report['accepted'])
    partial.rename(output)
    failed = generated.with_name(generated.stem + '_failed' + generated.suffix)
    if failed.exists():
        with h5py.File(failed, 'r') as source:
            for name, episode in source['data'].items():
                report['generation_failed'].append(name)
                report['episode_details'].append({'file': failed.name, 'demo': name,
                                                   **physical_summary(episode)})
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
