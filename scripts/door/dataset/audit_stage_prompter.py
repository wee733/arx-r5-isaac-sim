#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Replay a stage-labelled export through the deployable stage prompter.

Stage language is only usable if the inference-time prompter, which sees
only commands, finger width and arm joints, reproduces the aligned labels.
Each frame's prompt is decided before that frame's command, as in rollout.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.policy.guards import GripperGate, RuleStagePrompter


def audit(dataset: Path) -> dict:
    import pyarrow.parquet as pq

    info = json.loads((dataset / 'meta/info.json').read_text())
    agreement, lags = [], {}
    for index in range(info['total_episodes']):
        name = f'episode_{index:06d}'
        table = pq.read_table(dataset / info['data_path'].format(
            episode_chunk=index // info['chunks_size'], episode_index=index)).to_pydict()
        state, action = np.asarray(table['observation.state']), np.asarray(table['action'])
        with np.load(dataset / 'diagnostics' / f'{name}.npz') as diagnostics:
            if 'stage' not in diagnostics.files:
                raise ValueError(f'{name}: export with --stage-language first')
            labels = diagnostics['stage']
        gate, prompter, predicted = GripperGate(), RuleStagePrompter(), []
        for row, command in zip(state, action):
            predicted.append(prompter.stage)
            gate(float(command[6]))
            prompter.update(row, float(command[6]), gate.closed)
        predicted = np.asarray(predicted)
        agreement.append(float((predicted == labels).mean()))
        for stage in range(1, int(labels.max()) + 1):
            if stage in predicted:
                lags.setdefault(stage, []).append(
                    int(np.argmax(predicted >= stage)) - int(np.argmax(labels >= stage)))
            else:
                lags.setdefault(stage, []).append(None)
    return {
        'episodes': len(agreement),
        'frame_agreement_mean': float(np.mean(agreement)),
        'frame_agreement_min': float(np.min(agreement)),
        'switch_lag_frames': {str(stage): {
            'missed': sum(value is None for value in values),
            'min': min(v for v in values if v is not None),
            'median': float(np.median([v for v in values if v is not None])),
            'max': max(v for v in values if v is not None)} for stage, values in lags.items()},
        'lag_sign': 'positive = prompter switches after the aligned label',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset.resolve()), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
