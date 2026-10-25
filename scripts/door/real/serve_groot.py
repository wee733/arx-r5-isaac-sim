#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Serve the released B2ARX EEF checkpoint using NVIDIA's saved processor."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from arx_door.policy.config import load_config, resource
from arx_door.real.contract import observation, from_policy_reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('serve', 'verify'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required; CPU fallback is disabled')
    torch.set_num_threads(4)
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                        model_path=str(resource(config, 'checkpoint')), device='cuda:0', strict=True)
    policy.model.action_head.num_inference_timesteps = config['denoising_steps']
    modalities = policy.get_modality_config()
    expected = {'video': ['wrist', 'ego'], 'state': ['eef', 'gripper', 'joints'],
                'action': ['eef', 'gripper'], 'language': ['annotation.human.task_description']}
    if any(modalities[k].modality_keys != keys for k, keys in expected.items()):
        raise ValueError('Expected the B2ARX EEF checkpoint, not the legacy joint-action model')
    if modalities['action'].delta_indices != list(range(16)):
        raise ValueError('Expected 16 decoded action steps')
    # Warm-up is deliberately synthetic and never connects to ROS or the robot.
    state = {'schema': 'arx_vla_observation_v1', 'calibrated': True,
             'session_id': 'warmup', 'observation_id': 1,
             'state': {'eef': [0.25, 0., 0.25, 0., 0., 0.], 'gripper': [0.02], 'joints': [0.]*6}}
    rgb = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in ('wrist', 'ego')}
    obs = observation(state, rgb, config['language'])
    times = []
    for _ in range(3):
        start = time.monotonic()
        reply, _ = policy.get_action(obs)
        torch.cuda.synchronize()
        times.append(time.monotonic()-start)
        from_policy_reply(reply, state, 1, execute_steps=16)
    print(json.dumps({'synthetic_warmup_only': True, 'inference_seconds': times,
                      'eef_shape': list(reply['eef'].shape), 'gripper_shape': list(reply['gripper'].shape),
                      'checkpoint': str(resource(config, 'checkpoint'))}), flush=True)
    policy.reset()
    if args.mode == 'serve':
        with PolicyServer(policy, host=config['host'], port=config['port']) as server:
            print('B2ARX_POLICY_READY', flush=True)
            server.run()


if __name__ == '__main__':
    main()
