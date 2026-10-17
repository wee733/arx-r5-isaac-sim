#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Load the original GR00T policy, verify real inputs, optionally serve it."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from arx_door.files import write_json
from arx_door.policy.config import DEFAULT_CONFIG, load_config, resource
from arx_door.policy.contract import DEFAULT_LANGUAGE_KEY, action_chunk, observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'serve'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'loading', 'checkpoint': str(resource(config, 'checkpoint')),
               'official_commit': config['official_commit'], 'scope': 'local_gpu_inference',
               'closed_loop_success_measured': False}
    started = time.monotonic()
    write_json(output / 'receipt.json', receipt)
    try:
        import pandas as pd
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy
        from gr00t.policy.server_client import PolicyServer
        from PIL import Image

        actual = subprocess.check_output(
            ['git', '-C', str(resource(config, 'official_source')), 'rev-parse', 'HEAD'],
            text=True).strip()
        if actual != config['official_commit']:
            raise ValueError('official source differs from the training revision')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; CPU fallback is disabled')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.set_num_threads(4)
        policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                             model_path=str(resource(config, 'checkpoint')),
                             device='cuda:0', strict=True)
        policy.model.action_head.num_inference_timesteps = config['denoising_steps']
        modality = policy.get_modality_config()
        video_keys = config.get('video_keys', ['wrist'])
        language_key = config.get('language_key', DEFAULT_LANGUAGE_KEY)
        expected = {'video': video_keys, 'state': ['single_arm', 'gripper'],
                    'action': ['single_arm', 'gripper'], 'language': [language_key]}
        if any(modality[k].modality_keys != v for k, v in expected.items()):
            raise ValueError(f'loaded modality keys differ from the configured contract {expected}')
        if len(modality['action'].delta_indices) != config['effective_action_horizon']:
            raise ValueError('checkpoint does not expose 16 effective action steps')
        statistics = deepcopy(policy.processor.state_action_processor.statistics['new_embodiment'])
        training = json.loads((resource(config, 'checkpoint') /
                               'experiment_cfg/dataset_statistics.json').read_text())['new_embodiment']
        if statistics != training:
            raise ValueError('loaded normalization differs from training statistics')
        receipt.update(gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__,
                       loading_seconds=time.monotonic() - started,
                       normalization='saved_training_statistics',
                       effective_action_horizon=len(modality['action'].delta_indices))
        dataset = resource(config, 'dataset')
        info = json.loads((dataset / 'meta/info.json').read_text())
        parquet = dataset / info['data_path'].format(episode_chunk=0, episode_index=0)
        videos = {key: dataset / info['video_path'].format(
            episode_chunk=0, episode_index=0, video_key=f'observation.images.{key}')
            for key in video_keys}
        rows = pd.read_parquet(parquet)
        samples = []
        for frame in (0, len(rows) // 2, len(rows) - 16):
            views = {}
            for key, video in videos.items():
                decoded = subprocess.check_output(
                    ['ffmpeg', '-v', 'error', '-threads', '1', '-ss', str(frame / info['fps']),
                     '-i', str(video), '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
                views[key] = np.frombuffer(decoded, dtype=np.uint8).reshape(480, 640, 3).copy()
            rgb = views['wrist']
            state = np.asarray(rows.iloc[frame]['observation.state'], dtype=np.float32)
            obs = observation(views, state, config['language'], language_key=language_key)
            torch.cuda.synchronize()
            tick = time.monotonic()
            reply, _ = policy.get_action(obs)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            chunk = action_chunk(reply)
            np.savez(output / f'frame_{frame:06d}.npz', state=state, prediction=chunk)
            Image.fromarray(rgb).save(output / f'frame_{frame:06d}.png')
            samples.append({'frame': frame, 'seconds': elapsed, 'shape': list(chunk.shape)})
            print(f'GPU_INFERENCE frame={frame} seconds={elapsed:.3f}', flush=True)
        if statistics != policy.processor.state_action_processor.statistics['new_embodiment']:
            raise RuntimeError('inference changed training normalization')
        receipt.update(status='passed', samples=samples,
                       allocated_vram_mib=torch.cuda.memory_allocated() / 2**20,
                       peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                       elapsed_seconds=time.monotonic() - started)
        write_json(output / 'receipt.json', receipt)
        print('LOCAL_GPU_POLICY_VERIFIED', flush=True)
        if args.mode == 'serve':
            policy.reset()
            with PolicyServer(policy, host=config['host'], port=config['port']) as server:
                write_json(output / 'ready.json', {'host': config['host'], 'port': config['port']})
                print('POLICY_SERVER_READY', flush=True)
                server.run()
            (output / 'ready.json').unlink(missing_ok=True)
            write_json(output / 'stopped.json', {'status': 'stopped'})
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                       elapsed_seconds=time.monotonic() - started)
        write_json(output / 'receipt.json', receipt)
        raise


if __name__ == '__main__':
    main()
