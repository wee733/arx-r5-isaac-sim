#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Keep the official PolicyClient in the GR00T venv, away from Kit libraries."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from arx_door.policy.config import DEFAULT_CONFIG, load_config
from arx_door.policy.contract import DEFAULT_LANGUAGE_KEY, action_chunk, observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        from gr00t.policy.server_client import PolicyClient
        with PolicyClient(config['host'], config['port'], timeout_ms=60000) as client:
            print(json.dumps({'ready': client.ping()}), file=protocol, flush=True)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request['op'] == 'reset':
                        client.reset()
                    elif request['op'] == 'infer':
                        with np.load(request['input'], allow_pickle=False) as arrays:
                            views = {key.removeprefix('rgb_'): arrays[key]
                                     for key in arrays.files if key.startswith('rgb_')}
                            obs = observation(views or arrays['rgb'], arrays['state'], request['language'],
                                              language_key=config.get('language_key', DEFAULT_LANGUAGE_KEY))
                        reply, _ = client.get_action(obs)
                        np.save(request['output'], action_chunk(reply), allow_pickle=False)
                    else:
                        raise ValueError('unknown bridge operation')
                    response = {'ok': True}
                except Exception as exc:  # noqa: BLE001 - return upstream errors across IPC.
                    response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                print(json.dumps(response), file=protocol, flush=True)


if __name__ == '__main__':
    main()
