#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Install the pinned PyTorch inference stack without touching Isaac or data venvs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.policy.config import DEFAULT_CONFIG, load_config, resource
from arx_door.runtime import clean_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--download-base-model', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    source = resource(config, 'official_source')
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.source_archive) as archive:
            archive.extractall(source.parent, filter='data')
        (source.parent / 'latest_review').rename(source)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    if actual != config['official_commit']:
        parser.error('existing source differs from the trained model; it will not be reset')
    environment = clean_environment()
    environment['UV_PROJECT_ENVIRONMENT'] = str(resource(config, 'policy_python').parent.parent)
    subprocess.run(['uv', 'sync', '--project', str(source), '--frozen', '--python', '3.12',
                    '--no-install-package', 'tensorrt-cu12',
                    '--no-install-package', 'tensorrt-cu12-libs',
                    '--no-install-package', 'tensorrt-cu12-bindings'],
                   env=environment, check=True)
    snapshot = '9ce19a195e423419c349abfc86fd07178b230561'
    cache = resource(config, 'hf_home') / 'hub/models--nvidia--Cosmos-Reason2-2B'
    destination = cache / 'snapshots' / snapshot
    if args.download_base_model:
        subprocess.run([str(resource(config, 'policy_python').parent / 'hf'), 'download',
                        'nvidia/Cosmos-Reason2-2B', '--revision', snapshot,
                        '--local-dir', str(destination), '--include', '*.json', '*.txt',
                        '*.safetensors'], env=environment, check=True)
    if not (destination / 'model.safetensors').is_file():
        parser.error('base snapshot missing; download it with --download-base-model')
    refs = cache / 'refs'
    refs.mkdir(parents=True, exist_ok=True)
    (refs / 'main').write_text(snapshot)
    print('Pinned PyTorch inference environment is ready. TensorRT is a separate optimization.')


if __name__ == '__main__':
    main()
