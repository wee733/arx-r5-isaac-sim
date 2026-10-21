#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Convert a fresh copy with LeRobot's official v2.1 to v3 tool; never replace GR00T input."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        raise ValueError('output must be outside the source dataset')
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v2.1':
        raise ValueError('source must be LeRobot v2.1')
    for name in (output.name, output.name + '_old', output.name + '_v30'):
        if (output.parent / name).exists():
            raise FileExistsError(output.parent / name)
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    shutil.copytree(source, output)
    convert_dataset(repo_id=output.name, root=output.parent, push_to_hub=False)
    # Upstream retains its v2 copy as OUTPUT_old. It is this run's valid input,
    # not a legacy training dataset. Keep it for inspection rather than deleting.
    print(output)


if __name__ == '__main__':
    main()
