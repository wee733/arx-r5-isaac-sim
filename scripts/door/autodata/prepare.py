#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Extract portable Arena assets without editing the recorded source USD."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from arx_door.autodata.arena.assets import prepare_assets
    manifest = prepare_assets(args.scene, args.output)
    print(manifest)


if __name__ == '__main__':
    main()
