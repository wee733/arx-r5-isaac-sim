#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Convert an ARX expert seed into an AutoData annotated source (robot-root frame)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True, help='expert seed HDF5 from scripts/door/demos')
    parser.add_argument('--assets', type=Path, required=True,
                        help='manifest.json of the bundle prepared from the same seed scene')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from arx_door.autodata.source import convert_seed
    from arx_door.runtime import planning_urdf
    report = convert_seed(args.seed, args.assets, args.output, planning_urdf())
    for name, episode in report['episodes'].items():
        print(name, 'base vs bundle:', episode['base_vs_bundle_placement'])
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
