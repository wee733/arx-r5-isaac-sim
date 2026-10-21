#!/usr/bin/env python3
"""Sample real scene conditions; each bundle is generation AND replay input."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from arx_door.autodata.conditions import prepare_condition, sample_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for condition in sample_conditions(args.count, args.seed):
        path = prepare_condition(args.assets, args.output / condition['condition_id'], condition)
        rows.append({'assets': str(path), **condition})
    (args.output / 'conditions.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(args.output / 'conditions.json')


if __name__ == '__main__':
    main()
