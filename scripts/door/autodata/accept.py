#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Filter an AutoData run to physically accepted episodes (strict ARX acceptance)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='output directory of generate.sh')
    parser.add_argument('--output', type=Path, help='defaults to RUN/accepted.hdf5')
    args = parser.parse_args()
    from arx_door.autodata.accept import filter_accepted
    report = filter_accepted(args.run / 'generated.hdf5', args.output or args.run / 'accepted.hdf5')
    print(f"accepted {len(report['accepted'])}, rejected {len(report['rejected'])}")
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
