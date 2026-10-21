#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Convert one approved door attempt to an expert seed HDF5 (joint actions, world-frame datagen).

scripts/door/autodata/convert_seed.py then turns the seed into an AutoData source.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from arx_door.demos.seed import (
    build_skillgen_arrays,
    conversion_report,
    load_seed,
    write_skillgen_hdf5,
)
from arx_door.kinematics import ArmKinematics
from arx_door.runtime import planning_urdf


def main() -> int:
    """Validate, enrich, and atomically write one approved seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-agent-reviewed-seeds', action='store_true',
                        help='explicitly accept agent-reviewed sources with checked evidence; human review stays pending')
    parser.add_argument(
        '--urdf',
        type=Path,
        help='ARX cuMotion URDF; defaults to ARX_DESCRIPTION_DIR or workspace discovery',
    )
    args = parser.parse_args()
    urdf = planning_urdf(args.urdf)
    arrays = build_skillgen_arrays(
        load_seed(args.input, allow_agent_reviewed=args.allow_agent_reviewed_seeds),
        ArmKinematics(urdf),
    )
    write_skillgen_hdf5(arrays, args.output)
    report = conversion_report(arrays, args.output)
    report['urdf_path'] = str(urdf)
    report_path = args.output.with_suffix('.conversion.json')
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
