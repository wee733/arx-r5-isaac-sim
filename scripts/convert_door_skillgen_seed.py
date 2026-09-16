#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Convert one approved door attempt to an Isaac Lab SkillGen HDF5 seed."""

import argparse
import json
from pathlib import Path

from arx_r5_isaac_sim_bringup.door_skillgen.seed import (
    build_skillgen_arrays,
    conversion_report,
    load_seed,
    write_skillgen_hdf5,
)
from arx_r5_isaac_sim_bringup.door_teaching_kinematics import ArmKinematics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTION = (
    ROOT.parent
    / 'arx-r5-moveit'
    / 'isaac_ros_manipulation_arx_r5a_robot_description'
)


def main() -> int:
    """Validate, enrich, and atomically write one approved seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--urdf',
        type=Path,
        default=DEFAULT_DESCRIPTION / 'urdf/r5a_cumotion.urdf',
    )
    args = parser.parse_args()
    arrays = build_skillgen_arrays(
        load_seed(args.input),
        ArmKinematics(args.urdf),
    )
    write_skillgen_hdf5(arrays, args.output)
    report = conversion_report(arrays, args.output)
    report_path = args.output.with_suffix('.conversion.json')
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
