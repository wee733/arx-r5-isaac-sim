#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Inspect AutoData source data and runtime readiness; does not launch Sim."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from arx_door.autodata.preflight import checkout_evidence, runtime_evidence, source_evidence, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--autodata-root', type=Path)
    parser.add_argument('--source-only', action='store_true', help='check the pinned submodule without Docker or GPU checks')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = checkout_evidence(args.autodata_root) if args.source_only else runtime_evidence(args.autodata_root)
    if args.input:
        report['source'] = source_evidence(args.input)
    if args.report:
        write_report(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['source_ready' if args.source_only else 'ready_for_simulation'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
