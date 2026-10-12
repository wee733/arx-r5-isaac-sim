#!/usr/bin/env python3
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Send a bounded command to the local Isaac Sim teaching session."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', help='JSON command, or @path/to/request.json')
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    value = Path(args.command[1:]).read_text() if args.command.startswith('@') else args.command
    request = Request(f'http://127.0.0.1:{args.port}', data=json.dumps(json.loads(value)).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    print(json.dumps(result, indent=2))
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
