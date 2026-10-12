# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0

"""Small JSON helpers shared by package modules and standalone scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, value: Any, **dumps_options) -> Path:
    """Atomically publish JSON; the per-process temporary avoids writer collisions."""
    path = Path(path)
    options = {'indent': 2, 'ensure_ascii': False, 'allow_nan': False, **dumps_options}
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, **options) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
