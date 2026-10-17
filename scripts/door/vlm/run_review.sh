#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
# Run review_door_episodes.py in the GR00T data venv with the project FFmpeg.
#   scripts/door/vlm/run_review.sh <trial dirs | batch_results.json> [--every N] [--dry-run]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${GROOT_DATA_PYTHON:-${repo_root}/generated/runtime/groot_data/.venv/bin/python}"
tools="${repo_root}/generated/runtime/tools/usr"
unset PYTHONPATH PYTHONHOME
if [[ -d "${tools}/bin" ]]; then
  export PATH="${tools}/bin:${PATH}"
  export LD_LIBRARY_PATH="${tools}/lib/x86_64-linux-gnu"
fi
exec "${python_bin}" "${repo_root}/scripts/door/vlm/review_door_episodes.py" "$@"
