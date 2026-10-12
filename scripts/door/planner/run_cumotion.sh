#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_dir="${CUMOTION_ENV_DIR:-${repo_root}/generated/runtime/cumotion}"
python_bin="${CUMOTION_PYTHON:-${runtime_dir}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python 3.12 executable not found: ${python_bin}." >&2
  echo "Run: python3 ${repo_root}/scripts/door/setup/setup_cumotion_runtime.py --wheel /path/to/official-cp312-wheel.whl" >&2
  echo 'For an existing isolated planner environment, set CUMOTION_PYTHON.' >&2
  exit 2
fi
# Neither Sim 5.1's Python 3.11 packages nor Sim 6.1's native libraries may
# leak into this process. Explicit extracted-wheel overrides remain supported.
unset PYTHONHOME LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CUMOTION_RUNTIME_DIR:-}"
export CUMOTION_RUNTIME_MANIFEST="${runtime_dir}/runtime.json"
exec "${python_bin}" -s "${repo_root}/scripts/door/planner/door_cumotion_worker.py" "$@"
