#!/usr/bin/env bash
# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CUMOTION_PYTHON:-/var/lib/isaac-ros-cli/isaac-ros/bin/python}"
runtime_dir="${CUMOTION_RUNTIME_DIR:-${repo_root}/generated/runtime/cumotion}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python 3.12 executable not found: ${python_bin}; set CUMOTION_PYTHON." >&2
  exit 2
fi
if [[ ! -f "${runtime_dir}/cumotion/__init__.py" ]]; then
  echo "cuMotion runtime not found in ${runtime_dir}." >&2
  echo 'Extract the official cp312 cuMotion wheel here, or set CUMOTION_RUNTIME_DIR.' >&2
  exit 2
fi
# Do not inherit Isaac Sim's Python 3.11 packages into this Python 3.12 process.
export PYTHONPATH="${runtime_dir}"
exec "${python_bin}" "${repo_root}/scripts/door_cumotion_worker.py" "$@"
